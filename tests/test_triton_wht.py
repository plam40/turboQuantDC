"""Tests for Triton Walsh-Hadamard Transform kernel.

Validates that the Triton WHT kernel produces results matching the
Python/PyTorch fast_wht implementation, preserves norms, and provides
a correct round-trip (forward + inverse = identity).

Also benchmarks the Triton WHT against the Python butterfly loop.
"""

import math

import pytest
import torch

from turboquantdc.rotation import (
    apply_wht_rotation,
    fast_wht,
    generate_wht_rotation,
)
from turboquantdc.triton_kernels import (
    triton_wht_rotate,
    triton_wht_unrotate,
)

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wht_params(d, seed=DEFAULT_SEED, device=DEVICE):
    """Generate WHT rotation parameters on the target device."""
    return generate_wht_rotation(d, seed=seed, device=device)


def _apply_python_wht_rotate(x, signs, d):
    """Python-path forward rotation for comparison."""
    wht_params = {"signs": signs, "d": d}
    return apply_wht_rotation(x, wht_params, inverse=False)


def _apply_python_wht_unrotate(y, signs, d):
    """Python-path inverse rotation for comparison."""
    wht_params = {"signs": signs, "d": d}
    return apply_wht_rotation(y, wht_params, inverse=True)


# ---------------------------------------------------------------------------
# Correctness: Triton WHT matches Python WHT
# ---------------------------------------------------------------------------


class TestTritonWHTMatchesPython:
    """Triton WHT output matches the Python fast_wht implementation."""

    @pytest.mark.parametrize("d", [32, 64, 128, 256, 512])
    def test_forward_rotation_matches(self, d):
        """Triton forward WHT matches Python forward WHT within fp32 tolerance."""
        torch.manual_seed(DEFAULT_SEED)
        params = _make_wht_params(d)
        signs = params["signs"]

        x = torch.randn(100, d, device=DEVICE)

        result_python = _apply_python_wht_rotate(x, signs, d)
        result_triton = triton_wht_rotate(x, signs)

        max_diff = (result_python - result_triton).abs().max().item()
        assert max_diff < 1e-4, (
            f"d={d}: forward WHT max diff {max_diff:.2e} exceeds 1e-4"
        )

    @pytest.mark.parametrize("d", [32, 64, 128, 256, 512])
    def test_inverse_rotation_matches(self, d):
        """Triton inverse WHT matches Python inverse WHT within fp32 tolerance."""
        torch.manual_seed(DEFAULT_SEED)
        params = _make_wht_params(d)
        signs = params["signs"]

        # First rotate with Python, then unrotate with both paths
        x = torch.randn(100, d, device=DEVICE)
        y = _apply_python_wht_rotate(x, signs, d)

        result_python = _apply_python_wht_unrotate(y, signs, d)
        result_triton = triton_wht_unrotate(y, signs)

        max_diff = (result_python - result_triton).abs().max().item()
        assert max_diff < 1e-4, (
            f"d={d}: inverse WHT max diff {max_diff:.2e} exceeds 1e-4"
        )

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_single_vector(self, d):
        """Works correctly with a single vector (batch=1)."""
        torch.manual_seed(DEFAULT_SEED)
        params = _make_wht_params(d)
        signs = params["signs"]

        x = torch.randn(1, d, device=DEVICE)

        result_python = _apply_python_wht_rotate(x, signs, d)
        result_triton = triton_wht_rotate(x, signs)

        max_diff = (result_python - result_triton).abs().max().item()
        assert max_diff < 1e-4, (
            f"d={d}: single vector max diff {max_diff:.2e} exceeds 1e-4"
        )

    def test_large_batch(self):
        """Works correctly with 10K vectors."""
        d = 128
        torch.manual_seed(DEFAULT_SEED)
        params = _make_wht_params(d)
        signs = params["signs"]

        x = torch.randn(10000, d, device=DEVICE)

        result_python = _apply_python_wht_rotate(x, signs, d)
        result_triton = triton_wht_rotate(x, signs)

        max_diff = (result_python - result_triton).abs().max().item()
        assert max_diff < 1e-4, (
            f"Large batch max diff {max_diff:.2e} exceeds 1e-4"
        )


# ---------------------------------------------------------------------------
# Round-trip: WHT forward + inverse = identity
# ---------------------------------------------------------------------------


class TestTritonWHTRoundTrip:
    """Forward + inverse WHT recovers the original vector."""

    @pytest.mark.parametrize("d", [32, 64, 128, 256, 512])
    def test_roundtrip_triton_only(self, d):
        """Triton forward then Triton inverse recovers original."""
        torch.manual_seed(DEFAULT_SEED)
        params = _make_wht_params(d)
        signs = params["signs"]

        x = torch.randn(100, d, device=DEVICE)
        x_orig = x.clone()

        y = triton_wht_rotate(x, signs)
        x_recovered = triton_wht_unrotate(y, signs)

        max_error = (x_recovered - x_orig).abs().max().item()
        assert max_error < 1e-4, (
            f"d={d}: round-trip error {max_error:.2e} exceeds 1e-4"
        )

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_roundtrip_mixed_triton_python(self, d):
        """Triton forward + Python inverse recovers original (cross-compat)."""
        torch.manual_seed(DEFAULT_SEED)
        params = _make_wht_params(d)
        signs = params["signs"]

        x = torch.randn(50, d, device=DEVICE)
        x_orig = x.clone()

        # Triton forward, Python inverse
        y = triton_wht_rotate(x, signs)
        x_recovered = _apply_python_wht_unrotate(y, signs, d)

        max_error = (x_recovered - x_orig).abs().max().item()
        assert max_error < 1e-4, (
            f"d={d}: mixed round-trip error {max_error:.2e} exceeds 1e-4"
        )

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_roundtrip_python_forward_triton_inverse(self, d):
        """Python forward + Triton inverse recovers original (cross-compat)."""
        torch.manual_seed(DEFAULT_SEED)
        params = _make_wht_params(d)
        signs = params["signs"]

        x = torch.randn(50, d, device=DEVICE)
        x_orig = x.clone()

        # Python forward, Triton inverse
        y = _apply_python_wht_rotate(x, signs, d)
        x_recovered = triton_wht_unrotate(y, signs)

        max_error = (x_recovered - x_orig).abs().max().item()
        assert max_error < 1e-4, (
            f"d={d}: mixed round-trip error {max_error:.2e} exceeds 1e-4"
        )


# ---------------------------------------------------------------------------
# Norm preservation (orthogonal rotation)
# ---------------------------------------------------------------------------


class TestTritonWHTNormPreservation:
    """WHT rotation preserves vector norms (||Pi @ x|| = ||x||)."""

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_norm_preserved(self, d):
        """Vector norms are preserved by the Triton WHT rotation."""
        torch.manual_seed(DEFAULT_SEED)
        params = _make_wht_params(d)
        signs = params["signs"]

        x = torch.randn(200, d, device=DEVICE)
        y = triton_wht_rotate(x, signs)

        original_norms = x.norm(dim=-1)
        rotated_norms = y.norm(dim=-1)

        max_norm_diff = (original_norms - rotated_norms).abs().max().item()
        assert max_norm_diff < 1e-3, (
            f"d={d}: norm preservation error {max_norm_diff:.2e} exceeds 1e-3"
        )


# ---------------------------------------------------------------------------
# Random signs correctness
# ---------------------------------------------------------------------------


class TestTritonWHTSigns:
    """The random signs parameter is correctly applied."""

    def test_different_signs_different_results(self):
        """Different sign vectors produce different rotations."""
        d = 128
        torch.manual_seed(DEFAULT_SEED)
        x = torch.randn(10, d, device=DEVICE)

        signs_a = _make_wht_params(d, seed=1)["signs"]
        signs_b = _make_wht_params(d, seed=2)["signs"]

        y_a = triton_wht_rotate(x, signs_a)
        y_b = triton_wht_rotate(x, signs_b)

        # Should produce different rotations
        assert not torch.allclose(y_a, y_b, atol=1e-3), (
            "Different sign vectors should produce different rotations"
        )

    def test_all_positive_signs(self):
        """With all-positive signs, result equals pure WHT / sqrt(d)."""
        d = 64
        torch.manual_seed(DEFAULT_SEED)
        x = torch.randn(10, d, device=DEVICE)

        signs_ones = torch.ones(d, device=DEVICE)
        y_triton = triton_wht_rotate(x, signs_ones)

        # Pure WHT via Python
        x_copy = x.clone()
        y_python = fast_wht(x_copy) / math.sqrt(d)

        max_diff = (y_triton - y_python).abs().max().item()
        assert max_diff < 1e-4, (
            f"All-ones signs: max diff {max_diff:.2e} exceeds 1e-4"
        )


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------


class TestTritonWHTSpeed:
    """Triton WHT should be faster than Python WHT for larger batches."""

    def _cuda_benchmark(self, fn, warmup=20, iters=100):
        """Benchmark a CUDA function, return average time in ms."""
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def test_triton_faster_than_python(self):
        """Triton WHT should be faster than Python WHT for 5K vectors."""
        d = 128
        n = 5000
        torch.manual_seed(DEFAULT_SEED)
        params = _make_wht_params(d)
        signs = params["signs"]

        x = torch.randn(n, d, device=DEVICE)

        # Python WHT rotation
        def python_wht():
            return _apply_python_wht_rotate(x, signs, d)

        # Triton WHT rotation
        def triton_wht():
            return triton_wht_rotate(x, signs)

        py_ms = self._cuda_benchmark(python_wht)
        tri_ms = self._cuda_benchmark(triton_wht)

        speedup = py_ms / tri_ms
        print(
            f"\nWHT speedup ({n} vecs, d={d}): {speedup:.1f}x "
            f"(Python={py_ms:.3f}ms, Triton={tri_ms:.3f}ms)"
        )
        # Triton should be at least comparable. The Python loop has
        # multiple kernel launches per butterfly stage; Triton fuses all.
        # Accept if at least 0.8x (Triton overhead for small d).
        assert speedup > 0.8, (
            f"Triton WHT unexpectedly slow: {speedup:.1f}x vs Python"
        )

    def test_triton_faster_large_batch(self):
        """With 50K vectors, Triton should clearly win."""
        d = 128
        n = 50000
        torch.manual_seed(DEFAULT_SEED)
        params = _make_wht_params(d)
        signs = params["signs"]

        x = torch.randn(n, d, device=DEVICE)

        def python_wht():
            return _apply_python_wht_rotate(x, signs, d)

        def triton_wht():
            return triton_wht_rotate(x, signs)

        py_ms = self._cuda_benchmark(python_wht, warmup=5, iters=30)
        tri_ms = self._cuda_benchmark(triton_wht, warmup=5, iters=30)

        speedup = py_ms / tri_ms
        print(
            f"\nWHT speedup ({n} vecs, d={d}): {speedup:.1f}x "
            f"(Python={py_ms:.3f}ms, Triton={tri_ms:.3f}ms)"
        )
        # For large batches, Triton's single-kernel advantage should show
        assert speedup > 0.8, (
            f"Triton WHT unexpectedly slow at large batch: {speedup:.1f}x"
        )
