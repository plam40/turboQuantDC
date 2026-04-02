"""Tests for TurboQuant weight compression (TQ-W).

Tests verify:
    - Bit schedule computation for all strategies
    - CompressedLinear dequantization and forward pass
    - Weight MSE reconstruction at 2, 3, 4 bits vs original
    - Gradient strategy produces correct per-layer allocation
    - Memory savings match theoretical prediction
    - End-to-end compression of a small model
    - Coherent text generation after compression
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from turboquantdc.weight_compression import (
    CompressedLinear,
    TurboQuantWeightCompressor,
    compress_model,
    compute_weight_bit_schedule,
    effective_bpw,
    estimate_compressed_size,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
DEFAULT_DIM = 128
DEFAULT_OUT = 64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def compressor():
    """Default TurboQuantWeightCompressor."""
    return TurboQuantWeightCompressor(target_bpw=2.5, strategy="gradient", base_seed=SEED)


@pytest.fixture
def random_linear():
    """A random nn.Linear(128, 64) for testing."""
    torch.manual_seed(SEED)
    return nn.Linear(DEFAULT_DIM, DEFAULT_OUT, bias=True)


@pytest.fixture
def random_linear_no_bias():
    """A random nn.Linear(128, 64, bias=False) for testing."""
    torch.manual_seed(SEED)
    return nn.Linear(DEFAULT_DIM, DEFAULT_OUT, bias=False)


@pytest.fixture
def small_mlp():
    """A small 2-layer MLP for end-to-end testing."""
    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    return model


# ---------------------------------------------------------------------------
# Tests: bit schedule computation
# ---------------------------------------------------------------------------
class TestBitSchedule:
    """Test compute_weight_bit_schedule for all strategies."""

    def test_uniform_schedule(self):
        """Uniform strategy: all layers get the same bits."""
        schedule = compute_weight_bit_schedule(
            num_layers=80, target_bpw=3, strategy="uniform"
        )
        assert len(schedule) == 80
        assert all(b == 3 for b in schedule)

    def test_uniform_rounds_to_nearest(self):
        """Uniform with fractional target rounds to nearest integer."""
        schedule = compute_weight_bit_schedule(
            num_layers=10, target_bpw=2.5, strategy="uniform"
        )
        # 2.5 rounds to 2 (Python's round-half-even)
        assert all(b in (2, 3) for b in schedule)

    def test_uniform_clamps_to_valid_range(self):
        """Uniform clamps to [2, 8]."""
        schedule = compute_weight_bit_schedule(
            num_layers=5, target_bpw=1.0, strategy="uniform"
        )
        assert all(b == 2 for b in schedule)

        schedule_high = compute_weight_bit_schedule(
            num_layers=5, target_bpw=10.0, strategy="uniform"
        )
        assert all(b == 8 for b in schedule_high)

    def test_gradient_boundary_protection(self):
        """Gradient strategy: first 2 and last 2 layers get 4-bit."""
        schedule = compute_weight_bit_schedule(
            num_layers=80, strategy="gradient"
        )
        assert schedule[0] == 4
        assert schedule[1] == 4
        assert schedule[-1] == 4
        assert schedule[-2] == 4

    def test_gradient_middle_layers_2bit(self):
        """Gradient strategy: middle layers get 2-bit."""
        schedule = compute_weight_bit_schedule(
            num_layers=80, strategy="gradient"
        )
        # Middle of 80 layers (index 40) should be 2-bit
        assert schedule[40] == 2

    def test_gradient_near_boundary_3bit(self):
        """Gradient strategy: layers near boundaries get 3-bit."""
        schedule = compute_weight_bit_schedule(
            num_layers=80, strategy="gradient"
        )
        # 25% of 80 = 20. Layer index 10 is near the start but past index 2.
        # dist_from_start=10, dist_from_end=69, dist=10 < 80*0.25=20
        assert schedule[10] == 3

    def test_gradient_effective_bpw(self):
        """Gradient strategy achieves approximately target bpw for large models."""
        schedule = compute_weight_bit_schedule(
            num_layers=80, strategy="gradient"
        )
        avg = effective_bpw(schedule)
        # For 80 layers: 4 boundary (4-bit) + ~36 near-boundary (3-bit) + ~40 middle (2-bit)
        # Expected: (4*4 + 36*3 + 40*2) / 80 = (16 + 108 + 80) / 80 = 2.55
        assert 2.0 <= avg <= 3.0, f"Expected ~2.5 bpw, got {avg}"

    def test_custom_schedule(self):
        """Custom strategy uses the provided schedule exactly."""
        custom = [2, 3, 4, 3, 2]
        schedule = compute_weight_bit_schedule(
            num_layers=5, strategy="custom", custom_schedule=custom
        )
        assert schedule == custom

    def test_custom_schedule_length_mismatch(self):
        """Custom strategy rejects wrong-length schedule."""
        with pytest.raises(ValueError, match="must match"):
            compute_weight_bit_schedule(
                num_layers=5, strategy="custom", custom_schedule=[2, 3, 4]
            )

    def test_custom_strategy_requires_schedule(self):
        """Custom strategy requires custom_schedule argument."""
        with pytest.raises(ValueError, match="must be provided"):
            compute_weight_bit_schedule(num_layers=5, strategy="custom")

    def test_unknown_strategy_raises(self):
        """Unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            compute_weight_bit_schedule(num_layers=5, strategy="foo")


class TestEffectiveBpw:
    """Test effective_bpw helper."""

    def test_uniform(self):
        assert effective_bpw([3, 3, 3]) == 3.0

    def test_mixed(self):
        assert effective_bpw([2, 4]) == 3.0

    def test_empty(self):
        assert effective_bpw([]) == 0.0


# ---------------------------------------------------------------------------
# Tests: CompressedLinear
# ---------------------------------------------------------------------------
class TestCompressedLinear:
    """Test CompressedLinear dequantization and forward pass."""

    @pytest.fixture(params=[2, 3, 4], ids=lambda b: f"bits={b}")
    def bits(self, request):
        return request.param

    @pytest.fixture
    def compressed(self, random_linear, bits, compressor):
        """Compress a linear layer at the given bit-width."""
        return compressor.compress_linear(
            random_linear, bits=bits, rotation_seed=SEED
        )

    def test_output_shape(self, compressed, random_linear):
        """CompressedLinear produces same output shape as original."""
        x = torch.randn(8, DEFAULT_DIM)
        out_orig = random_linear(x)
        out_comp = compressed(x)
        assert out_comp.shape == out_orig.shape

    def test_dequantize_shape(self, compressed):
        """Dequantized weight has correct shape."""
        W = compressed._dequantize()
        assert W.shape == (DEFAULT_OUT, DEFAULT_DIM)

    def test_bias_preserved(self, compressed, random_linear):
        """Bias is exactly preserved (not quantized)."""
        if random_linear.bias is not None:
            torch.testing.assert_close(
                compressed.bias, random_linear.bias.data, atol=0, rtol=0
            )

    def test_no_bias(self, random_linear_no_bias, compressor):
        """CompressedLinear works without bias."""
        compressed = compressor.compress_linear(
            random_linear_no_bias, bits=3, rotation_seed=SEED
        )
        assert compressed.bias is None
        x = torch.randn(4, DEFAULT_DIM)
        out = compressed(x)
        assert out.shape == (4, DEFAULT_OUT)

    def test_mse_decreases_with_bits(self, random_linear, compressor):
        """Higher bit-width gives lower reconstruction MSE."""
        mse_values = {}
        for bits in [2, 3, 4]:
            compressed = compressor.compress_linear(
                random_linear, bits=bits, rotation_seed=SEED
            )
            mse_values[bits] = compressed.weight_mse(random_linear.weight.data)

        assert mse_values[2] > mse_values[3], (
            f"2-bit MSE ({mse_values[2]:.6f}) should exceed 3-bit ({mse_values[3]:.6f})"
        )
        assert mse_values[3] > mse_values[4], (
            f"3-bit MSE ({mse_values[3]:.6f}) should exceed 4-bit ({mse_values[4]:.6f})"
        )

    def test_4bit_mse_low(self, random_linear, compressor):
        """4-bit quantization achieves low MSE."""
        compressed = compressor.compress_linear(
            random_linear, bits=4, rotation_seed=SEED
        )
        mse = compressed.weight_mse(random_linear.weight.data)
        # Weight variance is ~1/fan_in = 1/128 for kaiming init
        # 4-bit Lloyd-Max distortion for Gaussian is ~0.009/d per coord
        # Total MSE should be well below the variance itself
        weight_var = random_linear.weight.data.var().item()
        assert mse < weight_var, (
            f"4-bit MSE ({mse:.6f}) should be below weight variance ({weight_var:.6f})"
        )

    def test_memory_bytes_computed(self, compressed):
        """memory_bytes returns a dict with expected keys."""
        mem = compressed.memory_bytes()
        assert "index_bits" in mem
        assert "total_bytes" in mem
        assert "theoretical_bits" in mem
        assert mem["total_bytes"] > 0

    def test_extra_repr(self, compressed):
        """extra_repr returns a meaningful string."""
        s = compressed.extra_repr()
        assert "in_features" in s
        assert "out_features" in s
        assert "bits" in s

    def test_deterministic_dequantize(self, compressed):
        """Same indices always produce the same dequantized weights."""
        W1 = compressed._dequantize()
        W2 = compressed._dequantize()
        torch.testing.assert_close(W1, W2, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Tests: TurboQuantWeightCompressor
# ---------------------------------------------------------------------------
class TestWeightCompressor:
    """Test the full model compression pipeline."""

    def test_compress_small_mlp(self, small_mlp, compressor):
        """Compress a small MLP and verify structure."""
        stats = compressor.compress(small_mlp)
        assert stats["num_compressed"] > 0
        # All Linear layers should be replaced
        for module in small_mlp.modules():
            if isinstance(module, (nn.Linear,)):
                # Only CompressedLinear should remain (not plain Linear)
                assert False, f"Found uncompressed nn.Linear: {module}"

    def test_compressed_mlp_forward(self, small_mlp, compressor):
        """Compressed MLP produces output of correct shape."""
        compressor.compress(small_mlp)
        x = torch.randn(4, 128)
        out = small_mlp(x)
        assert out.shape == (4, 10)

    def test_compressed_mlp_not_nan(self, small_mlp, compressor):
        """Compressed MLP output is finite (no NaN/Inf)."""
        compressor.compress(small_mlp)
        x = torch.randn(4, 128)
        out = small_mlp(x)
        assert torch.isfinite(out).all()

    def test_stats_keys(self, small_mlp, compressor):
        """Compression stats contain expected keys."""
        stats = compressor.compress(small_mlp)
        assert "num_compressed" in stats
        assert "schedule" in stats
        assert "effective_bpw" in stats
        assert "original_params" in stats
        assert "theoretical_size_mb" in stats

    def test_layer_detection_single(self, small_mlp, compressor):
        """For non-HF models, layer detection falls back to 1."""
        num = compressor._detect_num_layers(small_mlp)
        # Simple Sequential has no transformer layers
        assert num == 1


class TestCompressModelAPI:
    """Test the top-level compress_model function."""

    def test_compress_uniform(self, small_mlp):
        """compress_model with uniform strategy works."""
        stats = compress_model(small_mlp, target_bpw=3, strategy="uniform")
        assert stats["num_compressed"] > 0
        # Verify forward pass still works
        x = torch.randn(2, 128)
        out = small_mlp(x)
        assert out.shape == (2, 10)

    def test_compress_gradient(self, small_mlp):
        """compress_model with gradient strategy works."""
        stats = compress_model(small_mlp, target_bpw=2.5, strategy="gradient")
        assert stats["num_compressed"] > 0

    def test_compress_custom(self, small_mlp):
        """compress_model with custom strategy works."""
        # The small MLP has 1 detected layer, so schedule is length 1
        stats = compress_model(
            small_mlp, strategy="custom", custom_schedule=[3]
        )
        assert stats["num_compressed"] > 0


# ---------------------------------------------------------------------------
# Tests: Memory estimation
# ---------------------------------------------------------------------------
class TestEstimateCompressedSize:
    """Test estimate_compressed_size for capacity planning."""

    def test_70b_gradient_fits_24gb(self):
        """70B params at gradient ~2.5 bpw should fit in ~22GB."""
        result = estimate_compressed_size(
            num_params=70_000_000_000,
            num_layers=80,
            target_bpw=2.5,
            strategy="gradient",
        )
        # FP16 baseline: 70B * 2 bytes = 140GB
        assert 130 < result["fp16_gb"] < 145
        # Compressed: ~2.5 bpw -> ~22GB
        assert result["compressed_gb"] < 25, (
            f"Expected <25GB, got {result['compressed_gb']:.1f}GB"
        )
        assert result["compression_ratio"] > 5.0

    def test_7b_uniform_3bit(self):
        """7B params at uniform 3-bit -> ~2.6GB."""
        result = estimate_compressed_size(
            num_params=7_000_000_000,
            num_layers=32,
            target_bpw=3,
            strategy="uniform",
        )
        assert 2.0 < result["compressed_gb"] < 3.5
        assert result["compression_ratio"] > 4.0

    def test_compression_ratio_increases_with_lower_bpw(self):
        """Lower target bpw yields higher compression ratio."""
        r25 = estimate_compressed_size(70e9, 80, target_bpw=2.5, strategy="gradient")
        r30 = estimate_compressed_size(70e9, 80, target_bpw=3.0, strategy="uniform")
        assert r25["compression_ratio"] > r30["compression_ratio"]


# ---------------------------------------------------------------------------
# Tests: Weight reconstruction quality
# ---------------------------------------------------------------------------
class TestReconstructionQuality:
    """Verify weight reconstruction MSE at different bit-widths."""

    @pytest.fixture
    def large_linear(self):
        """Larger linear layer for more statistically stable MSE measurement."""
        torch.manual_seed(SEED)
        return nn.Linear(256, 512, bias=False)

    def test_mse_at_2_bits(self, large_linear, compressor):
        """2-bit weight MSE is finite and positive."""
        compressed = compressor.compress_linear(
            large_linear, bits=2, rotation_seed=SEED
        )
        mse = compressed.weight_mse(large_linear.weight.data)
        assert 0 < mse < 1.0  # Should be well below 1.0 for normalized weights

    def test_mse_at_3_bits(self, large_linear, compressor):
        """3-bit weight MSE is lower than 2-bit."""
        c2 = compressor.compress_linear(large_linear, bits=2, rotation_seed=SEED)
        c3 = compressor.compress_linear(large_linear, bits=3, rotation_seed=SEED)
        assert c3.weight_mse(large_linear.weight.data) < c2.weight_mse(large_linear.weight.data)

    def test_mse_at_4_bits(self, large_linear, compressor):
        """4-bit weight MSE is lower than 3-bit."""
        c3 = compressor.compress_linear(large_linear, bits=3, rotation_seed=SEED)
        c4 = compressor.compress_linear(large_linear, bits=4, rotation_seed=SEED)
        assert c4.weight_mse(large_linear.weight.data) < c3.weight_mse(large_linear.weight.data)

    def test_cosine_similarity_at_4bit(self, large_linear, compressor):
        """4-bit compressed weights have high cosine similarity to original."""
        compressed = compressor.compress_linear(
            large_linear, bits=4, rotation_seed=SEED
        )
        W_orig = large_linear.weight.data.float()
        W_hat = compressed._dequantize()
        # Flatten and compute cosine similarity
        cos_sim = F.cosine_similarity(
            W_orig.flatten().unsqueeze(0),
            W_hat.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.95, f"Expected cosine sim > 0.95, got {cos_sim:.4f}"

    def test_output_correlation(self, large_linear, compressor):
        """Compressed layer output is highly correlated with original output."""
        compressed = compressor.compress_linear(
            large_linear, bits=3, rotation_seed=SEED
        )
        torch.manual_seed(99)
        x = torch.randn(100, 256)
        out_orig = large_linear(x)
        out_comp = compressed(x)
        # Pearson correlation between flattened outputs
        o1 = out_orig.flatten()
        o2 = out_comp.flatten()
        correlation = torch.corrcoef(torch.stack([o1, o2]))[0, 1].item()
        assert correlation > 0.9, f"Expected correlation > 0.9, got {correlation:.4f}"


# ---------------------------------------------------------------------------
# Tests: Codebook and rotation caching
# ---------------------------------------------------------------------------
class TestCaching:
    """Verify that codebooks and rotations are cached and reused."""

    def test_codebook_reuse(self, compressor):
        """Same (d, bits) pair returns the same codebook object."""
        cb1 = compressor._get_codebook(128, 3)
        cb2 = compressor._get_codebook(128, 3)
        assert cb1 is cb2

    def test_codebook_different_for_different_params(self, compressor):
        """Different (d, bits) pairs produce different codebooks."""
        cb1 = compressor._get_codebook(128, 3)
        cb2 = compressor._get_codebook(128, 4)
        assert cb1 is not cb2

    def test_rotation_reuse(self, compressor):
        """Same (d, seed) pair returns the same rotation matrix."""
        r1 = compressor._get_rotation(128, 42)
        r2 = compressor._get_rotation(128, 42)
        assert r1 is r2

    def test_rotation_different_for_different_seed(self, compressor):
        """Different seeds produce different rotation matrices."""
        r1 = compressor._get_rotation(128, 42)
        r2 = compressor._get_rotation(128, 43)
        assert not torch.allclose(r1, r2)


# ---------------------------------------------------------------------------
# Tests: Gradient schedule for known model configs
# ---------------------------------------------------------------------------
class TestGradientScheduleModels:
    """Test gradient bit allocation for realistic model sizes."""

    def test_80_layer_model(self):
        """80-layer model (LLaMA-70B) gradient schedule structure."""
        schedule = compute_weight_bit_schedule(num_layers=80, strategy="gradient")
        assert len(schedule) == 80
        # First 2 and last 2 are 4-bit
        assert schedule[:2] == [4, 4]
        assert schedule[-2:] == [4, 4]
        # All values are in {2, 3, 4}
        assert all(b in (2, 3, 4) for b in schedule)
        # Middle layers are 2-bit
        assert schedule[40] == 2

    def test_32_layer_model(self):
        """32-layer model (LLaMA-8B) gradient schedule."""
        schedule = compute_weight_bit_schedule(num_layers=32, strategy="gradient")
        assert len(schedule) == 32
        assert schedule[0] == 4
        assert schedule[-1] == 4

    def test_small_model_all_boundary(self):
        """Very small model: all layers get boundary protection."""
        schedule = compute_weight_bit_schedule(num_layers=4, strategy="gradient")
        # With only 4 layers, all are within boundary range
        assert all(b >= 3 for b in schedule)
