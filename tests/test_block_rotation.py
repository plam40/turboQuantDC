"""Tests for block-diagonal rotations (Givens and Quaternion)."""

import math

import pytest
import torch

from turboquantdc.block_rotation import (
    GivensRotation,
    QuaternionRotation,
    _quat_conjugate,
    _quat_multiply,
)


# ---------------------------------------------------------------------------
# Quaternion helper tests
# ---------------------------------------------------------------------------


class TestQuaternionHelpers:
    """Tests for quaternion math primitives."""

    def test_quat_conjugate(self):
        q = torch.tensor([1.0, 2.0, 3.0, 4.0])
        qc = _quat_conjugate(q)
        assert torch.allclose(qc, torch.tensor([1.0, -2.0, -3.0, -4.0]))

    def test_quat_multiply_identity(self):
        """Multiplying by identity quaternion (1,0,0,0) is identity."""
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q = torch.tensor([0.5, 0.5, 0.5, 0.5])
        result = _quat_multiply(identity, q)
        assert torch.allclose(result, q, atol=1e-6)

    def test_quat_multiply_inverse(self):
        """q * conj(q) = |q|^2 * (1,0,0,0) for unit quaternion."""
        q = torch.randn(4)
        q = q / q.norm()
        result = _quat_multiply(q, _quat_conjugate(q))
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_quat_multiply_batched(self):
        """Batched quaternion multiplication works."""
        a = torch.randn(10, 4)
        b = torch.randn(10, 4)
        result = _quat_multiply(a, b)
        assert result.shape == (10, 4)
        # Check one element against scalar version
        r0 = _quat_multiply(a[0], b[0])
        assert torch.allclose(result[0], r0, atol=1e-5)


# ---------------------------------------------------------------------------
# GivensRotation tests
# ---------------------------------------------------------------------------


class TestGivensRotation:
    """Tests for 2D block-diagonal Givens rotation."""

    @pytest.fixture
    def rot128(self):
        return GivensRotation(d=128, seed=42)

    @pytest.fixture
    def rot127(self):
        """Odd dimension -- tests padding."""
        return GivensRotation(d=127, seed=42)

    def test_init_dimensions(self, rot128):
        assert rot128.d == 128
        assert rot128.n_groups == 64
        assert rot128.d_padded == 128
        assert rot128.cs.shape == (64, 2)

    def test_init_odd_dimension(self, rot127):
        assert rot127.d == 127
        assert rot127.n_groups == 64
        assert rot127.d_padded == 128

    def test_rotate_shape(self, rot128):
        x = torch.randn(10, 128)
        y = rot128.rotate(x)
        assert y.shape == (10, 128)

    def test_rotate_single_vector(self, rot128):
        x = torch.randn(128)
        y = rot128.rotate(x)
        assert y.shape == (128,)

    def test_roundtrip_identity(self, rot128):
        """rotate then unrotate should be identity."""
        x = torch.randn(50, 128)
        y = rot128.rotate(x)
        x_recovered = rot128.unrotate(y)
        assert torch.allclose(x, x_recovered, atol=1e-5)

    def test_roundtrip_odd_dim(self, rot127):
        """Roundtrip with odd dimension."""
        x = torch.randn(20, 127)
        y = rot127.rotate(x)
        x_recovered = rot127.unrotate(y)
        assert torch.allclose(x, x_recovered, atol=1e-5)

    def test_norm_preservation(self, rot128):
        """Rotation should preserve vector norms."""
        x = torch.randn(30, 128)
        y = rot128.rotate(x)
        x_norms = x.norm(dim=-1)
        y_norms = y.norm(dim=-1)
        assert torch.allclose(x_norms, y_norms, atol=1e-5)

    def test_deterministic_with_seed(self):
        """Same seed produces same rotation."""
        r1 = GivensRotation(d=128, seed=123)
        r2 = GivensRotation(d=128, seed=123)
        x = torch.randn(10, 128)
        y1 = r1.rotate(x)
        y2 = r2.rotate(x)
        assert torch.allclose(y1, y2)

    def test_different_seeds(self):
        """Different seeds produce different rotations."""
        r1 = GivensRotation(d=128, seed=1)
        r2 = GivensRotation(d=128, seed=2)
        x = torch.randn(10, 128)
        y1 = r1.rotate(x)
        y2 = r2.rotate(x)
        assert not torch.allclose(y1, y2)

    def test_explicit_matrix_orthogonal(self, rot128):
        """The explicit Pi matrix should be orthogonal."""
        Pi = rot128.Pi
        I = torch.eye(128)
        product = Pi @ Pi.T
        assert torch.allclose(product, I, atol=1e-5)

    def test_explicit_matrix_matches_rotate(self, rot128):
        """Pi @ x should match rotate(x)."""
        x = torch.randn(128)
        y_func = rot128.rotate(x)
        y_matrix = x @ rot128.Pi.T  # x @ Pi^T = (Pi @ x^T)^T in row convention
        assert torch.allclose(y_func, y_matrix, atol=1e-5)

    def test_3d_batch(self, rot128):
        """Works with 3D batches (batch, seq, d)."""
        x = torch.randn(4, 16, 128)
        y = rot128.rotate(x)
        assert y.shape == (4, 16, 128)
        x_back = rot128.unrotate(y)
        assert torch.allclose(x, x_back, atol=1e-5)


# ---------------------------------------------------------------------------
# QuaternionRotation tests
# ---------------------------------------------------------------------------


class TestQuaternionRotation:
    """Tests for 4D block-diagonal quaternion rotation."""

    @pytest.fixture
    def rot128(self):
        return QuaternionRotation(d=128, seed=42)

    @pytest.fixture
    def rot130(self):
        """Dimension not divisible by 4 -- tests padding."""
        return QuaternionRotation(d=130, seed=42)

    def test_init_dimensions(self, rot128):
        assert rot128.d == 128
        assert rot128.n_groups == 32
        assert rot128.d_padded == 128
        assert rot128.q_L.shape == (32, 4)
        assert rot128.q_R.shape == (32, 4)

    def test_init_non_divisible_dim(self, rot130):
        assert rot130.d == 130
        assert rot130.n_groups == 33  # ceil(130/4) = 33
        assert rot130.d_padded == 132

    def test_quaternions_are_unit(self, rot128):
        """Stored quaternions should be unit quaternions."""
        assert torch.allclose(rot128.q_L.norm(dim=-1), torch.ones(32), atol=1e-5)
        assert torch.allclose(rot128.q_R.norm(dim=-1), torch.ones(32), atol=1e-5)

    def test_rotate_shape(self, rot128):
        x = torch.randn(10, 128)
        y = rot128.rotate(x)
        assert y.shape == (10, 128)

    def test_rotate_single_vector(self, rot128):
        x = torch.randn(128)
        y = rot128.rotate(x)
        assert y.shape == (128,)

    def test_roundtrip_identity(self, rot128):
        """rotate then unrotate should be identity."""
        x = torch.randn(50, 128)
        y = rot128.rotate(x)
        x_recovered = rot128.unrotate(y)
        assert torch.allclose(x, x_recovered, atol=1e-5)

    def test_roundtrip_non_divisible(self, rot130):
        """Roundtrip with dimension not divisible by 4."""
        x = torch.randn(20, 130)
        y = rot130.rotate(x)
        x_recovered = rot130.unrotate(y)
        assert torch.allclose(x, x_recovered, atol=1e-5)

    def test_norm_preservation(self, rot128):
        """Quaternion rotation should preserve vector norms."""
        x = torch.randn(30, 128)
        y = rot128.rotate(x)
        x_norms = x.norm(dim=-1)
        y_norms = y.norm(dim=-1)
        assert torch.allclose(x_norms, y_norms, atol=1e-4)

    def test_deterministic_with_seed(self):
        """Same seed produces same rotation."""
        r1 = QuaternionRotation(d=128, seed=123)
        r2 = QuaternionRotation(d=128, seed=123)
        x = torch.randn(10, 128)
        y1 = r1.rotate(x)
        y2 = r2.rotate(x)
        assert torch.allclose(y1, y2)

    def test_different_seeds(self):
        """Different seeds produce different rotations."""
        r1 = QuaternionRotation(d=128, seed=1)
        r2 = QuaternionRotation(d=128, seed=2)
        x = torch.randn(10, 128)
        y1 = r1.rotate(x)
        y2 = r2.rotate(x)
        assert not torch.allclose(y1, y2)

    def test_explicit_matrix_orthogonal(self, rot128):
        """The explicit Pi matrix should be orthogonal."""
        Pi = rot128.Pi
        I = torch.eye(128)
        product = Pi @ Pi.T
        assert torch.allclose(product, I, atol=1e-4)

    def test_explicit_matrix_matches_rotate(self, rot128):
        """Pi @ x should match rotate(x)."""
        x = torch.randn(128)
        y_func = rot128.rotate(x)
        y_matrix = x @ rot128.Pi.T
        assert torch.allclose(y_func, y_matrix, atol=1e-4)

    def test_3d_batch(self, rot128):
        """Works with 3D batches (batch, seq, d)."""
        x = torch.randn(4, 16, 128)
        y = rot128.rotate(x)
        assert y.shape == (4, 16, 128)
        x_back = rot128.unrotate(y)
        assert torch.allclose(x, x_back, atol=1e-5)

    def test_rotation_not_identity(self, rot128):
        """The rotation should actually change the vector."""
        x = torch.randn(10, 128)
        y = rot128.rotate(x)
        assert not torch.allclose(x, y, atol=1e-3)
