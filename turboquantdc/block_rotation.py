"""Block-diagonal rotation matrices for KV cache quantization.

Two variants of block-diagonal rotations that provide O(d) compute instead of
O(d log d) (WHT) or O(d^2) (QR):

1. **GivensRotation** -- 2D block-diagonal (d/2 random Givens rotations).
   Each pair (x_{2i}, x_{2i+1}) is independently rotated by angle theta_i.
   1 DOF per block, 2 FMAs per block.

2. **QuaternionRotation** -- 4D block-diagonal (d/4 random quaternion rotations).
   Each 4-tuple (x_{4i}..x_{4i+3}) is rotated via full SO(4) quaternion
   sandwich: T(v) = q_L * v * conj(q_R), giving 6 DOF per block.
   ~16 FMAs per block but covers all of SO(4).

Key insight from RotorQuant: block-diagonal rotations distribute quantization
errors more *randomly* across coordinates, preserving attention score ranking
better than full-rank rotations (WHT/QR) which minimize MSE but can scramble
the ordering of inner products.

Both classes implement the same API:
    - rotate(x)   -- apply forward rotation, x shape (..., d)
    - unrotate(x)  -- apply inverse rotation, x shape (..., d)
    - Pi property   -- explicit d x d rotation matrix (for compatibility)

Reference: RotorQuant isoquant.py / planarquant.py (reimplemented, not copied).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Quaternion helpers (pure functions, no state)
# ---------------------------------------------------------------------------

def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate: (w, x, y, z) -> (w, -x, -y, -z)."""
    signs = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)
    return q * signs


def _quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions.

    a, b: (..., 4) as [w, x, y, z]
    Returns: (..., 4)
    """
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)

    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw

    return torch.stack([rw, rx, ry, rz], dim=-1)


# ---------------------------------------------------------------------------
# GivensRotation -- 2D block-diagonal
# ---------------------------------------------------------------------------


class GivensRotation(nn.Module):
    """2D block-diagonal rotation (d/2 independent Givens rotations).

    Each pair of adjacent coordinates is rotated by an independent random
    angle theta_i in [0, 2*pi). Compute cost is O(d): exactly 4 multiplies
    and 2 adds per pair.

    For odd d, the last coordinate is left unrotated (padded internally).

    Args:
        d: Vector dimension. Any positive integer.
        seed: Random seed for reproducible angle generation.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d = d
        self.seed = seed

        # Number of 2D blocks (ceil(d/2))
        self.n_groups = (d + 1) // 2
        self.d_padded = self.n_groups * 2

        # Generate random angles on CPU for reproducibility
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        angles = torch.rand(self.n_groups, generator=gen) * (2.0 * math.pi)
        # Store cos/sin pairs for efficient rotation
        cs = torch.stack([angles.cos(), angles.sin()], dim=-1)  # (n_groups, 2)
        self.register_buffer("cs", cs.to(device))

        # Build explicit Pi matrix for API compatibility
        Pi = self._build_explicit_matrix()
        self.register_buffer("Pi", Pi.to(device))

    def _build_explicit_matrix(self) -> torch.Tensor:
        """Build the full d x d rotation matrix (for compatibility/testing)."""
        Pi = torch.eye(self.d, dtype=torch.float32)
        cs_cpu = self.cs.cpu() if hasattr(self, "cs") else self.cs
        for i in range(self.n_groups):
            row = 2 * i
            if row + 1 >= self.d:
                break  # odd dimension, last element stays
            c = cs_cpu[i, 0].item()
            s = cs_cpu[i, 1].item()
            Pi[row, row] = c
            Pi[row, row + 1] = -s
            Pi[row + 1, row] = s
            Pi[row + 1, row + 1] = c
        return Pi

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad to even dimension if needed."""
        if self.d_padded > self.d:
            return torch.nn.functional.pad(x, (0, self.d_padded - self.d))
        return x

    def _unpad(self, x: torch.Tensor) -> torch.Tensor:
        """Remove padding."""
        return x[..., : self.d]

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward Givens rotation to vectors.

        Each pair (x_{2i}, x_{2i+1}) is rotated by angle theta_i:
            y_{2i}   = cos(theta_i) * x_{2i} - sin(theta_i) * x_{2i+1}
            y_{2i+1} = sin(theta_i) * x_{2i} + cos(theta_i) * x_{2i+1}

        Trailing unpaired elements (odd d) are left unchanged.

        Args:
            x: Input vectors, shape (..., d).

        Returns:
            Rotated vectors, same shape.
        """
        # Number of complete pairs
        n_paired = (self.d // 2) * 2  # largest even number <= d
        if n_paired == 0:
            return x

        paired = x[..., :n_paired]
        shape = paired.shape
        pairs = paired.reshape(*shape[:-1], n_paired // 2, 2)

        cs = self.cs[:n_paired // 2]
        c = cs[..., 0]
        s = cs[..., 1]
        v0 = pairs[..., 0]
        v1 = pairs[..., 1]

        r0 = c * v0 - s * v1
        r1 = s * v0 + c * v1
        result_paired = torch.stack([r0, r1], dim=-1).reshape(shape)

        if n_paired < self.d:
            # Odd dimension: keep trailing element unchanged
            return torch.cat([result_paired, x[..., n_paired:]], dim=-1)
        return result_paired

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse Givens rotation.

        Inverse = transpose of Givens = negate the sine component.
        Trailing unpaired elements (odd d) are left unchanged.

        Args:
            y: Rotated vectors, shape (..., d).

        Returns:
            Unrotated vectors, same shape.
        """
        n_paired = (self.d // 2) * 2
        if n_paired == 0:
            return y

        paired = y[..., :n_paired]
        shape = paired.shape
        pairs = paired.reshape(*shape[:-1], n_paired // 2, 2)

        cs = self.cs[:n_paired // 2]
        c = cs[..., 0]
        s = cs[..., 1]
        v0 = pairs[..., 0]
        v1 = pairs[..., 1]

        # Inverse: transpose = (c, -s; s, c)^T = (c, s; -s, c)
        r0 = c * v0 + s * v1
        r1 = -s * v0 + c * v1
        result_paired = torch.stack([r0, r1], dim=-1).reshape(shape)

        if n_paired < self.d:
            return torch.cat([result_paired, y[..., n_paired:]], dim=-1)
        return result_paired


# ---------------------------------------------------------------------------
# QuaternionRotation -- 4D block-diagonal
# ---------------------------------------------------------------------------


class QuaternionRotation(nn.Module):
    """4D block-diagonal rotation (d/4 independent quaternion sandwiches).

    Each 4-tuple of coordinates is rotated via the full SO(4) action:
        T(v) = q_L * v * conj(q_R)

    This gives 6 degrees of freedom per block (3 from q_L + 3 from q_R,
    minus the unit constraint on each). The full SO(4) coverage means
    every possible 4D rotation is reachable, giving maximum decorrelation
    within each block.

    For dimensions not divisible by 4, the trailing coordinates are padded
    with zeros before rotation and unpadded after.

    Args:
        d: Vector dimension. Any positive integer.
        seed: Random seed for reproducible quaternion generation.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d = d
        self.seed = seed

        # Number of 4D blocks (ceil(d/4))
        self.n_groups = (d + 3) // 4
        self.d_padded = self.n_groups * 4

        # Generate random unit quaternions on CPU for reproducibility
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        q_L = torch.randn(self.n_groups, 4, generator=gen)
        q_L = q_L / q_L.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("q_L", q_L.to(device))

        gen.manual_seed(seed + 10000)  # Different seed for q_R
        q_R = torch.randn(self.n_groups, 4, generator=gen)
        q_R = q_R / q_R.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("q_R", q_R.to(device))

        # Build explicit Pi matrix for API compatibility.
        # Must be done AFTER buffers are registered since rotate() uses them.
        Pi = self._build_explicit_matrix()
        self.register_buffer("Pi", Pi.to(device))

    def _build_explicit_matrix(self) -> torch.Tensor:
        """Build the full d x d rotation matrix satisfying rotate(x) = x @ Pi.T.

        We need Pi.T[:, i] = rotate(e_i), i.e., Pi[i, :] = rotate(e_i).
        But since rotate is applied row-wise, rotating the identity gives us
        a matrix R where R[i, :] = rotate(e_i). Then rotate(x) = x @ R.T
        would require Pi = R. However, our convention is rotate(x) = x @ Pi.T,
        so Pi = R.T and x @ Pi.T = x @ R.
        """
        I_d = torch.eye(self.d, dtype=torch.float32, device=self.q_L.device)
        R = self.rotate(I_d)  # R[i, :] = rotate(e_i)
        # rotate(x) = sum_i x_i * rotate(e_i) = x @ R
        # For convention rotate(x) = x @ Pi.T, we need Pi.T = R, so Pi = R.T
        return R.T.cpu()

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad to multiple of 4 if needed."""
        if self.d_padded > self.d:
            return torch.nn.functional.pad(x, (0, self.d_padded - self.d))
        return x

    def _unpad(self, x: torch.Tensor) -> torch.Tensor:
        """Remove padding."""
        return x[..., : self.d]

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward quaternion rotation: T(v) = q_L * v * conj(q_R).

        Each complete 4-tuple is treated as a quaternion and sandwiched between
        q_L and conj(q_R). Trailing elements (if d is not divisible by 4) are
        left unchanged.

        Args:
            x: Input vectors, shape (..., d).

        Returns:
            Rotated vectors, same shape.
        """
        n_aligned = (self.d // 4) * 4
        if n_aligned == 0:
            return x

        aligned = x[..., :n_aligned]
        shape = aligned.shape
        blocks = aligned.reshape(*shape[:-1], n_aligned // 4, 4)

        # T(v) = q_L * v * conj(q_R)
        q_L = self.q_L[:n_aligned // 4]
        q_R = self.q_R[:n_aligned // 4]
        temp = _quat_multiply(q_L, blocks)
        result = _quat_multiply(temp, _quat_conjugate(q_R))

        result_aligned = result.reshape(shape)

        if n_aligned < self.d:
            return torch.cat([result_aligned, x[..., n_aligned:]], dim=-1)
        return result_aligned

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse quaternion rotation: T^{-1}(v) = conj(q_L) * v * q_R.

        Trailing elements (if d is not divisible by 4) are left unchanged.

        Args:
            y: Rotated vectors, shape (..., d).

        Returns:
            Unrotated vectors, same shape.
        """
        n_aligned = (self.d // 4) * 4
        if n_aligned == 0:
            return y

        aligned = y[..., :n_aligned]
        shape = aligned.shape
        blocks = aligned.reshape(*shape[:-1], n_aligned // 4, 4)

        q_L = self.q_L[:n_aligned // 4]
        q_R = self.q_R[:n_aligned // 4]
        temp = _quat_multiply(_quat_conjugate(q_L), blocks)
        result = _quat_multiply(temp, q_R)

        result_aligned = result.reshape(shape)

        if n_aligned < self.d:
            return torch.cat([result_aligned, y[..., n_aligned:]], dim=-1)
        return result_aligned
