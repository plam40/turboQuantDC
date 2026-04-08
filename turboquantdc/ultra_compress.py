"""Ultra-compression: 1-bit KV cache experiments.

Three approaches to making 1-bit work for autoregressive generation:

1. Multi-scale residual chain: cascaded 1-bit stages (N stages of 1-bit)
2. Sign prediction with error coding: predict signs from neighbors, store errors
3. Attention-gated refinement: 1-bit base with dynamic 3-bit upgrade

All approaches build on the existing PolarQuant/codebook infrastructure.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook import LloydMaxCodebook
from .polarquant import PolarQuant

# ---------------------------------------------------------------------------
# Approach 1: Multi-Scale Residual Chain
# ---------------------------------------------------------------------------


class MultiScaleResidualChain(nn.Module):
    """Cascaded 1-bit quantization: each stage quantizes the previous residual.

    Stage 1: 1-bit quantize the rotated vector -> 2 centroids
    Stage 2: 1-bit quantize the stage-1 residual -> 2 centroids (finer)
    Stage 3: 1-bit quantize the stage-2 residual -> 2 centroids (finest)
    ...

    Total bits = num_stages * 1 bit per coordinate.
    Each stage sees a different distribution (residual of previous stage),
    so each stage gets its own codebook optimized for that distribution.

    The hypothesis: hierarchical 1-bit captures structure that flat N-bit misses
    because each stage's codebook is optimized for the specific residual shape.

    Args:
        d: Head dimension.
        num_stages: Number of 1-bit stages (total bits = num_stages).
        seed: Random seed.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        num_stages: int = 3,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d = d
        self.num_stages = num_stages

        # Stage 1: rotation + 1-bit Lloyd-Max
        self.polar = PolarQuant(d, bits=1, seed=seed, device=device)

        # Build residual codebooks for stages 2..N
        # Each stage's residual has variance = MSE distortion of previous stage
        self._stage_codebooks: List[LloydMaxCodebook] = []
        sigma_sq = LloydMaxCodebook(d, 1).compute_distortion()

        for s in range(1, num_stages):
            # d_eff = 1/sigma_sq gives us a codebook for N(0, sigma_sq)
            d_eff = max(int(round(1.0 / max(sigma_sq, 1e-12))), 2)
            cb = LloydMaxCodebook(d_eff, bits=1)
            cb = cb.to(device)
            self._stage_codebooks.append(cb)
            # Next stage's residual variance = this codebook's distortion
            sigma_sq = cb.compute_distortion()

    def quantize(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Multi-stage 1-bit quantization.

        Returns:
            Dict with stage1_indices, stage2_indices, ..., vec_norm.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        vec_norm = x.norm(dim=-1, keepdim=True)
        x_normalized = x / (vec_norm + 1e-8)

        # Rotate
        x_rotated = self.polar.rotate(x_normalized)

        result: Dict[str, torch.Tensor] = {"vec_norm": vec_norm.squeeze(-1)}

        # Stage 1
        indices_1 = self.polar.codebook.quantize(x_rotated)
        recon_1 = self.polar.centroids[indices_1]
        residual = x_rotated - recon_1
        result["stage1_indices"] = indices_1

        # Stages 2..N
        for s, cb in enumerate(self._stage_codebooks):
            indices_s = cb.quantize(residual)
            recon_s = cb.centroids.to(residual.device)[indices_s]
            residual = residual - recon_s
            result[f"stage{s+2}_indices"] = indices_s

        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

    def dequantize(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct from multi-stage indices."""
        squeeze = compressed["stage1_indices"].dim() == 1

        stage1_idx = compressed["stage1_indices"]
        vec_norm = compressed["vec_norm"]
        if squeeze:
            stage1_idx = stage1_idx.unsqueeze(0)
            vec_norm = vec_norm.unsqueeze(0)

        # Stage 1 reconstruction in rotated space
        combined_rotated = self.polar.centroids[stage1_idx]

        # Add stage 2..N corrections
        for s, cb in enumerate(self._stage_codebooks):
            key = f"stage{s+2}_indices"
            idx = compressed[key]
            if squeeze:
                idx = idx.unsqueeze(0)
            recon_s = cb.centroids.to(combined_rotated.device)[idx]
            combined_rotated = combined_rotated + recon_s

        # Unrotate and rescale
        x_recon = self.polar.unrotate(combined_rotated)
        result = x_recon * vec_norm.unsqueeze(-1)

        if squeeze:
            result = result.squeeze(0)
        return result


# ---------------------------------------------------------------------------
# Approach 2: Sign Prediction with Error Coding
# ---------------------------------------------------------------------------


class SignPredictionCompressor(nn.Module):
    """1-bit quantization with learned sign prediction from neighbors.

    Core idea: after rotation, neighboring coordinates are weakly correlated
    (imperfect decorrelation). We predict each coordinate's sign from its
    neighbors and only store the PREDICTION ERRORS (flipped bits).

    If prediction accuracy is P, effective bits = H(1-P) where H is binary
    entropy. At P=0.8, H(0.2) ~ 0.72 bits, so effective rate is 0.72 bits.

    For storage we still use 1 bit per coordinate (the prediction error),
    but the reconstruction uses sign predictions as the baseline, flipping
    where the error bit says to flip. The value comes from better initial
    reconstruction by using neighbor context.

    The prediction uses a simple sliding window mean of neighboring signs
    (no learned parameters needed).

    Args:
        d: Head dimension.
        window: Prediction window size (neighbors on each side).
        seed: Random seed.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        window: int = 4,
        use_residual_signs: bool = True,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d = d
        self.window = window
        self.use_residual_signs = use_residual_signs

        # 1-bit PolarQuant for base quantization
        self.polar = PolarQuant(d, bits=1, seed=seed, device=device)

    def _predict_signs(self, x_rotated: torch.Tensor) -> torch.Tensor:
        """Predict sign of each coordinate from neighboring values.

        Uses a simple averaging kernel: the predicted sign for coordinate j
        is sign(mean(x[j-w:j] + x[j+1:j+w+1])). This captures local trend.

        Returns predicted signs as {-1, +1} tensor.
        """
        batch, d = x_rotated.shape
        w = self.window

        # Pad with zeros for boundary handling
        padded = F.pad(x_rotated, (w, w), mode='constant', value=0)

        # Compute local mean excluding the center element
        # Use a 1D average pool then subtract the center contribution
        kernel_size = 2 * w + 1
        # Sum of all elements in window (including center)
        local_sum = F.avg_pool1d(
            padded.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=0
        ).squeeze(1) * kernel_size

        # Subtract center element to get neighbor sum
        neighbor_sum = local_sum - x_rotated
        neighbor_count = kernel_size - 1

        # Predicted sign from neighbors
        neighbor_mean = neighbor_sum / neighbor_count
        predicted_signs = (neighbor_mean >= 0).float() * 2 - 1

        return predicted_signs

    def quantize(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """1-bit quantize with sign prediction error coding.

        Stores:
            - mse_indices: 1-bit codebook indices (1 bit/coord)
            - sign_errors: XOR of actual vs predicted residual signs (1 bit/coord)
            - residual_scale: mean |residual| (16 bits/vector)
            - vec_norm: ||x|| (16 bits/vector)

        Total: 2 bits/coord + 32 bits overhead per vector.
        But the sign_errors are sparse (many zeros) -> entropy < 1 bit/coord.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        vec_norm = x.norm(dim=-1, keepdim=True)
        x_normalized = x / (vec_norm + 1e-8)

        x_rotated = self.polar.rotate(x_normalized)

        # Stage 1: 1-bit MSE quantize
        mse_indices = self.polar.codebook.quantize(x_rotated)
        x_mse_rotated = self.polar.centroids[mse_indices]

        # Residual in rotated space
        residual = x_rotated - x_mse_rotated
        actual_signs = (residual >= 0).float() * 2 - 1

        if self.use_residual_signs:
            # Predict signs from the residual pattern
            predicted_signs = self._predict_signs(residual)

            # Error = places where prediction is wrong (XOR-like)
            # Store as: +1 means prediction was correct, -1 means flip needed
            sign_agreement = (actual_signs == predicted_signs).float() * 2 - 1
        else:
            predicted_signs = torch.ones_like(actual_signs)
            sign_agreement = actual_signs

        residual_scale = residual.abs().mean(dim=-1)

        result = {
            "mse_indices": mse_indices,
            "sign_errors": sign_agreement,
            "predicted_signs": predicted_signs,
            "residual_scale": residual_scale,
            "vec_norm": vec_norm.squeeze(-1),
        }

        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}
        return result

    def dequantize(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct using predicted signs corrected by error bits."""
        squeeze = compressed["mse_indices"].dim() == 1

        mse_indices = compressed["mse_indices"]
        sign_errors = compressed["sign_errors"]
        predicted_signs = compressed["predicted_signs"]
        residual_scale = compressed["residual_scale"]
        vec_norm = compressed["vec_norm"]

        if squeeze:
            mse_indices = mse_indices.unsqueeze(0)
            sign_errors = sign_errors.unsqueeze(0)
            predicted_signs = predicted_signs.unsqueeze(0)
            residual_scale = residual_scale.unsqueeze(0)
            vec_norm = vec_norm.unsqueeze(0)

        # Recover actual signs: where agreement is +1, use prediction;
        # where agreement is -1, flip the prediction
        actual_signs = predicted_signs * sign_errors

        # Reconstruct in rotated space
        x_mse_rotated = self.polar.centroids[mse_indices]
        correction = residual_scale.unsqueeze(-1) * actual_signs
        x_corrected_rotated = x_mse_rotated + correction

        # Unrotate and rescale
        x_recon = self.polar.unrotate(x_corrected_rotated)
        result = x_recon * vec_norm.unsqueeze(-1)

        if squeeze:
            result = result.squeeze(0)
        return result

    def get_prediction_accuracy(self, x: torch.Tensor) -> float:
        """Measure what fraction of signs are correctly predicted.

        This determines the effective bit rate:
            If accuracy = P, effective bits = H(1-P) per coordinate.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        vec_norm = x.norm(dim=-1, keepdim=True)
        x_normalized = x / (vec_norm + 1e-8)
        x_rotated = self.polar.rotate(x_normalized)

        mse_indices = self.polar.codebook.quantize(x_rotated)
        x_mse_rotated = self.polar.centroids[mse_indices]
        residual = x_rotated - x_mse_rotated

        actual_signs = (residual >= 0).float() * 2 - 1
        predicted_signs = self._predict_signs(residual)

        accuracy = (actual_signs == predicted_signs).float().mean().item()
        return accuracy


# ---------------------------------------------------------------------------
# Approach 3: Attention-Gated Refinement
# ---------------------------------------------------------------------------


class AttentionGatedCache(nn.Module):
    """1-bit base with dynamic refinement to 3-bit for high-attention tokens.

    All tokens start at 1-bit. During each attention computation, tokens
    that receive attention weight above a threshold are "promoted" to 3-bit
    on the fly. This is analogous to mixed-precision in training.

    The key observation: in long context, most tokens are never highly
    attended (attention is sparse). So most tokens stay at 1-bit,
    and only the important ones get refined.

    For benchmarking purposes, we simulate this by:
    1. Storing everything at both 1-bit and 3-bit
    2. Using attention scores to decide which version to use
    3. Measuring effective bits as the weighted average

    Args:
        d: Head dimension.
        refine_bits: Bits for refined tokens (default 3).
        attention_threshold: Attention weight threshold for refinement.
        max_refined_fraction: Maximum fraction of tokens to refine.
        seed: Random seed.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        refine_bits: int = 3,
        attention_threshold: float = 0.05,
        max_refined_fraction: float = 0.2,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d = d
        self.refine_bits = refine_bits
        self.attention_threshold = attention_threshold
        self.max_refined_fraction = max_refined_fraction

        # 1-bit base quantizer
        self.base_polar = PolarQuant(d, bits=1, seed=seed, device=device)

        # 3-bit refined quantizer (same rotation for consistency)
        self.refine_polar = PolarQuant(d, bits=refine_bits, seed=seed, device=device)

    def quantize_both(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Store both 1-bit and refined representations.

        Returns:
            Dict containing both compression levels plus metadata.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        vec_norm = x.norm(dim=-1, keepdim=True)
        x_normalized = x / (vec_norm + 1e-8)
        x_rotated = self.base_polar.rotate(x_normalized)

        # 1-bit path
        base_indices = self.base_polar.codebook.quantize(x_rotated)
        base_recon_rot = self.base_polar.centroids[base_indices]
        base_residual = x_rotated - base_recon_rot
        base_signs = (base_residual >= 0).float() * 2 - 1
        base_scale = base_residual.abs().mean(dim=-1)

        # Refined path (same rotation)
        refine_indices = self.refine_polar.codebook.quantize(x_rotated)
        refine_recon_rot = self.refine_polar.centroids[refine_indices]
        refine_residual = x_rotated - refine_recon_rot
        refine_signs = (refine_residual >= 0).float() * 2 - 1
        refine_scale = refine_residual.abs().mean(dim=-1)

        result = {
            "base_indices": base_indices,
            "base_signs": base_signs,
            "base_scale": base_scale,
            "refine_indices": refine_indices,
            "refine_signs": refine_signs,
            "refine_scale": refine_scale,
            "vec_norm": vec_norm.squeeze(-1),
        }

        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}
        return result

    def dequantize_selective(
        self,
        compressed: Dict[str, torch.Tensor],
        refine_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct using 1-bit for most tokens, refined for selected ones.

        Args:
            compressed: Output from quantize_both().
            refine_mask: Boolean tensor (batch,) — True for tokens to refine.

        Returns:
            Reconstructed vectors.
        """
        squeeze = compressed["base_indices"].dim() == 1

        base_indices = compressed["base_indices"]
        base_signs = compressed["base_signs"]
        base_scale = compressed["base_scale"]
        refine_indices = compressed["refine_indices"]
        refine_signs = compressed["refine_signs"]
        refine_scale = compressed["refine_scale"]
        vec_norm = compressed["vec_norm"]

        if squeeze:
            base_indices = base_indices.unsqueeze(0)
            base_signs = base_signs.unsqueeze(0)
            base_scale = base_scale.unsqueeze(0)
            refine_indices = refine_indices.unsqueeze(0)
            refine_signs = refine_signs.unsqueeze(0)
            refine_scale = refine_scale.unsqueeze(0)
            vec_norm = vec_norm.unsqueeze(0)
            refine_mask = refine_mask.unsqueeze(0)

        # Base reconstruction (1-bit + residual signs)
        base_recon_rot = self.base_polar.centroids[base_indices]
        base_correction = base_scale.unsqueeze(-1) * base_signs
        base_full_rot = base_recon_rot + base_correction

        # Refined reconstruction (3-bit + residual signs)
        refine_recon_rot = self.refine_polar.centroids[refine_indices]
        refine_correction = refine_scale.unsqueeze(-1) * refine_signs
        refine_full_rot = refine_recon_rot + refine_correction

        # Select: use refined where mask is True
        mask_expanded = refine_mask.unsqueeze(-1).expand_as(base_full_rot)
        combined_rot = torch.where(mask_expanded, refine_full_rot, base_full_rot)

        # Unrotate and rescale
        x_recon = self.base_polar.unrotate(combined_rot)
        result = x_recon * vec_norm.unsqueeze(-1)

        if squeeze:
            result = result.squeeze(0)
        return result

    def dequantize_base_only(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct using only 1-bit (no refinement)."""
        squeeze = compressed["base_indices"].dim() == 1

        base_indices = compressed["base_indices"]
        base_signs = compressed["base_signs"]
        base_scale = compressed["base_scale"]
        vec_norm = compressed["vec_norm"]

        if squeeze:
            base_indices = base_indices.unsqueeze(0)
            base_signs = base_signs.unsqueeze(0)
            base_scale = base_scale.unsqueeze(0)
            vec_norm = vec_norm.unsqueeze(0)

        base_recon_rot = self.base_polar.centroids[base_indices]
        base_correction = base_scale.unsqueeze(-1) * base_signs
        base_full_rot = base_recon_rot + base_correction

        x_recon = self.base_polar.unrotate(base_full_rot)
        result = x_recon * vec_norm.unsqueeze(-1)

        if squeeze:
            result = result.squeeze(0)
        return result

    def dequantize_refine_all(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct using 3-bit for all tokens (upper bound quality)."""
        squeeze = compressed["refine_indices"].dim() == 1

        refine_indices = compressed["refine_indices"]
        refine_signs = compressed["refine_signs"]
        refine_scale = compressed["refine_scale"]
        vec_norm = compressed["vec_norm"]

        if squeeze:
            refine_indices = refine_indices.unsqueeze(0)
            refine_signs = refine_signs.unsqueeze(0)
            refine_scale = refine_scale.unsqueeze(0)
            vec_norm = vec_norm.unsqueeze(0)

        refine_recon_rot = self.refine_polar.centroids[refine_indices]
        refine_correction = refine_scale.unsqueeze(-1) * refine_signs
        refine_full_rot = refine_recon_rot + refine_correction

        x_recon = self.base_polar.unrotate(refine_full_rot)
        result = x_recon * vec_norm.unsqueeze(-1)

        if squeeze:
            result = result.squeeze(0)
        return result

    @staticmethod
    def compute_effective_bits(
        refine_mask: torch.Tensor,
        base_bits: float = 2.0,  # 1-bit MSE + 1-bit residual sign
        refine_bits: float = 4.0,  # 3-bit MSE + 1-bit residual sign
    ) -> float:
        """Compute effective bits per coordinate given the refinement mask."""
        frac_refined = refine_mask.float().mean().item()
        return base_bits * (1 - frac_refined) + refine_bits * frac_refined


# ---------------------------------------------------------------------------
# Combined 1-bit + residual sign (no chain, no prediction — pure 1+1 bit)
# ---------------------------------------------------------------------------


class OneBitResidualQuant(nn.Module):
    """Pure 1-bit MSE + 1-bit residual sign = 2 bits total.

    This is the simplest 1-bit approach: same as ResidualQuant but at 1-bit
    MSE instead of 2-bit or 3-bit. Serves as the baseline for all
    ultra-compression experiments.

    Args:
        d: Head dimension.
        seed: Random seed.
        device: Target device.
    """

    def __init__(self, d: int, seed: int = 42, device: str | torch.device = "cpu"):
        super().__init__()
        self.d = d
        self.polar = PolarQuant(d, bits=1, seed=seed, device=device)

    def quantize(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        vec_norm = x.norm(dim=-1, keepdim=True)
        x_normalized = x / (vec_norm + 1e-8)
        x_rotated = self.polar.rotate(x_normalized)

        mse_indices = self.polar.codebook.quantize(x_rotated)
        x_mse_rotated = self.polar.centroids[mse_indices]
        residual = x_rotated - x_mse_rotated

        residual_signs = (residual >= 0).float() * 2 - 1
        residual_scale = residual.abs().mean(dim=-1)

        result = {
            "mse_indices": mse_indices,
            "residual_signs": residual_signs,
            "residual_scale": residual_scale,
            "vec_norm": vec_norm.squeeze(-1),
        }
        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}
        return result

    def dequantize(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        squeeze = compressed["mse_indices"].dim() == 1

        mse_indices = compressed["mse_indices"]
        residual_signs = compressed["residual_signs"]
        residual_scale = compressed["residual_scale"]
        vec_norm = compressed["vec_norm"]

        if squeeze:
            mse_indices = mse_indices.unsqueeze(0)
            residual_signs = residual_signs.unsqueeze(0)
            residual_scale = residual_scale.unsqueeze(0)
            vec_norm = vec_norm.unsqueeze(0)

        x_mse_rotated = self.polar.centroids[mse_indices]
        correction = residual_scale.unsqueeze(-1) * residual_signs
        x_corrected_rotated = x_mse_rotated + correction

        x_recon = self.polar.unrotate(x_corrected_rotated)
        result = x_recon * vec_norm.unsqueeze(-1)

        if squeeze:
            result = result.squeeze(0)
        return result
