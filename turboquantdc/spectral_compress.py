"""Spectral KV Cache Compression — frequency-domain alternative to spatial quantization.

Explores two fundamentally different compression strategies:

1. **DCT-based compression**: Transform each KV vector to frequency domain via
   Discrete Cosine Transform, then keep only top-K coefficients that capture
   most of the energy. Remaining coefficients are zeroed or quantized coarsely.

2. **SVD subspace projection**: For each layer, compute the principal subspace
   of the KV cache and project to a k-dimensional subspace. If k << d, this
   achieves dimensionality reduction before any quantization.

Both approaches exploit structure in KV vectors that coordinate-independent
quantization (PolarQuant/ResidualQuant) ignores.

The key question: do real LLM KV vectors have concentrated energy spectra?
If 90% of energy is in 30% of DCT coefficients, we can achieve much higher
compression than uniform-bit quantization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import dct as scipy_dct
from scipy.fft import idct as scipy_idct

# ---------------------------------------------------------------------------
# DCT via PyTorch (GPU-compatible, no scipy dependency at inference)
# ---------------------------------------------------------------------------

def dct_type2(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal DCT Type-II via FFT.

    Transforms each vector along the last dimension to frequency domain.
    Uses the standard mirror-and-FFT approach for O(d log d) computation.

    Args:
        x: Input tensor of shape (..., d).

    Returns:
        DCT coefficients of shape (..., d), orthonormally scaled.
    """
    d = x.shape[-1]
    # Create mirrored signal for DCT via FFT
    # v[0..d-1] = x, v[d..2d-1] = x reversed
    v = torch.cat([x, x.flip(-1)], dim=-1)
    V = torch.fft.rfft(v, dim=-1)

    # Twiddle factors: exp(-j*pi*k/(2*d)) for k = 0..d-1
    k = torch.arange(d, device=x.device, dtype=x.dtype)
    W = torch.exp(-1j * math.pi * k / (2 * d))

    # Extract DCT coefficients
    coeffs = (V[..., :d] * W).real

    # Orthonormal scaling: c[0] *= 1/sqrt(4d), c[k>0] *= 1/sqrt(2d)
    scale = torch.ones(d, device=x.device, dtype=x.dtype) / math.sqrt(2 * d)
    scale[0] = 1.0 / math.sqrt(4 * d)
    return coeffs * scale


def idct_type2(X: torch.Tensor) -> torch.Tensor:
    """Inverse orthonormal DCT Type-II.

    Recovers the original signal from DCT coefficients.

    Args:
        X: DCT coefficients of shape (..., d).

    Returns:
        Reconstructed signal of shape (..., d).
    """
    d = X.shape[-1]

    # Undo orthonormal scaling
    scale = torch.ones(d, device=X.device, dtype=X.dtype) / math.sqrt(2 * d)
    scale[0] = 1.0 / math.sqrt(4 * d)
    X_unscaled = X / scale

    # Inverse twiddle
    k = torch.arange(d, device=X.device, dtype=X.dtype)
    W_inv = torch.exp(1j * math.pi * k / (2 * d))

    # Reconstruct via inverse FFT
    V_partial = X_unscaled.to(torch.complex64) * W_inv
    # Pad with zeros for the full FFT
    V_full = torch.zeros(*X.shape[:-1], d + 1, device=X.device, dtype=torch.complex64)
    V_full[..., :d] = V_partial
    v = torch.fft.irfft(V_full, n=2 * d, dim=-1)
    return v[..., :d]


def dct_scipy(x_np: np.ndarray) -> np.ndarray:
    """Canonical scipy DCT Type-II with orthonormal scaling (for validation)."""
    return scipy_dct(x_np, type=2, norm='ortho', axis=-1)


def idct_scipy(X_np: np.ndarray) -> np.ndarray:
    """Canonical scipy inverse DCT Type-II with orthonormal scaling."""
    return scipy_idct(X_np, type=2, norm='ortho', axis=-1)


# ---------------------------------------------------------------------------
# Energy Analysis
# ---------------------------------------------------------------------------

@dataclass
class EnergyProfile:
    """Energy distribution analysis of DCT coefficients."""
    total_energy: float
    cumulative_energy: np.ndarray  # shape (d,), cumulative fraction
    energy_per_coeff: np.ndarray  # shape (d,), energy at each frequency
    k_for_90: int  # coefficients needed for 90% energy
    k_for_95: int  # coefficients needed for 95% energy
    k_for_99: int  # coefficients needed for 99% energy
    k_for_999: int  # coefficients needed for 99.9% energy


def analyze_energy_spectrum(
    vectors: torch.Tensor,
    use_scipy: bool = True,
) -> EnergyProfile:
    """Analyze DCT energy distribution of a batch of vectors.

    Computes what fraction of total energy is captured by the top-K
    DCT coefficients, averaged across all vectors in the batch.

    Args:
        vectors: Tensor of shape (n_vectors, d).
        use_scipy: Use scipy DCT (exact) vs PyTorch DCT (approximate).

    Returns:
        EnergyProfile with cumulative energy statistics.
    """
    if use_scipy:
        v_np = vectors.float().cpu().numpy()
        coeffs = dct_scipy(v_np)
        # Average energy per coefficient index across all vectors
        energy_per_coeff = np.mean(coeffs ** 2, axis=0)
    else:
        coeffs = dct_type2(vectors.float())
        energy_per_coeff = (coeffs ** 2).mean(dim=0).cpu().numpy()

    total_energy = float(np.sum(energy_per_coeff))

    # Sort by energy (descending) for cumulative analysis
    sorted_energy = np.sort(energy_per_coeff)[::-1]
    cumulative = np.cumsum(sorted_energy) / (total_energy + 1e-12)

    # Also compute frequency-order cumulative (low freq first)
    freq_cumulative = np.cumsum(energy_per_coeff) / (total_energy + 1e-12)

    d = vectors.shape[-1]

    def k_for_threshold(cum, threshold):
        idx = np.searchsorted(cum, threshold)
        return min(int(idx) + 1, d)

    return EnergyProfile(
        total_energy=total_energy,
        cumulative_energy=cumulative,
        energy_per_coeff=energy_per_coeff,
        k_for_90=k_for_threshold(cumulative, 0.90),
        k_for_95=k_for_threshold(cumulative, 0.95),
        k_for_99=k_for_threshold(cumulative, 0.99),
        k_for_999=k_for_threshold(cumulative, 0.999),
    )


def analyze_svd_spectrum(
    vectors: torch.Tensor,
) -> Dict[str, any]:
    """Analyze singular value spectrum of a matrix of vectors.

    Args:
        vectors: Tensor of shape (n_vectors, d).

    Returns:
        Dict with singular values, explained variance ratios, etc.
    """
    # Center the data
    mean = vectors.float().mean(dim=0, keepdim=True)
    centered = vectors.float() - mean

    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    total_var = (S ** 2).sum().item()
    explained = (S ** 2).cumsum(0) / (total_var + 1e-12)
    explained_np = explained.cpu().numpy()

    d = vectors.shape[-1]

    def k_for_threshold(threshold):
        idx = np.searchsorted(explained_np, threshold)
        return min(int(idx) + 1, d)

    return {
        "singular_values": S.cpu().numpy(),
        "explained_variance_ratio": explained_np,
        "total_variance": total_var,
        "k_for_90": k_for_threshold(0.90),
        "k_for_95": k_for_threshold(0.95),
        "k_for_99": k_for_threshold(0.99),
        "k_for_999": k_for_threshold(0.999),
        "mean": mean.squeeze(0),
        "Vh": Vh,  # Right singular vectors (principal directions)
    }


# ---------------------------------------------------------------------------
# DCT Compressor
# ---------------------------------------------------------------------------

class DCTCompressor:
    """Compress vectors by keeping top-K DCT coefficients.

    Strategy:
        1. Transform to DCT domain
        2. Keep only the K coefficients with largest magnitude
        3. Optionally quantize the kept coefficients
        4. Store: coefficient indices + values (+ optional quantized values)
        5. Reconstruct via inverse DCT of sparse coefficient vector

    Storage per vector:
        - K coefficient indices: K * ceil(log2(d)) bits
        - K coefficient values: K * value_bits bits
        - Total: K * (ceil(log2(d)) + value_bits) bits

    For d=128, K=40, value_bits=8:
        40 * (7 + 8) = 600 bits = 4.7 bits/dim average
    For d=128, K=20, value_bits=8:
        20 * (7 + 8) = 300 bits = 2.3 bits/dim average
    """

    def __init__(
        self,
        d: int,
        keep_k: int,
        value_bits: int = 16,
        use_scipy: bool = True,
    ):
        self.d = d
        self.keep_k = min(keep_k, d)
        self.value_bits = value_bits
        self.use_scipy = use_scipy

    def compress(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compress vectors by keeping top-K DCT coefficients.

        Args:
            x: Input vectors of shape (batch, d) or (d,).

        Returns:
            Dict with:
                - indices: (batch, K) indices of kept coefficients
                - values: (batch, K) values of kept coefficients
                - norms: (batch,) original vector norms (for quality tracking)
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        # Transform to DCT domain
        if self.use_scipy:
            x_np = x.float().cpu().numpy()
            coeffs_np = dct_scipy(x_np)
            coeffs = torch.from_numpy(coeffs_np).to(x.device)
        else:
            coeffs = dct_type2(x.float())

        # Find top-K by magnitude
        _, top_indices = torch.topk(coeffs.abs(), self.keep_k, dim=-1)

        # Gather the coefficient values at those indices
        top_values = torch.gather(coeffs, -1, top_indices)

        # Optionally quantize values
        if self.value_bits < 16:
            top_values = self._quantize_values(top_values)

        norms = x.float().norm(dim=-1)

        result = {
            "indices": top_indices,
            "values": top_values,
            "norms": norms,
        }
        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}
        return result

    def decompress(
        self, compressed: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Reconstruct vectors from sparse DCT coefficients.

        Args:
            compressed: Output from compress().

        Returns:
            Reconstructed vectors of shape (batch, d) or (d,).
        """
        indices = compressed["indices"]
        values = compressed["values"]

        squeeze = indices.dim() == 1
        if squeeze:
            indices = indices.unsqueeze(0)
            values = values.unsqueeze(0)

        batch = indices.shape[0]

        # Reconstruct sparse DCT coefficient vector
        coeffs = torch.zeros(batch, self.d, device=values.device, dtype=values.dtype)
        coeffs.scatter_(-1, indices, values)

        # Inverse DCT
        if self.use_scipy:
            coeffs_np = coeffs.cpu().numpy()
            x_np = idct_scipy(coeffs_np)
            x = torch.from_numpy(x_np).to(values.device).float()
        else:
            x = idct_type2(coeffs)

        if squeeze:
            x = x.squeeze(0)
        return x

    def _quantize_values(self, values: torch.Tensor) -> torch.Tensor:
        """Simple min-max quantization of coefficient values."""
        n_levels = (1 << self.value_bits) - 1
        vmin = values.min(dim=-1, keepdim=True).values
        vmax = values.max(dim=-1, keepdim=True).values
        scale = (vmax - vmin + 1e-8)
        normalized = (values - vmin) / scale
        quantized = torch.round(normalized * n_levels) / n_levels
        return quantized * scale + vmin

    def bits_per_dim(self) -> float:
        """Compute average bits per dimension for this configuration."""
        index_bits = math.ceil(math.log2(self.d))  # bits per index
        total_bits_per_vector = self.keep_k * (index_bits + self.value_bits)
        return total_bits_per_vector / self.d


# ---------------------------------------------------------------------------
# SVD Subspace Compressor
# ---------------------------------------------------------------------------

class SVDCompressor:
    """Compress vectors by projecting to a learned low-rank subspace.

    Strategy:
        1. Compute SVD of training vectors: X = U S V^T
        2. Keep top-k right singular vectors V_k (shape: k x d)
        3. Project new vectors: z = (x - mean) @ V_k^T (shape: batch x k)
        4. Reconstruct: x_hat = z @ V_k + mean
        5. Optionally quantize the k-dimensional projections

    Storage per vector (after fitting):
        - k projected coordinates at value_bits each: k * value_bits
        - Shared across all vectors: V_k (k x d x 32 bits) + mean (d x 32 bits)

    For d=128, k=32, value_bits=16:
        32 * 16 = 512 bits = 4.0 bits/dim average (ignoring shared overhead)
    For d=128, k=16, value_bits=16:
        16 * 16 = 256 bits = 2.0 bits/dim average
    """

    def __init__(self, d: int, k: int, value_bits: int = 16):
        self.d = d
        self.k = min(k, d)
        self.value_bits = value_bits
        self.Vk: Optional[torch.Tensor] = None  # (k, d)
        self.mean: Optional[torch.Tensor] = None  # (d,)
        self._fitted = False

    def fit(self, training_vectors: torch.Tensor) -> Dict[str, any]:
        """Fit the subspace from training data.

        Args:
            training_vectors: (n, d) tensor of representative vectors.

        Returns:
            Dict with fitting statistics (explained variance, etc.).
        """
        self.mean = training_vectors.float().mean(dim=0)
        centered = training_vectors.float() - self.mean

        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

        self.Vk = Vh[:self.k]  # (k, d) — top-k principal directions

        total_var = (S ** 2).sum().item()
        kept_var = (S[:self.k] ** 2).sum().item()

        self._fitted = True
        return {
            "explained_variance": kept_var / (total_var + 1e-12),
            "singular_values": S.cpu().numpy(),
            "k": self.k,
            "d": self.d,
        }

    def compress(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Project vectors to low-rank subspace.

        Args:
            x: Input vectors of shape (batch, d) or (d,).

        Returns:
            Dict with projections and metadata.
        """
        assert self._fitted, "Must call fit() before compress()"

        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        centered = x.float() - self.mean.to(x.device)
        # Project: z = centered @ Vk^T, shape (batch, k)
        z = centered @ self.Vk.to(x.device).T

        norms = x.float().norm(dim=-1)

        result = {
            "projections": z,
            "norms": norms,
        }
        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}
        return result

    def decompress(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct vectors from subspace projections.

        Args:
            compressed: Output from compress().

        Returns:
            Reconstructed vectors.
        """
        z = compressed["projections"]

        squeeze = z.dim() == 1
        if squeeze:
            z = z.unsqueeze(0)

        # Reconstruct: x_hat = z @ Vk + mean
        x_hat = z @ self.Vk.to(z.device) + self.mean.to(z.device)

        if squeeze:
            x_hat = x_hat.squeeze(0)
        return x_hat

    def bits_per_dim(self, n_vectors: int = 1000) -> float:
        """Compute average bits per dimension (amortizing shared overhead).

        Args:
            n_vectors: Number of vectors (for amortizing V_k and mean storage).
        """
        per_vector_bits = self.k * self.value_bits
        # Shared cost: V_k is k*d*32 bits, mean is d*32 bits
        shared_bits = self.k * self.d * 32 + self.d * 32
        total = n_vectors * per_vector_bits + shared_bits
        return total / (n_vectors * self.d)


# ---------------------------------------------------------------------------
# Hybrid: SVD + DCT
# ---------------------------------------------------------------------------

class HybridCompressor:
    """SVD subspace projection followed by DCT of residual.

    Two-stage compression:
        Stage 1: Project to top-k SVD subspace (captures global structure)
        Stage 2: Apply DCT to the residual and keep top-K2 coefficients

    This combines the best of both worlds:
        - SVD captures inter-vector correlations (layer-specific patterns)
        - DCT captures intra-vector frequency structure
    """

    def __init__(
        self,
        d: int,
        svd_k: int,
        dct_keep: int = 0,
        value_bits: int = 16,
    ):
        self.d = d
        self.svd = SVDCompressor(d, svd_k, value_bits)
        self.dct = DCTCompressor(d, dct_keep, value_bits) if dct_keep > 0 else None
        self.dct_keep = dct_keep

    def fit(self, training_vectors: torch.Tensor) -> Dict:
        return self.svd.fit(training_vectors)

    def compress(self, x: torch.Tensor) -> Dict[str, any]:
        svd_comp = self.svd.compress(x)
        result = {"svd": svd_comp}

        if self.dct is not None:
            # Compute residual
            x_svd = self.svd.decompress(svd_comp)
            residual = x.float() - x_svd
            dct_comp = self.dct.compress(residual)
            result["dct_residual"] = dct_comp

        return result

    def decompress(self, compressed: Dict) -> torch.Tensor:
        x_svd = self.svd.decompress(compressed["svd"])
        if self.dct is not None and "dct_residual" in compressed:
            residual = self.dct.decompress(compressed["dct_residual"])
            return x_svd + residual
        return x_svd


# ---------------------------------------------------------------------------
# Quality Metrics
# ---------------------------------------------------------------------------

def compute_quality_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    queries: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute reconstruction and inner-product quality metrics.

    Args:
        original: Original vectors, shape (n, d).
        reconstructed: Reconstructed vectors, shape (n, d).
        queries: Optional query vectors for attention-quality metrics.

    Returns:
        Dict of metric name -> value.
    """
    orig = original.float()
    recon = reconstructed.float()

    # Per-vector cosine similarity
    cos_sim = F.cosine_similarity(orig, recon, dim=-1)

    # MSE
    mse = ((orig - recon) ** 2).mean().item()

    # Relative error
    orig_norm = orig.norm(dim=-1)
    error_norm = (orig - recon).norm(dim=-1)
    rel_error = (error_norm / (orig_norm + 1e-8)).mean().item()

    metrics = {
        "cosine_sim_mean": cos_sim.mean().item(),
        "cosine_sim_min": cos_sim.min().item(),
        "cosine_sim_std": cos_sim.std().item(),
        "mse": mse,
        "relative_error": rel_error,
    }

    if queries is not None:
        # Attention score quality
        # True attention scores
        true_scores = queries.float() @ orig.T  # (n_q, n_k)
        recon_scores = queries.float() @ recon.T

        # Cosine similarity of full attention score vectors
        attn_cos = F.cosine_similarity(
            true_scores, recon_scores, dim=-1
        )
        metrics["attention_cosine_mean"] = attn_cos.mean().item()

        # Top-1 match rate
        true_top1 = true_scores.argmax(dim=-1)
        recon_top1 = recon_scores.argmax(dim=-1)
        metrics["top1_match"] = (true_top1 == recon_top1).float().mean().item()

        # Top-5 match rate
        if true_scores.shape[-1] >= 5:
            true_top5 = true_scores.topk(5, dim=-1).indices
            recon_top5 = recon_scores.topk(5, dim=-1).indices
            # Check if true top-1 is in reconstructed top-5
            true_top1_expanded = true_top1.unsqueeze(-1)
            in_top5 = (recon_top5 == true_top1_expanded).any(dim=-1)
            metrics["top1_in_top5"] = in_top5.float().mean().item()

    return metrics
