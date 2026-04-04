"""Entropy analysis of quantized indices -- measuring free compression.

After Lloyd-Max quantization on a Gaussian distribution, the quantization
indices are NOT uniformly distributed. Middle centroids (near the distribution
peak) are selected far more often than tail centroids. This means the Shannon
entropy H < allocated bits, and the gap is free lossless compression.

This module provides:
    1. Per-head and per-layer entropy measurement on real KV caches
    2. Sequential correlation analysis (do adjacent coordinates predict each other?)
    3. WHT vs QR rotation comparison (does rotation type affect entropy?)
    4. Run-length analysis (are there exploitable runs of the same index?)
    5. Actual compressed size measurement using ANS and zlib

The key insight: Lloyd-Max is MSE-optimal for a GIVEN number of levels, but
it wastes bits because it assigns equal bit-width to every level. Entropy
coding reclaims those wasted bits losslessly.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .codebook import LloydMaxCodebook, gaussian_pdf
from .entropy_coding import (
    ANSEncoder,
    EntropyEncoder,
    ZlibEncoder,
    _symbol_probabilities,
    measure_index_entropy,
    theoretical_index_entropy,
)
from .polarquant import PolarQuant


# ---------------------------------------------------------------------------
# Core entropy measurement on real data
# ---------------------------------------------------------------------------


def measure_real_entropy(
    vectors: torch.Tensor,
    bits: int,
    d: int,
    seed: int = 42,
    rotation_type: str | None = None,
) -> Dict[str, float]:
    """Measure the entropy of quantized indices from real vectors.

    Quantizes the input vectors using PolarQuant and measures the empirical
    Shannon entropy of the resulting index stream.

    Args:
        vectors: Input vectors of shape (n, d), NOT necessarily unit vectors.
                 Will be normalized internally.
        bits: Bit-width for PolarQuant.
        d: Head dimension.
        seed: Random seed for rotation matrix.
        rotation_type: "wht" or "qr" (None = auto).

    Returns:
        Dict with:
            empirical_entropy: H(indices) in bits
            allocated_bits: nominal bit-width
            savings_pct: (1 - H/bits) * 100
            n_vectors: number of vectors quantized
            index_counts: per-index frequency counts
    """
    device = vectors.device
    pq = PolarQuant(d=d, bits=bits, seed=seed, device=device, rotation_type=rotation_type)

    # Normalize vectors (PolarQuant expects unit vectors)
    norms = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_normalized = vectors / norms

    # Quantize
    indices = pq.quantize(x_normalized)
    n_levels = 1 << bits

    # Measure entropy
    entropy = measure_index_entropy(indices, n_levels)

    # Count per-index usage
    flat = indices.reshape(-1).cpu().numpy()
    counts = np.bincount(flat, minlength=n_levels)

    return {
        "empirical_entropy": entropy,
        "allocated_bits": float(bits),
        "savings_pct": (1.0 - entropy / bits) * 100.0 if bits > 0 else 0.0,
        "n_vectors": vectors.shape[0],
        "index_counts": counts.tolist(),
    }


def measure_per_coordinate_entropy(
    vectors: torch.Tensor,
    bits: int,
    d: int,
    seed: int = 42,
) -> np.ndarray:
    """Measure entropy per coordinate position after rotation.

    Checks whether some coordinate positions have more or less entropy
    than others. If entropy varies by position, per-coordinate coding
    could provide additional gains.

    Args:
        vectors: Input vectors of shape (n, d).
        bits: Bit-width.
        d: Head dimension.
        seed: Random seed.

    Returns:
        Array of shape (d,) with entropy per coordinate in bits.
    """
    device = vectors.device
    pq = PolarQuant(d=d, bits=bits, seed=seed, device=device)
    n_levels = 1 << bits

    norms = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_normalized = vectors / norms
    indices = pq.quantize(x_normalized)  # (n, d)

    entropies = np.zeros(d, dtype=np.float64)
    for j in range(d):
        col = indices[:, j]
        entropies[j] = measure_index_entropy(col, n_levels)

    return entropies


# ---------------------------------------------------------------------------
# Sequential correlation analysis
# ---------------------------------------------------------------------------


def measure_sequential_correlation(
    vectors: torch.Tensor,
    bits: int,
    d: int,
    seed: int = 42,
    max_lag: int = 4,
) -> Dict[str, float]:
    """Measure how much adjacent coordinate indices predict each other.

    If index[j] is correlated with index[j+1], we could use context-based
    coding (like PPM or FSE with context) for additional compression beyond
    the zeroth-order entropy.

    Computes conditional entropy H(X_{j+1} | X_j) and compares with H(X).
    If H(X|Y) << H(X), there is exploitable sequential structure.

    Args:
        vectors: Input vectors of shape (n, d).
        bits: Bit-width.
        d: Head dimension.
        seed: Random seed.
        max_lag: Maximum lag to check (1 = adjacent pairs).

    Returns:
        Dict with:
            zeroth_order_entropy: H(X) -- marginal entropy per symbol
            conditional_entropy_lag1: H(X_{j+1} | X_j)
            correlation_gain_lag1: H(X) - H(X|X_{-1}) -- additional bits from context
            conditional_entropies: list of H(X | X_{-k}) for k=1..max_lag
    """
    device = vectors.device
    pq = PolarQuant(d=d, bits=bits, seed=seed, device=device)
    n_levels = 1 << bits

    norms = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_normalized = vectors / norms
    indices = pq.quantize(x_normalized)  # (n, d)
    flat = indices.reshape(-1).cpu().numpy()

    # Zeroth-order entropy
    h0 = measure_index_entropy(indices, n_levels)

    # Conditional entropies at various lags
    conditional_entropies = []
    for lag in range(1, max_lag + 1):
        # Build joint counts (prev_symbol, current_symbol)
        prev = flat[:-lag]
        curr = flat[lag:]
        joint_counts = np.zeros((n_levels, n_levels), dtype=np.int64)
        for p, c in zip(prev, curr):
            joint_counts[p, c] += 1

        # H(Y|X) = H(X,Y) - H(X)
        # H(X,Y) = -sum p(x,y) log2 p(x,y)
        total = joint_counts.sum()
        if total == 0:
            conditional_entropies.append(h0)
            continue

        joint_probs = joint_counts.astype(np.float64) / total
        nonzero = joint_probs[joint_probs > 0]
        h_joint = -np.sum(nonzero * np.log2(nonzero))

        # H(X) for the conditioning variable
        marginal = joint_counts.sum(axis=1).astype(np.float64) / total
        nonzero_m = marginal[marginal > 0]
        h_marginal = -np.sum(nonzero_m * np.log2(nonzero_m))

        h_cond = h_joint - h_marginal
        conditional_entropies.append(float(h_cond))

    result = {
        "zeroth_order_entropy": h0,
        "conditional_entropy_lag1": conditional_entropies[0] if conditional_entropies else h0,
        "correlation_gain_lag1": h0 - conditional_entropies[0] if conditional_entropies else 0.0,
        "conditional_entropies": conditional_entropies,
    }
    return result


# ---------------------------------------------------------------------------
# Run-length analysis
# ---------------------------------------------------------------------------


def measure_run_lengths(
    vectors: torch.Tensor,
    bits: int,
    d: int,
    seed: int = 42,
) -> Dict[str, float]:
    """Analyze run-length patterns in quantized indices.

    If there are long runs of the same index (e.g., many consecutive 3's),
    run-length encoding could be combined with entropy coding.

    Args:
        vectors: Input vectors of shape (n, d).
        bits: Bit-width.
        d: Head dimension.
        seed: Random seed.

    Returns:
        Dict with:
            avg_run_length: average length of same-symbol runs
            max_run_length: longest run
            run_count: total number of runs
            expected_random_run_length: what random (memoryless) would give
            rle_compressible: True if runs are significantly longer than random
    """
    device = vectors.device
    pq = PolarQuant(d=d, bits=bits, seed=seed, device=device)

    norms = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_normalized = vectors / norms
    indices = pq.quantize(x_normalized)
    flat = indices.reshape(-1).cpu().numpy()

    if len(flat) == 0:
        return {
            "avg_run_length": 0.0,
            "max_run_length": 0,
            "run_count": 0,
            "expected_random_run_length": 0.0,
            "rle_compressible": False,
        }

    # Count runs
    runs = []
    current_val = flat[0]
    current_len = 1
    for i in range(1, len(flat)):
        if flat[i] == current_val:
            current_len += 1
        else:
            runs.append(current_len)
            current_val = flat[i]
            current_len = 1
    runs.append(current_len)

    avg_run = np.mean(runs) if runs else 0.0
    max_run = max(runs) if runs else 0

    # Expected run length for memoryless source: 1 / (1 - max(p_i))
    n_levels = 1 << bits
    counts = np.bincount(flat, minlength=n_levels).astype(np.float64)
    probs = counts / counts.sum()
    max_p = probs.max()
    expected_random = 1.0 / (1.0 - max_p) if max_p < 1.0 else len(flat)

    return {
        "avg_run_length": float(avg_run),
        "max_run_length": int(max_run),
        "run_count": len(runs),
        "expected_random_run_length": float(expected_random),
        "rle_compressible": avg_run > expected_random * 1.5,
    }


# ---------------------------------------------------------------------------
# WHT vs QR rotation entropy comparison
# ---------------------------------------------------------------------------


def compare_rotation_entropy(
    vectors: torch.Tensor,
    bits: int,
    d: int,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compare entropy of indices under WHT vs QR rotation.

    The WHT rotation uses a structured (non-Haar-uniform) orthogonal matrix.
    This might produce indices with slightly different entropy characteristics
    compared to the fully random QR rotation.

    Args:
        vectors: Input vectors of shape (n, d). d must be power of 2.
        bits: Bit-width.
        d: Head dimension (must be power of 2 for WHT).
        seed: Random seed.

    Returns:
        Dict with "wht" and "qr" keys, each containing entropy stats.
    """
    results = {}
    for rot_type in ("wht", "qr"):
        stats = measure_real_entropy(
            vectors, bits=bits, d=d, seed=seed, rotation_type=rot_type
        )
        results[rot_type] = stats
    return results


# ---------------------------------------------------------------------------
# Actual compression measurement
# ---------------------------------------------------------------------------


def measure_actual_compression(
    vectors: torch.Tensor,
    bits: int,
    d: int,
    seed: int = 42,
) -> Dict[str, float]:
    """Measure the ACTUAL compressed size using ANS and zlib.

    This is the ground truth: encode real indices, measure the byte stream,
    and compare to raw storage.

    Args:
        vectors: Input vectors of shape (n, d).
        bits: Bit-width.
        d: Head dimension.
        seed: Random seed.

    Returns:
        Dict with:
            raw_bytes: uncompressed size (bits * n * d / 8)
            ans_bytes: ANS compressed size
            zlib_bytes: zlib compressed size
            lzma_bytes: lzma compressed size (best general-purpose)
            ans_ratio: raw / ans
            zlib_ratio: raw / zlib
            lzma_ratio: raw / lzma
            ans_bps: actual bits per symbol from ANS
            zlib_bps: actual bits per symbol from zlib
            lzma_bps: actual bits per symbol from lzma
    """
    import lzma
    import zlib

    device = vectors.device
    pq = PolarQuant(d=d, bits=bits, seed=seed, device=device)
    codebook = pq.codebook

    norms = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_normalized = vectors / norms
    indices = pq.quantize(x_normalized)

    n_symbols = indices.numel()

    # Raw size: bits * n_symbols, packed
    raw_bits = bits * n_symbols
    raw_bytes = (raw_bits + 7) // 8

    # ANS encoding
    ans_enc = ANSEncoder(codebook)
    flat_indices = indices.reshape(-1)
    ans_compressed = ans_enc.encode(flat_indices)
    ans_bytes = len(ans_compressed)

    # Zlib encoding
    flat_np = flat_indices.cpu().numpy().astype(np.uint8)
    raw_stream = flat_np.tobytes()
    zlib_compressed = zlib.compress(raw_stream, 9)  # max compression
    zlib_bytes = len(zlib_compressed)

    # LZMA encoding (best general-purpose compressor)
    lzma_compressed = lzma.compress(raw_stream, preset=9)
    lzma_bytes = len(lzma_compressed)

    return {
        "raw_bytes": raw_bytes,
        "raw_bytes_byte_per_idx": len(raw_stream),
        "ans_bytes": ans_bytes,
        "zlib_bytes": zlib_bytes,
        "lzma_bytes": lzma_bytes,
        "ans_ratio": len(raw_stream) / ans_bytes if ans_bytes > 0 else 1.0,
        "zlib_ratio": len(raw_stream) / zlib_bytes if zlib_bytes > 0 else 1.0,
        "lzma_ratio": len(raw_stream) / lzma_bytes if lzma_bytes > 0 else 1.0,
        "ans_bps": (ans_bytes * 8) / n_symbols if n_symbols > 0 else float(bits),
        "zlib_bps": (zlib_bytes * 8) / n_symbols if n_symbols > 0 else float(bits),
        "lzma_bps": (lzma_bytes * 8) / n_symbols if n_symbols > 0 else float(bits),
        "n_symbols": n_symbols,
    }


# ---------------------------------------------------------------------------
# Full per-layer/per-head analysis for a KV cache
# ---------------------------------------------------------------------------


def analyze_kv_cache_entropy(
    key_cache: List[torch.Tensor],
    bits_list: Tuple[int, ...] = (2, 3, 4),
    head_dim: int = 128,
    max_layers: Optional[int] = None,
    max_heads: Optional[int] = None,
) -> Dict:
    """Analyze entropy across all layers and heads of a real KV cache.

    Args:
        key_cache: List of key tensors, one per layer.
                   Each tensor has shape (batch, n_kv_heads, seq_len, head_dim).
        bits_list: Bit-widths to analyze.
        head_dim: Head dimension.
        max_layers: Limit to first N layers (None = all).
        max_heads: Limit to first N heads per layer (None = all).

    Returns:
        Nested dict with per-bits, per-layer, per-head entropy data,
        plus aggregates.
    """
    n_layers = len(key_cache)
    if max_layers is not None:
        n_layers = min(n_layers, max_layers)

    results = {}
    for bits in bits_list:
        layer_results = []
        all_entropies = []

        for layer_idx in range(n_layers):
            keys = key_cache[layer_idx]  # (batch, n_kv_heads, seq, head_dim)
            n_kv_heads = keys.shape[1]
            if max_heads is not None:
                n_kv_heads = min(n_kv_heads, max_heads)

            head_results = []
            for h in range(n_kv_heads):
                k = keys[0, h].float()  # (seq, head_dim)
                seed = layer_idx * 10000 + h

                stats = measure_real_entropy(
                    k, bits=bits, d=head_dim, seed=seed
                )
                stats["layer"] = layer_idx
                stats["head"] = h
                head_results.append(stats)
                all_entropies.append(stats["empirical_entropy"])

            layer_results.append({
                "layer_idx": layer_idx,
                "heads": head_results,
                "avg_entropy": np.mean([h["empirical_entropy"] for h in head_results]),
            })

        all_entropies_arr = np.array(all_entropies)
        results[bits] = {
            "layers": layer_results,
            "global_avg_entropy": float(all_entropies_arr.mean()),
            "global_std_entropy": float(all_entropies_arr.std()),
            "global_min_entropy": float(all_entropies_arr.min()),
            "global_max_entropy": float(all_entropies_arr.max()),
            "global_savings_pct": float((1.0 - all_entropies_arr.mean() / bits) * 100),
            "allocated_bits": bits,
        }

    return results
