"""Attention-optimal quantization -- minimize attention score error, not MSE.

Standard approach: min ||K - K_hat||^2 (reconstruction error)
This module: min ||softmax(Q@K^T) - softmax(Q@K_hat^T)|| (attention error)

Three strategies:

1. Mean-Removed Quantization
   Softmax is shift-invariant: softmax(x + c) = softmax(x).
   So the per-head mean of K is invisible to attention. Removing it before
   quantization reduces variance -> better codebook utilization -> lower
   attention error at the same bit budget.

2. Importance-Weighted Quantization
   Weight quantization error by attention mass: tokens that receive high
   attention should be quantized more accurately. Implemented via
   importance-weighted Lloyd-Max codebook design.

3. Rank-Preserving Quantization
   Attention only cares about the RANKING of dot products, not absolute
   values. Measure Spearman correlation between true and quantized
   attention scores as the metric that actually matters.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .polarquant import PolarQuant
from .residual_quant import ResidualQuantEstimator

# ---------------------------------------------------------------------------
# Attention metrics
# ---------------------------------------------------------------------------

def compute_attention_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute scaled attention scores: softmax(Q @ K^T / sqrt(d)).

    Args:
        query: (n_q, d) or (d,) query vectors.
        keys: (seq_len, d) key vectors.
        scale: Scaling factor. Defaults to 1/sqrt(d).

    Returns:
        Attention weights of shape (n_q, seq_len) or (seq_len,).
    """
    squeeze = query.dim() == 1
    if squeeze:
        query = query.unsqueeze(0)

    d = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # (n_q, d) @ (d, seq) -> (n_q, seq)
    logits = query @ keys.T * scale
    attn = F.softmax(logits, dim=-1)

    if squeeze:
        attn = attn.squeeze(0)
    return attn


def attention_metrics(
    attn_true: torch.Tensor,
    attn_quant: torch.Tensor,
) -> Dict[str, float]:
    """Compute suite of attention quality metrics.

    Args:
        attn_true: True attention weights (seq_len,) or (n_q, seq_len).
        attn_quant: Quantized attention weights, same shape.

    Returns:
        Dict with: cosine_sim, l1_error, top1_match, top5_match, kl_div,
        spearman_rho, attention_mass_error.
    """
    if attn_true.dim() == 1:
        attn_true = attn_true.unsqueeze(0)
        attn_quant = attn_quant.unsqueeze(0)

    n_q = attn_true.shape[0]
    results = {}

    # Cosine similarity of attention vectors
    cos_sims = F.cosine_similarity(attn_true, attn_quant, dim=-1)
    results["cosine_sim"] = cos_sims.mean().item()

    # L1 error (total variation)
    l1 = (attn_true - attn_quant).abs().sum(dim=-1)
    results["l1_error"] = l1.mean().item()

    # Top-1 match
    top1_true = attn_true.argmax(dim=-1)
    top1_quant = attn_quant.argmax(dim=-1)
    results["top1_match"] = (top1_true == top1_quant).float().mean().item()

    # Top-5 match (true top-1 in quantized top-5)
    seq_len = attn_true.shape[-1]
    k5 = min(5, seq_len)
    top5_quant = attn_quant.topk(k5, dim=-1).indices  # (n_q, 5)
    top1_in_top5 = (top5_quant == top1_true.unsqueeze(-1)).any(dim=-1)
    results["top5_match"] = top1_in_top5.float().mean().item()

    # Top-10 match
    k10 = min(10, seq_len)
    top10_quant = attn_quant.topk(k10, dim=-1).indices
    top1_in_top10 = (top10_quant == top1_true.unsqueeze(-1)).any(dim=-1)
    results["top10_match"] = top1_in_top10.float().mean().item()

    # KL divergence (true || quant), clamped for stability
    attn_quant_safe = attn_quant.clamp(min=1e-10)
    attn_true_safe = attn_true.clamp(min=1e-10)
    kl = (attn_true_safe * (attn_true_safe.log() - attn_quant_safe.log())).sum(dim=-1)
    results["kl_div"] = kl.mean().item()

    # Spearman rank correlation (averaged across queries)
    spearman_sum = 0.0
    for i in range(n_q):
        true_ranks = attn_true[i].argsort(descending=True).argsort().float()
        quant_ranks = attn_quant[i].argsort(descending=True).argsort().float()
        # Spearman = Pearson correlation of ranks
        rho = F.cosine_similarity(
            true_ranks - true_ranks.mean(),
            quant_ranks - quant_ranks.mean(),
            dim=0,
        )
        spearman_sum += rho.item()
    results["spearman_rho"] = spearman_sum / n_q

    # Attention mass error: difference in cumulative mass of top-K tokens
    # (how much probability mass shifts between true and quant top-10%)
    k_mass = max(1, seq_len // 10)  # top 10%
    true_topk_mass = attn_true.topk(k_mass, dim=-1).values.sum(dim=-1)
    quant_topk_mass = attn_quant.topk(k_mass, dim=-1).values.sum(dim=-1)
    results["mass_error"] = (true_topk_mass - quant_topk_mass).abs().mean().item()

    return results


# ---------------------------------------------------------------------------
# Experiment 1: Mean-Removed Quantization
# ---------------------------------------------------------------------------

class MeanRemovedQuantizer:
    """Quantize (K - mean_K) instead of K for better codebook utilization.

    Insight: softmax(Q @ K^T) = softmax(Q @ (K - mean_K)^T) because
    Q @ mean_K^T is a constant added to every logit, and softmax is
    shift-invariant.

    Benefits:
    - Lower variance in K after mean removal -> fewer outliers
    - Codebook centroids better utilized (centered around 0)
    - No extra storage: mean_K can be discarded since attention doesn't need it

    The mean is computed per-head across the sequence dimension.
    """

    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu"):
        self.d = d
        self.bits = bits
        self.device = device
        self.estimator = ResidualQuantEstimator(d=d, bits=bits, seed=seed, device=device)

    def quantize_and_score(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Quantize keys with mean removal and compute attention scores.

        Args:
            query: (n_q, d) or (d,)
            keys: (seq_len, d)

        Returns:
            (attn_quant, metrics) where attn_quant has same shape as
            compute_attention_scores output.
        """
        device = keys.device

        # Compute and remove per-head mean
        mean_k = keys.mean(dim=0, keepdim=True)  # (1, d)
        keys_centered = keys - mean_k

        # Quantize the centered keys
        compressed = self.estimator.quantize(keys_centered.to(device))
        keys_deq = self.estimator.dequantize(compressed)  # reconstructed centered keys

        # For attention: use centered keys directly (mean doesn't matter)
        # But we ALSO need to compare vs the centered ground truth
        attn_true = compute_attention_scores(query, keys_centered)
        attn_quant = compute_attention_scores(query, keys_deq)

        # Also compute vs original (non-centered) ground truth — should be identical
        attn_true_orig = compute_attention_scores(query, keys)

        metrics = attention_metrics(attn_true, attn_quant)
        # Verify shift-invariance: attn_true should equal attn_true_orig
        shift_error = (attn_true - attn_true_orig).abs().max().item()
        metrics["shift_invariance_error"] = shift_error

        return attn_quant, metrics


class StandardQuantizer:
    """Baseline: standard ResidualQuant without mean removal."""

    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu"):
        self.d = d
        self.bits = bits
        self.device = device
        self.estimator = ResidualQuantEstimator(d=d, bits=bits, seed=seed, device=device)

    def quantize_and_score(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Standard quantization and attention scoring."""
        compressed = self.estimator.quantize(keys)
        keys_deq = self.estimator.dequantize(compressed)

        attn_true = compute_attention_scores(query, keys)
        attn_quant = compute_attention_scores(query, keys_deq)

        metrics = attention_metrics(attn_true, attn_quant)
        return attn_quant, metrics


# ---------------------------------------------------------------------------
# Experiment 2: Importance-Weighted Quantization
# ---------------------------------------------------------------------------

class ImportanceWeightedQuantizer:
    """Weight quantization error by attention importance.

    Idea: tokens that receive high attention probability should have lower
    quantization error. We implement this by:

    1. Run a cheap "pilot" pass to estimate attention weights
       (using MSE-only quantized keys, no residual correction)
    2. Use these weights to allocate bit budget:
       - High-importance tokens: quantize at (bits+1) bits
       - Medium-importance tokens: quantize at (bits) bits
       - Low-importance tokens: quantize at (bits-1) bits

    The allocation boundaries are at the 90th and 50th percentile of
    attention mass.
    """

    def __init__(
        self,
        d: int,
        bits: int,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.d = d
        self.bits = bits
        self.device = device

        # Three quantizers at different bit levels
        bits_high = min(bits + 1, 8)
        bits_low = max(bits - 1, 1)

        self.quant_high = ResidualQuantEstimator(d=d, bits=bits_high, seed=seed, device=device)
        self.quant_mid = ResidualQuantEstimator(d=d, bits=bits, seed=seed, device=device)
        self.quant_low = ResidualQuantEstimator(d=d, bits=bits_low, seed=seed, device=device)

        # Pilot quantizer: cheap MSE-only for estimating importance
        self.pilot = PolarQuant(d=d, bits=max(bits - 1, 1), seed=seed, device=device)

        self.bits_high = bits_high
        self.bits_mid = bits
        self.bits_low = bits_low

    def quantize_and_score(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        high_pct: float = 0.90,
        mid_pct: float = 0.50,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Importance-weighted quantization.

        Args:
            query: (n_q, d) or (d,)
            keys: (seq_len, d)
            high_pct: Percentile threshold for high-importance tokens.
            mid_pct: Percentile threshold for medium-importance tokens.

        Returns:
            (attn_quant, metrics)
        """
        squeeze = query.dim() == 1
        if squeeze:
            query = query.unsqueeze(0)

        seq_len = keys.shape[0]

        # Step 1: Pilot pass — estimate importance via cheap quantization
        norms = keys.norm(dim=-1, keepdim=True)
        keys_norm = keys / (norms + 1e-8)
        pilot_indices = self.pilot.quantize(keys_norm)
        keys_pilot = self.pilot.dequantize(pilot_indices) * norms
        pilot_attn = compute_attention_scores(query, keys_pilot)  # (n_q, seq)

        # Average importance across queries
        importance = pilot_attn.mean(dim=0)  # (seq_len,)

        # Step 2: Classify tokens into importance tiers
        sorted_imp, _ = importance.sort(descending=True)
        cumsum = sorted_imp.cumsum(dim=0)
        total_mass = importance.sum()

        # Find cutoffs: top tokens capturing high_pct of mass
        high_threshold = sorted_imp[(cumsum <= total_mass * (1 - high_pct + 0.01)).sum().clamp(max=seq_len - 1)]
        mid_threshold = sorted_imp[(cumsum <= total_mass * (1 - mid_pct + 0.01)).sum().clamp(max=seq_len - 1)]

        is_high = importance >= high_threshold
        is_mid = (~is_high) & (importance >= mid_threshold)
        is_low = ~(is_high | is_mid)

        n_high = is_high.sum().item()
        n_mid = is_mid.sum().item()
        n_low = is_low.sum().item()

        # Step 3: Quantize each tier at its assigned bit-width
        keys_deq = torch.zeros_like(keys)

        if n_high > 0:
            comp_h = self.quant_high.quantize(keys[is_high])
            keys_deq[is_high] = self.quant_high.dequantize(comp_h)
        if n_mid > 0:
            comp_m = self.quant_mid.quantize(keys[is_mid])
            keys_deq[is_mid] = self.quant_mid.dequantize(comp_m)
        if n_low > 0:
            comp_l = self.quant_low.quantize(keys[is_low])
            keys_deq[is_low] = self.quant_low.dequantize(comp_l)

        # Compute attention scores
        attn_true = compute_attention_scores(query, keys)
        attn_quant = compute_attention_scores(query, keys_deq)

        metrics = attention_metrics(attn_true, attn_quant)
        metrics["n_high"] = n_high
        metrics["n_mid"] = n_mid
        metrics["n_low"] = n_low

        # Effective average bits
        avg_bits = (
            n_high * self.bits_high + n_mid * self.bits_mid + n_low * self.bits_low
        ) / max(seq_len, 1)
        metrics["avg_bits"] = avg_bits

        if squeeze:
            attn_quant = attn_quant.squeeze(0)
        return attn_quant, metrics


# ---------------------------------------------------------------------------
# Experiment 3: Rank-Preserving Analysis
# ---------------------------------------------------------------------------

def rank_preservation_analysis(
    query: torch.Tensor,
    keys: torch.Tensor,
    bits_list: List[int] = [2, 3, 4],
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """Analyze how well different quantization methods preserve attention rank order.

    Computes Spearman correlation and other rank metrics across bit widths
    for standard, mean-removed, and importance-weighted quantization.

    Args:
        query: (d,) or (n_q, d) query vectors.
        keys: (seq_len, d) key vectors.
        bits_list: Bit-widths to test.
        seed: Random seed.
        device: Target device.

    Returns:
        Nested dict: results[method_name][metric_name] = value
    """
    results = {}

    for bits in bits_list:
        # Standard
        std = StandardQuantizer(d=keys.shape[-1], bits=bits, seed=seed, device=device)
        _, std_metrics = std.quantize_and_score(query, keys)
        results[f"standard_{bits}bit"] = std_metrics

        # Mean-removed
        mr = MeanRemovedQuantizer(d=keys.shape[-1], bits=bits, seed=seed, device=device)
        _, mr_metrics = mr.quantize_and_score(query, keys)
        results[f"mean_removed_{bits}bit"] = mr_metrics

        # Importance-weighted (same average bits)
        iw = ImportanceWeightedQuantizer(d=keys.shape[-1], bits=bits, seed=seed, device=device)
        _, iw_metrics = iw.quantize_and_score(query, keys)
        results[f"importance_{bits}bit"] = iw_metrics

    return results


# ---------------------------------------------------------------------------
# Combined experiment: mean-removal + importance weighting
# ---------------------------------------------------------------------------

class CombinedOptimalQuantizer:
    """Mean-remove THEN importance-weight. Stacks both optimizations."""

    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu"):
        self.d = d
        self.bits = bits
        self.device = device
        self.iw = ImportanceWeightedQuantizer(d=d, bits=bits, seed=seed, device=device)

    def quantize_and_score(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Mean-remove, then importance-weight."""
        mean_k = keys.mean(dim=0, keepdim=True)
        keys_centered = keys - mean_k

        attn_quant, metrics = self.iw.quantize_and_score(query, keys_centered)

        # Ground truth uses centered keys (shift-invariant)
        attn_true = compute_attention_scores(query, keys)
        attn_true_centered = compute_attention_scores(query, keys_centered)

        # Recompute full metrics against original ground truth
        metrics = attention_metrics(attn_true, attn_quant)
        metrics["uses_mean_removal"] = True
        metrics["uses_importance_weighting"] = True

        return attn_quant, metrics
