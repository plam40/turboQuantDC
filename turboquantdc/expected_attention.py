"""Expected Attention pruning for KV cache -- predicts FUTURE importance.

Given recent query statistics q ~ N(mu_q, Sigma_q), the expected attention
weight for key k_i is analytically computable WITHOUT materializing the
attention matrix:

    score_i = exp(mu_q @ k_i / sqrt(d) + 0.5 * k_i^T @ Sigma_q @ k_i / d)
    importance_i = score_i / sum(scores)

This predicts which tokens FUTURE queries will attend to, enabling proactive
eviction/compression of tokens the model won't need. O(n*d) compute, no O(n^2).

Reference: arxiv 2510.00636

Comparison to adaptive_bits.py ImportanceScorer:
    - ImportanceScorer: EMA of PAST attention weights. Reactive -- can only
      observe what the model already attended to.
    - ExpectedAttentionScorer: Predicts FUTURE attention from the query
      distribution. Proactive -- identifies tokens that WILL be important
      even if they haven't been heavily attended to yet.

Usage::

    scorer = ExpectedAttentionScorer(d=128, window=64)
    # Feed queries as they arrive
    scorer.update_queries(q)  # (batch, d) query vectors
    # Score all cached keys
    importance = scorer.score(keys)  # (n_keys,) importance scores

    # Full cache wrapper
    cache = ExpectedAttentionCache(d=128, rescore_interval=32)
    # Use as drop-in replacement...
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Expected Attention Scorer
# ---------------------------------------------------------------------------


class ExpectedAttentionScorer:
    """Predict future attention importance from query distribution statistics.

    Tracks running mean and covariance of recent queries, then uses the
    analytic formula to compute expected softmax attention scores for any
    set of keys -- without materializing the n*n attention matrix.

    The math (arxiv 2510.00636):
        If q ~ N(mu, Sigma), then:
            E[softmax(q @ K^T / sqrt(d))]_i
            ~ exp(mu @ k_i / sqrt(d) + 0.5 * k_i^T Sigma k_i / d) / Z

        where Z is the normalizing sum.

    Args:
        d: Head dimension.
        window: Number of recent queries to track (default: 64).
            Uses a rolling window for mean/covariance estimation.
        use_diagonal_cov: If True, track only diagonal of covariance
            matrix. O(d) storage and O(n*d) scoring instead of O(d^2)
            and O(n*d^2). Slight accuracy loss but much faster for
            large d. Default: True.
        ema_decay: EMA decay for statistics update (default: 0.0,
            meaning use exact window statistics). If > 0, uses
            exponential moving average instead of rolling window.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        window: int = 64,
        use_diagonal_cov: bool = True,
        ema_decay: float = 0.0,
        device: str | torch.device = "cpu",
    ):
        self.d = d
        self.window = window
        self.use_diagonal_cov = use_diagonal_cov
        self.ema_decay = ema_decay
        self.device = torch.device(device)

        # Running statistics
        self._query_buffer: List[torch.Tensor] = []  # recent queries
        self._mu: Optional[torch.Tensor] = None  # (d,)
        self._cov: Optional[torch.Tensor] = None  # (d,) diagonal or (d, d) full
        self._n_queries: int = 0

    @property
    def is_ready(self) -> bool:
        """Whether enough queries have been seen to make predictions."""
        return self._mu is not None and self._n_queries >= 4

    @property
    def n_queries_seen(self) -> int:
        return self._n_queries

    def update_queries(self, queries: torch.Tensor) -> None:
        """Update running statistics with new query vectors.

        Args:
            queries: (batch, d) or (d,) query vectors.
                Can also be (batch, heads, d) -- will average over heads.
        """
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)
        if queries.dim() == 3:
            # (batch, heads, d) -> (batch, d) by averaging heads
            queries = queries.mean(dim=1)

        queries = queries.detach().float().to(self.device)

        if self.ema_decay > 0:
            self._update_ema(queries)
        else:
            self._update_window(queries)

        self._n_queries += queries.shape[0]

    def _update_window(self, queries: torch.Tensor) -> None:
        """Update statistics using a rolling window of recent queries."""
        # Add to buffer
        for i in range(queries.shape[0]):
            self._query_buffer.append(queries[i])

        # Trim to window size
        if len(self._query_buffer) > self.window:
            self._query_buffer = self._query_buffer[-self.window:]

        # Recompute statistics from the window
        if len(self._query_buffer) >= 2:
            window_q = torch.stack(self._query_buffer, dim=0)  # (W, d)
            self._mu = window_q.mean(dim=0)  # (d,)

            centered = window_q - self._mu.unsqueeze(0)  # (W, d)
            if self.use_diagonal_cov:
                # Diagonal covariance: variance per dimension
                self._cov = (centered ** 2).mean(dim=0)  # (d,)
            else:
                # Full covariance matrix
                self._cov = (centered.T @ centered) / centered.shape[0]  # (d, d)

    def _update_ema(self, queries: torch.Tensor) -> None:
        """Update statistics using exponential moving average."""
        alpha = self.ema_decay
        batch_mu = queries.mean(dim=0)  # (d,)

        if self._mu is None:
            self._mu = batch_mu
            centered = queries - batch_mu.unsqueeze(0)
            if self.use_diagonal_cov:
                self._cov = (centered ** 2).mean(dim=0)
            else:
                self._cov = (centered.T @ centered) / max(centered.shape[0], 1)
        else:
            self._mu = alpha * self._mu + (1 - alpha) * batch_mu
            centered = queries - self._mu.unsqueeze(0)
            if self.use_diagonal_cov:
                batch_var = (centered ** 2).mean(dim=0)
                self._cov = alpha * self._cov + (1 - alpha) * batch_var
            else:
                batch_cov = (centered.T @ centered) / max(centered.shape[0], 1)
                self._cov = alpha * self._cov + (1 - alpha) * batch_cov

    def score(self, keys: torch.Tensor) -> torch.Tensor:
        """Compute expected attention importance for each key vector.

        Uses the analytic formula:
            score_i = exp(mu @ k_i / sqrt(d) + 0.5 * k_i^T @ Sigma @ k_i / d)
            importance_i = score_i / sum(scores)

        Args:
            keys: (n_keys, d) key vectors.

        Returns:
            (n_keys,) importance scores summing to 1.
        """
        if not self.is_ready:
            # Uniform scores if not enough data
            n = keys.shape[0]
            return torch.ones(n, device=keys.device) / max(n, 1)

        keys = keys.float().to(self.device)
        n, d = keys.shape
        assert d == self.d, f"Key dim {d} != scorer dim {self.d}"

        scale = 1.0 / math.sqrt(d)

        # Term 1: mu @ k_i / sqrt(d)
        # (n,) = (n, d) @ (d,)
        mean_term = (keys @ self._mu) * scale  # (n,)

        # Term 2: 0.5 * k_i^T @ Sigma @ k_i / d
        if self.use_diagonal_cov:
            # Diagonal case: k_i^T diag(sigma) k_i = sum(sigma * k_i^2)
            # (n,) = (n, d) * (d,) -> sum over d
            var_term = 0.5 * ((keys ** 2) @ self._cov) / d  # (n,)
        else:
            # Full covariance: k_i^T @ Sigma @ k_i
            # (n, d) @ (d, d) -> (n, d), then element-wise * keys -> sum
            Sk = keys @ self._cov  # (n, d)
            var_term = 0.5 * (Sk * keys).sum(dim=-1) / d  # (n,)

        # Log-scores for numerical stability
        log_scores = mean_term + var_term  # (n,)

        # Softmax to get importance (numerically stable)
        importance = torch.softmax(log_scores, dim=0)  # (n,)

        return importance

    def score_with_details(self, keys: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Score keys and return detailed breakdown.

        Returns dict with:
            importance: (n,) normalized scores
            log_scores: (n,) unnormalized log-scores
            mean_term: (n,) contribution from query mean
            var_term: (n,) contribution from query variance
        """
        if not self.is_ready:
            n = keys.shape[0]
            uniform = torch.ones(n, device=keys.device) / max(n, 1)
            return {
                "importance": uniform,
                "log_scores": torch.zeros(n, device=keys.device),
                "mean_term": torch.zeros(n, device=keys.device),
                "var_term": torch.zeros(n, device=keys.device),
            }

        keys = keys.float().to(self.device)
        n, d = keys.shape
        scale = 1.0 / math.sqrt(d)

        mean_term = (keys @ self._mu) * scale
        if self.use_diagonal_cov:
            var_term = 0.5 * ((keys ** 2) @ self._cov) / d
        else:
            Sk = keys @ self._cov
            var_term = 0.5 * (Sk * keys).sum(dim=-1) / d

        log_scores = mean_term + var_term
        importance = torch.softmax(log_scores, dim=0)

        return {
            "importance": importance,
            "log_scores": log_scores,
            "mean_term": mean_term,
            "var_term": var_term,
        }

    def reset(self) -> None:
        """Reset all tracked statistics."""
        self._query_buffer.clear()
        self._mu = None
        self._cov = None
        self._n_queries = 0

    def stats(self) -> Dict[str, Any]:
        """Return current query distribution statistics."""
        result: Dict[str, Any] = {
            "n_queries_seen": self._n_queries,
            "window_size": self.window,
            "buffer_size": len(self._query_buffer),
            "is_ready": self.is_ready,
            "use_diagonal_cov": self.use_diagonal_cov,
        }
        if self._mu is not None:
            result["mu_norm"] = self._mu.norm().item()
            result["mu_mean"] = self._mu.mean().item()
        if self._cov is not None:
            if self.use_diagonal_cov:
                result["cov_trace"] = self._cov.sum().item()
                result["cov_max_var"] = self._cov.max().item()
                result["cov_min_var"] = self._cov.min().item()
            else:
                result["cov_trace"] = torch.trace(self._cov).item()
                result["cov_frobenius"] = self._cov.norm().item()
        return result


# ---------------------------------------------------------------------------
# Expected Attention Cache
# ---------------------------------------------------------------------------


class ExpectedAttentionCache:
    """KV cache with Expected Attention-based eviction and tiered compression.

    Wraps cached keys/values and periodically re-scores all tokens using
    the Expected Attention formula. Tokens are sorted into tiers:

        Top tier (top_pct):      Keep at FP16 (or current bit-width)
        Middle tier (mid_pct):   Keep at current compression (3-bit)
        Bottom tier (rest):      Evict entirely (or drop to 1-bit)

    This achieves effective compression of:
        5x (TurboQuant base) * 2-3x (eviction) = 10-15x

    Args:
        d: Head dimension.
        rescore_interval: Re-score all tokens every N decode steps (default: 32).
        scorer_window: Number of recent queries to track (default: 64).
        top_pct: Fraction of tokens in the top tier (default: 0.20).
        mid_pct: Fraction in middle tier (default: 0.30).
            Bottom tier = 1.0 - top_pct - mid_pct (default: 0.50).
        use_diagonal_cov: Use diagonal covariance (default: True).
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        rescore_interval: int = 32,
        scorer_window: int = 64,
        top_pct: float = 0.20,
        mid_pct: float = 0.30,
        use_diagonal_cov: bool = True,
        device: str | torch.device = "cpu",
    ):
        self.d = d
        self.rescore_interval = rescore_interval
        self.top_pct = top_pct
        self.mid_pct = mid_pct
        self.bot_pct = 1.0 - top_pct - mid_pct
        self.device = torch.device(device)

        self.scorer = ExpectedAttentionScorer(
            d=d,
            window=scorer_window,
            use_diagonal_cov=use_diagonal_cov,
            device=device,
        )

        # Storage
        self._keys: List[torch.Tensor] = []  # list of (batch, d) chunks
        self._values: List[torch.Tensor] = []
        self._seq_len: int = 0
        self._step_count: int = 0
        self._tier_assignments: Optional[torch.Tensor] = None
        self._evicted_count: int = 0

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Add new KV pairs.

        Args:
            keys: (batch, d) or (d,) key vectors.
            values: (batch, d) or (d,) value vectors.
        """
        if keys.dim() == 1:
            keys = keys.unsqueeze(0)
            values = values.unsqueeze(0)
        self._keys.append(keys.detach().float().to(self.device))
        self._values.append(values.detach().float().to(self.device))
        self._seq_len += keys.shape[0]

    def update_queries(self, queries: torch.Tensor) -> None:
        """Feed new query vectors to update the scorer.

        Args:
            queries: (batch, d) or (batch, heads, d) query vectors.
        """
        self.scorer.update_queries(queries)
        self._step_count += 1

        if (
            self._step_count > 0
            and self._step_count % self.rescore_interval == 0
            and self.scorer.is_ready
        ):
            self._rescore_and_evict()

    def _rescore_and_evict(self) -> None:
        """Re-score all cached keys and evict bottom tier."""
        if self._seq_len == 0:
            return

        all_keys = torch.cat(self._keys, dim=0)  # (seq_len, d)
        all_values = torch.cat(self._values, dim=0)

        importance = self.scorer.score(all_keys)  # (seq_len,)

        n = all_keys.shape[0]
        n_top = max(1, int(self.top_pct * n))
        n_mid = max(1, int(self.mid_pct * n))

        # Sort by importance (descending)
        sorted_idx = torch.argsort(importance, descending=True)

        # Tier assignments
        self._tier_assignments = torch.full((n,), 2, dtype=torch.long)  # default: evict
        self._tier_assignments[sorted_idx[:n_top]] = 0  # top tier
        self._tier_assignments[sorted_idx[n_top:n_top + n_mid]] = 1  # middle tier

        # Keep only top + middle tier
        keep_mask = self._tier_assignments < 2  # tiers 0 and 1

        # Always protect the first few tokens (prompt/instruction context)
        n_protect = max(int(0.05 * n), min(4, n))
        keep_mask[:n_protect] = True

        # Always protect the last few tokens (recent context)
        n_recent = min(32, n)
        keep_mask[-n_recent:] = True

        n_keep = keep_mask.sum().item()
        n_evict = n - n_keep
        self._evicted_count += n_evict

        # Apply eviction
        self._keys = [all_keys[keep_mask]]
        self._values = [all_values[keep_mask]]
        self._seq_len = n_keep

    def get_keys(self) -> torch.Tensor:
        """Return all cached keys."""
        if self._seq_len == 0:
            return torch.zeros(0, self.d, device=self.device)
        return torch.cat(self._keys, dim=0)

    def get_values(self) -> torch.Tensor:
        """Return all cached values."""
        if self._seq_len == 0:
            return torch.zeros(0, self.d, device=self.device)
        return torch.cat(self._values, dim=0)

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def effective_compression(self, base_compression: float = 5.0) -> float:
        """Estimate effective compression ratio.

        Args:
            base_compression: Base compression from TurboQuant (default: 5x).

        Returns:
            Effective compression = base * (total_seen / retained).
        """
        total_seen = self._seq_len + self._evicted_count
        if self._seq_len == 0:
            return base_compression
        eviction_factor = total_seen / max(self._seq_len, 1)
        return base_compression * eviction_factor

    def stats(self) -> Dict[str, Any]:
        """Report cache statistics."""
        total_seen = self._seq_len + self._evicted_count
        return {
            "seq_len": self._seq_len,
            "evicted": self._evicted_count,
            "total_seen": total_seen,
            "retention_rate": self._seq_len / max(total_seen, 1),
            "eviction_rate": self._evicted_count / max(total_seen, 1),
            "effective_compression": self.effective_compression(),
            "scorer": self.scorer.stats(),
        }

    def reset(self) -> None:
        """Clear all state."""
        self._keys.clear()
        self._values.clear()
        self._seq_len = 0
        self._step_count = 0
        self._tier_assignments = None
        self._evicted_count = 0
        self.scorer.reset()


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def compare_scorers(
    keys: torch.Tensor,
    queries_past: torch.Tensor,
    queries_future: torch.Tensor,
    d: int,
    ema_decay: float = 0.9,
    scorer_window: int = 64,
) -> Dict[str, Any]:
    """Compare Expected Attention vs EMA scoring against ground truth.

    Ground truth = actual attention weights from future queries.

    Args:
        keys: (n_keys, d) cached key vectors.
        queries_past: (n_past, d) queries used so far (for EMA scorer).
        queries_future: (n_future, d) future queries (ground truth).
        d: Head dimension.
        ema_decay: EMA decay for ImportanceScorer.
        scorer_window: Window for ExpectedAttentionScorer.

    Returns:
        Dict with Spearman correlation, top-K overlap, etc.
    """
    from .adaptive_bits import ImportanceScorer
    from scipy.stats import spearmanr

    n_keys = keys.shape[0]
    scale = 1.0 / math.sqrt(d)

    # --- Ground truth: actual future attention ---
    # Compute softmax attention from future queries to all keys
    # future_attn: (n_future, n_keys)
    future_scores = (queries_future @ keys.T) * scale
    future_attn = torch.softmax(future_scores, dim=-1)
    # Average attention each key receives from future queries
    ground_truth = future_attn.mean(dim=0)  # (n_keys,)
    ground_truth = ground_truth / ground_truth.sum().clamp(min=1e-10)

    # --- Expected Attention scorer ---
    ea_scorer = ExpectedAttentionScorer(d=d, window=scorer_window, device=keys.device)
    ea_scorer.update_queries(queries_past)
    ea_importance = ea_scorer.score(keys)

    # --- EMA scorer ---
    # Simulate EMA by computing actual attention from past queries
    ema_scorer = ImportanceScorer(ema_decay=ema_decay)
    # Feed past attention in chunks to simulate streaming
    chunk_size = min(32, queries_past.shape[0])
    for i in range(0, queries_past.shape[0], chunk_size):
        chunk_q = queries_past[i:i + chunk_size]
        # Compute attention for this chunk
        chunk_scores = (chunk_q @ keys.T) * scale
        chunk_attn = torch.softmax(chunk_scores, dim=-1)
        # Shape: (1, 1, chunk_size, n_keys) for ImportanceScorer
        ema_scorer.update(chunk_attn.unsqueeze(0).unsqueeze(0))

    ema_importance = ema_scorer.scores
    if ema_importance is not None:
        ema_importance = ema_importance / ema_importance.sum().clamp(min=1e-10)
    else:
        ema_importance = torch.ones(n_keys) / n_keys

    # --- Spearman correlation ---
    gt_np = ground_truth.cpu().numpy()
    ea_np = ea_importance.cpu().numpy()
    ema_np = ema_importance.cpu().numpy()

    ea_spearman = spearmanr(gt_np, ea_np).statistic
    ema_spearman = spearmanr(gt_np, ema_np).statistic

    # --- Top-K overlap ---
    results = {
        "ea_spearman": ea_spearman,
        "ema_spearman": ema_spearman,
        "spearman_advantage": ea_spearman - ema_spearman,
    }

    for k_pct in [0.10, 0.20, 0.30, 0.50]:
        k = max(1, int(k_pct * n_keys))
        gt_topk = set(torch.topk(ground_truth, k).indices.cpu().tolist())
        ea_topk = set(torch.topk(ea_importance, k).indices.cpu().tolist())
        ema_topk = set(torch.topk(ema_importance, k).indices.cpu().tolist())

        ea_overlap = len(gt_topk & ea_topk) / k
        ema_overlap = len(gt_topk & ema_topk) / k

        results[f"top{int(k_pct*100)}_ea_overlap"] = ea_overlap
        results[f"top{int(k_pct*100)}_ema_overlap"] = ema_overlap
        results[f"top{int(k_pct*100)}_advantage"] = ea_overlap - ema_overlap

    return results


def simulate_eviction(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    importance: torch.Tensor,
    eviction_rate: float,
    protect_recent: int = 32,
    protect_prompt: float = 0.05,
) -> Dict[str, float]:
    """Simulate token eviction and measure attention output quality.

    Args:
        keys: (n, d) key vectors.
        values: (n, d) value vectors.
        queries: (n_q, d) query vectors for quality measurement.
        importance: (n,) importance scores for eviction decisions.
        eviction_rate: Fraction of tokens to evict (0.0 to 1.0).
        protect_recent: Number of recent tokens to always keep.
        protect_prompt: Fraction of initial tokens to always keep.

    Returns:
        Dict with quality metrics after eviction.
    """
    n, d = keys.shape
    scale = 1.0 / math.sqrt(d)

    # Compute ground truth attention output (no eviction)
    gt_scores = (queries @ keys.T) * scale  # (n_q, n)
    gt_attn = torch.softmax(gt_scores, dim=-1)  # (n_q, n)
    gt_output = gt_attn @ values  # (n_q, d)

    # Decide which tokens to keep
    keep_mask = torch.zeros(n, dtype=torch.bool, device=keys.device)

    # Always protect prompt and recent tokens
    n_protect_prompt = max(int(protect_prompt * n), min(4, n))
    n_protect_recent = min(protect_recent, n)
    keep_mask[:n_protect_prompt] = True
    keep_mask[-n_protect_recent:] = True

    # From remaining (evictable) tokens, keep the most important ones
    evictable = ~keep_mask
    n_evictable = evictable.sum().item()
    n_to_evict = int(eviction_rate * n)
    n_to_keep_from_evictable = max(0, n_evictable - n_to_evict)

    if n_to_keep_from_evictable > 0 and n_evictable > 0:
        evictable_importance = importance[evictable]
        _, keep_local_idx = torch.topk(
            evictable_importance, k=min(n_to_keep_from_evictable, n_evictable)
        )
        evictable_indices = torch.where(evictable)[0]
        keep_mask[evictable_indices[keep_local_idx]] = True

    # Apply eviction
    kept_keys = keys[keep_mask]
    kept_values = values[keep_mask]

    # Compute attention output after eviction
    evict_scores = (queries @ kept_keys.T) * scale
    evict_attn = torch.softmax(evict_scores, dim=-1)
    evict_output = evict_attn @ kept_values

    # Quality metrics
    # Cosine similarity of attention outputs
    cos_sim = F.cosine_similarity(gt_output, evict_output, dim=-1).mean().item()

    # L2 error
    l2_error = (gt_output - evict_output).norm(dim=-1).mean().item()
    gt_norm = gt_output.norm(dim=-1).mean().item()
    relative_error = l2_error / max(gt_norm, 1e-10)

    # Top-K attention match (do the same tokens get highest attention?)
    gt_top10 = set(torch.topk(gt_attn.mean(dim=0), max(1, n // 10)).indices.cpu().tolist())
    # Map evicted indices back -- check which of the original top tokens survived
    kept_indices = torch.where(keep_mask)[0].cpu().tolist()
    evict_top10_local = torch.topk(
        evict_attn.mean(dim=0), min(max(1, n // 10), kept_keys.shape[0])
    ).indices.cpu().tolist()
    evict_top10 = set(kept_indices[i] for i in evict_top10_local)
    top10_match = len(gt_top10 & evict_top10) / max(len(gt_top10), 1)

    n_kept = keep_mask.sum().item()
    actual_eviction_rate = 1.0 - n_kept / n

    return {
        "cosine_similarity": cos_sim,
        "relative_error": relative_error,
        "l2_error": l2_error,
        "top10_attention_match": top10_match,
        "tokens_kept": n_kept,
        "tokens_evicted": n - n_kept,
        "actual_eviction_rate": actual_eviction_rate,
        "effective_compression": n / max(n_kept, 1),
    }
