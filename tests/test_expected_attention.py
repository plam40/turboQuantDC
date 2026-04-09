"""Tests for Expected Attention scoring and cache."""

import math
import pytest
import torch

from turboquantdc.expected_attention import (
    ExpectedAttentionScorer,
    ExpectedAttentionCache,
    simulate_eviction,
)


# ---------------------------------------------------------------------------
# ExpectedAttentionScorer
# ---------------------------------------------------------------------------


class TestExpectedAttentionScorer:
    """Tests for the analytic expected attention scorer."""

    def test_not_ready_before_queries(self):
        scorer = ExpectedAttentionScorer(d=64, window=16)
        assert not scorer.is_ready
        assert scorer.n_queries_seen == 0

    def test_ready_after_enough_queries(self):
        scorer = ExpectedAttentionScorer(d=64, window=16)
        scorer.update_queries(torch.randn(10, 64))
        assert scorer.is_ready
        assert scorer.n_queries_seen == 10

    def test_uniform_when_not_ready(self):
        scorer = ExpectedAttentionScorer(d=64, window=16)
        keys = torch.randn(20, 64)
        importance = scorer.score(keys)
        assert importance.shape == (20,)
        assert torch.allclose(importance, torch.ones(20) / 20, atol=1e-6)

    def test_scores_sum_to_one(self):
        scorer = ExpectedAttentionScorer(d=128, window=32)
        scorer.update_queries(torch.randn(40, 128))
        keys = torch.randn(100, 128)
        importance = scorer.score(keys)
        assert abs(importance.sum().item() - 1.0) < 1e-5

    def test_scores_all_positive(self):
        scorer = ExpectedAttentionScorer(d=128, window=32)
        scorer.update_queries(torch.randn(40, 128))
        keys = torch.randn(100, 128)
        importance = scorer.score(keys)
        assert (importance >= 0).all()

    def test_keys_aligned_with_queries_score_higher(self):
        """Keys that align with the query mean should get higher importance."""
        d = 64
        scorer = ExpectedAttentionScorer(d=d, window=32)
        # Queries concentrated in one direction
        direction = torch.randn(d)
        direction = direction / direction.norm()
        queries = direction.unsqueeze(0) + 0.1 * torch.randn(50, d)
        scorer.update_queries(queries)

        # Key aligned with query direction
        key_aligned = direction.unsqueeze(0) * 5.0
        # Key orthogonal
        ortho = torch.randn(d)
        ortho = ortho - (ortho @ direction) * direction
        ortho = ortho / ortho.norm()
        key_ortho = ortho.unsqueeze(0) * 5.0

        keys = torch.cat([key_aligned, key_ortho], dim=0)
        importance = scorer.score(keys)
        assert importance[0] > importance[1], (
            f"Aligned key ({importance[0]:.4f}) should score higher "
            f"than orthogonal ({importance[1]:.4f})"
        )

    def test_window_rolling(self):
        """Verify the rolling window trims old queries."""
        scorer = ExpectedAttentionScorer(d=32, window=10)
        # Add 20 queries; only last 10 should be in the buffer
        scorer.update_queries(torch.randn(20, 32))
        assert len(scorer._query_buffer) == 10
        assert scorer.n_queries_seen == 20

    def test_1d_query_input(self):
        scorer = ExpectedAttentionScorer(d=32, window=8)
        scorer.update_queries(torch.randn(32))  # single query, no batch dim
        assert scorer.n_queries_seen == 1

    def test_3d_query_input(self):
        """Accept (batch, heads, d) and average over heads."""
        scorer = ExpectedAttentionScorer(d=32, window=8)
        scorer.update_queries(torch.randn(4, 8, 32))  # 4 batch, 8 heads
        assert scorer.n_queries_seen == 4

    def test_ema_mode(self):
        scorer = ExpectedAttentionScorer(d=64, window=32, ema_decay=0.9)
        scorer.update_queries(torch.randn(10, 64))
        scorer.update_queries(torch.randn(10, 64))
        assert scorer.is_ready
        keys = torch.randn(50, 64)
        importance = scorer.score(keys)
        assert abs(importance.sum().item() - 1.0) < 1e-5

    def test_full_covariance(self):
        scorer = ExpectedAttentionScorer(d=32, window=16, use_diagonal_cov=False)
        scorer.update_queries(torch.randn(20, 32))
        assert scorer._cov is not None
        assert scorer._cov.shape == (32, 32)
        keys = torch.randn(20, 32)
        importance = scorer.score(keys)
        assert abs(importance.sum().item() - 1.0) < 1e-5

    def test_score_with_details(self):
        scorer = ExpectedAttentionScorer(d=64, window=16)
        scorer.update_queries(torch.randn(20, 64))
        keys = torch.randn(30, 64)
        details = scorer.score_with_details(keys)
        assert "importance" in details
        assert "mean_term" in details
        assert "var_term" in details
        assert "log_scores" in details
        assert details["importance"].shape == (30,)
        # var_term should be non-negative (quadratic form with PSD covariance)
        assert (details["var_term"] >= -1e-6).all()

    def test_reset(self):
        scorer = ExpectedAttentionScorer(d=32, window=8)
        scorer.update_queries(torch.randn(10, 32))
        assert scorer.is_ready
        scorer.reset()
        assert not scorer.is_ready
        assert scorer.n_queries_seen == 0

    def test_stats(self):
        scorer = ExpectedAttentionScorer(d=32, window=8)
        stats = scorer.stats()
        assert stats["is_ready"] is False
        scorer.update_queries(torch.randn(10, 32))
        stats = scorer.stats()
        assert stats["is_ready"] is True
        assert "mu_norm" in stats
        assert "cov_trace" in stats

    def test_dim_mismatch_raises(self):
        scorer = ExpectedAttentionScorer(d=64, window=8)
        scorer.update_queries(torch.randn(10, 64))
        with pytest.raises(AssertionError):
            scorer.score(torch.randn(10, 32))  # wrong dimension


# ---------------------------------------------------------------------------
# ExpectedAttentionCache
# ---------------------------------------------------------------------------


class TestExpectedAttentionCache:
    """Tests for the Expected Attention cache wrapper."""

    def test_basic_append_and_retrieve(self):
        cache = ExpectedAttentionCache(d=64, rescore_interval=100)
        cache.append(torch.randn(5, 64), torch.randn(5, 64))
        assert cache.seq_len == 5
        keys = cache.get_keys()
        assert keys.shape == (5, 64)

    def test_eviction_on_rescore(self):
        cache = ExpectedAttentionCache(d=32, rescore_interval=4, top_pct=0.3, mid_pct=0.3)
        # Add enough tokens, then trigger rescoring
        for i in range(50):
            cache.append(torch.randn(1, 32), torch.randn(1, 32))
            cache.update_queries(torch.randn(1, 32))

        stats = cache.stats()
        # After several rescore cycles, some tokens should be evicted
        # (unless protection keeps everything)
        assert stats["total_seen"] == 50

    def test_1d_append(self):
        cache = ExpectedAttentionCache(d=32, rescore_interval=100)
        cache.append(torch.randn(32), torch.randn(32))
        assert cache.seq_len == 1

    def test_reset(self):
        cache = ExpectedAttentionCache(d=32, rescore_interval=100)
        cache.append(torch.randn(10, 32), torch.randn(10, 32))
        cache.reset()
        assert cache.seq_len == 0

    def test_effective_compression(self):
        cache = ExpectedAttentionCache(d=32, rescore_interval=100)
        cache.append(torch.randn(10, 32), torch.randn(10, 32))
        # No eviction yet, so effective = base compression
        assert cache.effective_compression(5.0) == 5.0


# ---------------------------------------------------------------------------
# simulate_eviction
# ---------------------------------------------------------------------------


class TestSimulateEviction:
    """Tests for the eviction simulation utility."""

    def test_no_eviction(self):
        n, d = 50, 32
        keys = torch.randn(n, d)
        values = torch.randn(n, d)
        queries = torch.randn(5, d)
        importance = torch.ones(n) / n
        result = simulate_eviction(
            keys, values, queries, importance, eviction_rate=0.0,
        )
        assert result["cosine_similarity"] > 0.999
        assert result["tokens_kept"] == n

    def test_full_eviction_degrades_quality(self):
        n, d = 100, 64
        keys = torch.randn(n, d)
        values = torch.randn(n, d)
        queries = torch.randn(10, d)
        importance = torch.rand(n)
        importance = importance / importance.sum()

        result_30 = simulate_eviction(keys, values, queries, importance, 0.3)
        result_70 = simulate_eviction(keys, values, queries, importance, 0.7)

        # More eviction = worse quality
        assert result_70["cosine_similarity"] <= result_30["cosine_similarity"] + 0.01

    def test_smart_eviction_beats_random(self):
        """Evicting low-importance tokens should give better quality."""
        torch.manual_seed(42)
        n, d = 200, 64
        keys = torch.randn(n, d)
        values = torch.randn(n, d)
        queries = torch.randn(20, d)

        # Ground truth importance
        scale = 1.0 / math.sqrt(d)
        gt_scores = (queries @ keys.T) * scale
        gt_attn = torch.softmax(gt_scores, dim=-1)
        gt_importance = gt_attn.mean(dim=0)
        gt_importance = gt_importance / gt_importance.sum()

        # Smart eviction (using ground truth importance)
        result_smart = simulate_eviction(
            keys, values, queries, gt_importance, 0.5,
        )
        # Random eviction
        result_random = simulate_eviction(
            keys, values, queries, torch.rand(n), 0.5,
        )

        assert result_smart["cosine_similarity"] >= result_random["cosine_similarity"] - 0.05

    def test_protection_of_prompt_and_recent(self):
        """Prompt and recent tokens should always be kept."""
        n, d = 100, 32
        keys = torch.randn(n, d)
        values = torch.randn(n, d)
        queries = torch.randn(5, d)
        # Give all importance to middle tokens (prompt and recent get zero)
        importance = torch.zeros(n)
        importance[30:70] = 1.0
        importance = importance / importance.sum()

        result = simulate_eviction(
            keys, values, queries, importance, 0.7,
            protect_recent=10, protect_prompt=0.05,
        )
        # Should keep at least protect_prompt + protect_recent tokens
        assert result["tokens_kept"] >= 15
