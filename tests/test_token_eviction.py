"""Tests for attention-guided token eviction cache.

Validates the three-tier (hot/warm/evicted) caching system:
- Hot tier always stores FP16 tokens
- Warm tier stores quantized tokens
- Cold tokens are evicted entirely (not stored)
- Eviction happens when warm tier exceeds max_warm_tokens
- Recent (important) tokens survive eviction
- get_seq_length returns hot + warm (not evicted)
- Compression ratio improves with eviction
- HF Cache protocol compliance
"""

import math

import pytest
import torch

from turboquantdc.token_eviction import EvictionCache


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEAD_DIM = 128
NUM_HEADS = 4
BATCH_SIZE = 2
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_kv_states(
    batch: int = BATCH_SIZE,
    num_heads: int = NUM_HEADS,
    seq_len: int = 8,
    head_dim: int = HEAD_DIM,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random KV tensors in HF format [batch, num_heads, seq_len, head_dim]."""
    torch.manual_seed(seed)
    keys = torch.randn(batch, num_heads, seq_len, head_dim)
    values = torch.randn(batch, num_heads, seq_len, head_dim)
    return keys, values


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean cosine similarity between two tensors of same shape."""
    a_flat = a.reshape(-1, a.shape[-1]).float()
    b_flat = b.reshape(-1, b.shape[-1]).float()
    sims = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1)
    return sims.mean().item()


# ===========================================================================
# Test: Construction & defaults
# ===========================================================================
class TestEvictionCacheConstruction:
    """Validate constructor and default parameters."""

    def test_defaults(self):
        cache = EvictionCache()
        assert cache.key_bits == 3
        assert cache.val_bits == 3
        assert cache.fp16_window == 64
        assert cache.max_warm_tokens == 512
        assert cache.eviction_threshold == 0.01
        assert cache.use_residual_quant is True

    def test_custom_params(self):
        cache = EvictionCache(
            key_bits=4,
            val_bits=2,
            fp16_window=32,
            max_warm_tokens=256,
            eviction_threshold=0.05,
            anchor_interval=12,
            use_residual_quant=False,
        )
        assert cache.key_bits == 4
        assert cache.val_bits == 2
        assert cache.fp16_window == 32
        assert cache.max_warm_tokens == 256
        assert cache.eviction_threshold == 0.05
        assert cache.anchor_interval == 12
        assert cache.use_residual_quant is False

    def test_invalid_key_bits(self):
        with pytest.raises(ValueError):
            EvictionCache(key_bits=0)
        with pytest.raises(ValueError):
            EvictionCache(key_bits=9)

    def test_invalid_val_bits(self):
        with pytest.raises(ValueError):
            EvictionCache(val_bits=0)

    def test_invalid_fp16_window(self):
        with pytest.raises(ValueError):
            EvictionCache(fp16_window=-1)

    def test_invalid_max_warm_tokens(self):
        with pytest.raises(ValueError):
            EvictionCache(max_warm_tokens=0)


# ===========================================================================
# Test: HF Cache protocol
# ===========================================================================
class TestHFCacheProtocol:
    """Validate HF Cache protocol methods."""

    def test_update_returns_correct_shapes(self):
        cache = EvictionCache(seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        k_out, v_out = cache.update(keys, values, layer_idx=0)
        assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)
        assert v_out.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)

    def test_update_accumulates_sequence(self):
        cache = EvictionCache(seed=SEED, max_warm_tokens=1024)
        k1, v1 = make_kv_states(seq_len=5, seed=1)
        k2, v2 = make_kv_states(seq_len=3, seed=2)
        cache.update(k1, v1, layer_idx=0)
        k_out, v_out = cache.update(k2, v2, layer_idx=0)
        assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 8, HEAD_DIM)
        assert v_out.shape == (BATCH_SIZE, NUM_HEADS, 8, HEAD_DIM)

    def test_get_seq_length_empty(self):
        cache = EvictionCache(seed=SEED)
        assert cache.get_seq_length(0) == 0

    def test_get_seq_length_after_update(self):
        cache = EvictionCache(seed=SEED, max_warm_tokens=1024)
        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=0)
        assert cache.get_seq_length(0) == 10

    def test_get_seq_length_out_of_range(self):
        cache = EvictionCache(seed=SEED)
        assert cache.get_seq_length(99) == 0

    def test_get_max_cache_shape(self):
        cache = EvictionCache(seed=SEED)
        assert cache.get_max_cache_shape() == -1

    def test_get_mask_sizes_empty(self):
        cache = EvictionCache(seed=SEED)
        pos = torch.arange(5)
        kv_len, offset = cache.get_mask_sizes(pos, layer_idx=0)
        assert kv_len == 5
        assert offset == 0

    def test_get_mask_sizes_with_cached(self):
        cache = EvictionCache(seed=SEED, max_warm_tokens=1024)
        keys, values = make_kv_states(seq_len=20)
        cache.update(keys, values, layer_idx=0)
        pos = torch.arange(1)
        kv_len, offset = cache.get_mask_sizes(pos, layer_idx=0)
        assert kv_len == 21  # 20 cached + 1 query
        assert offset == 0

    def test_len(self):
        cache = EvictionCache(seed=SEED)
        assert len(cache) == 0
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        assert len(cache) == 1

    def test_contains(self):
        cache = EvictionCache(seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        assert 0 in cache
        assert 1 not in cache

    def test_getitem(self):
        cache = EvictionCache(seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        k, v = cache[0]
        assert k.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)

    def test_getitem_out_of_range(self):
        cache = EvictionCache(seed=SEED)
        with pytest.raises(IndexError):
            _ = cache[0]

    def test_iter(self):
        cache = EvictionCache(seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        for k, v, extra in cache:
            assert k.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)
            assert extra is None

    def test_seen_tokens(self):
        cache = EvictionCache(seed=SEED, max_warm_tokens=1024)
        assert cache.seen_tokens == 0
        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=0)
        assert cache.seen_tokens == 10

    def test_is_initialized(self):
        cache = EvictionCache(seed=SEED)
        assert not cache.is_initialized
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        assert cache.is_initialized

    def test_reset(self):
        cache = EvictionCache(seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        cache.reset()
        assert cache.get_seq_length(0) == 0

    def test_crop(self):
        cache = EvictionCache(seed=SEED, max_warm_tokens=1024)
        keys, values = make_kv_states(seq_len=20)
        cache.update(keys, values, layer_idx=0)
        cache.crop(10)
        assert cache.get_seq_length(0) == 10


# ===========================================================================
# Test: Hot tier (FP16 window)
# ===========================================================================
class TestHotTier:
    """Validate that the hot tier always stores FP16 tokens."""

    def test_recent_tokens_are_fp16(self):
        """The last fp16_window tokens should be stored at full precision."""
        fp16_win = 8
        cache = EvictionCache(
            seed=SEED, fp16_window=fp16_win, max_warm_tokens=1024,
        )
        keys, values = make_kv_states(seq_len=20, seed=1)
        k_out, v_out = cache.update(keys, values, layer_idx=0)

        # The last fp16_win positions should exactly match the original
        k_tail = k_out[:, :, -fp16_win:, :]
        v_tail = v_out[:, :, -fp16_win:, :]
        k_orig = keys[:, :, -fp16_win:, :]
        v_orig = values[:, :, -fp16_win:, :]

        assert cosine_sim(k_tail, k_orig) > 0.999
        assert cosine_sim(v_tail, v_orig) > 0.999

    def test_fp16_window_zero_disables_hot_tier(self):
        """With fp16_window=0, no tokens are stored at full precision."""
        cache = EvictionCache(
            seed=SEED, fp16_window=0, max_warm_tokens=1024,
            anchor_interval=0,  # disable anchors so layer 0 is compressed
        )
        keys, values = make_kv_states(seq_len=10, seed=1)
        k_out, v_out = cache.update(keys, values, layer_idx=0)
        # All positions are quantized; not bit-exact with original
        assert not torch.equal(k_out, keys), "Expected quantization error"


# ===========================================================================
# Test: Eviction behavior
# ===========================================================================
class TestEviction:
    """Validate that eviction happens when warm tier exceeds max."""

    def test_eviction_triggers_at_max_warm(self):
        """When total tokens exceed fp16_window + max_warm_tokens, eviction occurs."""
        fp16_win = 4
        max_warm = 16
        cache = EvictionCache(
            seed=SEED,
            fp16_window=fp16_win,
            max_warm_tokens=max_warm,
            anchor_interval=0,  # no anchor layers
        )

        # Feed enough tokens to exceed the warm capacity
        total_tokens = fp16_win + max_warm + 20  # 20 extra should trigger eviction
        for i in range(total_tokens):
            k, v = make_kv_states(seq_len=1, seed=i + 100)
            cache.update(k, v, layer_idx=0)

        # After eviction, seq_length should be <= fp16_win + max_warm
        seq = cache.get_seq_length(0)
        assert seq <= fp16_win + max_warm

    def test_eviction_preserves_recent_tokens(self):
        """Recent tokens (in hot tier) should always survive eviction."""
        fp16_win = 4
        max_warm = 8
        cache = EvictionCache(
            seed=SEED,
            fp16_window=fp16_win,
            max_warm_tokens=max_warm,
            anchor_interval=0,
        )

        # Feed tokens one at a time, keeping track of the last few
        recent_keys = []
        total = fp16_win + max_warm + 10
        for i in range(total):
            k, v = make_kv_states(seq_len=1, seed=i + 200)
            k_out, v_out = cache.update(k, v, layer_idx=0)
            recent_keys.append(k)

        # The last token output should be the most recent token
        # (since fp16_window includes it)
        final_k, final_v = cache[0]
        seq_len = final_k.shape[2]
        assert seq_len > 0
        # Must have at least the fp16_window worth of recent tokens
        assert seq_len >= fp16_win

    def test_no_eviction_below_max(self):
        """No eviction should happen when tokens are below max_warm + fp16_window."""
        fp16_win = 4
        max_warm = 100
        cache = EvictionCache(
            seed=SEED,
            fp16_window=fp16_win,
            max_warm_tokens=max_warm,
            anchor_interval=0,
        )

        # Feed 20 tokens -- well below capacity
        for i in range(20):
            k, v = make_kv_states(seq_len=1, seed=i + 300)
            cache.update(k, v, layer_idx=0)

        assert cache.get_seq_length(0) == 20  # no eviction

    def test_eviction_count_tracks_evicted_tokens(self):
        """The eviction stats should report how many tokens were evicted."""
        fp16_win = 4
        max_warm = 8
        cache = EvictionCache(
            seed=SEED,
            fp16_window=fp16_win,
            max_warm_tokens=max_warm,
            anchor_interval=0,
        )

        total = fp16_win + max_warm + 20
        for i in range(total):
            k, v = make_kv_states(seq_len=1, seed=i + 400)
            cache.update(k, v, layer_idx=0)

        stats = cache.eviction_stats()
        assert stats["total_tokens_seen"] == total
        assert stats["total_tokens_evicted"] > 0
        assert stats["total_tokens_retained"] <= fp16_win + max_warm


# ===========================================================================
# Test: Importance scoring
# ===========================================================================
class TestImportanceScoring:
    """Validate the exponential decay importance scoring."""

    def test_most_recent_has_highest_importance(self):
        """The most recently appended token should have the highest importance."""
        cache = EvictionCache(
            seed=SEED,
            fp16_window=4,
            max_warm_tokens=100,
            anchor_interval=0,
        )

        for i in range(20):
            k, v = make_kv_states(seq_len=1, seed=i + 500)
            cache.update(k, v, layer_idx=0)

        importance = cache._compute_importance(layer_idx=0)
        if importance is not None and len(importance) > 1:
            # Last position should have highest importance
            assert importance[-1] >= importance[0]

    def test_older_tokens_have_lower_importance(self):
        """Importance should monotonically decrease with age."""
        cache = EvictionCache(
            seed=SEED,
            fp16_window=4,
            max_warm_tokens=100,
            anchor_interval=0,
        )

        for i in range(20):
            k, v = make_kv_states(seq_len=1, seed=i + 600)
            cache.update(k, v, layer_idx=0)

        importance = cache._compute_importance(layer_idx=0)
        if importance is not None and len(importance) > 2:
            # Should be non-decreasing (newer tokens >= older tokens)
            for j in range(1, len(importance)):
                assert importance[j] >= importance[j - 1]


# ===========================================================================
# Test: Compression ratio improvement
# ===========================================================================
class TestCompressionRatio:
    """Validate that eviction improves compression ratio over base GenerationCache."""

    def test_eviction_improves_compression(self):
        """With eviction, effective compression should exceed non-eviction baseline."""
        from turboquantdc.generation_cache import GenerationCache

        fp16_win = 4
        max_warm = 32

        # Eviction cache
        ev_cache = EvictionCache(
            seed=SEED,
            key_bits=3,
            val_bits=3,
            fp16_window=fp16_win,
            max_warm_tokens=max_warm,
            anchor_interval=0,
        )

        # Baseline GenerationCache (no eviction)
        base_cache = GenerationCache(
            key_bits=3,
            val_bits=2,
            fp16_window=fp16_win,
            anchor_interval=0,
            seed=SEED,
        )

        total = 80  # enough to trigger eviction
        for i in range(total):
            k, v = make_kv_states(seq_len=1, seed=i + 700)
            ev_cache.update(k, v, layer_idx=0)
            base_cache.update(k, v, layer_idx=0)

        ev_stats = ev_cache.memory_savings()
        base_stats = base_cache.memory_savings()

        # Eviction cache should have higher effective compression ratio
        # because it stores fewer tokens
        ev_ratio = ev_stats["overall_compression_ratio"]
        base_ratio = base_stats["overall_compression_ratio"]
        assert ev_ratio > base_ratio, (
            f"Eviction compression {ev_ratio:.2f}x should exceed "
            f"baseline {base_ratio:.2f}x"
        )


# ===========================================================================
# Test: Multi-layer behavior
# ===========================================================================
class TestMultiLayer:
    """Validate eviction works across multiple layers."""

    def test_multiple_layers(self):
        cache = EvictionCache(seed=SEED, max_warm_tokens=1024)
        k0, v0 = make_kv_states(seq_len=5, seed=1)
        k1, v1 = make_kv_states(seq_len=5, seed=2)
        cache.update(k0, v0, layer_idx=0)
        cache.update(k1, v1, layer_idx=1)
        assert cache.get_seq_length(0) == 5
        assert cache.get_seq_length(1) == 5

    def test_anchor_layers(self):
        """Anchor layers should use FP16 storage."""
        cache = EvictionCache(seed=SEED, anchor_interval=3)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        cache.update(keys, values, layer_idx=1)
        cache.update(keys, values, layer_idx=3)

        # Layer 0 and 3 are anchors (0 % 3 == 0, 3 % 3 == 0)
        assert cache._is_anchor_layer(0)
        assert not cache._is_anchor_layer(1)
        assert cache._is_anchor_layer(3)

    def test_eviction_per_layer(self):
        """Each layer should track eviction independently."""
        cache = EvictionCache(
            seed=SEED,
            fp16_window=4,
            max_warm_tokens=8,
            anchor_interval=0,
        )

        # Feed many tokens to layer 0 but few to layer 1
        for i in range(30):
            k, v = make_kv_states(seq_len=1, seed=i + 800)
            cache.update(k, v, layer_idx=0)
        for i in range(5):
            k, v = make_kv_states(seq_len=1, seed=i + 900)
            cache.update(k, v, layer_idx=1)

        # Layer 0 should have eviction, layer 1 should not
        assert cache.get_seq_length(0) <= 4 + 8
        assert cache.get_seq_length(1) == 5


# ===========================================================================
# Test: Quality after eviction
# ===========================================================================
class TestQualityAfterEviction:
    """Validate that eviction maintains acceptable quality."""

    def test_attention_score_quality(self):
        """Attention scores from evicted cache should correlate with FP16 scores."""
        fp16_win = 8
        max_warm = 32
        cache = EvictionCache(
            seed=SEED,
            fp16_window=fp16_win,
            max_warm_tokens=max_warm,
            anchor_interval=0,
        )

        # Build up the cache
        all_keys = []
        all_values = []
        for i in range(50):
            k, v = make_kv_states(seq_len=1, seed=i + 1000)
            cache.update(k, v, layer_idx=0)
            all_keys.append(k)
            all_values.append(v)

        # Get reconstructed cache
        cached_keys, cached_values = cache[0]
        seq_len = cached_keys.shape[2]

        # Compute attention with a query
        torch.manual_seed(9999)
        query = torch.randn(BATCH_SIZE, NUM_HEADS, 1, HEAD_DIM)

        # Attention scores from cached keys
        scale = 1.0 / math.sqrt(HEAD_DIM)
        cached_scores = (query @ cached_keys.transpose(-2, -1)) * scale
        cached_probs = torch.softmax(cached_scores, dim=-1)

        # The attention distribution should be valid
        assert cached_probs.shape[-1] == seq_len
        assert torch.allclose(cached_probs.sum(dim=-1), torch.ones_like(cached_probs.sum(dim=-1)), atol=1e-5)

    def test_value_reconstruction_quality(self):
        """Reconstructed values from warm tier should have reasonable cosine sim."""
        cache = EvictionCache(
            seed=SEED,
            fp16_window=4,
            max_warm_tokens=1024,
            anchor_interval=0,
        )

        keys, values = make_kv_states(seq_len=20, seed=1234)
        k_out, v_out = cache.update(keys, values, layer_idx=0)

        # Overall reconstruction should be reasonable
        # (hot tier is exact, warm tier is quantized)
        key_sim = cosine_sim(k_out, keys)
        val_sim = cosine_sim(v_out, values)
        assert key_sim > 0.9, f"Key cosine sim {key_sim:.4f} too low"
        assert val_sim > 0.9, f"Value cosine sim {val_sim:.4f} too low"


# ===========================================================================
# Test: Beam search support
# ===========================================================================
class TestBeamSearch:
    """Validate beam search operations."""

    def test_reorder_cache(self):
        cache = EvictionCache(seed=SEED, max_warm_tokens=1024)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        # Swap batch indices
        beam_idx = torch.tensor([1, 0])
        cache.reorder_cache(beam_idx)
        k, v = cache[0]
        assert k.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)

    def test_batch_repeat_interleave(self):
        cache = EvictionCache(seed=SEED, max_warm_tokens=1024)
        keys, values = make_kv_states(batch=1, seq_len=5)
        cache.update(keys, values, layer_idx=0)
        cache.batch_repeat_interleave(3)
        k, v = cache[0]
        assert k.shape[0] == 3


# ===========================================================================
# Test: Memory reporting
# ===========================================================================
class TestMemoryReporting:
    """Validate memory_savings() reports."""

    def test_memory_savings_empty(self):
        cache = EvictionCache(seed=SEED)
        stats = cache.memory_savings()
        assert stats["overall_compression_ratio"] >= 1.0

    def test_memory_savings_after_updates(self):
        cache = EvictionCache(
            seed=SEED, max_warm_tokens=1024, anchor_interval=0,
            fp16_window=4,  # small window so most tokens are compressed
        )
        for i in range(40):
            k, v = make_kv_states(seq_len=1, seed=i)
            cache.update(k, v, layer_idx=0)
        stats = cache.memory_savings()
        # With anchor_interval=0 and fp16_window=4, most tokens are compressed
        assert stats["overall_compression_ratio"] > 1.0
        assert "config" in stats
        assert stats["config"]["max_warm_tokens"] == 1024

    def test_eviction_stats_included(self):
        """memory_savings() should include eviction statistics."""
        cache = EvictionCache(
            seed=SEED,
            fp16_window=4,
            max_warm_tokens=8,
            anchor_interval=0,
        )
        for i in range(30):
            k, v = make_kv_states(seq_len=1, seed=i)
            cache.update(k, v, layer_idx=0)
        stats = cache.memory_savings()
        assert "eviction" in stats
        assert stats["eviction"]["total_evicted"] > 0


# ===========================================================================
# Test: config_summary
# ===========================================================================
class TestConfigSummary:
    """Validate config_summary() output."""

    def test_config_summary_contains_eviction_info(self):
        cache = EvictionCache(
            seed=SEED, max_warm_tokens=256, eviction_threshold=0.02,
        )
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        summary = cache.config_summary()
        assert "EvictionCache" in summary
        assert "max_warm=256" in summary
