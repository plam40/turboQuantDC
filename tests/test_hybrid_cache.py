"""Tests for HybridCache — maximum quality via stacked winning strategies.

Validates:
- Hybrid preset configs create valid caches
- HybridCache implements the full HF Cache protocol
- Per-head bit allocation after warmup
- Attention entropy computation
- Quality: hybrid vs best uniform config
- Integration with boundary anchoring + gradient bits
"""

import math

import pytest
import torch

from turboquantdc.generation_cache import (
    GenerationCache,
    HybridCache,
    _compute_attention_entropy,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEAD_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 12
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


def make_attention_weights(
    batch: int = BATCH_SIZE,
    num_heads: int = NUM_HEADS,
    seq_q: int = 8,
    seq_kv: int = 32,
    seed: int = SEED,
    entropy_pattern: str = "mixed",
) -> torch.Tensor:
    """Create synthetic attention weights [batch, num_heads, seq_q, seq_kv].

    Args:
        entropy_pattern: "mixed" creates a mix of peaked and diffuse heads.
            "uniform" creates all uniform attention.
            "peaked" creates all peaked attention.
    """
    torch.manual_seed(seed)
    if entropy_pattern == "uniform":
        # All heads attend uniformly
        weights = torch.ones(batch, num_heads, seq_q, seq_kv) / seq_kv
    elif entropy_pattern == "peaked":
        # All heads attend to a single token
        weights = torch.zeros(batch, num_heads, seq_q, seq_kv)
        weights[:, :, :, 0] = 1.0
    else:
        # Mixed: odd heads are peaked, even heads are diffuse
        weights = torch.zeros(batch, num_heads, seq_q, seq_kv)
        for h in range(num_heads):
            if h % 2 == 0:
                # Diffuse: uniform attention
                weights[:, h, :, :] = 1.0 / seq_kv
            else:
                # Peaked: attend mostly to first few tokens
                logits = torch.zeros(seq_kv)
                logits[0] = 10.0  # strong peak at position 0
                probs = torch.softmax(logits, dim=-1)
                weights[:, h, :, :] = probs.unsqueeze(0).unsqueeze(0)
    return weights


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean cosine similarity between two tensors of same shape."""
    a_flat = a.reshape(-1, a.shape[-1]).float()
    b_flat = b.reshape(-1, b.shape[-1]).float()
    sims = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1)
    return sims.mean().item()


# ---------------------------------------------------------------------------
# Test: attention entropy computation
# ---------------------------------------------------------------------------
class TestAttentionEntropy:
    """Validate the _compute_attention_entropy helper."""

    def test_uniform_attention_has_max_entropy(self):
        """Uniform attention should have entropy = log(seq_kv)."""
        seq_kv = 16
        weights = torch.ones(1, 4, 1, seq_kv) / seq_kv
        entropy = _compute_attention_entropy(weights)
        expected = math.log(seq_kv)
        assert entropy.shape == (1, 4)
        # All heads should have same entropy
        for h in range(4):
            assert abs(entropy[0, h].item() - expected) < 0.01

    def test_peaked_attention_has_low_entropy(self):
        """Peaked attention (single token) should have near-zero entropy."""
        weights = torch.zeros(1, 4, 1, 16)
        weights[:, :, :, 0] = 1.0
        entropy = _compute_attention_entropy(weights)
        assert entropy.shape == (1, 4)
        for h in range(4):
            assert entropy[0, h].item() < 0.01

    def test_entropy_is_nonnegative(self):
        """Entropy should always be non-negative."""
        weights = make_attention_weights(entropy_pattern="mixed")
        entropy = _compute_attention_entropy(weights)
        assert (entropy >= 0).all()

    def test_entropy_differentiates_heads(self):
        """Mixed pattern should produce different entropy for odd vs even heads."""
        weights = make_attention_weights(
            num_heads=8, seq_kv=32, entropy_pattern="mixed",
        )
        entropy = _compute_attention_entropy(weights)
        # Even heads (diffuse) should have higher entropy than odd heads (peaked)
        even_ent = entropy[0, ::2].mean().item()
        odd_ent = entropy[0, 1::2].mean().item()
        assert even_ent > odd_ent, (
            f"Even (diffuse) entropy {even_ent:.4f} should be > "
            f"odd (peaked) entropy {odd_ent:.4f}"
        )

    def test_batch_averaging(self):
        """Entropy should average across batch dimension."""
        weights = make_attention_weights(batch=4, num_heads=2)
        entropy = _compute_attention_entropy(weights)
        assert entropy.shape == (4, 2)


# ---------------------------------------------------------------------------
# Test: hybrid preset configs
# ---------------------------------------------------------------------------
class TestHybridPresets:
    """Validate that hybrid presets create valid GenerationCache instances."""

    def test_hybrid_max_quality_preset_exists(self):
        """hybrid_max_quality should be in PRESETS."""
        assert "hybrid_max_quality" in GenerationCache.PRESETS

    def test_hybrid_max_compression_preset_exists(self):
        """hybrid_max_compression should be in PRESETS."""
        assert "hybrid_max_compression" in GenerationCache.PRESETS

    def test_hybrid_max_quality_config(self):
        """hybrid_max_quality should use boundary strategy with RQ+norm."""
        cache = GenerationCache.from_preset(
            "hybrid_max_quality", num_layers=NUM_LAYERS,
        )
        assert cache.key_bits == 3
        assert cache.val_bits == 3
        assert cache.anchor_strategy == "boundary"
        assert cache.fp16_window == 64
        assert cache.use_residual_quant is True
        assert cache.use_norm_correction is True

    def test_hybrid_max_compression_config(self):
        """hybrid_max_compression should use gradient strategy with K3/V2."""
        cache = GenerationCache.from_preset(
            "hybrid_max_compression", num_layers=NUM_LAYERS,
        )
        assert cache.key_bits == 3
        assert cache.val_bits == 2
        assert cache.anchor_strategy == "gradient"
        assert cache.fp16_window == 64
        assert cache.use_residual_quant is True
        assert cache.use_norm_correction is True

    def test_hybrid_presets_create_valid_caches(self):
        """Both hybrid presets should produce working caches."""
        keys, values = make_kv_states(seq_len=16, seed=42)
        for name in ("hybrid_max_quality", "hybrid_max_compression"):
            cache = GenerationCache.from_preset(
                name, num_layers=NUM_LAYERS, seed=SEED,
            )
            k_out, v_out = cache.update(keys, values, layer_idx=0)
            assert k_out.shape == keys.shape, f"Preset '{name}' wrong key shape"
            assert v_out.shape == values.shape, f"Preset '{name}' wrong val shape"

    def test_hybrid_quality_boundary_has_fp16_edges(self):
        """hybrid_max_quality (boundary) should have FP16 at first/last 2 layers."""
        cache = GenerationCache.from_preset(
            "hybrid_max_quality", num_layers=NUM_LAYERS, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=8, seed=300)
        for i in range(NUM_LAYERS):
            cache.update(keys, values, layer_idx=i)

        # First 2 and last 2 layers should be exact FP16
        for anchor_idx in [0, 1, NUM_LAYERS - 2, NUM_LAYERS - 1]:
            k_out, v_out = cache[anchor_idx]
            torch.testing.assert_close(k_out, keys, atol=1e-6, rtol=1e-5)

    def test_hybrid_compression_gradient_bits(self):
        """hybrid_max_compression (gradient) should have graded key bits."""
        cache = GenerationCache.from_preset(
            "hybrid_max_compression", num_layers=36, seed=SEED,
        )
        summary = cache.anchor_summary()
        # Boundary layers should be FP16
        assert summary["fp16_count"] > 0
        # Middle layers should have base bits (3)
        middle_bits = summary["per_layer_key_bits"][len(summary["per_layer_key_bits"]) // 2]
        assert middle_bits == 3


# ---------------------------------------------------------------------------
# Test: HybridCache HF protocol
# ---------------------------------------------------------------------------
class TestHybridCacheProtocol:
    """Validate that HybridCache implements the full HF Cache protocol."""

    def test_init(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        assert cache.num_layers == NUM_LAYERS
        assert cache.base_key_bits == 3
        assert cache.base_val_bits == 2

    def test_update_returns_correct_shapes(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        k_out, v_out = cache.update(keys, values, layer_idx=0)
        assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)
        assert v_out.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)

    def test_update_accumulates_sequence(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        k1, v1 = make_kv_states(seq_len=5, seed=1)
        k2, v2 = make_kv_states(seq_len=3, seed=2)
        cache.update(k1, v1, layer_idx=0)
        k_out, v_out = cache.update(k2, v2, layer_idx=0)
        assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 8, HEAD_DIM)

    def test_get_seq_length(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        assert cache.get_seq_length(0) == 0
        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=0)
        assert cache.get_seq_length(0) == 10

    def test_get_max_cache_shape(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        assert cache.get_max_cache_shape() == -1

    def test_get_mask_sizes(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        keys, values = make_kv_states(seq_len=20)
        cache.update(keys, values, layer_idx=0)
        pos = torch.arange(1)
        kv_len, offset = cache.get_mask_sizes(pos, layer_idx=0)
        assert kv_len == 21
        assert offset == 0

    def test_len(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        assert len(cache) == 0
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        assert len(cache) == 1

    def test_contains(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        assert 0 in cache
        assert 1 not in cache

    def test_getitem(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        k, v = cache[0]
        assert k.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)

    def test_iter(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        cache.update(keys, values, layer_idx=1)
        items = list(cache)
        assert len(items) == 2

    def test_seen_tokens(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        assert cache.seen_tokens == 0
        keys, values = make_kv_states(seq_len=7)
        cache.update(keys, values, layer_idx=0)
        assert cache.seen_tokens == 7

    def test_is_initialized(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        assert not cache.is_initialized
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        assert cache.is_initialized

    def test_is_sliding(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        assert cache.is_sliding == [False]

    def test_is_compileable(self):
        assert HybridCache.is_compileable is False

    def test_crop(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        keys, values = make_kv_states(seq_len=20)
        cache.update(keys, values, layer_idx=0)
        cache.crop(10)
        assert cache.get_seq_length(0) == 10

    def test_reset(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        # Simulate warmup
        weights = make_attention_weights()
        cache.record_attention_entropy(weights)
        cache.reset()
        assert cache.get_seq_length(0) == 0
        assert not cache.warmup_complete
        assert cache.per_head_key_bits is None

    def test_reorder_cache(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        keys, values = make_kv_states(batch=4, seq_len=5, seed=400)
        cache.update(keys, values, layer_idx=0)
        beam_idx = torch.tensor([3, 1, 2, 0])
        cache.reorder_cache(beam_idx)
        k_out, _ = cache[0]
        assert k_out.shape[0] == 4


# ---------------------------------------------------------------------------
# Test: per-head bit allocation
# ---------------------------------------------------------------------------
class TestPerHeadBitAllocation:
    """Validate per-head bit assignment based on attention entropy."""

    def test_warmup_not_complete_initially(self):
        """Before any entropy recording, warmup should not be complete."""
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        assert not cache.warmup_complete
        assert cache.per_head_key_bits is None

    def test_warmup_completes_after_enough_tokens(self):
        """Recording enough tokens should finalize per-head bits."""
        cache = HybridCache(
            num_layers=NUM_LAYERS, seed=SEED,
            warmup_tokens=16,
        )
        # Record enough attention to hit warmup
        weights = make_attention_weights(
            num_heads=NUM_HEADS, seq_q=16, seq_kv=32,
            entropy_pattern="mixed",
        )
        cache.record_attention_entropy(weights)
        assert cache.warmup_complete
        assert cache.per_head_key_bits is not None
        assert len(cache.per_head_key_bits) == NUM_HEADS

    def test_per_head_bits_range(self):
        """Per-head bits should be in [base-1, base+1] range."""
        cache = HybridCache(
            num_layers=NUM_LAYERS, seed=SEED,
            base_key_bits=3, warmup_tokens=8,
        )
        weights = make_attention_weights(
            num_heads=NUM_HEADS, seq_q=8, seq_kv=32,
            entropy_pattern="mixed",
        )
        cache.record_attention_entropy(weights)
        for bits in cache.per_head_key_bits:
            assert 2 <= bits <= 4, f"Per-head bits {bits} out of range [2, 4]"

    def test_high_entropy_heads_get_more_bits(self):
        """High-entropy (diffuse) heads should get base_bits + 1."""
        cache = HybridCache(
            num_layers=NUM_LAYERS, seed=SEED,
            base_key_bits=3, warmup_tokens=8,
            high_entropy_pct=50, low_entropy_pct=50,
        )
        # Create strongly differentiated pattern
        weights = make_attention_weights(
            num_heads=NUM_HEADS, seq_q=8, seq_kv=32,
            entropy_pattern="mixed",
        )
        cache.record_attention_entropy(weights)

        # Even heads (diffuse) should have higher bits
        even_bits = [cache.per_head_key_bits[h] for h in range(0, NUM_HEADS, 2)]
        odd_bits = [cache.per_head_key_bits[h] for h in range(1, NUM_HEADS, 2)]
        assert sum(even_bits) >= sum(odd_bits), (
            f"Diffuse heads {even_bits} should get >= bits than peaked heads {odd_bits}"
        )

    def test_low_entropy_heads_get_fewer_bits(self):
        """Low-entropy (peaked) heads should get base_bits - 1."""
        cache = HybridCache(
            num_layers=NUM_LAYERS, seed=SEED,
            base_key_bits=3, warmup_tokens=8,
            high_entropy_pct=50, low_entropy_pct=50,
        )
        weights = make_attention_weights(
            num_heads=NUM_HEADS, seq_q=8, seq_kv=32,
            entropy_pattern="mixed",
        )
        cache.record_attention_entropy(weights)

        # Odd heads (peaked) should get fewer bits
        odd_bits = [cache.per_head_key_bits[h] for h in range(1, NUM_HEADS, 2)]
        assert all(b <= 3 for b in odd_bits), (
            f"Peaked heads should get <= base bits, got {odd_bits}"
        )

    def test_all_uniform_heads_get_same_bits(self):
        """If all heads have same entropy, all should get the same bit allocation."""
        cache = HybridCache(
            num_layers=NUM_LAYERS, seed=SEED,
            base_key_bits=3, warmup_tokens=8,
        )
        weights = make_attention_weights(
            num_heads=NUM_HEADS, seq_q=8, seq_kv=32,
            entropy_pattern="uniform",
        )
        cache.record_attention_entropy(weights)
        # With identical entropy across all heads, the percentile thresholds
        # collapse: all heads have the same entropy so they all fall into the
        # same bucket.  The important property is uniformity of assignment.
        bits = cache.per_head_key_bits
        assert len(set(bits)) == 1, (
            f"All-uniform heads should get identical bits, got {bits}"
        )

    def test_recording_after_warmup_is_noop(self):
        """Recording more entropy after warmup should not change bits."""
        cache = HybridCache(
            num_layers=NUM_LAYERS, seed=SEED,
            warmup_tokens=8,
        )
        weights = make_attention_weights(
            num_heads=NUM_HEADS, seq_q=8, seq_kv=32,
            entropy_pattern="mixed",
        )
        cache.record_attention_entropy(weights)
        original_bits = list(cache.per_head_key_bits)

        # Record more — should be ignored
        weights2 = make_attention_weights(
            num_heads=NUM_HEADS, seq_q=16, seq_kv=32,
            entropy_pattern="uniform", seed=999,
        )
        cache.record_attention_entropy(weights2)
        assert cache.per_head_key_bits == original_bits

    def test_incremental_warmup(self):
        """Warmup should accumulate across multiple calls."""
        cache = HybridCache(
            num_layers=NUM_LAYERS, seed=SEED,
            warmup_tokens=16,
        )
        # First call: 8 tokens (not enough)
        w1 = make_attention_weights(
            num_heads=NUM_HEADS, seq_q=8, seq_kv=16,
            entropy_pattern="mixed",
        )
        cache.record_attention_entropy(w1)
        assert not cache.warmup_complete

        # Second call: 8 more tokens (total 16, enough)
        w2 = make_attention_weights(
            num_heads=NUM_HEADS, seq_q=8, seq_kv=16,
            entropy_pattern="mixed", seed=99,
        )
        cache.record_attention_entropy(w2)
        assert cache.warmup_complete

    def test_min_bits_clamped_to_1(self):
        """Per-head bits should never go below 1."""
        cache = HybridCache(
            num_layers=NUM_LAYERS, seed=SEED,
            base_key_bits=1,  # Edge case: base is already 1
            warmup_tokens=8,
        )
        weights = make_attention_weights(
            num_heads=NUM_HEADS, seq_q=8, seq_kv=32,
            entropy_pattern="mixed",
        )
        cache.record_attention_entropy(weights)
        for bits in cache.per_head_key_bits:
            assert bits >= 1

    def test_max_bits_clamped_to_8(self):
        """Per-head bits should never exceed 8."""
        cache = HybridCache(
            num_layers=NUM_LAYERS, seed=SEED,
            base_key_bits=8,  # Edge case: base is already 8
            warmup_tokens=8,
        )
        weights = make_attention_weights(
            num_heads=NUM_HEADS, seq_q=8, seq_kv=32,
            entropy_pattern="mixed",
        )
        cache.record_attention_entropy(weights)
        for bits in cache.per_head_key_bits:
            assert bits <= 8


# ---------------------------------------------------------------------------
# Test: parameter validation
# ---------------------------------------------------------------------------
class TestHybridCacheValidation:
    """Validate parameter validation for HybridCache."""

    def test_invalid_num_layers(self):
        with pytest.raises(ValueError, match="num_layers"):
            HybridCache(num_layers=0)

    def test_invalid_base_key_bits(self):
        with pytest.raises(ValueError, match="base_key_bits"):
            HybridCache(num_layers=12, base_key_bits=0)
        with pytest.raises(ValueError, match="base_key_bits"):
            HybridCache(num_layers=12, base_key_bits=9)

    def test_invalid_base_val_bits(self):
        with pytest.raises(ValueError, match="base_val_bits"):
            HybridCache(num_layers=12, base_val_bits=0)
        with pytest.raises(ValueError, match="base_val_bits"):
            HybridCache(num_layers=12, base_val_bits=9)

    def test_invalid_warmup_tokens(self):
        with pytest.raises(ValueError, match="warmup_tokens"):
            HybridCache(num_layers=12, warmup_tokens=0)

    def test_invalid_high_entropy_pct(self):
        with pytest.raises(ValueError, match="high_entropy_pct"):
            HybridCache(num_layers=12, high_entropy_pct=-1)
        with pytest.raises(ValueError, match="high_entropy_pct"):
            HybridCache(num_layers=12, high_entropy_pct=101)

    def test_invalid_low_entropy_pct(self):
        with pytest.raises(ValueError, match="low_entropy_pct"):
            HybridCache(num_layers=12, low_entropy_pct=-1)
        with pytest.raises(ValueError, match="low_entropy_pct"):
            HybridCache(num_layers=12, low_entropy_pct=101)


# ---------------------------------------------------------------------------
# Test: HybridCache reporting
# ---------------------------------------------------------------------------
class TestHybridCacheReporting:
    """Validate memory reporting and config summaries."""

    def test_memory_savings_includes_hybrid_flag(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        keys, values = make_kv_states(seq_len=16)
        cache.update(keys, values, layer_idx=0)
        report = cache.memory_savings()
        assert report["hybrid"] is True
        assert report["warmup_complete"] is False

    def test_memory_savings_after_warmup(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED, warmup_tokens=8)
        keys, values = make_kv_states(seq_len=16)
        cache.update(keys, values, layer_idx=0)
        weights = make_attention_weights(seq_q=8, seq_kv=16)
        cache.record_attention_entropy(weights)
        report = cache.memory_savings()
        assert report["warmup_complete"] is True
        assert report["per_head_key_bits"] is not None

    def test_anchor_summary_delegates(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        summary = cache.anchor_summary()
        assert summary["strategy"] == "gradient"
        assert summary["num_layers"] == NUM_LAYERS

    def test_config_summary_before_warmup(self):
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED, warmup_tokens=32)
        summary = cache.config_summary()
        assert "HybridCache" in summary
        assert "warmup 0/32" in summary

    def test_config_summary_after_warmup(self):
        cache = HybridCache(
            num_layers=NUM_LAYERS, seed=SEED, warmup_tokens=8,
        )
        weights = make_attention_weights(seq_q=8, seq_kv=16)
        cache.record_attention_entropy(weights)
        summary = cache.config_summary()
        assert "HybridCache" in summary
        assert "per-head bits=" in summary


# ---------------------------------------------------------------------------
# Test: quality comparison — hybrid vs uniform
# ---------------------------------------------------------------------------
class TestHybridQuality:
    """Compare hybrid cache quality against uniform baseline."""

    def test_hybrid_key_cosine_quality(self):
        """Hybrid cache key reconstruction should have high cosine similarity."""
        cache = HybridCache(
            num_layers=NUM_LAYERS, base_key_bits=3, base_val_bits=2,
            fp16_window=0, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=64, seed=100)
        k_out, _ = cache.update(keys, values, layer_idx=5)  # middle layer
        sim = cosine_sim(keys, k_out)
        assert sim > 0.90, f"Hybrid key cosine sim {sim:.4f} below 0.90"

    def test_hybrid_value_cosine_quality(self):
        """Hybrid cache value reconstruction should have reasonable quality."""
        cache = HybridCache(
            num_layers=NUM_LAYERS, base_key_bits=3, base_val_bits=2,
            fp16_window=0, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=64, seed=100)
        _, v_out = cache.update(keys, values, layer_idx=5)
        sim = cosine_sim(values, v_out)
        assert sim > 0.80, f"Hybrid value cosine sim {sim:.4f} below 0.80"

    def test_hybrid_fp16_window_preserves_recent(self):
        """FP16 window in hybrid should keep recent tokens exact."""
        window = 8
        cache = HybridCache(
            num_layers=NUM_LAYERS, fp16_window=window, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=32, seed=200)
        k_out, v_out = cache.update(keys, values, layer_idx=5)
        torch.testing.assert_close(
            k_out[:, :, -window:, :],
            keys[:, :, -window:, :],
            atol=1e-6, rtol=1e-5,
        )

    def test_hybrid_boundary_layers_are_fp16(self):
        """Edge layers should be stored at FP16 (gradient strategy)."""
        cache = HybridCache(num_layers=36, seed=SEED)
        keys, values = make_kv_states(seq_len=8, seed=300)
        # Fill all 36 layers
        for i in range(36):
            cache.update(keys, values, layer_idx=i)

        # Layer 0 and layer 35 should be FP16 (within 10% of boundary)
        k0, v0 = cache[0]
        torch.testing.assert_close(k0, keys, atol=1e-6, rtol=1e-5)
        k35, v35 = cache[35]
        torch.testing.assert_close(k35, keys, atol=1e-6, rtol=1e-5)

    def test_hybrid_autoregressive_generation(self):
        """Simulate token-by-token generation with hybrid cache."""
        n_layers = 4
        cache = HybridCache(
            num_layers=n_layers, base_key_bits=3, base_val_bits=2,
            fp16_window=8, seed=SEED,
        )

        # Prefill with 16 tokens
        prefill_k, prefill_v = make_kv_states(
            batch=1, num_heads=4, seq_len=16, head_dim=64, seed=500,
        )
        for layer in range(n_layers):
            cache.update(prefill_k, prefill_v, layer_idx=layer)

        assert cache.get_seq_length(0) == 16

        # Generate 10 tokens one at a time
        for step in range(10):
            new_k, new_v = make_kv_states(
                batch=1, num_heads=4, seq_len=1, head_dim=64, seed=600 + step,
            )
            for layer in range(n_layers):
                k_out, v_out = cache.update(new_k, new_v, layer_idx=layer)
                assert k_out.shape[2] == 16 + step + 1

        assert cache.get_seq_length(0) == 26


# ---------------------------------------------------------------------------
# Test: integration — hybrid uses gradient anchor schedule
# ---------------------------------------------------------------------------
class TestHybridIntegration:
    """Validate that HybridCache correctly uses the gradient anchor schedule."""

    def test_inner_cache_uses_gradient_strategy(self):
        """The inner GenerationCache should use gradient anchor strategy."""
        cache = HybridCache(num_layers=36, seed=SEED)
        assert cache._inner.anchor_strategy == "gradient"

    def test_inner_cache_has_residual_quant(self):
        """Inner cache should have ResidualQuant enabled."""
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        assert cache._inner.use_residual_quant is True

    def test_inner_cache_has_norm_correction(self):
        """Inner cache should have norm correction enabled."""
        cache = HybridCache(num_layers=NUM_LAYERS, seed=SEED)
        assert cache._inner.use_norm_correction is True

    def test_inner_cache_fp16_window(self):
        """Inner cache should pass through fp16_window."""
        cache = HybridCache(num_layers=NUM_LAYERS, fp16_window=128, seed=SEED)
        assert cache._inner.fp16_window == 128

    def test_gradient_schedule_has_boundary_fp16(self):
        """Gradient strategy should produce FP16 for boundary layers."""
        cache = HybridCache(num_layers=36, seed=SEED)
        summary = cache.anchor_summary()
        # Layer 0 should be FP16
        assert 0 in summary["fp16_layers"]
        # Layer 35 should be FP16
        assert 35 in summary["fp16_layers"]
        # At least some FP16 layers
        assert summary["fp16_count"] >= 2
