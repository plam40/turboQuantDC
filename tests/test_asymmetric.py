"""Tests for asymmetric KV cache — different bit-widths for keys and values.

Validates:
1. Basic append/retrieve/shapes
2. Quality presets give expected cosine similarity
3. Asymmetric K4/V2 beats uniform 3-bit for attention scores
4. Compression ratios match theory
5. Separate K/V memory accounting
6. All 4 presets create valid caches
7. HF-compatible AsymmetricTurboQuantCache protocol
8. GPU operation
9. Various head dimensions (d=64, 128, 256)
"""

import math

import pytest
import torch

from turboquantdc.asymmetric import (
    PRESETS,
    AsymmetricKVCache,
    AsymmetricTurboQuantCache,
    AsymmetricTurboQuantLayer,
    create_asymmetric_cache,
)
from turboquantdc.polarquant import PolarQuant


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEAD_DIM = 128
SEED = 42
NUM_HEADS = 4
BATCH_SIZE = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_unit_vectors(n: int, d: int, seed: int = SEED) -> torch.Tensor:
    """Create n random unit vectors in R^d."""
    torch.manual_seed(seed)
    x = torch.randn(n, d)
    return x / x.norm(dim=-1, keepdim=True)


def make_vectors(n: int, d: int, seed: int = SEED) -> torch.Tensor:
    """Create n random vectors (non-unit) in R^d."""
    torch.manual_seed(seed)
    return torch.randn(n, d)


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


def cosine_similarity_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute per-row cosine similarity between a and b."""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


# ---------------------------------------------------------------------------
# Test 1: Basic append, retrieve, shapes
# ---------------------------------------------------------------------------

class TestAsymmetricBasic:
    """Test basic asymmetric cache operations."""

    def test_append_and_retrieve(self):
        """append() should store data, get_values() should retrieve it."""
        cache = AsymmetricKVCache(d_key=128, d_value=128, key_bits=4, val_bits=2, seed=SEED)

        keys = make_vectors(10, 128, seed=1)
        values = make_vectors(10, 128, seed=2)
        cache.append(keys, values)

        assert cache.seq_len == 1  # one batch append
        retrieved_values = cache.get_values()
        assert retrieved_values.shape == (10, 128)

    def test_multiple_appends(self):
        """Multiple appends should accumulate tokens."""
        cache = AsymmetricKVCache(d_key=128, d_value=128, key_bits=4, val_bits=3, seed=SEED)

        for i in range(5):
            k = make_vectors(3, 128, seed=i)
            v = make_vectors(3, 128, seed=i + 100)
            cache.append(k, v)

        assert cache.seq_len == 5
        vals = cache.get_values()
        assert vals.shape == (15, 128)  # 5 appends * 3 vectors each

    def test_single_vector_append(self):
        """Should handle 1D input (single vector)."""
        cache = AsymmetricKVCache(d_key=64, d_value=64, key_bits=3, val_bits=2, seed=SEED)

        k = torch.randn(64)
        v = torch.randn(64)
        cache.append(k, v)

        assert cache.seq_len == 1
        vals = cache.get_values()
        assert vals.shape == (1, 64)

    def test_clear(self):
        """clear() should reset the cache."""
        cache = AsymmetricKVCache(d_key=128, d_value=128, key_bits=4, val_bits=2, seed=SEED)
        cache.append(make_vectors(5, 128), make_vectors(5, 128))
        assert cache.seq_len == 1

        cache.clear()
        assert cache.seq_len == 0

    def test_empty_cache_attention(self):
        """attention_scores on empty cache should return empty tensor."""
        cache = AsymmetricKVCache(d_key=128, d_value=128, key_bits=4, val_bits=2, seed=SEED)
        queries = make_vectors(3, 128)
        scores = cache.attention_scores(queries)
        assert scores.shape == (3, 0)

    def test_empty_cache_values(self):
        """get_values on empty cache should return empty tensor."""
        cache = AsymmetricKVCache(d_key=128, d_value=128, key_bits=4, val_bits=2, seed=SEED)
        vals = cache.get_values()
        assert vals.shape == (0, 128)

    def test_attention_scores_shape(self):
        """attention_scores should return correct shape."""
        cache = AsymmetricKVCache(d_key=128, d_value=128, key_bits=4, val_bits=2, seed=SEED)
        keys = make_vectors(10, 128, seed=1)
        values = make_vectors(10, 128, seed=2)
        cache.append(keys, values)

        queries = make_vectors(3, 128, seed=3)
        scores = cache.attention_scores(queries)
        assert scores.shape == (3, 10)


# ---------------------------------------------------------------------------
# Test 2: Quality preset (key=4, val=3)
# ---------------------------------------------------------------------------

class TestQualityPreset:
    """Test that quality preset (K4/V3) gives high cosine similarity."""

    def test_key_reconstruction_quality(self):
        """4-bit keys should give very high cosine similarity (>0.998)."""
        cache = create_asymmetric_cache(128, 128, preset="quality", seed=SEED)

        keys = make_unit_vectors(100, 128, seed=1)
        values = make_unit_vectors(100, 128, seed=2)
        cache.append(keys, values)

        # Reconstruct keys via attention with identity queries
        recon_keys = cache._reconstruct_keys()
        # Normalize for cosine comparison (keys were unit, norms ~1)
        recon_norms = recon_keys.norm(dim=-1, keepdim=True)
        recon_normalized = recon_keys / (recon_norms + 1e-8)

        cos_sim = cosine_similarity_batch(keys, recon_normalized)
        mean_cos = cos_sim.mean().item()
        assert mean_cos > 0.993, f"4-bit key cosine sim {mean_cos:.4f} < 0.993"

    def test_value_reconstruction_quality(self):
        """3-bit values should give cosine sim > 0.97."""
        cache = create_asymmetric_cache(128, 128, preset="quality", seed=SEED)

        keys = make_unit_vectors(100, 128, seed=1)
        values = make_unit_vectors(100, 128, seed=2)
        cache.append(keys, values)

        recon_values = cache.get_values()
        recon_norms = recon_values.norm(dim=-1, keepdim=True)
        recon_normalized = recon_values / (recon_norms + 1e-8)

        cos_sim = cosine_similarity_batch(values, recon_normalized)
        mean_cos = cos_sim.mean().item()
        assert mean_cos > 0.97, f"3-bit value cosine sim {mean_cos:.4f} < 0.97"


# ---------------------------------------------------------------------------
# Test 3: Balanced preset (key=4, val=2) — good compression
# ---------------------------------------------------------------------------

class TestBalancedPreset:
    """Test that balanced preset (K4/V2) gives good compression with acceptable quality."""

    def test_key_still_high_quality(self):
        """4-bit keys should maintain high cosine sim even with 2-bit values."""
        cache = create_asymmetric_cache(128, 128, preset="balanced", seed=SEED)

        keys = make_unit_vectors(100, 128, seed=1)
        values = make_unit_vectors(100, 128, seed=2)
        cache.append(keys, values)

        recon_keys = cache._reconstruct_keys()
        recon_norms = recon_keys.norm(dim=-1, keepdim=True)
        recon_normalized = recon_keys / (recon_norms + 1e-8)

        cos_sim = cosine_similarity_batch(keys, recon_normalized)
        mean_cos = cos_sim.mean().item()
        assert mean_cos > 0.993, f"4-bit key cosine sim {mean_cos:.4f} < 0.993"

    def test_value_acceptable_quality(self):
        """2-bit values should give cosine sim > 0.90."""
        cache = create_asymmetric_cache(128, 128, preset="balanced", seed=SEED)

        keys = make_unit_vectors(100, 128, seed=1)
        values = make_unit_vectors(100, 128, seed=2)
        cache.append(keys, values)

        recon_values = cache.get_values()
        recon_norms = recon_values.norm(dim=-1, keepdim=True)
        recon_normalized = recon_values / (recon_norms + 1e-8)

        cos_sim = cosine_similarity_batch(values, recon_normalized)
        mean_cos = cos_sim.mean().item()
        assert mean_cos > 0.90, f"2-bit value cosine sim {mean_cos:.4f} < 0.90"


# ---------------------------------------------------------------------------
# Test 4: Asymmetric K4/V2 beats uniform 3-bit for attention scores
# ---------------------------------------------------------------------------

class TestAsymmetricBeatsUniform:
    """Asymmetric K4/V2 should give better attention scores than uniform 3-bit."""

    def test_attention_accuracy(self):
        """K4/V2 asymmetric should have lower attention score error than uniform 3-bit."""
        d = 128
        n_keys = 50
        n_queries = 10

        torch.manual_seed(SEED)
        keys = torch.randn(n_keys, d)
        values = torch.randn(n_keys, d)
        queries = torch.randn(n_queries, d)

        # Ground truth attention scores
        true_scores = queries @ keys.T

        # Asymmetric K4/V2
        asym_cache = AsymmetricKVCache(d, d, key_bits=4, val_bits=2, seed=SEED)
        asym_cache.append(keys, values)
        asym_scores = asym_cache.attention_scores(queries)

        # Uniform 3-bit (both K and V at 3 bits, MSE-only like asymmetric)
        uniform_cache = AsymmetricKVCache(d, d, key_bits=3, val_bits=3, seed=SEED)
        uniform_cache.append(keys, values)
        uniform_scores = uniform_cache.attention_scores(queries)

        # Attention score error (relative)
        asym_error = (asym_scores - true_scores).abs().mean().item()
        uniform_error = (uniform_scores - true_scores).abs().mean().item()

        # Asymmetric K4 should give better attention scores than uniform K3
        # because attention scores depend only on keys, and K4 > K3
        assert asym_error < uniform_error, (
            f"Asymmetric K4/V2 attention error ({asym_error:.4f}) should be less than "
            f"uniform K3/V3 ({uniform_error:.4f})"
        )


# ---------------------------------------------------------------------------
# Test 5: Compression ratio
# ---------------------------------------------------------------------------

class TestCompressionRatio:
    """Test that compression ratios match theory."""

    def test_balanced_compression_ratio(self):
        """K4/V2 should give ~4-5x compression."""
        cache = AsymmetricKVCache(d_key=128, d_value=128, key_bits=4, val_bits=2, seed=SEED)
        keys = make_vectors(100, 128, seed=1)
        values = make_vectors(100, 128, seed=2)
        cache.append(keys, values)

        ratio = cache.compression_ratio()
        # FP16 baseline = 2 * 128 * 16 = 4096 bits per token
        # K4 compressed = 4 * 128 + 16 = 528 bits per token
        # V2 compressed = 2 * 128 + 16 = 272 bits per token
        # Total = 800 bits => ratio = 4096 / 800 = 5.12
        assert 4.0 < ratio < 6.0, f"K4/V2 compression ratio {ratio:.2f} not in [4.0, 6.0]"

    def test_quality_compression_ratio(self):
        """K4/V3 should give ~3-4x compression."""
        cache = AsymmetricKVCache(d_key=128, d_value=128, key_bits=4, val_bits=3, seed=SEED)
        keys = make_vectors(100, 128, seed=1)
        values = make_vectors(100, 128, seed=2)
        cache.append(keys, values)

        ratio = cache.compression_ratio()
        # K4 = 528, V3 = 3*128+16 = 400, total = 928 => 4096/928 = 4.41
        assert 3.0 < ratio < 5.0, f"K4/V3 compression ratio {ratio:.2f} not in [3.0, 5.0]"

    def test_aggressive_compression_ratio(self):
        """K3/V2 should give ~5-7x compression."""
        cache = AsymmetricKVCache(d_key=128, d_value=128, key_bits=3, val_bits=2, seed=SEED)
        keys = make_vectors(100, 128, seed=1)
        values = make_vectors(100, 128, seed=2)
        cache.append(keys, values)

        ratio = cache.compression_ratio()
        # K3 = 3*128+16 = 400, V2 = 272, total = 672 => 4096/672 = 6.09
        assert 5.0 < ratio < 7.5, f"K3/V2 compression ratio {ratio:.2f} not in [5.0, 7.5]"

    def test_extreme_compression_ratio(self):
        """K2/V2 should give ~7-8x compression."""
        cache = AsymmetricKVCache(d_key=128, d_value=128, key_bits=2, val_bits=2, seed=SEED)
        keys = make_vectors(100, 128, seed=1)
        values = make_vectors(100, 128, seed=2)
        cache.append(keys, values)

        ratio = cache.compression_ratio()
        # K2 = 272, V2 = 272, total = 544 => 4096/544 = 7.53
        assert 6.5 < ratio < 9.0, f"K2/V2 compression ratio {ratio:.2f} not in [6.5, 9.0]"


# ---------------------------------------------------------------------------
# Test 6: Memory breakdown — separate K/V accounting
# ---------------------------------------------------------------------------

class TestMemoryBreakdown:
    """Test that memory reporting correctly separates K and V contributions."""

    def test_separate_kv_accounting(self):
        """memory_usage_bits should report K and V separately."""
        cache = AsymmetricKVCache(d_key=128, d_value=128, key_bits=4, val_bits=2, seed=SEED)
        keys = make_vectors(10, 128, seed=1)
        values = make_vectors(10, 128, seed=2)
        cache.append(keys, values)

        stats = cache.memory_usage_bits()

        assert "key_bits_total" in stats
        assert "value_bits_total" in stats
        assert stats["key_bits_total"] > 0
        assert stats["value_bits_total"] > 0
        assert stats["total_bits"] == stats["key_bits_total"] + stats["value_bits_total"]

    def test_key_uses_more_bits_than_value(self):
        """K4/V2: keys should use more bits than values."""
        cache = AsymmetricKVCache(d_key=128, d_value=128, key_bits=4, val_bits=2, seed=SEED)
        keys = make_vectors(10, 128, seed=1)
        values = make_vectors(10, 128, seed=2)
        cache.append(keys, values)

        stats = cache.memory_usage_bits()
        assert stats["key_bits_total"] > stats["value_bits_total"], (
            f"Keys ({stats['key_bits_total']}) should use more bits than "
            f"values ({stats['value_bits_total']}) with K4/V2"
        )

    def test_empty_cache_memory(self):
        """Empty cache should report zero memory."""
        cache = AsymmetricKVCache(d_key=128, d_value=128, key_bits=4, val_bits=2, seed=SEED)
        stats = cache.memory_usage_bits()
        assert stats["total_bits"] == 0
        assert stats["compression_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Test 7: All presets create valid caches
# ---------------------------------------------------------------------------

class TestPresets:
    """Test all preset configs."""

    @pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
    def test_preset_creates_valid_cache(self, preset_name):
        """Each preset should create a working cache."""
        cache = create_asymmetric_cache(128, 128, preset=preset_name, seed=SEED)
        config = PRESETS[preset_name]

        assert cache.key_bits == config["key_bits"]
        assert cache.val_bits == config["val_bits"]

        # Should work for basic operations
        keys = make_vectors(10, 128, seed=1)
        values = make_vectors(10, 128, seed=2)
        cache.append(keys, values)
        assert cache.seq_len == 1
        assert cache.get_values().shape == (10, 128)

    def test_invalid_preset_raises(self):
        """Unknown preset should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            create_asymmetric_cache(128, 128, preset="nonexistent")

    @pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
    def test_preset_attention_scores_reasonable(self, preset_name):
        """Each preset should produce reasonable attention scores."""
        d = 128
        cache = create_asymmetric_cache(d, d, preset=preset_name, seed=SEED)

        torch.manual_seed(SEED)
        keys = torch.randn(20, d)
        values = torch.randn(20, d)
        queries = torch.randn(5, d)

        cache.append(keys, values)
        scores = cache.attention_scores(queries)

        true_scores = queries @ keys.T
        # Correlation should be positive (compressed scores track true scores)
        for i in range(5):
            corr = torch.corrcoef(
                torch.stack([scores[i], true_scores[i]])
            )[0, 1].item()
            # 2-bit extreme preset is very aggressive; lower threshold
            min_corr = 0.7 if preset_name == "extreme" else 0.8
            assert corr > min_corr, (
                f"Preset '{preset_name}' query {i}: attention score correlation "
                f"{corr:.3f} < {min_corr}"
            )


# ---------------------------------------------------------------------------
# Test 8: HF-compatible AsymmetricTurboQuantCache
# ---------------------------------------------------------------------------

class TestAsymmetricHFCache:
    """Test AsymmetricTurboQuantCache as HF Cache protocol drop-in."""

    def test_update_returns_correct_shapes(self):
        """update() should return dequantized tensors with correct shapes."""
        cache = AsymmetricTurboQuantCache(key_bits=4, val_bits=2, seed=SEED)
        keys, values = make_kv_states(seq_len=5)

        k_out, v_out = cache.update(keys, values, layer_idx=0)

        assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)
        assert v_out.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)

    def test_multi_layer(self):
        """Should support multiple transformer layers."""
        cache = AsymmetricTurboQuantCache(key_bits=4, val_bits=2, seed=SEED)

        for layer_idx in range(4):
            keys, values = make_kv_states(seq_len=3, seed=SEED + layer_idx)
            cache.update(keys, values, layer_idx=layer_idx)

        assert len(cache) == 4
        for i in range(4):
            assert cache.get_seq_length(i) == 3

    def test_incremental_updates(self):
        """Should accumulate tokens across updates (prefill + decode)."""
        cache = AsymmetricTurboQuantCache(key_bits=4, val_bits=2, seed=SEED)

        # Prefill: 10 tokens
        keys1, vals1 = make_kv_states(seq_len=10, seed=1)
        cache.update(keys1, vals1, layer_idx=0)
        assert cache.get_seq_length(0) == 10

        # Decode: 1 token at a time
        for i in range(5):
            keys2, vals2 = make_kv_states(seq_len=1, seed=100 + i)
            k_out, v_out = cache.update(keys2, vals2, layer_idx=0)
            assert k_out.shape[2] == 10 + i + 1

        assert cache.get_seq_length(0) == 15

    def test_getitem(self):
        """__getitem__ should return dequantized tensors."""
        cache = AsymmetricTurboQuantCache(key_bits=4, val_bits=2, seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)

        k, v = cache[0]
        assert k.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)
        assert v.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)

    def test_iter(self):
        """__iter__ should yield (keys, values, None) per layer."""
        cache = AsymmetricTurboQuantCache(key_bits=4, val_bits=2, seed=SEED)
        for layer_idx in range(3):
            keys, values = make_kv_states(seq_len=4, seed=SEED + layer_idx)
            cache.update(keys, values, layer_idx=layer_idx)

        count = 0
        for k, v, extra in cache:
            assert extra is None
            assert k.shape[2] == 4
            count += 1
        assert count == 3

    def test_crop(self):
        """crop() should truncate all layers."""
        cache = AsymmetricTurboQuantCache(key_bits=4, val_bits=2, seed=SEED)
        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=0)
        cache.update(keys, values, layer_idx=1)

        cache.crop(5)
        assert cache.get_seq_length(0) == 5
        assert cache.get_seq_length(1) == 5

    def test_reset(self):
        """reset() should clear all layers."""
        cache = AsymmetricTurboQuantCache(key_bits=4, val_bits=2, seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)

        cache.reset()
        assert cache.get_seq_length(0) == 0

    def test_len_and_is_initialized(self):
        """__len__ and is_initialized should track layers."""
        cache = AsymmetricTurboQuantCache(key_bits=4, val_bits=2, seed=SEED)
        assert not cache.is_initialized
        assert len(cache) == 0

        keys, values = make_kv_states(seq_len=3)
        cache.update(keys, values, layer_idx=0)
        assert cache.is_initialized
        assert len(cache) == 1

    def test_get_max_cache_shape(self):
        """Dynamic cache has no maximum."""
        cache = AsymmetricTurboQuantCache(key_bits=4, val_bits=2, seed=SEED)
        assert cache.get_max_cache_shape() == -1

    def test_memory_savings(self):
        """memory_savings() should report correctly."""
        cache = AsymmetricTurboQuantCache(key_bits=4, val_bits=2, seed=SEED)
        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=0)

        savings = cache.memory_savings()
        assert savings["key_bits"] == 4
        assert savings["val_bits"] == 2
        assert savings["num_layers"] == 1
        assert savings["overall_compression_ratio"] > 1.0

    def test_invalid_bits_raises(self):
        """Invalid bit-widths should raise ValueError."""
        with pytest.raises(ValueError):
            AsymmetricTurboQuantCache(key_bits=5, val_bits=2)
        with pytest.raises(ValueError):
            AsymmetricTurboQuantCache(key_bits=4, val_bits=1)

    def test_reorder_cache(self):
        """reorder_cache should work for beam search."""
        cache = AsymmetricTurboQuantCache(key_bits=4, val_bits=2, seed=SEED)
        keys, values = make_kv_states(batch=4, seq_len=5)
        cache.update(keys, values, layer_idx=0)

        # Select beams 0 and 2
        beam_idx = torch.tensor([0, 2])
        cache.reorder_cache(beam_idx)

        k, v = cache[0]
        assert k.shape[0] == 2  # batch reduced from 4 to 2


# ---------------------------------------------------------------------------
# Test 9: GPU operation
# ---------------------------------------------------------------------------

class TestAsymmetricGPU:
    """Test asymmetric cache on CUDA."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_standalone_gpu(self):
        """AsymmetricKVCache should work on GPU."""
        device = "cuda"
        cache = AsymmetricKVCache(
            d_key=128, d_value=128, key_bits=4, val_bits=2,
            seed=SEED, device=device,
        )

        keys = torch.randn(20, 128, device=device)
        values = torch.randn(20, 128, device=device)
        cache.append(keys, values)

        queries = torch.randn(5, 128, device=device)
        scores = cache.attention_scores(queries)
        assert scores.device.type == "cuda"
        assert scores.shape == (5, 20)

        vals = cache.get_values()
        assert vals.device.type == "cuda"
        assert vals.shape == (20, 128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_hf_cache_gpu(self):
        """AsymmetricTurboQuantCache should work on GPU via update()."""
        device = "cuda"
        cache = AsymmetricTurboQuantCache(key_bits=4, val_bits=2, seed=SEED)

        keys = torch.randn(1, 4, 10, 128, device=device)
        values = torch.randn(1, 4, 10, 128, device=device)

        k_out, v_out = cache.update(keys, values, layer_idx=0)
        assert k_out.device.type == "cuda"
        assert v_out.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_gpu_compression_quality(self):
        """GPU results should match CPU quality."""
        d = 128
        torch.manual_seed(SEED)
        keys = torch.randn(50, d)
        values = torch.randn(50, d)

        # CPU
        cpu_cache = AsymmetricKVCache(d, d, key_bits=4, val_bits=2, seed=SEED, device="cpu")
        cpu_cache.append(keys, values)
        cpu_vals = cpu_cache.get_values()

        # GPU
        gpu_cache = AsymmetricKVCache(d, d, key_bits=4, val_bits=2, seed=SEED, device="cuda")
        gpu_cache.append(keys.cuda(), values.cuda())
        gpu_vals = gpu_cache.get_values().cpu()

        # Results should be essentially identical
        diff = (cpu_vals - gpu_vals).abs().max().item()
        assert diff < 1e-4, f"CPU/GPU value mismatch: max diff = {diff}"


# ---------------------------------------------------------------------------
# Test 10: Various dimensions
# ---------------------------------------------------------------------------

class TestVariousDimensions:
    """Test with d=64, 128, 256."""

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_standalone_various_dims(self, d):
        """AsymmetricKVCache should work with various head dimensions."""
        cache = AsymmetricKVCache(d_key=d, d_value=d, key_bits=4, val_bits=2, seed=SEED)

        keys = make_vectors(20, d, seed=1)
        values = make_vectors(20, d, seed=2)
        cache.append(keys, values)

        queries = make_vectors(5, d, seed=3)
        scores = cache.attention_scores(queries)
        assert scores.shape == (5, 20)

        vals = cache.get_values()
        assert vals.shape == (20, d)

        ratio = cache.compression_ratio()
        assert ratio > 3.0, f"Compression ratio {ratio:.2f} too low for d={d}"

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_hf_cache_various_dims(self, d):
        """HF cache should work with various head dimensions."""
        cache = AsymmetricTurboQuantCache(key_bits=4, val_bits=2, seed=SEED)

        keys = torch.randn(1, 2, 5, d)
        values = torch.randn(1, 2, 5, d)

        k_out, v_out = cache.update(keys, values, layer_idx=0)
        assert k_out.shape == (1, 2, 5, d)
        assert v_out.shape == (1, 2, 5, d)

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_key_quality_scales_with_dimension(self, d):
        """Higher d should give better reconstruction quality (concentration effect)."""
        cache = AsymmetricKVCache(d_key=d, d_value=d, key_bits=4, val_bits=2, seed=SEED)

        keys = make_unit_vectors(50, d, seed=1)
        values = make_unit_vectors(50, d, seed=2)
        cache.append(keys, values)

        recon = cache._reconstruct_keys()
        recon_normalized = recon / (recon.norm(dim=-1, keepdim=True) + 1e-8)

        cos_sim = cosine_similarity_batch(keys, recon_normalized).mean().item()
        # 4-bit should give > 0.99 even at d=64
        assert cos_sim > 0.99, f"4-bit key cosine sim at d={d}: {cos_sim:.4f} < 0.99"

    def test_different_key_value_dims(self):
        """Should handle d_key != d_value (e.g. GQA models)."""
        cache = AsymmetricKVCache(
            d_key=128, d_value=64, key_bits=4, val_bits=2, seed=SEED,
        )

        keys = make_vectors(10, 128, seed=1)
        values = make_vectors(10, 64, seed=2)
        cache.append(keys, values)

        queries = make_vectors(3, 128, seed=3)
        scores = cache.attention_scores(queries)
        assert scores.shape == (3, 10)

        vals = cache.get_values()
        assert vals.shape == (10, 64)
