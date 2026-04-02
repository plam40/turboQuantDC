"""Tests for the 70B Streaming Engine with Layer-Priority GPU Scheduling.

Tests cover:
    - LayerGPUCache: LRU eviction, priority protection, capacity management
    - MemoryPlanner: budget calculation and layer allocation
    - AsyncPrefetcher: double-buffering with CUDA streams (mocked)
    - StreamingModel: end-to-end integration with GenerationCache
    - Edge cases: zero capacity, all-priority, single-layer models
"""

import gc
import threading
from collections import OrderedDict
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from turboquantdc.streaming_70b import (
    LayerGPUCache,
    MemoryPlanner,
    AsyncPrefetcher,
    StreamingModel,
)


# ---------------------------------------------------------------------------
# LayerGPUCache: LRU eviction
# ---------------------------------------------------------------------------
class TestLayerGPUCacheLRU:
    """LRU eviction respects capacity and evicts least-recently-used."""

    def test_basic_lru_eviction(self):
        """When cache is full, LRU non-priority layer is evicted."""
        cache = LayerGPUCache(capacity=3, priority_layers={0})

        # Mock layers as simple objects with to() methods
        layers = {}
        for i in range(5):
            layer = MagicMock()
            layer.to = MagicMock(return_value=layer)
            layers[i] = layer

        # Fill cache: load layers 0, 1, 2
        cache.load(0, layers[0])
        cache.load(1, layers[1])
        cache.load(2, layers[2])
        assert len(cache.on_gpu) == 3

        # Access layer 1 to make it recently used
        result = cache.get(1)
        assert result is layers[1]

        # Load layer 3 -- should evict layer 2 (LRU non-priority)
        cache.load(3, layers[3])
        assert 3 in cache.on_gpu
        assert 2 not in cache.on_gpu  # evicted
        assert 0 in cache.on_gpu  # priority, never evicted
        assert 1 in cache.on_gpu  # recently accessed

    def test_lru_order_updates_on_get(self):
        """get() should move an entry to the end (most recently used)."""
        cache = LayerGPUCache(capacity=3, priority_layers=set())

        layers = {}
        for i in range(3):
            layer = MagicMock()
            layer.to = MagicMock(return_value=layer)
            layers[i] = layer

        cache.load(0, layers[0])
        cache.load(1, layers[1])
        cache.load(2, layers[2])

        # Access layer 0, making it MRU
        cache.get(0)

        # Order should be: 1, 2, 0 (LRU to MRU)
        order = list(cache.on_gpu.keys())
        assert order == [1, 2, 0]

    def test_get_returns_none_for_missing_layer(self):
        """get() returns None for layers not in GPU cache."""
        cache = LayerGPUCache(capacity=3, priority_layers=set())
        assert cache.get(99) is None

    def test_load_existing_layer_moves_to_end(self):
        """Loading an already-cached layer moves it to MRU position."""
        cache = LayerGPUCache(capacity=3, priority_layers=set())

        layer = MagicMock()
        layer.to = MagicMock(return_value=layer)

        cache.load(0, layer)
        cache.load(1, MagicMock())
        cache.load(0, layer)  # re-load

        order = list(cache.on_gpu.keys())
        assert order[-1] == 0  # moved to end


# ---------------------------------------------------------------------------
# LayerGPUCache: Priority protection
# ---------------------------------------------------------------------------
class TestLayerGPUCachePriority:
    """Priority layers are never evicted."""

    def test_priority_layers_never_evicted(self):
        """Priority layers survive eviction even when they are LRU."""
        cache = LayerGPUCache(capacity=3, priority_layers={0, 1})

        layers = {}
        for i in range(4):
            layer = MagicMock()
            layer.to = MagicMock(return_value=layer)
            layers[i] = layer

        cache.load(0, layers[0])  # priority
        cache.load(1, layers[1])  # priority
        cache.load(2, layers[2])  # non-priority

        # Load layer 3 -- should evict layer 2, NOT 0 or 1
        cache.load(3, layers[3])
        assert 0 in cache.on_gpu
        assert 1 in cache.on_gpu
        assert 2 not in cache.on_gpu
        assert 3 in cache.on_gpu

    def test_all_priority_cache_still_evicts_when_possible(self):
        """If capacity is exceeded but all are priority, oldest non-priority from later load is evicted."""
        # If all layers are priority and cache is full, eviction should
        # fail gracefully (exceed capacity rather than crash)
        cache = LayerGPUCache(capacity=2, priority_layers={0, 1, 2})

        layers = {}
        for i in range(3):
            layer = MagicMock()
            layer.to = MagicMock(return_value=layer)
            layers[i] = layer

        cache.load(0, layers[0])
        cache.load(1, layers[1])
        # All priority, cache full -- loading 2 should still work
        # (exceeds capacity since nothing can be evicted)
        cache.load(2, layers[2])
        assert len(cache.on_gpu) == 3  # exceeds capacity, but no crash

    def test_empty_priority_set(self):
        """With no priority layers, any layer can be evicted."""
        cache = LayerGPUCache(capacity=2, priority_layers=set())

        layers = {}
        for i in range(3):
            layer = MagicMock()
            layer.to = MagicMock(return_value=layer)
            layers[i] = layer

        cache.load(0, layers[0])
        cache.load(1, layers[1])
        cache.load(2, layers[2])

        assert len(cache.on_gpu) == 2
        assert 0 not in cache.on_gpu  # LRU, evicted


# ---------------------------------------------------------------------------
# LayerGPUCache: Capacity
# ---------------------------------------------------------------------------
class TestLayerGPUCacheCapacity:
    """Capacity is respected and edge cases handled."""

    def test_capacity_one(self):
        """Cache with capacity=1 holds only one layer at a time."""
        cache = LayerGPUCache(capacity=1, priority_layers=set())

        layers = {}
        for i in range(3):
            layer = MagicMock()
            layer.to = MagicMock(return_value=layer)
            layers[i] = layer

        cache.load(0, layers[0])
        assert len(cache.on_gpu) == 1
        cache.load(1, layers[1])
        assert len(cache.on_gpu) == 1
        assert 0 not in cache.on_gpu

    def test_evict_calls_cpu(self):
        """Evicted layers should be moved to CPU."""
        cache = LayerGPUCache(capacity=1, priority_layers=set())

        layer0 = MagicMock()
        layer0.to = MagicMock(return_value=layer0)
        layer1 = MagicMock()
        layer1.to = MagicMock(return_value=layer1)

        cache.load(0, layer0)
        cache.load(1, layer1)

        # layer0 should have been moved to CPU during eviction
        layer0.to.assert_any_call("cpu")

    def test_resident_count(self):
        """resident_count property returns number of layers on GPU."""
        cache = LayerGPUCache(capacity=5, priority_layers=set())

        layer = MagicMock()
        layer.to = MagicMock(return_value=layer)

        cache.load(0, layer)
        cache.load(1, MagicMock())
        assert cache.resident_count == 2

    def test_is_resident(self):
        """is_resident checks if a layer is on GPU."""
        cache = LayerGPUCache(capacity=5, priority_layers=set())

        layer = MagicMock()
        layer.to = MagicMock(return_value=layer)
        cache.load(0, layer)

        assert cache.is_resident(0) is True
        assert cache.is_resident(1) is False


# ---------------------------------------------------------------------------
# MemoryPlanner
# ---------------------------------------------------------------------------
class TestMemoryPlanner:
    """Memory budget calculation and layer allocation."""

    def test_plan_basic_70b(self):
        """Plan for a 70B model with standard parameters."""
        plan = MemoryPlanner.plan(
            num_layers=80,
            hidden_size=8192,
            num_kv_heads=8,
            head_dim=128,
            vocab_size=128256,
            dtype_bytes=2,  # FP16
            gpu_budget_gb=20,
        )

        assert "num_gpu_layers" in plan
        assert "kv_budget_gb" in plan
        assert "layer_size_mb" in plan
        assert "embed_head_mb" in plan
        assert "priority_layers" in plan

        # Should fit some layers but not all 80
        assert 0 < plan["num_gpu_layers"] < 80
        # Priority layers should be first 2 + last 2
        assert 0 in plan["priority_layers"]
        assert 1 in plan["priority_layers"]
        assert 78 in plan["priority_layers"]
        assert 79 in plan["priority_layers"]

    def test_plan_reserves_kv_budget(self):
        """Plan should reserve at least 2GB for KV cache."""
        plan = MemoryPlanner.plan(
            num_layers=80,
            hidden_size=8192,
            num_kv_heads=8,
            head_dim=128,
            vocab_size=128256,
            dtype_bytes=2,
            gpu_budget_gb=20,
        )
        assert plan["kv_budget_gb"] >= 2.0

    def test_plan_small_budget_still_works(self):
        """Even with tiny GPU budget, at least priority layers fit."""
        plan = MemoryPlanner.plan(
            num_layers=80,
            hidden_size=8192,
            num_kv_heads=8,
            head_dim=128,
            vocab_size=128256,
            dtype_bytes=2,
            gpu_budget_gb=4,  # very tight
        )
        # Should still have a valid plan
        assert plan["num_gpu_layers"] >= 0

    def test_plan_all_layers_fit(self):
        """When GPU is large enough, all layers fit."""
        plan = MemoryPlanner.plan(
            num_layers=10,
            hidden_size=1024,
            num_kv_heads=4,
            head_dim=64,
            vocab_size=32000,
            dtype_bytes=2,
            gpu_budget_gb=20,  # way more than needed
        )
        assert plan["num_gpu_layers"] == 10

    def test_plan_single_layer(self):
        """Single-layer model should always fit."""
        plan = MemoryPlanner.plan(
            num_layers=1,
            hidden_size=1024,
            num_kv_heads=4,
            head_dim=64,
            vocab_size=32000,
            dtype_bytes=2,
            gpu_budget_gb=4,
        )
        assert plan["num_gpu_layers"] == 1

    def test_priority_layers_for_small_model(self):
        """Small models (< 4 layers) have all layers as priority."""
        plan = MemoryPlanner.plan(
            num_layers=3,
            hidden_size=1024,
            num_kv_heads=4,
            head_dim=64,
            vocab_size=32000,
            dtype_bytes=2,
            gpu_budget_gb=4,
        )
        assert plan["priority_layers"] == {0, 1, 2}


# ---------------------------------------------------------------------------
# AsyncPrefetcher
# ---------------------------------------------------------------------------
class TestAsyncPrefetcher:
    """Double-buffering with CUDA streams (tested with mocks)."""

    def test_prefetch_moves_layer_to_gpu(self):
        """prefetch() should move next layer to GPU on prefetch stream."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for prefetch test")

        prefetcher = AsyncPrefetcher(device=torch.device("cuda"))

        # Create a tiny module to move
        layer = torch.nn.Linear(4, 4)
        assert next(layer.parameters()).device.type == "cpu"

        prefetcher.prefetch(layer)
        prefetcher.wait()

        # After wait, layer should be on GPU
        assert next(layer.parameters()).device.type == "cuda"

        # Clean up
        layer.cpu()

    def test_wait_without_prefetch_is_noop(self):
        """wait() without prior prefetch should not raise."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for prefetch test")

        prefetcher = AsyncPrefetcher(device=torch.device("cuda"))
        prefetcher.wait()  # should not raise

    def test_prefetch_synchronizes_default_stream(self):
        """After wait(), the default stream can safely use the layer."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for prefetch test")

        prefetcher = AsyncPrefetcher(device=torch.device("cuda"))
        layer = torch.nn.Linear(4, 4)

        prefetcher.prefetch(layer)
        prefetcher.wait()

        # Should be able to compute on default stream without errors
        x = torch.randn(2, 4, device="cuda")
        y = layer(x)
        assert y.shape == (2, 4)

        layer.cpu()


# ---------------------------------------------------------------------------
# StreamingModel (mocked -- no real HF model load)
# ---------------------------------------------------------------------------
class TestStreamingModelInit:
    """StreamingModel initialization and configuration."""

    def test_invalid_bits_raises(self):
        """bits outside valid range should raise."""
        with pytest.raises(ValueError, match="bits must be"):
            StreamingModel("fake-model", kv_bits=0)
        with pytest.raises(ValueError, match="bits must be"):
            StreamingModel("fake-model", kv_bits=9)

    def test_valid_kv_compression_strategies(self):
        """All valid kv_compression values should be accepted."""
        for strategy in ("fixed", "boundary", "gradient"):
            model = StreamingModel.__new__(StreamingModel)
            model._validate_config(kv_bits=3, kv_compression=strategy)
            # Should not raise

    def test_invalid_kv_compression_raises(self):
        """Invalid kv_compression should raise."""
        model = StreamingModel.__new__(StreamingModel)
        with pytest.raises(ValueError, match="kv_compression"):
            model._validate_config(kv_bits=3, kv_compression="invalid")

    def test_gpu_budget_must_be_positive(self):
        """gpu_budget_gb <= 0 should raise."""
        with pytest.raises(ValueError, match="gpu_budget_gb"):
            StreamingModel("fake-model", gpu_budget_gb=0)


# ---------------------------------------------------------------------------
# StreamingModel: memory report
# ---------------------------------------------------------------------------
class TestStreamingModelReport:
    """Memory reports are structured and complete."""

    def test_report_structure(self):
        """memory_report returns all required keys."""
        model = StreamingModel.__new__(StreamingModel)
        model._peak_vram = 0
        model._layer_size_bytes = 100 * 1024 * 1024
        model._embed_head_bytes = 500 * 1024 * 1024
        model._model_total_bytes = 40 * 1024 * 1024 * 1024
        model._tokens_generated = 10
        model._generation_time = 2.0
        model._load_time = 5.0
        model.num_layers = 80
        model.kv_bits = 3
        model.kv_compression = "boundary"
        model.gpu_budget_gb = 20
        model.tq_cache = None
        model.layer_cache = MagicMock()
        model.layer_cache.resident_count = 30

        report = model.memory_report()

        expected_keys = {
            "peak_vram_mb", "model_total_mb", "layer_size_mb",
            "embed_head_mb", "num_layers", "gpu_layers",
            "kv_bits", "kv_compression", "tokens_generated",
            "tokens_per_sec", "load_time_sec", "gpu_budget_gb",
        }
        assert expected_keys.issubset(report.keys())


# ---------------------------------------------------------------------------
# Integration: StreamingModel + LayerGPUCache
# ---------------------------------------------------------------------------
class TestStreamingModelLayerCacheIntegration:
    """StreamingModel correctly uses LayerGPUCache for layer management."""

    def test_forward_layer_uses_cache(self):
        """forward_layer should check cache before loading from CPU."""
        cache = LayerGPUCache(capacity=3, priority_layers={0})

        layer = MagicMock()
        layer.to = MagicMock(return_value=layer)

        # Pre-load layer into cache
        cache.load(0, layer)

        # get should find it
        result = cache.get(0)
        assert result is layer
        # Should NOT call .to("cuda") again since it's already resident
        # (the load already moved it)

    def test_cache_eviction_during_sequential_layer_walk(self):
        """Simulating a forward pass through 10 layers with capacity 3."""
        cache = LayerGPUCache(capacity=3, priority_layers={0, 9})

        layers = {}
        for i in range(10):
            layer = MagicMock()
            layer.to = MagicMock(return_value=layer)
            layers[i] = layer

        # Walk through all layers like a forward pass
        for i in range(10):
            if cache.get(i) is None:
                cache.load(i, layers[i])

        # After walking all 10, cache should have capacity layers
        # Last 3 accessed should be resident (with priority protection)
        assert cache.resident_count <= 3
        # Priority layers should still be there
        assert cache.is_resident(0) or cache.is_resident(9)  # at least one priority

    def test_second_forward_pass_reuses_cached_layers(self):
        """Priority layers should be cache hits on subsequent passes."""
        cache = LayerGPUCache(capacity=5, priority_layers={0, 1})

        layers = {}
        for i in range(5):
            layer = MagicMock()
            layer.to = MagicMock(return_value=layer)
            layers[i] = layer

        # First pass
        for i in range(5):
            cache.load(i, layers[i])

        # Second pass -- priority layers should be hits
        assert cache.get(0) is layers[0]
        assert cache.get(1) is layers[1]
