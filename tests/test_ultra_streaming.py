"""Tests for the Ultra-Streaming Engine for 200B+ models.

Tests cover:
    - ModelAnalyzer: dense vs MoE detection, size calculations
    - WeightManager: LRU eviction, priority pinning, stats
    - KVManager: auto-selection, cache creation
    - Memory planning: 70B, 200B, 405B, MoE models
    - UltraStreamingEngine: config validation, analysis, reports
    - Known architectures: all entries are well-formed
    - Integration: end-to-end plan + weight manager + KV manager flow
"""

import sys
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch

from turboquantdc.ultra_streaming import (
    KNOWN_ARCHITECTURES,
    KVManager,
    ModelAnalyzer,
    UltraStreamingEngine,
    WeightManager,
    format_plan_report,
    plan_memory,
)


# ---------------------------------------------------------------------------
# ModelAnalyzer: dense models
# ---------------------------------------------------------------------------


class TestModelAnalyzerDense:
    """ModelAnalyzer correctly identifies and measures dense architectures."""

    def test_from_dict_70b(self):
        """70B dense model is correctly analyzed."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-70b"], weight_bits=4,
        )
        assert analyzer.is_moe is False
        assert analyzer.num_layers == 80
        assert analyzer.hidden_size == 8192
        assert analyzer.num_kv_heads == 8
        assert analyzer.head_dim == 128
        assert analyzer.weight_bits == 4

    def test_from_dict_7b(self):
        """7B dense model basic properties."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-7b"], weight_bits=4,
        )
        assert analyzer.is_moe is False
        assert analyzer.num_layers == 32
        assert analyzer.hidden_size == 4096
        assert analyzer.num_kv_heads == 32  # no GQA for 7B

    def test_embedding_size_positive(self):
        """Embedding size should be positive and reasonable."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-70b"], weight_bits=4,
        )
        assert analyzer.embedding_size_bytes > 0
        # 128256 * 8192 * 2 bytes * 2 (embed + head) ~ 4 GB
        assert analyzer.embedding_size_gb > 1.0
        assert analyzer.embedding_size_gb < 10.0

    def test_layer_size_70b(self):
        """Layer size should be ~0.9 GB at FP16 for 70B."""
        # At FP16 (16-bit), one 70B layer is ~0.9 GB
        analyzer_fp16 = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-70b"], weight_bits=16,
        )
        layer_gb = analyzer_fp16.layer_size_gb
        assert 0.5 < layer_gb < 2.0  # reasonable range for 70B

    def test_layer_size_scales_with_bits(self):
        """4-bit layer should be ~4x smaller than 16-bit."""
        analyzer_16 = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-70b"], weight_bits=16,
        )
        analyzer_4 = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-70b"], weight_bits=4,
        )
        # 4-bit is 4x smaller than 16-bit
        ratio = analyzer_16.layer_size_bytes / analyzer_4.layer_size_bytes
        assert 3.5 < ratio < 4.5

    def test_active_layer_equals_layer_for_dense(self):
        """For dense models, active layer size == full layer size."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-70b"], weight_bits=4,
        )
        assert analyzer.active_layer_size_bytes == analyzer.layer_size_bytes

    def test_total_size_70b_4bit(self):
        """70B at 4-bit should be ~35 GB total."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-70b"], weight_bits=4,
        )
        # 70B params * 0.5 bytes/param = 35 GB + embeddings
        assert 30 < analyzer.total_size_gb < 50

    def test_total_size_405b_4bit(self):
        """405B at 4-bit should be ~200 GB total."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-405b"], weight_bits=4,
        )
        # Massive -- over 100 GB even at 4-bit
        assert analyzer.total_size_gb > 100

    def test_kv_bytes_per_token_positive(self):
        """KV bytes per token should be positive."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-70b"], weight_bits=4,
        )
        assert analyzer.kv_bytes_per_token_fp16 > 0

    def test_kv_bytes_scales_with_layers(self):
        """More layers = more KV bytes per token."""
        a_70b = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-70b"], weight_bits=4,
        )
        a_405b = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-405b"], weight_bits=4,
        )
        assert a_405b.kv_bytes_per_token_fp16 > a_70b.kv_bytes_per_token_fp16

    def test_summary_has_required_keys(self):
        """Summary dict should have all required keys for dense models."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-70b"], weight_bits=4,
        )
        s = analyzer.summary()
        required = {
            "type", "num_layers", "hidden_size", "num_attention_heads",
            "num_kv_heads", "head_dim", "intermediate_size", "vocab_size",
            "weight_bits", "embedding_size_mb", "layer_size_mb",
            "total_size_gb", "kv_bytes_per_token_fp16",
        }
        assert required.issubset(s.keys())
        assert s["type"] == "dense"


# ---------------------------------------------------------------------------
# ModelAnalyzer: MoE models
# ---------------------------------------------------------------------------


class TestModelAnalyzerMoE:
    """ModelAnalyzer correctly identifies and measures MoE architectures."""

    def test_moe_detection(self):
        """MoE models are correctly identified."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama4-scout"], weight_bits=4,
        )
        assert analyzer.is_moe is True
        assert analyzer.num_experts == 16
        assert analyzer.num_active_experts == 1

    def test_deepseek_v3_moe(self):
        """DeepSeek V3 (685B, 256 experts) is correctly analyzed."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["deepseek-v3"], weight_bits=4,
        )
        assert analyzer.is_moe is True
        assert analyzer.num_experts == 256
        assert analyzer.num_active_experts == 8
        assert analyzer.total_size_gb > 100

    def test_mixtral_moe(self):
        """Mixtral 8x7B is correctly analyzed."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["mixtral-8x7b"], weight_bits=4,
        )
        assert analyzer.is_moe is True
        assert analyzer.num_experts == 8
        assert analyzer.num_active_experts == 2

    def test_active_layer_smaller_than_full_for_moe(self):
        """For MoE, active layer size < full layer size."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["deepseek-v3"], weight_bits=4,
        )
        assert analyzer.active_layer_size_bytes < analyzer.layer_size_bytes

    def test_expert_size_positive(self):
        """Expert size should be positive for MoE models."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama4-scout"], weight_bits=4,
        )
        assert analyzer.expert_size_bytes > 0

    def test_moe_summary_has_expert_keys(self):
        """MoE summary should include expert-specific fields."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["deepseek-v3"], weight_bits=4,
        )
        s = analyzer.summary()
        assert s["type"] == "moe"
        assert "num_experts" in s
        assert "num_active_experts" in s
        assert "expert_size_mb" in s
        assert "active_layer_size_mb" in s

    def test_moe_layer_includes_all_experts(self):
        """Full layer size should include all experts, not just active ones."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["mixtral-8x7b"], weight_bits=4,
        )
        # Full layer includes 8 experts
        # Active layer includes only 2
        ratio = analyzer.layer_size_bytes / analyzer.active_layer_size_bytes
        assert ratio > 2.0  # Should be significantly larger


# ---------------------------------------------------------------------------
# ModelAnalyzer: from HF config
# ---------------------------------------------------------------------------


class TestModelAnalyzerFromConfig:
    """ModelAnalyzer works with HuggingFace AutoConfig objects."""

    def _make_mock_config(self, is_moe=False):
        """Create a mock HF config using a simple namespace (not MagicMock)."""

        class SimpleConfig:
            pass

        config = SimpleConfig()
        config.num_hidden_layers = 32
        config.hidden_size = 4096
        config.num_attention_heads = 32
        config.num_key_value_heads = 8
        config.head_dim = 128
        config.intermediate_size = 11008
        config.vocab_size = 32000
        if is_moe:
            config.num_local_experts = 8
            config.num_experts_per_tok = 2
        else:
            config.num_local_experts = 0
            config.num_experts_per_tok = 0
        return config

    def test_from_config_dense(self):
        """Dense config is correctly parsed."""
        config = self._make_mock_config(is_moe=False)
        analyzer = ModelAnalyzer(config=config, weight_bits=4)
        assert analyzer.is_moe is False
        assert analyzer.num_layers == 32
        assert analyzer.hidden_size == 4096
        assert analyzer.num_kv_heads == 8

    def test_from_config_moe(self):
        """MoE config is correctly parsed."""
        config = self._make_mock_config(is_moe=True)
        analyzer = ModelAnalyzer(config=config, weight_bits=4)
        assert analyzer.is_moe is True
        assert analyzer.num_experts == 8
        assert analyzer.num_active_experts == 2

    def test_neither_config_nor_dict_raises(self):
        """Must provide either config or arch_dict."""
        with pytest.raises(ValueError, match="Either config or arch_dict"):
            ModelAnalyzer(config=None, arch_dict=None)


# ---------------------------------------------------------------------------
# WeightManager: LRU cache
# ---------------------------------------------------------------------------


class TestWeightManagerLRU:
    """WeightManager LRU eviction and caching."""

    def _make_module(self, size_bytes=1000):
        """Create a mock module for testing."""
        module = MagicMock()
        module.to = MagicMock(return_value=module)
        return module

    def test_basic_cache_hit(self):
        """Loaded module should be retrievable."""
        mgr = WeightManager(gpu_budget_bytes=10000)
        module = self._make_module()
        mgr.load((0,), module, 1000)
        result = mgr.get((0,))
        assert result is module

    def test_cache_miss_returns_none(self):
        """Non-cached key returns None."""
        mgr = WeightManager(gpu_budget_bytes=10000)
        assert mgr.get((99,)) is None

    def test_lru_eviction(self):
        """When budget is exceeded, LRU entry is evicted."""
        mgr = WeightManager(gpu_budget_bytes=2500)
        m0 = self._make_module()
        m1 = self._make_module()
        m2 = self._make_module()

        mgr.load((0,), m0, 1000)
        mgr.load((1,), m1, 1000)
        # Cache has 2000 bytes. Loading m2 (1000) exceeds 2500, so evict LRU
        mgr.load((2,), m2, 1000)

        # m0 should have been evicted (LRU)
        assert mgr.get((0,)) is None
        assert mgr.get((1,)) is m1
        assert mgr.get((2,)) is m2

    def test_get_updates_lru_order(self):
        """get() should move entry to MRU position."""
        mgr = WeightManager(gpu_budget_bytes=2500)
        m0 = self._make_module()
        m1 = self._make_module()
        m2 = self._make_module()

        mgr.load((0,), m0, 1000)
        mgr.load((1,), m1, 1000)

        # Access m0, making it MRU
        mgr.get((0,))

        # Loading m2 should evict m1 (now LRU), not m0
        mgr.load((2,), m2, 1000)

        assert mgr.get((0,)) is m0
        assert mgr.get((1,)) is None  # evicted
        assert mgr.get((2,)) is m2

    def test_priority_never_evicted(self):
        """Priority entries should never be evicted."""
        mgr = WeightManager(gpu_budget_bytes=2500)
        m_priority = self._make_module()
        m1 = self._make_module()
        m2 = self._make_module()

        mgr.pin_priority((-1,), m_priority, 1000)
        mgr.load((0,), m1, 1000)
        mgr.load((1,), m2, 1000)

        # Priority should survive
        assert mgr.is_cached((-1,))
        assert mgr.is_cached((1,))
        # m1 (key (0,)) might be evicted
        # But priority is safe
        assert mgr.is_cached((-1,))

    def test_evict_all_clears_non_priority(self):
        """evict_all() should remove all non-priority entries."""
        mgr = WeightManager(gpu_budget_bytes=10000)
        m_pri = self._make_module()
        m1 = self._make_module()
        m2 = self._make_module()

        mgr.pin_priority((-1,), m_pri, 1000)
        mgr.load((0,), m1, 1000)
        mgr.load((1,), m2, 1000)

        mgr.evict_all()

        assert mgr.is_cached((-1,))  # priority survives
        assert not mgr.is_cached((0,))
        assert not mgr.is_cached((1,))


# ---------------------------------------------------------------------------
# WeightManager: statistics
# ---------------------------------------------------------------------------


class TestWeightManagerStats:
    """WeightManager statistics tracking."""

    def _make_module(self):
        module = MagicMock()
        module.to = MagicMock(return_value=module)
        return module

    def test_stats_structure(self):
        """Stats dict has all required keys."""
        mgr = WeightManager(gpu_budget_bytes=10000)
        stats = mgr.stats()
        required = {
            "cached_entries", "priority_entries", "current_bytes",
            "budget_bytes", "utilization_pct", "cache_hits",
            "cache_misses", "hit_rate_pct", "total_transfers",
            "total_gb_transferred",
        }
        assert required.issubset(stats.keys())

    def test_hit_rate_tracking(self):
        """Cache hits and misses are tracked correctly."""
        mgr = WeightManager(gpu_budget_bytes=10000)
        m = self._make_module()
        mgr.load((0,), m, 1000)

        mgr.get((0,))  # hit
        mgr.get((0,))  # hit
        mgr.get((1,))  # miss

        assert mgr.cache_hits == 2
        assert mgr.cache_misses == 1
        assert abs(mgr.hit_rate - 2 / 3) < 0.01

    def test_utilization_tracking(self):
        """Utilization reflects current cache usage."""
        mgr = WeightManager(gpu_budget_bytes=10000)
        assert mgr.utilization == 0.0

        m = self._make_module()
        mgr.load((0,), m, 5000)
        assert abs(mgr.utilization - 0.5) < 0.01

    def test_transfer_stats(self):
        """Transfer count and bytes are tracked."""
        mgr = WeightManager(gpu_budget_bytes=10000)
        m = self._make_module()
        mgr.load((0,), m, 1000)
        mgr.load((1,), self._make_module(), 2000)

        assert mgr.total_transfers == 2
        assert mgr.total_bytes_transferred == 3000

    def test_load_existing_does_not_increment_transfers(self):
        """Re-loading an already cached module should not count as a transfer."""
        mgr = WeightManager(gpu_budget_bytes=10000)
        m = self._make_module()
        mgr.load((0,), m, 1000)
        mgr.load((0,), m, 1000)  # already cached

        assert mgr.total_transfers == 1  # only counted once


# ---------------------------------------------------------------------------
# KVManager: auto-selection
# ---------------------------------------------------------------------------


class TestKVManagerAutoSelect:
    """KVManager auto-selects the best cache strategy."""

    def _make_kv_manager(self, kv_budget_gb=10.0, kv_bytes_per_token=327680):
        """Create a KVManager with typical 70B parameters."""
        return KVManager(
            kv_budget_gb=kv_budget_gb,
            num_layers=80,
            kv_bytes_per_token_fp16=kv_bytes_per_token,
            kv_compression="boundary",
            kv_bits=3,
            max_context=32768,
        )

    def test_auto_select_returns_strategy(self):
        """auto_select() returns a strategy dict."""
        mgr = self._make_kv_manager()
        strategy = mgr.auto_select()
        assert "cache_type" in strategy
        assert "compression_ratio" in strategy
        assert "max_tokens" in strategy
        assert "kwargs" in strategy
        assert "description" in strategy

    def test_auto_select_picks_least_aggressive(self):
        """With generous budget, should pick GenerationCache (least aggressive)."""
        mgr = self._make_kv_manager(kv_budget_gb=20.0)
        strategy = mgr.auto_select()
        # With 20GB and 327680 bytes/token, GenerationCache at 5x should
        # easily handle 32K tokens
        assert strategy["cache_type"] == "GenerationCache"

    def test_auto_select_picks_eviction_for_tight_budget(self):
        """With tight budget, should pick EvictionCache."""
        # Very tight: 0.5 GB for 80 layers
        mgr = KVManager(
            kv_budget_gb=0.5,
            num_layers=80,
            kv_bytes_per_token_fp16=327680,
            max_context=32768,
        )
        strategy = mgr.auto_select()
        # Should use aggressive eviction
        assert strategy["cache_type"] == "EvictionCache"

    def test_create_cache_returns_instance(self):
        """create_cache() returns a usable cache object."""
        mgr = self._make_kv_manager()
        cache = mgr.create_cache()
        assert cache is not None
        assert hasattr(cache, "update")
        assert hasattr(cache, "get_seq_length")

    def test_create_cache_generation(self):
        """When GenerationCache is selected, it's created correctly."""
        from turboquantdc.generation_cache import GenerationCache

        mgr = self._make_kv_manager(kv_budget_gb=20.0)
        cache = mgr.create_cache()
        assert isinstance(cache, GenerationCache)

    def test_create_cache_eviction(self):
        """When EvictionCache is selected, it's created correctly."""
        from turboquantdc.token_eviction import EvictionCache

        mgr = KVManager(
            kv_budget_gb=0.5,
            num_layers=80,
            kv_bytes_per_token_fp16=327680,
            max_context=32768,
        )
        cache = mgr.create_cache()
        assert isinstance(cache, EvictionCache)

    def test_cache_accepts_tensor_update(self):
        """Created cache should accept a tensor update (HF protocol)."""
        mgr = self._make_kv_manager()
        cache = mgr.create_cache()
        k = torch.randn(1, 8, 1, 128)
        v = torch.randn(1, 8, 1, 128)
        result = cache.update(k, v, layer_idx=0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_strategy_property(self):
        """Strategy property reflects the selected strategy."""
        mgr = self._make_kv_manager()
        assert mgr.strategy is None
        mgr.auto_select()
        assert mgr.strategy is not None

    def test_cache_property(self):
        """Cache property reflects the created cache."""
        mgr = self._make_kv_manager()
        assert mgr.cache is None
        mgr.create_cache()
        assert mgr.cache is not None


# ---------------------------------------------------------------------------
# Memory planning
# ---------------------------------------------------------------------------


class TestPlanMemory:
    """plan_memory() produces feasible memory plans."""

    def test_plan_70b_on_24gb(self):
        """70B at 4-bit on 24GB RTX 4090 should be feasible."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-70b"], weight_bits=4,
        )
        plan = plan_memory(analyzer, gpu_budget_gb=24.0, cpu_budget_gb=64.0)

        assert plan["feasible"] is True
        assert plan["is_moe"] is False
        assert plan["total_layers"] == 80
        assert plan["kv_budget_gb"] > 0
        assert plan["weight_cache_gb"] > 0
        assert plan["context_at_5x"] > 0
        assert plan["estimated_tok_per_sec"] > 0

    def test_plan_405b_on_24gb(self):
        """405B at 4-bit on 24GB -- needs ~200GB CPU RAM."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-405b"], weight_bits=4,
        )
        plan = plan_memory(analyzer, gpu_budget_gb=24.0, cpu_budget_gb=256.0)

        assert plan["total_layers"] == 126
        # 405B at 4-bit is ~200 GB -- needs lots of CPU RAM
        assert plan["model_size_gb"] > 100
        # With 256 GB CPU, should be feasible
        assert plan["feasible"] is True

    def test_plan_405b_insufficient_cpu(self):
        """405B on 24GB GPU with only 32GB CPU should not be feasible."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-405b"], weight_bits=4,
        )
        plan = plan_memory(analyzer, gpu_budget_gb=24.0, cpu_budget_gb=32.0)

        assert plan["feasible"] is False

    def test_plan_moe_scout(self):
        """Llama 4 Scout (109B MoE, 17B active) on 24GB."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama4-scout"], weight_bits=4,
        )
        plan = plan_memory(analyzer, gpu_budget_gb=24.0, cpu_budget_gb=64.0)

        assert plan["is_moe"] is True
        assert plan["total_layers"] == 48
        assert plan["kv_budget_gb"] > 0
        assert plan["estimated_tok_per_sec"] > 0

    def test_plan_deepseek_v3(self):
        """DeepSeek V3 (685B, 256 experts) on 24GB GPU + 1TB CPU."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["deepseek-v3"], weight_bits=4,
        )
        # DeepSeek V3 is huge even at 4-bit -- needs lots of CPU RAM
        plan = plan_memory(analyzer, gpu_budget_gb=24.0, cpu_budget_gb=1024.0)

        assert plan["is_moe"] is True
        assert plan["total_layers"] == 61
        assert plan["model_size_gb"] > 100
        assert plan["feasible"] is True

    def test_plan_small_model_all_layers_fit(self):
        """A small 7B model at 4-bit should fit entirely on 24GB GPU."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-7b"], weight_bits=4,
        )
        plan = plan_memory(analyzer, gpu_budget_gb=24.0, cpu_budget_gb=32.0)

        assert plan["feasible"] is True
        assert plan["layers_cached"] == 32  # all layers fit
        assert plan["layers_streamed"] == 0

    def test_context_scales_with_kv_budget(self):
        """More GPU VRAM = more context capacity."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-70b"], weight_bits=4,
        )
        plan_small = plan_memory(analyzer, gpu_budget_gb=16.0)
        plan_large = plan_memory(analyzer, gpu_budget_gb=48.0)

        assert plan_large["context_at_5x"] >= plan_small["context_at_5x"]

    def test_plan_has_all_required_keys(self):
        """Plan dict has all the keys we depend on."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-70b"], weight_bits=4,
        )
        plan = plan_memory(analyzer, gpu_budget_gb=24.0)

        required = {
            "model_size_gb", "fits_cpu", "cpu_needed_gb",
            "embedding_gb", "layer_size_gb", "active_layer_size_gb",
            "overhead_gb", "active_buffer_gb",
            "weight_cache_gb", "kv_budget_gb",
            "layers_cached", "layers_streamed", "total_layers",
            "context_at_5x", "context_at_7_5x", "context_at_10x",
            "estimated_tok_per_sec", "is_moe",
            "gpu_budget_gb", "cpu_budget_gb", "target_context", "feasible",
        }
        assert required.issubset(plan.keys())

    def test_kv_budget_at_least_2gb(self):
        """KV budget should always be at least 2 GB."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-405b"], weight_bits=4,
        )
        plan = plan_memory(analyzer, gpu_budget_gb=8.0)
        assert plan["kv_budget_gb"] >= 2.0

    def test_higher_compression_means_more_context(self):
        """10x compression should give more context than 5x."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-70b"], weight_bits=4,
        )
        plan = plan_memory(analyzer, gpu_budget_gb=24.0)
        assert plan["context_at_10x"] >= plan["context_at_7_5x"]
        assert plan["context_at_7_5x"] >= plan["context_at_5x"]


# ---------------------------------------------------------------------------
# format_plan_report
# ---------------------------------------------------------------------------


class TestFormatPlanReport:
    """format_plan_report() produces readable reports."""

    def _make_report(self, arch_key="llama-70b", gpu_gb=24.0):
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES[arch_key], weight_bits=4,
        )
        plan = plan_memory(analyzer, gpu_budget_gb=gpu_gb, cpu_budget_gb=64.0)
        return format_plan_report(f"meta/{arch_key}", analyzer, plan)

    def test_report_mentions_model(self):
        """Report should mention the model name."""
        report = self._make_report("llama-70b")
        assert "llama-70b" in report

    def test_report_mentions_dense(self):
        """Dense model report should say 'Dense'."""
        report = self._make_report("llama-70b")
        assert "Dense" in report

    def test_report_mentions_moe(self):
        """MoE model report should say 'MoE'."""
        report = self._make_report("deepseek-v3")
        assert "MoE" in report

    def test_report_contains_context_info(self):
        """Report should show context capacity."""
        report = self._make_report("llama-70b")
        assert "compression" in report

    def test_report_contains_speed_estimate(self):
        """Report should show speed estimate."""
        report = self._make_report("llama-70b")
        assert "tok/s" in report

    def test_report_feasible_status(self):
        """Report should show feasibility status."""
        report = self._make_report("llama-70b")
        assert "FEASIBLE" in report

    def test_report_infeasible_has_warning(self):
        """Infeasible plan should show a warning."""
        analyzer = ModelAnalyzer(
            arch_dict=KNOWN_ARCHITECTURES["llama-405b"], weight_bits=4,
        )
        plan = plan_memory(analyzer, gpu_budget_gb=24.0, cpu_budget_gb=16.0)
        report = format_plan_report("llama-405b", analyzer, plan)
        assert "WARNING" in report or "NOT FEASIBLE" in report


# ---------------------------------------------------------------------------
# KNOWN_ARCHITECTURES: consistency
# ---------------------------------------------------------------------------


class TestKnownArchitectures:
    """All entries in KNOWN_ARCHITECTURES are well-formed."""

    def test_all_have_required_keys(self):
        """Every architecture has the minimum required keys."""
        required = {
            "num_layers", "hidden_size", "num_attention_heads",
            "head_dim", "vocab_size",
        }
        for name, arch in KNOWN_ARCHITECTURES.items():
            missing = required - set(arch.keys())
            assert not missing, f"{name} missing keys: {missing}"

    def test_moe_have_expert_keys(self):
        """MoE architectures have expert-specific keys."""
        for name, arch in KNOWN_ARCHITECTURES.items():
            if arch.get("type") == "moe":
                assert "num_experts" in arch, f"{name} missing num_experts"
                assert "num_active_experts" in arch, f"{name} missing num_active_experts"
                assert arch["num_experts"] > 1, f"{name} has <= 1 expert"
                assert arch["num_active_experts"] > 0, f"{name} has 0 active experts"
                assert arch["num_active_experts"] <= arch["num_experts"]

    def test_all_values_positive(self):
        """All numeric values should be positive."""
        for name, arch in KNOWN_ARCHITECTURES.items():
            for key in ["num_layers", "hidden_size", "num_attention_heads",
                        "head_dim", "vocab_size"]:
                assert arch[key] > 0, f"{name}.{key} is not positive"

    def test_analyzable(self):
        """Every known architecture can be analyzed without errors."""
        for name, arch in KNOWN_ARCHITECTURES.items():
            analyzer = ModelAnalyzer(arch_dict=arch, weight_bits=4)
            summary = analyzer.summary()
            assert summary["total_size_gb"] > 0, f"{name} has 0 total size"

    def test_at_least_one_dense_and_one_moe(self):
        """Should have at least one dense and one MoE architecture."""
        types = {arch.get("type", "dense") for arch in KNOWN_ARCHITECTURES.values()}
        assert "dense" in types
        assert "moe" in types


# ---------------------------------------------------------------------------
# UltraStreamingEngine: configuration validation
# ---------------------------------------------------------------------------


class TestUltraStreamingEngineConfig:
    """UltraStreamingEngine validates configuration parameters."""

    def test_invalid_kv_bits_low(self):
        """kv_bits < 2 should raise."""
        with pytest.raises(ValueError, match="kv_bits"):
            UltraStreamingEngine("fake-model", kv_bits=1)

    def test_invalid_kv_bits_high(self):
        """kv_bits > 8 should raise."""
        with pytest.raises(ValueError, match="kv_bits"):
            UltraStreamingEngine("fake-model", kv_bits=9)

    def test_invalid_gpu_budget(self):
        """gpu_budget_gb <= 0 should raise."""
        with pytest.raises(ValueError, match="gpu_budget_gb"):
            UltraStreamingEngine("fake-model", gpu_budget_gb=0)

    def test_invalid_cpu_budget(self):
        """cpu_budget_gb <= 0 should raise."""
        with pytest.raises(ValueError, match="cpu_budget_gb"):
            UltraStreamingEngine("fake-model", cpu_budget_gb=-1)

    def test_invalid_weight_bits(self):
        """weight_bits not in (4, 8, 16) should raise."""
        with pytest.raises(ValueError, match="weight_bits"):
            UltraStreamingEngine("fake-model", weight_bits=3)

    def test_invalid_kv_compression(self):
        """Invalid kv_compression strategy should raise."""
        with pytest.raises(ValueError, match="kv_compression"):
            UltraStreamingEngine("fake-model", kv_compression="magic")

    def test_valid_construction(self):
        """Valid parameters should construct without error."""
        engine = UltraStreamingEngine(
            "meta-llama/Llama-3.3-70B-Instruct",
            gpu_budget_gb=22,
            cpu_budget_gb=64,
            kv_compression="boundary",
            kv_bits=3,
            weight_bits=4,
        )
        assert engine.model_name == "meta-llama/Llama-3.3-70B-Instruct"
        assert engine.gpu_budget_gb == 22
        assert engine.cpu_budget_gb == 64


# ---------------------------------------------------------------------------
# UltraStreamingEngine: offline analysis
# ---------------------------------------------------------------------------


class TestUltraStreamingEngineAnalysis:
    """UltraStreamingEngine.analyze() works without loading the model."""

    def test_analyze_known_model(self):
        """analyze() with a known model name should work offline."""
        engine = UltraStreamingEngine(
            "some-model-70b", gpu_budget_gb=24, cpu_budget_gb=64,
        )
        plan = engine.analyze()
        assert plan["total_layers"] == 80  # matched to llama-70b
        assert plan["is_moe"] is False
        assert plan["feasible"] is True

    def test_analyze_moe_model(self):
        """analyze() with a known MoE model name."""
        engine = UltraStreamingEngine(
            "llama4-scout-model", gpu_budget_gb=24, cpu_budget_gb=64,
        )
        plan = engine.analyze()
        assert plan["is_moe"] is True
        assert plan["total_layers"] == 48

    def test_analyze_405b(self):
        """analyze() with a 405B model name."""
        engine = UltraStreamingEngine(
            "big-405b-model", gpu_budget_gb=24, cpu_budget_gb=256,
        )
        plan = engine.analyze()
        assert plan["total_layers"] == 126
        assert plan["model_size_gb"] > 100

    def test_analyze_with_config(self):
        """analyze() with an explicit config object."""

        class SimpleConfig:
            pass

        config = SimpleConfig()
        config.num_hidden_layers = 40
        config.hidden_size = 5120
        config.num_attention_heads = 40
        config.num_key_value_heads = 8
        config.head_dim = 128
        config.intermediate_size = 13824
        config.vocab_size = 32000
        config.num_local_experts = 0
        config.num_experts_per_tok = 0

        engine = UltraStreamingEngine(
            "custom-model", gpu_budget_gb=24, cpu_budget_gb=64,
        )
        plan = engine.analyze(config=config)
        assert plan["total_layers"] == 40
        assert plan["is_moe"] is False

    def test_analysis_report_string(self):
        """analysis_report() returns a non-empty string."""
        engine = UltraStreamingEngine(
            "some-model-70b", gpu_budget_gb=24, cpu_budget_gb=64,
        )
        report = engine.analysis_report()
        assert isinstance(report, str)
        assert len(report) > 100
        assert "70b" in report.lower() or "Dense" in report

    def test_match_known_architecture(self):
        """_match_known_architecture matches common model name patterns."""
        match = UltraStreamingEngine._match_known_architecture
        assert match("meta-llama/Llama-3.3-70B-Instruct") is not None
        assert match("deepseek-ai/DeepSeek-V3") is not None
        assert match("mistralai/Mixtral-8x7B-v0.1") is not None
        assert match("totally-unknown-model") is None

    def test_match_case_insensitive(self):
        """Architecture matching should be case-insensitive."""
        match = UltraStreamingEngine._match_known_architecture
        assert match("LLAMA-70B") is not None
        assert match("DeepSeek-V3") is not None


# ---------------------------------------------------------------------------
# UltraStreamingEngine: memory report
# ---------------------------------------------------------------------------


class TestUltraStreamingEngineReport:
    """UltraStreamingEngine.memory_report() returns structured data."""

    def test_report_before_load(self):
        """memory_report() before load() should still work."""
        engine = UltraStreamingEngine(
            "some-model-70b", gpu_budget_gb=24, cpu_budget_gb=64,
        )
        # Initialize metrics to defaults
        report = engine.memory_report()
        required = {
            "peak_vram_mb", "load_time_sec", "tokens_generated",
            "tokens_per_sec", "gpu_budget_gb", "cpu_budget_gb",
            "kv_bits", "kv_compression", "weight_bits",
        }
        assert required.issubset(report.keys())

    def test_report_after_analyze(self):
        """memory_report() after analyze() includes plan data."""
        engine = UltraStreamingEngine(
            "some-model-70b", gpu_budget_gb=24, cpu_budget_gb=64,
        )
        engine.analyze()
        report = engine.memory_report()
        assert "model_size_gb" in report
        assert "is_moe" in report
        assert "estimated_tok_per_sec" in report


# ---------------------------------------------------------------------------
# Integration: full flow without model loading
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end flow: analyze + plan + weight manager + KV manager."""

    def test_full_analysis_flow_dense(self):
        """Full analysis for a dense 70B model."""
        engine = UltraStreamingEngine(
            "some-model-70b", gpu_budget_gb=24, cpu_budget_gb=64,
            kv_bits=3, kv_compression="boundary", weight_bits=4,
        )
        plan = engine.analyze()

        # Set up weight manager
        weight_cache_bytes = int(plan["weight_cache_gb"] * (1024 ** 3))
        weight_mgr = WeightManager(gpu_budget_bytes=weight_cache_bytes)
        assert weight_mgr.gpu_budget_bytes > 0

        # Set up KV manager
        kv_mgr = KVManager(
            kv_budget_gb=plan["kv_budget_gb"],
            num_layers=plan["total_layers"],
            kv_bytes_per_token_fp16=engine.analyzer.kv_bytes_per_token_fp16,
            kv_compression="boundary",
            kv_bits=3,
        )
        cache = kv_mgr.create_cache()
        assert cache is not None

        # Feed a tensor through the cache
        k = torch.randn(1, 8, 1, 128)
        v = torch.randn(1, 8, 1, 128)
        cache.update(k, v, layer_idx=0)
        assert cache.get_seq_length(0) == 1

    def test_full_analysis_flow_moe(self):
        """Full analysis for a MoE model."""
        engine = UltraStreamingEngine(
            "llama4-scout-test", gpu_budget_gb=24, cpu_budget_gb=64,
            kv_bits=3, kv_compression="boundary", weight_bits=4,
        )
        plan = engine.analyze()
        assert plan["is_moe"] is True

        # KV manager
        kv_mgr = KVManager(
            kv_budget_gb=plan["kv_budget_gb"],
            num_layers=plan["total_layers"],
            kv_bytes_per_token_fp16=engine.analyzer.kv_bytes_per_token_fp16,
        )
        cache = kv_mgr.create_cache()
        assert cache is not None

    def test_weight_manager_expert_keys(self):
        """WeightManager supports (layer, expert) tuple keys for MoE."""
        mgr = WeightManager(gpu_budget_bytes=100000, is_moe=True)
        m0 = MagicMock()
        m0.to = MagicMock(return_value=m0)
        m1 = MagicMock()
        m1.to = MagicMock(return_value=m1)

        # Load expert 3 from layer 5
        mgr.load((5, 3), m0, 1000)
        # Load expert 7 from layer 5
        mgr.load((5, 7), m1, 1000)

        assert mgr.get((5, 3)) is m0
        assert mgr.get((5, 7)) is m1
        assert mgr.get((5, 0)) is None  # not loaded

    def test_multi_model_comparison(self):
        """Compare feasibility across multiple model sizes."""
        models = ["llama-7b", "llama-70b", "llama-405b"]
        plans = {}
        for name in models:
            analyzer = ModelAnalyzer(
                arch_dict=KNOWN_ARCHITECTURES[name], weight_bits=4,
            )
            plans[name] = plan_memory(
                analyzer, gpu_budget_gb=24.0, cpu_budget_gb=64.0,
            )

        # 7B should have more context capacity than 70B at same budget
        assert plans["llama-7b"]["context_at_5x"] > plans["llama-70b"]["context_at_5x"]

        # All should have positive estimated throughput
        for name in models:
            assert plans[name]["estimated_tok_per_sec"] > 0

    def test_generate_requires_load(self):
        """generate() before load() should raise RuntimeError."""
        engine = UltraStreamingEngine(
            "some-model-70b", gpu_budget_gb=24, cpu_budget_gb=64,
        )
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.generate("hello")
