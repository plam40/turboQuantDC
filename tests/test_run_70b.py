"""Tests for the unified 70B launcher (run_70b.py).

Tests cover:
    - GPU VRAM detection returns positive numbers
    - Memory calculation for 70B at various context targets
    - Speed mode selection and validation
    - KV strategy selection per mode
    - KV cache creation per mode
    - Startup report formatting
    - CLI argument parsing
    - Edge cases: tiny VRAM, huge context, invalid modes
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, "/home/dhawal/turboQuantDC")

from run_70b import (
    COMPRESSION_RATIOS,
    DEFAULT_70B_CONFIG,
    DEFAULT_MODEL,
    ESTIMATED_SPEEDS,
    SPEED_MODES,
    calculate_memory_plan,
    create_kv_cache,
    detect_gpu,
    format_startup_report,
    select_kv_strategy,
)


# ---------------------------------------------------------------------------
# GPU Detection
# ---------------------------------------------------------------------------


class TestDetectGPU:
    """detect_gpu() returns valid GPU info."""

    def test_returns_dict_with_required_keys(self):
        """Result always has name, vram_gb, vram_bytes, compute_capability, available."""
        info = detect_gpu()
        assert "name" in info
        assert "vram_gb" in info
        assert "vram_bytes" in info
        assert "compute_capability" in info
        assert "available" in info

    def test_vram_positive_when_gpu_available(self):
        """If CUDA is available, VRAM should be a positive number."""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU available")
        info = detect_gpu()
        assert info["vram_gb"] > 0
        assert info["vram_bytes"] > 0
        assert info["available"] is True

    def test_vram_zero_when_no_gpu(self):
        """When CUDA is not available, VRAM should be 0."""
        with patch("run_70b.torch.cuda.is_available", return_value=False):
            info = detect_gpu()
            assert info["vram_gb"] == 0.0
            assert info["vram_bytes"] == 0
            assert info["available"] is False
            assert info["name"] == "none"

    def test_compute_capability_is_tuple(self):
        """compute_capability should be a (major, minor) tuple."""
        info = detect_gpu()
        cc = info["compute_capability"]
        assert isinstance(cc, tuple)
        assert len(cc) == 2


# ---------------------------------------------------------------------------
# Memory Calculation
# ---------------------------------------------------------------------------


class TestCalculateMemoryPlan:
    """calculate_memory_plan() produces valid allocation plans."""

    def test_basic_70b_plan(self):
        """Default 70B config at 24GB returns a valid plan."""
        plan = calculate_memory_plan(
            context_target=144_000,
            gpu_vram_gb=24.0,
            speed_mode="balanced",
        )
        assert plan["total_layers"] == 80
        assert plan["num_gpu_layers"] > 0
        assert plan["num_gpu_layers"] <= 80
        assert plan["weight_budget_gb"] > 0
        assert plan["kv_budget_gb"] > 0
        assert plan["compression_ratio"] == 5.0
        assert plan["context_target"] == 144_000
        assert plan["context_achievable"] > 0

    def test_plan_with_48gb_gpu(self):
        """More VRAM should allow more GPU layers."""
        plan_24 = calculate_memory_plan(gpu_vram_gb=24.0, speed_mode="safe")
        plan_48 = calculate_memory_plan(gpu_vram_gb=48.0, speed_mode="safe")
        assert plan_48["num_gpu_layers"] >= plan_24["num_gpu_layers"]

    def test_fast_mode_has_higher_compression(self):
        """Fast mode should have higher compression ratio than safe."""
        plan_safe = calculate_memory_plan(speed_mode="safe")
        plan_fast = calculate_memory_plan(speed_mode="fast")
        assert plan_fast["compression_ratio"] > plan_safe["compression_ratio"]

    def test_context_achievable_scales_with_vram(self):
        """More VRAM should enable more context."""
        plan_16 = calculate_memory_plan(gpu_vram_gb=16.0, speed_mode="balanced")
        plan_48 = calculate_memory_plan(gpu_vram_gb=48.0, speed_mode="balanced")
        assert plan_48["context_achievable"] >= plan_16["context_achievable"]

    def test_custom_model_config(self):
        """Custom config with fewer layers should be reflected."""
        small_config = {
            "num_layers": 32,
            "hidden_size": 4096,
            "num_kv_heads": 8,
            "head_dim": 128,
            "vocab_size": 32000,
        }
        plan = calculate_memory_plan(
            model_config=small_config,
            speed_mode="balanced",
        )
        assert plan["total_layers"] == 32

    def test_weight_bits_8_vs_4(self):
        """8-bit weights should use more VRAM per layer than 4-bit."""
        plan_4 = calculate_memory_plan(weight_quant_bits=4, speed_mode="safe")
        plan_8 = calculate_memory_plan(weight_quant_bits=8, speed_mode="safe")
        # 8-bit layers are larger, so fewer fit on GPU
        assert plan_8["num_gpu_layers"] <= plan_4["num_gpu_layers"]

    def test_invalid_speed_mode_raises(self):
        """Invalid speed mode should raise ValueError."""
        with pytest.raises(ValueError, match="speed_mode"):
            calculate_memory_plan(speed_mode="turbo")

    def test_tiny_vram(self):
        """Even with tiny VRAM, plan should not crash and should have >= 1 layer."""
        plan = calculate_memory_plan(gpu_vram_gb=6.0, speed_mode="safe")
        assert plan["num_gpu_layers"] >= 1
        assert plan["weight_budget_gb"] >= 1.0

    def test_all_speed_modes(self):
        """Every defined speed mode should produce a valid plan."""
        for mode in SPEED_MODES:
            plan = calculate_memory_plan(speed_mode=mode)
            assert plan["speed_mode"] == mode
            assert plan["estimated_tok_per_sec"] == ESTIMATED_SPEEDS[mode]
            assert plan["compression_ratio"] == COMPRESSION_RATIOS[mode]


# ---------------------------------------------------------------------------
# KV Strategy Selection
# ---------------------------------------------------------------------------


class TestSelectKVStrategy:
    """select_kv_strategy() picks the right cache type per mode."""

    def test_safe_uses_generation_cache(self):
        """Safe mode should use GenerationCache with hybrid_max_quality preset."""
        strategy = select_kv_strategy("safe")
        assert strategy["cache_type"] == "GenerationCache"
        assert strategy["preset"] == "hybrid_max_quality"
        assert strategy["quality_pct"] == 100

    def test_balanced_uses_generation_cache(self):
        """Balanced mode should use GenerationCache with boundary anchoring."""
        strategy = select_kv_strategy("balanced")
        assert strategy["cache_type"] == "GenerationCache"
        assert strategy["kwargs"]["anchor_strategy"] == "boundary"
        assert strategy["kwargs"]["key_bits"] == 3
        assert strategy["kwargs"]["val_bits"] == 3

    def test_fast_uses_eviction_cache(self):
        """Fast mode should use EvictionCache with K3/V2."""
        strategy = select_kv_strategy("fast")
        assert strategy["cache_type"] == "EvictionCache"
        assert strategy["kwargs"]["key_bits"] == 3
        assert strategy["kwargs"]["val_bits"] == 2
        assert strategy["kwargs"]["max_warm_tokens"] == 1024
        assert strategy["quality_pct"] == 96

    def test_all_modes_have_description(self):
        """Every mode should return a human-readable description."""
        for mode in SPEED_MODES:
            strategy = select_kv_strategy(mode)
            assert len(strategy["description"]) > 0
            assert strategy["quality_pct"] > 0

    def test_invalid_mode_raises(self):
        """Invalid speed mode should raise ValueError."""
        with pytest.raises(ValueError, match="speed_mode"):
            select_kv_strategy("insane")


# ---------------------------------------------------------------------------
# KV Cache Creation
# ---------------------------------------------------------------------------


class TestCreateKVCache:
    """create_kv_cache() instantiates the correct cache type."""

    def test_safe_creates_generation_cache(self):
        """Safe mode should create a GenerationCache from preset."""
        from turboquantdc.generation_cache import GenerationCache

        cache = create_kv_cache("safe", num_layers=80)
        assert isinstance(cache, GenerationCache)

    def test_balanced_creates_generation_cache(self):
        """Balanced mode should create a GenerationCache with boundary anchoring."""
        from turboquantdc.generation_cache import GenerationCache

        cache = create_kv_cache("balanced", num_layers=80)
        assert isinstance(cache, GenerationCache)
        assert cache.anchor_strategy == "boundary"
        assert cache.key_bits == 3
        assert cache.val_bits == 3

    def test_fast_creates_eviction_cache(self):
        """Fast mode should create an EvictionCache."""
        from turboquantdc.token_eviction import EvictionCache

        cache = create_kv_cache("fast", num_layers=80)
        assert isinstance(cache, EvictionCache)
        assert cache.key_bits == 3
        assert cache.val_bits == 2

    def test_safe_cache_has_residual_quant(self):
        """Safe mode preset should have residual quant enabled."""
        cache = create_kv_cache("safe", num_layers=80)
        assert cache.use_residual_quant is True

    def test_all_caches_have_update_method(self):
        """All cache types must duck-type the HF Cache update() protocol."""
        for mode in SPEED_MODES:
            cache = create_kv_cache(mode, num_layers=80)
            assert hasattr(cache, "update")
            assert hasattr(cache, "get_seq_length")

    def test_all_caches_have_reset_method(self):
        """All cache types should support reset() for /clear command."""
        for mode in SPEED_MODES:
            cache = create_kv_cache(mode, num_layers=80)
            assert hasattr(cache, "reset")


# ---------------------------------------------------------------------------
# Startup Report Formatting
# ---------------------------------------------------------------------------


class TestFormatStartupReport:
    """format_startup_report() produces a well-formatted box."""

    def _make_inputs(self, speed_mode="balanced"):
        gpu_info = {
            "name": "NVIDIA RTX 4090",
            "vram_gb": 24.0,
            "vram_bytes": 24 * 1024**3,
            "compute_capability": (8, 9),
            "available": True,
        }
        memory_plan = calculate_memory_plan(
            context_target=144_000,
            gpu_vram_gb=24.0,
            speed_mode=speed_mode,
        )
        kv_strategy = select_kv_strategy(speed_mode)
        return gpu_info, memory_plan, kv_strategy

    def test_report_contains_model_name(self):
        """Report should mention the model name."""
        gpu_info, plan, kv = self._make_inputs()
        report = format_startup_report(DEFAULT_MODEL, gpu_info, plan, kv)
        assert "Llama-3.3-70B-Instruct" in report

    def test_report_contains_gpu_name(self):
        """Report should mention the GPU."""
        gpu_info, plan, kv = self._make_inputs()
        report = format_startup_report(DEFAULT_MODEL, gpu_info, plan, kv)
        assert "RTX 4090" in report

    def test_report_contains_kv_description(self):
        """Report should describe the KV cache strategy."""
        gpu_info, plan, kv = self._make_inputs()
        report = format_startup_report(DEFAULT_MODEL, gpu_info, plan, kv)
        assert "K3/V3" in report or "boundary" in report

    def test_report_contains_context_target(self):
        """Report should show the context target."""
        gpu_info, plan, kv = self._make_inputs()
        report = format_startup_report(DEFAULT_MODEL, gpu_info, plan, kv)
        assert "144K" in report or "144000" in report

    def test_report_has_box_structure(self):
        """Report should have box delimiters."""
        gpu_info, plan, kv = self._make_inputs()
        report = format_startup_report(DEFAULT_MODEL, gpu_info, plan, kv)
        lines = report.strip().split("\n")
        # First and last lines should be box borders
        assert lines[0].startswith("+")
        assert lines[-1].startswith("+")

    def test_report_mentions_speed_mode(self):
        """Report should mention the speed mode."""
        gpu_info, plan, kv = self._make_inputs("fast")
        report = format_startup_report(DEFAULT_MODEL, gpu_info, plan, kv)
        assert "fast" in report

    def test_report_all_speed_modes(self):
        """Every speed mode should produce a valid report (no crashes)."""
        for mode in SPEED_MODES:
            gpu_info, plan, kv = self._make_inputs(mode)
            report = format_startup_report(DEFAULT_MODEL, gpu_info, plan, kv)
            assert len(report) > 100

    def test_report_handles_short_model_name(self):
        """Model name without org prefix should work."""
        gpu_info, plan, kv = self._make_inputs()
        report = format_startup_report("my-local-model", gpu_info, plan, kv)
        assert "my-local-model" in report

    def test_report_lines_are_consistent_width(self):
        """All | lines should have the same width (proper box alignment)."""
        gpu_info, plan, kv = self._make_inputs()
        report = format_startup_report(DEFAULT_MODEL, gpu_info, plan, kv)
        pipe_lines = [l for l in report.split("\n") if l.startswith("|")]
        if pipe_lines:
            widths = [len(l) for l in pipe_lines]
            assert len(set(widths)) == 1, f"Inconsistent widths: {widths}"


# ---------------------------------------------------------------------------
# Constants / Validation
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants are consistent."""

    def test_speed_modes_tuple(self):
        """SPEED_MODES should be a tuple of strings."""
        assert isinstance(SPEED_MODES, tuple)
        assert all(isinstance(m, str) for m in SPEED_MODES)
        assert "safe" in SPEED_MODES
        assert "balanced" in SPEED_MODES
        assert "fast" in SPEED_MODES

    def test_compression_ratios_cover_all_modes(self):
        """COMPRESSION_RATIOS should have an entry for every speed mode."""
        for mode in SPEED_MODES:
            assert mode in COMPRESSION_RATIOS
            assert COMPRESSION_RATIOS[mode] > 1.0

    def test_estimated_speeds_cover_all_modes(self):
        """ESTIMATED_SPEEDS should have an entry for every speed mode."""
        for mode in SPEED_MODES:
            assert mode in ESTIMATED_SPEEDS
            assert ESTIMATED_SPEEDS[mode] > 0

    def test_default_model_is_string(self):
        """DEFAULT_MODEL should be a non-empty string."""
        assert isinstance(DEFAULT_MODEL, str)
        assert len(DEFAULT_MODEL) > 0

    def test_default_config_has_required_keys(self):
        """DEFAULT_70B_CONFIG should have the keys MemoryPlanner expects."""
        required = {"num_layers", "hidden_size", "num_kv_heads", "head_dim", "vocab_size"}
        assert required.issubset(set(DEFAULT_70B_CONFIG.keys()))


# ---------------------------------------------------------------------------
# Integration: end-to-end plan + cache flow
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end: plan -> strategy -> cache -> report."""

    def test_full_flow_safe(self):
        """Safe mode: plan + strategy + cache + report all work together."""
        plan = calculate_memory_plan(speed_mode="safe", gpu_vram_gb=24.0)
        strategy = select_kv_strategy("safe")
        cache = create_kv_cache("safe", num_layers=plan["total_layers"])
        gpu_info = {"name": "RTX 4090", "vram_gb": 24.0, "vram_bytes": 24 * 1024**3,
                     "compute_capability": (8, 9), "available": True}
        report = format_startup_report(DEFAULT_MODEL, gpu_info, plan, strategy)
        assert len(report) > 0
        assert cache.get_seq_length(0) == 0

    def test_full_flow_fast(self):
        """Fast mode: plan + strategy + cache + report all work together."""
        plan = calculate_memory_plan(speed_mode="fast", gpu_vram_gb=24.0)
        strategy = select_kv_strategy("fast")
        cache = create_kv_cache("fast", num_layers=plan["total_layers"])
        gpu_info = {"name": "RTX 4090", "vram_gb": 24.0, "vram_bytes": 24 * 1024**3,
                     "compute_capability": (8, 9), "available": True}
        report = format_startup_report(DEFAULT_MODEL, gpu_info, plan, strategy)
        assert len(report) > 0
        assert hasattr(cache, "eviction_stats")

    def test_cache_accepts_tensor_update(self):
        """Created cache should accept a tensor update call (HF protocol)."""
        cache = create_kv_cache("balanced", num_layers=32)
        # Simulate a single token update: [batch=1, heads=8, seq=1, dim=128]
        k = torch.randn(1, 8, 1, 128)
        v = torch.randn(1, 8, 1, 128)
        result = cache.update(k, v, layer_idx=0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert cache.get_seq_length(0) == 1
