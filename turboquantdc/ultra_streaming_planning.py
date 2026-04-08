from __future__ import annotations

import gc
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from .generation_cache import ANCHOR_STRATEGIES, GenerationCache
from .streaming_70b import AsyncPrefetcher
from .token_eviction import EvictionCache

from .ultra_streaming_analyzer import ModelAnalyzer



def plan_memory(
    analyzer: ModelAnalyzer,
    gpu_budget_gb: float = 24.0,
    cpu_budget_gb: float = 64.0,
    target_context: int = 32768,
) -> Dict[str, Any]:
    """Calculate what's achievable for a given model on a given GPU.

    Determines:
    - Whether the model fits in CPU RAM (after quantization)
    - How much VRAM is available for weight cache vs KV cache
    - Achievable context lengths at different compression levels
    - Estimated throughput

    Args:
        analyzer: ModelAnalyzer for the target model.
        gpu_budget_gb: GPU VRAM in GB.
        cpu_budget_gb: CPU RAM available in GB.
        target_context: Desired context length.

    Returns:
        Dict with detailed memory plan and feasibility analysis.
    """
    GB = 1024 ** 3

    model_size_gb = analyzer.total_size_gb
    embed_gb = analyzer.embedding_size_gb

    # Check CPU feasibility
    fits_cpu = model_size_gb <= cpu_budget_gb * 0.9  # 90% threshold

    # GPU allocation:
    # 1. Embeddings + LM head (always resident, FP16)
    # 2. Overhead (CUDA context, activations, fragmentation)
    # 3. Active layer buffer (double-buffered for async prefetch)
    # 4. Remaining: split between weight cache and KV cache

    overhead_gb = 1.5  # CUDA context + activations + fragmentation

    if analyzer.is_moe:
        # For MoE, we cache individual experts, not full layers
        active_buffer_gb = analyzer.active_layer_size_gb * 2  # double buffer
    else:
        active_buffer_gb = analyzer.layer_size_gb * 2  # double buffer

    # What's left after fixed allocations
    fixed_gb = embed_gb + overhead_gb + active_buffer_gb
    remaining_gb = max(0, gpu_budget_gb - fixed_gb)

    # Split remaining between weight cache and KV cache
    # Heuristic: give 40% to weight cache, 60% to KV for very large models
    # For smaller models that fit more layers, give more to weights
    layers_that_fit = 0
    if analyzer.layer_size_gb > 0:
        layers_that_fit = int(remaining_gb / analyzer.layer_size_gb)
    layers_that_fit = min(layers_that_fit, analyzer.num_layers)

    if layers_that_fit >= analyzer.num_layers:
        # All layers fit -- all remaining goes to KV
        weight_cache_gb = layers_that_fit * analyzer.layer_size_gb
        kv_budget_gb = remaining_gb - weight_cache_gb
    else:
        # Not all layers fit -- split intelligently
        # More weight cache = fewer CPU transfers = faster
        # More KV cache = longer context
        weight_cache_gb = min(remaining_gb * 0.4, 10.0)  # Cap at 10GB
        kv_budget_gb = remaining_gb - weight_cache_gb

    kv_budget_gb = max(kv_budget_gb, 2.0)  # Minimum 2GB for KV

    # Context capacity at different compression levels
    kv_bpt = analyzer.kv_bytes_per_token_fp16
    context_at = {}
    for comp in [5.0, 7.5, 10.0]:
        if kv_bpt > 0:
            tokens = int(kv_budget_gb * GB / (kv_bpt / comp))
        else:
            tokens = 0
        context_at[f"{comp}x"] = tokens

    # Throughput estimate
    pcie_bw = ModelAnalyzer.PCIE4_BANDWIDTH

    if analyzer.is_moe:
        # MoE: only transfer active experts per token
        transfer_per_token = analyzer.active_layer_size_bytes * analyzer.num_layers
        # But with caching, we hit ~50-80% of experts from cache
        cache_hit_rate = 0.6  # conservative estimate
        effective_transfer = transfer_per_token * (1 - cache_hit_rate)
    else:
        # Dense: transfer all non-cached layers
        layers_to_stream = max(0, analyzer.num_layers - layers_that_fit)
        effective_transfer = layers_to_stream * analyzer.layer_size_bytes
        # Double-buffering hides ~50% of transfer
        effective_transfer *= 0.5

    if effective_transfer > 0:
        time_per_token = effective_transfer / pcie_bw
        estimated_tok_s = 1.0 / max(time_per_token, 0.001)
    else:
        estimated_tok_s = 50.0  # all layers cached

    return {
        "model_size_gb": round(model_size_gb, 1),
        "fits_cpu": fits_cpu,
        "cpu_needed_gb": round(model_size_gb * 1.1, 1),  # 10% overhead
        "embedding_gb": round(embed_gb, 2),
        "layer_size_gb": round(analyzer.layer_size_gb, 3),
        "active_layer_size_gb": round(analyzer.active_layer_size_gb, 3),
        "overhead_gb": overhead_gb,
        "active_buffer_gb": round(active_buffer_gb, 2),
        "weight_cache_gb": round(weight_cache_gb, 1),
        "kv_budget_gb": round(kv_budget_gb, 1),
        "layers_cached": min(layers_that_fit, analyzer.num_layers),
        "layers_streamed": max(0, analyzer.num_layers - layers_that_fit),
        "total_layers": analyzer.num_layers,
        "context_at_5x": context_at.get("5.0x", 0),
        "context_at_7_5x": context_at.get("7.5x", 0),
        "context_at_10x": context_at.get("10.0x", 0),
        "estimated_tok_per_sec": round(estimated_tok_s, 1),
        "is_moe": analyzer.is_moe,
        "gpu_budget_gb": gpu_budget_gb,
        "cpu_budget_gb": cpu_budget_gb,
        "target_context": target_context,
        "feasible": fits_cpu,
    }


def format_plan_report(
    model_name: str,
    analyzer: ModelAnalyzer,
    plan: Dict[str, Any],
) -> str:
    """Format a memory plan into a human-readable report.

    Args:
        model_name: Name of the model.
        analyzer: ModelAnalyzer instance.
        plan: Output of plan_memory().

    Returns:
        Multi-line formatted string.
    """
    short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    arch_type = "MoE" if plan["is_moe"] else "Dense"

    lines = []
    lines.append(f"Model: {short_name} ({arch_type})")
    lines.append(f"  Total size: {plan['model_size_gb']:.1f} GB at {analyzer.weight_bits}-bit")
    lines.append(f"  Layers: {plan['total_layers']}")
    if plan["is_moe"]:
        lines.append(f"  Experts: {analyzer.num_experts} total, {analyzer.num_active_experts} active")
    lines.append("")

    lines.append(f"GPU Budget: {plan['gpu_budget_gb']:.0f} GB")
    lines.append(f"  Embeddings + LM head: {plan['embedding_gb']:.2f} GB")
    lines.append(f"  Active buffer:        {plan['active_buffer_gb']:.2f} GB")
    lines.append(f"  Overhead:             {plan['overhead_gb']:.1f} GB")
    lines.append(f"  Weight cache:         {plan['weight_cache_gb']:.1f} GB")
    lines.append(f"  KV cache budget:      {plan['kv_budget_gb']:.1f} GB")
    lines.append("")

    lines.append(f"Streaming: {plan['layers_cached']}/{plan['total_layers']} layers cached")
    lines.append(f"  Layers to stream per token: {plan['layers_streamed']}")
    lines.append(f"  Estimated speed: ~{plan['estimated_tok_per_sec']:.1f} tok/s")
    lines.append("")

    lines.append("Context capacity (KV cache):")
    lines.append(f"   5.0x compression: {plan['context_at_5x']:,} tokens")
    lines.append(f"   7.5x compression: {plan['context_at_7_5x']:,} tokens")
    lines.append(f"  10.0x compression: {plan['context_at_10x']:,} tokens")
    lines.append("")

    if not plan["fits_cpu"]:
        lines.append(
            f"WARNING: Model needs {plan['cpu_needed_gb']:.0f} GB CPU RAM "
            f"(only {plan['cpu_budget_gb']:.0f} GB available)"
        )
    else:
        lines.append(
            f"CPU RAM: {plan['cpu_needed_gb']:.0f} GB needed "
            f"({plan['cpu_budget_gb']:.0f} GB available) -- OK"
        )

    if plan["feasible"]:
        lines.append("Status: FEASIBLE")
    else:
        lines.append("Status: NOT FEASIBLE (insufficient CPU RAM)")

    return "\n".join(lines)

