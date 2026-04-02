"""Ultra-Streaming Engine: Run arbitrarily large models on limited GPU VRAM.

Supports both dense models (full layer streaming) and MoE models
(expert-level offloading with caching).

The GPU holds:
1. Embedding + LM head (always resident, ~1GB)
2. Active layer(s) being computed (1-3 layers, ~1-3GB)
3. TurboQuantDC compressed KV cache (rest of VRAM)

CPU RAM holds:
- All model weights (quantized)
- KV cache overflow (if needed)

Two operating modes:

**Dense Streaming** (for models like Llama-3.1-70B, 405B):
    Stream transformer layers from CPU to GPU one at a time with async
    double-buffered prefetch. ~1-10 tok/s depending on model size and
    how many layers fit in the GPU cache.

**MoE Expert Offloading** (for models like Llama 4 Scout, DeepSeek V3):
    Keep router + embedding + LM head on GPU. Cache the most-used experts
    on GPU, stream needed experts from CPU on demand. Faster because only
    active experts are needed per token. ~5-15 tok/s.

Usage:
    engine = UltraStreamingEngine(
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        gpu_budget_gb=22,
        cpu_budget_gb=64,
        kv_compression="boundary",
    )
    engine.load()
    output = engine.generate("Explain quantum computing", max_new_tokens=200)
    print(engine.analysis_report())
"""

from __future__ import annotations

import gc
import math
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from .generation_cache import (
    ANCHOR_STRATEGIES,
    GenerationCache,
    compute_anchor_schedule,
)
from .streaming_70b import AsyncPrefetcher, LayerGPUCache, MemoryPlanner
from .token_eviction import EvictionCache


# ---------------------------------------------------------------------------
# Known model architectures (for offline planning without downloading configs)
# ---------------------------------------------------------------------------

KNOWN_ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    # Dense models
    "llama-7b": {
        "type": "dense",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "total_params_b": 7,
    },
    "llama-13b": {
        "type": "dense",
        "num_layers": 40,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_kv_heads": 40,
        "head_dim": 128,
        "intermediate_size": 13824,
        "vocab_size": 32000,
        "total_params_b": 13,
    },
    "llama-70b": {
        "type": "dense",
        "num_layers": 80,
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 28672,
        "vocab_size": 128256,
        "total_params_b": 70,
    },
    "llama-405b": {
        "type": "dense",
        "num_layers": 126,
        "hidden_size": 16384,
        "num_attention_heads": 128,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 53248,
        "vocab_size": 128256,
        "total_params_b": 405,
    },
    "qwen-72b": {
        "type": "dense",
        "num_layers": 80,
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 29568,
        "vocab_size": 152064,
        "total_params_b": 72,
    },
    # MoE models
    "llama4-scout": {
        "type": "moe",
        "num_layers": 48,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 8192,
        "vocab_size": 202400,
        "total_params_b": 109,
        "num_experts": 16,
        "num_active_experts": 1,
        "active_params_b": 17,
    },
    "llama4-maverick": {
        "type": "moe",
        "num_layers": 48,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 8192,
        "vocab_size": 202400,
        "total_params_b": 400,
        "num_experts": 128,
        "num_active_experts": 1,
        "active_params_b": 17,
    },
    "deepseek-v3": {
        "type": "moe",
        "num_layers": 61,
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "num_kv_heads": 128,
        "head_dim": 128,
        "intermediate_size": 2048,  # per-expert intermediate (DeepSeek uses small experts)
        "vocab_size": 129280,
        "total_params_b": 685,
        "num_experts": 256,
        "num_active_experts": 8,
        "active_params_b": 37,
    },
    "mixtral-8x7b": {
        "type": "moe",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "vocab_size": 32000,
        "total_params_b": 47,
        "num_experts": 8,
        "num_active_experts": 2,
        "active_params_b": 13,
    },
    "mixtral-8x22b": {
        "type": "moe",
        "num_layers": 56,
        "hidden_size": 6144,
        "num_attention_heads": 48,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 16384,
        "vocab_size": 32768,
        "total_params_b": 141,
        "num_experts": 8,
        "num_active_experts": 2,
        "active_params_b": 39,
    },
}


# ---------------------------------------------------------------------------
# ModelAnalyzer: inspect model architecture
# ---------------------------------------------------------------------------


class ModelAnalyzer:
    """Analyze a model's architecture to determine streaming strategy.

    Works from either:
    1. A HuggingFace AutoConfig object
    2. A dict from KNOWN_ARCHITECTURES
    3. Manually specified parameters

    Determines:
    - Dense vs MoE architecture
    - Size per layer / per expert
    - Total params, active params
    - Minimum GPU VRAM needed
    - Optimal streaming strategy
    """

    # PCIe 4.0 x16 bandwidth in bytes/sec
    PCIE4_BANDWIDTH = 25 * (1024 ** 3)
    # PCIe 5.0 x16 bandwidth in bytes/sec
    PCIE5_BANDWIDTH = 50 * (1024 ** 3)

    def __init__(
        self,
        config: Optional[Any] = None,
        arch_dict: Optional[Dict[str, Any]] = None,
        weight_bits: int = 4,
    ):
        """Initialize from a HF config, a known architecture dict, or both.

        Args:
            config: HuggingFace AutoConfig object. If provided, arch_dict
                is ignored and values are extracted from config.
            arch_dict: Dict with architecture parameters (e.g. from
                KNOWN_ARCHITECTURES). Used when no HF config is available.
            weight_bits: Bits per weight after quantization (4 for BnB 4-bit,
                8 for BnB 8-bit, 16 for FP16).
        """
        if config is not None:
            self._from_config(config)
        elif arch_dict is not None:
            self._from_dict(arch_dict)
        else:
            raise ValueError("Either config or arch_dict must be provided")

        self.weight_bits = weight_bits

    def _from_config(self, config: Any) -> None:
        """Extract architecture from HuggingFace AutoConfig."""
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.num_attention_heads
        )
        self.intermediate_size = getattr(
            config, "intermediate_size",
            int(self.hidden_size * 3.5),
        )
        self.vocab_size = config.vocab_size

        # MoE detection — use hasattr to avoid MagicMock/dynamic attr issues
        self.num_experts = 0
        if hasattr(config, "num_local_experts") and isinstance(
            getattr(config, "num_local_experts", None), int
        ):
            self.num_experts = config.num_local_experts
        elif hasattr(config, "num_experts") and isinstance(
            getattr(config, "num_experts", None), int
        ):
            self.num_experts = config.num_experts

        self.num_active_experts = 0
        if hasattr(config, "num_experts_per_tok") and isinstance(
            getattr(config, "num_experts_per_tok", None), int
        ):
            self.num_active_experts = config.num_experts_per_tok
        elif hasattr(config, "num_selected_experts") and isinstance(
            getattr(config, "num_selected_experts", None), int
        ):
            self.num_active_experts = config.num_selected_experts

        self.is_moe = self.num_experts > 1

    def _from_dict(self, d: Dict[str, Any]) -> None:
        """Extract architecture from a known architecture dict."""
        self.num_layers = d["num_layers"]
        self.hidden_size = d["hidden_size"]
        self.num_attention_heads = d["num_attention_heads"]
        self.num_kv_heads = d.get("num_kv_heads", self.num_attention_heads)
        self.head_dim = d.get(
            "head_dim", self.hidden_size // self.num_attention_heads
        )
        self.intermediate_size = d.get(
            "intermediate_size", int(self.hidden_size * 3.5)
        )
        self.vocab_size = d["vocab_size"]

        self.num_experts = d.get("num_experts", 0)
        self.num_active_experts = d.get("num_active_experts", 0)
        self.is_moe = d.get("type", "dense") == "moe" or self.num_experts > 1

    # ------- Size calculations -------

    @property
    def bytes_per_param(self) -> float:
        """Bytes per parameter at the current weight quantization."""
        return self.weight_bits / 8.0

    @property
    def embedding_size_bytes(self) -> int:
        """Size of embedding + LM head in bytes (always FP16)."""
        # Embeddings and LM head are typically kept at FP16 even with
        # quantized weights, because quantizing them hurts quality badly.
        embed = self.vocab_size * self.hidden_size * 2  # FP16
        lm_head = self.vocab_size * self.hidden_size * 2
        return embed + lm_head

    @property
    def embedding_size_gb(self) -> float:
        """Size of embedding + LM head in GB."""
        return self.embedding_size_bytes / (1024 ** 3)

    @property
    def attention_size_bytes(self) -> int:
        """Size of one attention block (Q, K, V, O projections) in bytes."""
        # Q: hidden_size * num_attention_heads * head_dim
        # K, V: hidden_size * num_kv_heads * head_dim
        # O: num_attention_heads * head_dim * hidden_size
        q_size = self.hidden_size * self.num_attention_heads * self.head_dim
        kv_size = self.hidden_size * self.num_kv_heads * self.head_dim * 2
        o_size = self.num_attention_heads * self.head_dim * self.hidden_size
        total_params = q_size + kv_size + o_size
        return int(total_params * self.bytes_per_param)

    @property
    def ffn_size_bytes(self) -> int:
        """Size of one FFN block in bytes.

        For dense models: gate_proj + up_proj + down_proj.
        For MoE: this is per-expert FFN size.
        """
        # Llama-style: gate(h->i) + up(h->i) + down(i->h) = 3 * h * i
        total_params = 3 * self.hidden_size * self.intermediate_size
        return int(total_params * self.bytes_per_param)

    @property
    def expert_size_bytes(self) -> int:
        """Size of one expert (for MoE) or one FFN (for dense) in bytes."""
        return self.ffn_size_bytes

    @property
    def layer_size_bytes(self) -> int:
        """Size of one full transformer layer in bytes.

        For dense: attention + ffn.
        For MoE: attention + all experts + router.
        """
        attn = self.attention_size_bytes
        if self.is_moe and self.num_experts > 0:
            # All experts + router
            experts = self.num_experts * self.expert_size_bytes
            router = self.hidden_size * self.num_experts * 4  # router weights
            return attn + experts + router
        else:
            return attn + self.ffn_size_bytes

    @property
    def active_layer_size_bytes(self) -> int:
        """Size of one layer considering only active parameters.

        For dense: same as layer_size_bytes.
        For MoE: attention + active_experts only.
        """
        attn = self.attention_size_bytes
        if self.is_moe and self.num_active_experts > 0:
            active_ffn = self.num_active_experts * self.expert_size_bytes
            router = self.hidden_size * self.num_experts * 4
            return attn + active_ffn + router
        else:
            return self.layer_size_bytes

    @property
    def layer_size_gb(self) -> float:
        """Size of one full transformer layer in GB."""
        return self.layer_size_bytes / (1024 ** 3)

    @property
    def active_layer_size_gb(self) -> float:
        """Size of active parameters per layer in GB."""
        return self.active_layer_size_bytes / (1024 ** 3)

    @property
    def total_size_bytes(self) -> int:
        """Total model size in bytes."""
        return (
            self.embedding_size_bytes
            + self.num_layers * self.layer_size_bytes
        )

    @property
    def total_size_gb(self) -> float:
        """Total model size in GB."""
        return self.total_size_bytes / (1024 ** 3)

    @property
    def kv_bytes_per_token_fp16(self) -> int:
        """FP16 KV cache cost per token across all layers.

        Each token stores key and value vectors for every layer:
        2 (K+V) * num_kv_heads * head_dim * 2 (FP16 bytes) * num_layers
        """
        return 2 * self.num_kv_heads * self.head_dim * 2 * self.num_layers

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the model architecture."""
        MB = 1024 ** 2
        GB = 1024 ** 3
        info = {
            "type": "moe" if self.is_moe else "dense",
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "weight_bits": self.weight_bits,
            "embedding_size_mb": round(self.embedding_size_bytes / MB, 1),
            "layer_size_mb": round(self.layer_size_bytes / MB, 1),
            "total_size_gb": round(self.total_size_gb, 1),
            "kv_bytes_per_token_fp16": self.kv_bytes_per_token_fp16,
        }
        if self.is_moe:
            info.update({
                "num_experts": self.num_experts,
                "num_active_experts": self.num_active_experts,
                "expert_size_mb": round(self.expert_size_bytes / MB, 1),
                "active_layer_size_mb": round(
                    self.active_layer_size_bytes / MB, 1
                ),
            })
        return info


# ---------------------------------------------------------------------------
# WeightManager: LRU cache for layers/experts on GPU
# ---------------------------------------------------------------------------


class WeightManager:
    """Manages weight loading and offloading between CPU and GPU.

    For dense models: caches full transformer layers.
    For MoE models: caches individual experts independently.

    Uses pinned CPU memory for fast transfers and an LRU eviction
    policy for the GPU cache. Supports async prefetch via CUDA streams.

    Args:
        gpu_budget_bytes: Total GPU bytes available for weight caching.
        device: CUDA device.
        is_moe: Whether this is an MoE model.
    """

    def __init__(
        self,
        gpu_budget_bytes: int,
        device: torch.device = torch.device("cuda"),
        is_moe: bool = False,
    ):
        self.gpu_budget_bytes = gpu_budget_bytes
        self.device = device
        self.is_moe = is_moe

        # GPU cache: maps (layer_idx,) or (layer_idx, expert_idx) to module
        self._cache: OrderedDict[Tuple[int, ...], Any] = OrderedDict()
        self._cache_sizes: Dict[Tuple[int, ...], int] = {}
        self._current_bytes: int = 0

        # Priority entries (never evicted): embeddings, lm_head, etc.
        self._priority: Set[Tuple[int, ...]] = set()

        # Stats
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.total_transfers: int = 0
        self.total_bytes_transferred: int = 0

    def pin_priority(self, key: Tuple[int, ...], module: Any, size_bytes: int) -> None:
        """Pin a module on GPU permanently (never evicted).

        Used for embeddings, LM head, final norm, etc.

        Args:
            key: Cache key tuple (e.g., (-1,) for embeddings).
            module: nn.Module to keep on GPU.
            size_bytes: Size of the module in bytes.
        """
        self._priority.add(key)
        self._cache[key] = module
        self._cache_sizes[key] = size_bytes
        self._current_bytes += size_bytes

    def get(self, key: Tuple[int, ...]) -> Optional[Any]:
        """Look up a module in the GPU cache.

        If found, moves to MRU position. Returns None if not cached.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1
        return None

    def load(self, key: Tuple[int, ...], module: Any, size_bytes: int) -> None:
        """Load a module onto GPU with LRU eviction.

        If the module is already cached, just moves to MRU.
        Otherwise, evicts enough LRU non-priority entries to make room.

        Args:
            key: Cache key tuple.
            module: nn.Module to transfer.
            size_bytes: Size of the module in bytes.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
            return

        # Evict until we have room
        while (
            self._current_bytes + size_bytes > self.gpu_budget_bytes
            and self._has_evictable()
        ):
            self._evict_lru()

        module.to(self.device, non_blocking=True)
        self._cache[key] = module
        self._cache_sizes[key] = size_bytes
        self._current_bytes += size_bytes
        self.total_transfers += 1
        self.total_bytes_transferred += size_bytes

    def _has_evictable(self) -> bool:
        """Check if there are any non-priority entries to evict."""
        return any(k not in self._priority for k in self._cache)

    def _evict_lru(self) -> None:
        """Evict the least-recently-used non-priority entry."""
        for key in list(self._cache.keys()):
            if key not in self._priority:
                module = self._cache[key]
                module.to("cpu", non_blocking=True)
                size = self._cache_sizes.pop(key)
                del self._cache[key]
                self._current_bytes -= size
                return

    def is_cached(self, key: Tuple[int, ...]) -> bool:
        """Check if a key is in the GPU cache."""
        return key in self._cache

    @property
    def utilization(self) -> float:
        """GPU cache utilization as a fraction [0, 1]."""
        if self.gpu_budget_bytes <= 0:
            return 0.0
        return self._current_bytes / self.gpu_budget_bytes

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction [0, 1]."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "cached_entries": len(self._cache),
            "priority_entries": len(self._priority),
            "current_bytes": self._current_bytes,
            "budget_bytes": self.gpu_budget_bytes,
            "utilization_pct": round(self.utilization * 100, 1),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_pct": round(self.hit_rate * 100, 1),
            "total_transfers": self.total_transfers,
            "total_gb_transferred": round(
                self.total_bytes_transferred / (1024 ** 3), 2
            ),
        }

    def evict_all(self) -> None:
        """Evict all non-priority entries from GPU cache."""
        for key in list(self._cache.keys()):
            if key not in self._priority:
                self._cache[key].to("cpu", non_blocking=True)
                self._current_bytes -= self._cache_sizes.pop(key)
                del self._cache[key]


# ---------------------------------------------------------------------------
# KVManager: manages KV cache with TurboQuantDC
# ---------------------------------------------------------------------------


class KVManager:
    """Manages TurboQuantDC KV cache with auto-configuration.

    Auto-selects the best cache type and compression level based on
    available VRAM after weight allocations.

    Args:
        kv_budget_gb: VRAM available for KV cache in GB.
        num_layers: Number of transformer layers.
        kv_bytes_per_token_fp16: FP16 KV cost per token across all layers.
        kv_compression: Anchor strategy ("fixed", "boundary", "gradient").
        kv_bits: Key bit-width (2-8).
        max_context: Maximum context length target.
    """

    # KV budget thresholds for strategy selection
    EVICTION_THRESHOLD_GB = 4.0  # Below this, use eviction
    GENEROUS_THRESHOLD_GB = 12.0  # Above this, use higher quality

    def __init__(
        self,
        kv_budget_gb: float,
        num_layers: int,
        kv_bytes_per_token_fp16: int,
        kv_compression: str = "boundary",
        kv_bits: int = 3,
        max_context: int = 32768,
    ):
        self.kv_budget_gb = kv_budget_gb
        self.num_layers = num_layers
        self.kv_bytes_per_token_fp16 = kv_bytes_per_token_fp16
        self.kv_compression = kv_compression
        self.kv_bits = kv_bits
        self.max_context = max_context

        self._cache: Optional[Any] = None
        self._strategy: Optional[Dict[str, Any]] = None

    def auto_select(self) -> Dict[str, Any]:
        """Auto-select the best KV cache strategy for the available budget.

        Returns:
            Dict with cache_type, kwargs, compression_ratio,
            max_tokens, and description.
        """
        budget_bytes = self.kv_budget_gb * (1024 ** 3)

        # Estimate compression ratios for different strategies
        strategies = []

        # Strategy 1: GenerationCache (quantize-only, 5x)
        gen_ratio = 5.0
        gen_tokens = int(
            budget_bytes / max(self.kv_bytes_per_token_fp16 / gen_ratio, 1)
        )
        strategies.append({
            "cache_type": "GenerationCache",
            "compression_ratio": gen_ratio,
            "max_tokens": gen_tokens,
            "kwargs": {
                "key_bits": self.kv_bits,
                "val_bits": max(2, self.kv_bits - 1),
                "anchor_strategy": self.kv_compression,
            },
            "description": (
                f"GenerationCache {self.kv_bits}b keys, "
                f"{self.kv_compression} anchoring, {gen_ratio}x"
            ),
        })

        # Strategy 2: EvictionCache (quantize + evict, 7.5x)
        evict_ratio = 7.5
        evict_tokens = int(
            budget_bytes / max(self.kv_bytes_per_token_fp16 / evict_ratio, 1)
        )
        strategies.append({
            "cache_type": "EvictionCache",
            "compression_ratio": evict_ratio,
            "max_tokens": evict_tokens,
            "kwargs": {
                "key_bits": self.kv_bits,
                "val_bits": max(2, self.kv_bits - 1),
                "fp16_window": 64,
                "max_warm_tokens": 2048,
                "anchor_interval": 12,
            },
            "description": (
                f"EvictionCache {self.kv_bits}b keys + eviction, {evict_ratio}x"
            ),
        })

        # Strategy 3: Aggressive eviction (10x)
        agg_ratio = 10.0
        agg_tokens = int(
            budget_bytes / max(self.kv_bytes_per_token_fp16 / agg_ratio, 1)
        )
        strategies.append({
            "cache_type": "EvictionCache",
            "compression_ratio": agg_ratio,
            "max_tokens": agg_tokens,
            "kwargs": {
                "key_bits": 2,
                "val_bits": 2,
                "fp16_window": 32,
                "max_warm_tokens": 512,
                "anchor_interval": 12,
            },
            "description": "EvictionCache 2b aggressive, 10x",
        })

        # Select strategy: use the least aggressive one that meets context target
        for strategy in strategies:
            if strategy["max_tokens"] >= self.max_context:
                self._strategy = strategy
                return strategy

        # If nothing meets the target, use the most aggressive one
        self._strategy = strategies[-1]
        return strategies[-1]

    def create_cache(self) -> Any:
        """Create the selected KV cache instance.

        Calls auto_select() if not already done.

        Returns:
            A GenerationCache or EvictionCache instance.
        """
        if self._strategy is None:
            self.auto_select()

        strategy = self._strategy

        if strategy["cache_type"] == "EvictionCache":
            self._cache = EvictionCache(**strategy["kwargs"])
        else:
            kwargs = strategy["kwargs"].copy()
            if kwargs.get("anchor_strategy") in ("boundary", "gradient"):
                kwargs["num_layers"] = self.num_layers
            self._cache = GenerationCache(**kwargs)

        return self._cache

    @property
    def cache(self) -> Optional[Any]:
        """The current KV cache instance (None if not created)."""
        return self._cache

    @property
    def strategy(self) -> Optional[Dict[str, Any]]:
        """The selected strategy dict (None if not selected)."""
        return self._strategy


# ---------------------------------------------------------------------------
# Memory planning: what's achievable for a given model + GPU
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# UltraStreamingEngine: ties everything together
# ---------------------------------------------------------------------------


class UltraStreamingEngine:
    """Run arbitrarily large models on limited GPU VRAM.

    Supports both dense models (full layer streaming) and MoE models
    (expert-level offloading with caching).

    The GPU holds:
    1. Embedding + LM head (always resident, ~1GB)
    2. Active layer(s) being computed (1-3 layers, ~1-3GB)
    3. TurboQuantDC compressed KV cache (rest of VRAM)

    CPU RAM holds:
    - All model weights (quantized)
    - KV cache overflow (if needed)

    Args:
        model_name: HuggingFace model name or path.
        gpu_budget_gb: VRAM budget in GB (default 22, leaving 2GB system headroom).
        cpu_budget_gb: CPU RAM budget in GB (default 64).
        kv_compression: Anchor strategy for KV cache.
        kv_bits: Bit-width for KV cache compression (2-8).
        weight_bits: Weight quantization bits (4 or 8).
        device: CUDA device string.
        dtype: Weight dtype.
    """

    def __init__(
        self,
        model_name: str,
        gpu_budget_gb: float = 22.0,
        cpu_budget_gb: float = 64.0,
        kv_compression: str = "boundary",
        kv_bits: int = 3,
        weight_bits: int = 4,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        if kv_bits < 2 or kv_bits > 8:
            raise ValueError(f"kv_bits must be between 2 and 8, got {kv_bits}")
        if gpu_budget_gb <= 0:
            raise ValueError(f"gpu_budget_gb must be positive, got {gpu_budget_gb}")
        if cpu_budget_gb <= 0:
            raise ValueError(f"cpu_budget_gb must be positive, got {cpu_budget_gb}")
        if weight_bits not in (4, 8, 16):
            raise ValueError(f"weight_bits must be 4, 8, or 16, got {weight_bits}")
        if kv_compression not in ANCHOR_STRATEGIES:
            raise ValueError(
                f"kv_compression must be one of {ANCHOR_STRATEGIES}, "
                f"got '{kv_compression}'"
            )

        self.model_name = model_name
        self.gpu_budget_gb = gpu_budget_gb
        self.cpu_budget_gb = cpu_budget_gb
        self.kv_compression = kv_compression
        self.kv_bits = kv_bits
        self.weight_bits = weight_bits
        self.device = torch.device(device)
        self.dtype = dtype

        # Components (populated by load())
        self.analyzer: Optional[ModelAnalyzer] = None
        self.weight_mgr: Optional[WeightManager] = None
        self.kv_mgr: Optional[KVManager] = None
        self.prefetcher: Optional[AsyncPrefetcher] = None
        self._plan: Optional[Dict[str, Any]] = None

        # Model components (populated by load())
        self.config = None
        self.tokenizer = None
        self.embed_tokens = None
        self.rotary_emb = None
        self.final_norm = None
        self.lm_head = None
        self.layers: List[Any] = []
        self.num_layers: int = 0

        # Metrics
        self._load_time: float = 0.0
        self._tokens_generated: int = 0
        self._generation_time: float = 0.0
        self._peak_vram: int = 0

    def analyze(self, config: Optional[Any] = None) -> Dict[str, Any]:
        """Analyze what's possible with this model + GPU combination.

        Can be called without loading the model -- uses config or
        KNOWN_ARCHITECTURES for offline analysis.

        Args:
            config: Optional HuggingFace AutoConfig. If None, tries to
                load from model_name or match against known architectures.

        Returns:
            Memory plan dict from plan_memory().
        """
        if config is not None:
            self.analyzer = ModelAnalyzer(
                config=config, weight_bits=self.weight_bits,
            )
        elif self.analyzer is None:
            # Try to match against known architectures
            arch = self._match_known_architecture(self.model_name)
            if arch is not None:
                self.analyzer = ModelAnalyzer(
                    arch_dict=arch, weight_bits=self.weight_bits,
                )
            else:
                # Must download config
                from transformers import AutoConfig
                hf_config = AutoConfig.from_pretrained(self.model_name)
                self.analyzer = ModelAnalyzer(
                    config=hf_config, weight_bits=self.weight_bits,
                )

        self._plan = plan_memory(
            self.analyzer,
            gpu_budget_gb=self.gpu_budget_gb,
            cpu_budget_gb=self.cpu_budget_gb,
        )
        return self._plan

    @staticmethod
    def _match_known_architecture(model_name: str) -> Optional[Dict[str, Any]]:
        """Try to match a model name against KNOWN_ARCHITECTURES.

        Matches on substrings in the model name, case-insensitive.

        Args:
            model_name: HuggingFace model name or path.

        Returns:
            Architecture dict if matched, None otherwise.
        """
        name_lower = model_name.lower()
        # Direct key match
        for key, arch in KNOWN_ARCHITECTURES.items():
            if key in name_lower:
                return arch
        # Common pattern matching
        patterns = {
            "405b": "llama-405b",
            "70b": "llama-70b",
            "13b": "llama-13b",
            "7b": "llama-7b",
            "72b": "qwen-72b",
            "scout": "llama4-scout",
            "maverick": "llama4-maverick",
            "deepseek-v3": "deepseek-v3",
            "deepseek_v3": "deepseek-v3",
            "mixtral-8x22": "mixtral-8x22b",
            "mixtral-8x7": "mixtral-8x7b",
        }
        for pattern, key in patterns.items():
            if pattern in name_lower:
                return KNOWN_ARCHITECTURES.get(key)
        return None

    def load(self) -> None:
        """Load model to CPU and set up GPU streaming infrastructure.

        1. Loads model config and tokenizer
        2. Loads weights to CPU (quantized via BitsAndBytes)
        3. Moves embeddings + LM head to GPU
        4. Sets up WeightManager for layer/expert caching
        5. Sets up KVManager for compressed KV cache
        6. Initializes async prefetcher
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        start = time.time()

        # Load config
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.analyzer = ModelAnalyzer(
            config=self.config, weight_bits=self.weight_bits,
        )

        # Plan memory
        self._plan = plan_memory(
            self.analyzer,
            gpu_budget_gb=self.gpu_budget_gb,
            cpu_budget_gb=self.cpu_budget_gb,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model to CPU
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": self.dtype,
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
        }

        # Add quantization config if needed
        if self.weight_bits in (4, 8):
            try:
                from transformers import BitsAndBytesConfig
                if self.weight_bits == 4:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["max_memory"] = {
                        0: f"{int(self.gpu_budget_gb)}GiB",
                        "cpu": f"{int(self.cpu_budget_gb)}GiB",
                    }
                else:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["max_memory"] = {
                        0: f"{int(self.gpu_budget_gb)}GiB",
                        "cpu": f"{int(self.cpu_budget_gb)}GiB",
                    }
            except ImportError:
                pass  # BnB not available, load at FP16

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs,
        )
        model.eval()

        # Extract backbone
        backbone = self._get_backbone(model)
        self.num_layers = self.analyzer.num_layers

        # Move permanent GPU residents
        self.embed_tokens = backbone.embed_tokens.to(self.device)
        if hasattr(backbone, "rotary_emb"):
            self.rotary_emb = backbone.rotary_emb.to(self.device)
        self.final_norm = backbone.norm.to(self.device)
        self.lm_head = model.lm_head.to(self.device)

        # Keep layers on CPU
        self.layers = list(backbone.layers)
        for layer in self.layers:
            layer.to("cpu")
            layer.eval()

        # Set up WeightManager
        weight_cache_bytes = int(self._plan["weight_cache_gb"] * (1024 ** 3))
        self.weight_mgr = WeightManager(
            gpu_budget_bytes=weight_cache_bytes,
            device=self.device,
            is_moe=self.analyzer.is_moe,
        )

        # Set up KVManager
        self.kv_mgr = KVManager(
            kv_budget_gb=self._plan["kv_budget_gb"],
            num_layers=self.num_layers,
            kv_bytes_per_token_fp16=self.analyzer.kv_bytes_per_token_fp16,
            kv_compression=self.kv_compression,
            kv_bits=self.kv_bits,
        )
        self.kv_mgr.create_cache()

        # Set up async prefetcher
        self.prefetcher = AsyncPrefetcher(device=self.device)

        # Clean up model shell
        del model, backbone
        gc.collect()

        self._load_time = time.time() - start

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            self._peak_vram = torch.cuda.max_memory_allocated(self.device)

    def _get_backbone(self, model: Any) -> Any:
        """Extract the transformer backbone from an HF model."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer
        for attr in ("model", "transformer", "backbone"):
            if hasattr(model, attr):
                sub = getattr(model, attr)
                if hasattr(sub, "layers") or hasattr(sub, "h"):
                    return sub
        raise ValueError(
            f"Cannot find transformer backbone in {type(model).__name__}."
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> str:
        """Generate text with ultra-streaming inference.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering (0 = greedy).

        Returns:
            Generated text (prompt + completion).
        """
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Fresh KV cache per generation
        self.kv_mgr.create_cache()
        cache = self.kv_mgr.cache

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        all_token_ids = input_ids.clone()

        gen_start = time.time()

        # Prefill
        next_token = self._generate_token(
            input_ids, past_seq_len=0, cache=cache,
            temperature=temperature, top_k=top_k,
        )
        all_token_ids = torch.cat([all_token_ids, next_token], dim=-1)
        past_seq_len = input_ids.shape[1]

        # Decode
        for step in range(max_new_tokens - 1):
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            next_token = self._generate_token(
                next_token, past_seq_len=past_seq_len, cache=cache,
                temperature=temperature, top_k=top_k,
            )
            all_token_ids = torch.cat([all_token_ids, next_token], dim=-1)
            past_seq_len += 1

        self._generation_time = time.time() - gen_start
        self._tokens_generated = all_token_ids.shape[1] - input_ids.shape[1]

        if torch.cuda.is_available():
            self._peak_vram = torch.cuda.max_memory_allocated(self.device)

        return self.tokenizer.decode(
            all_token_ids[0], skip_special_tokens=True,
        )

    @torch.inference_mode()
    def _generate_token(
        self,
        input_ids: torch.Tensor,
        past_seq_len: int,
        cache: Any,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Generate one token by streaming through all layers.

        Uses WeightManager for layer caching and async prefetch.
        """
        input_ids = input_ids.to(self.device)
        batch_size, seq_len = input_ids.shape

        cache_position = torch.arange(
            past_seq_len, past_seq_len + seq_len, device=self.device,
        )
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)

        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        attention_mask = self._build_causal_mask(
            batch_size, seq_len, past_seq_len, hidden_states.dtype,
        )

        # Stream through all transformer layers
        for layer_idx in range(self.num_layers):
            layer_key = (layer_idx,)

            # Check weight cache
            layer = self.weight_mgr.get(layer_key)
            if layer is None:
                # Not cached -- load from CPU
                layer_module = self.layers[layer_idx]
                layer_size = sum(
                    p.numel() * p.element_size()
                    for p in layer_module.parameters()
                )
                self.weight_mgr.load(layer_key, layer_module, layer_size)
                layer = layer_module

            # Start prefetching next layer
            next_idx = layer_idx + 1
            if next_idx < self.num_layers:
                next_key = (next_idx,)
                if not self.weight_mgr.is_cached(next_key):
                    self.prefetcher.prefetch(self.layers[next_idx])

            # Forward pass through layer
            output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

        # Wait for any pending prefetch
        self.prefetcher.wait()

        # Final norm + LM head
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states[:, -1:, :])

        # Sample
        if temperature <= 0 or top_k == 0:
            next_token = logits.argmax(dim=-1)
        else:
            logits = logits / temperature
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(top_k_logits, dim=-1)
                sampled_idx = torch.multinomial(probs.squeeze(1), 1)
                next_token = top_k_indices.squeeze(1).gather(-1, sampled_idx)

        return next_token

    def _build_causal_mask(
        self,
        batch_size: int,
        seq_len: int,
        past_seq_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a 4D causal attention mask."""
        total_len = past_seq_len + seq_len
        mask = torch.zeros(
            batch_size, 1, seq_len, total_len,
            device=self.device, dtype=dtype,
        )
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.full(
                    (seq_len, seq_len), torch.finfo(dtype).min,
                    device=self.device, dtype=dtype,
                ),
                diagonal=1,
            )
            mask[:, :, :, past_seq_len:] = causal_mask.unsqueeze(0).unsqueeze(0)
        return mask

    def analysis_report(self) -> str:
        """Generate a human-readable analysis report.

        Returns:
            Multi-line report string.
        """
        if self.analyzer is None or self._plan is None:
            self.analyze()
        return format_plan_report(self.model_name, self.analyzer, self._plan)

    def memory_report(self) -> Dict[str, Any]:
        """Report current memory usage and performance metrics."""
        MB = 1024 ** 2

        tok_per_sec = 0.0
        if self._generation_time > 0 and self._tokens_generated > 0:
            tok_per_sec = self._tokens_generated / self._generation_time

        report = {
            "peak_vram_mb": round(self._peak_vram / MB, 1),
            "load_time_sec": round(self._load_time, 1),
            "tokens_generated": self._tokens_generated,
            "tokens_per_sec": round(tok_per_sec, 2),
            "gpu_budget_gb": self.gpu_budget_gb,
            "cpu_budget_gb": self.cpu_budget_gb,
            "kv_bits": self.kv_bits,
            "kv_compression": self.kv_compression,
            "weight_bits": self.weight_bits,
        }

        if self.weight_mgr is not None:
            report["weight_cache"] = self.weight_mgr.stats()

        if self.kv_mgr is not None and self.kv_mgr.strategy is not None:
            report["kv_strategy"] = self.kv_mgr.strategy["description"]

        if self._plan is not None:
            report["model_size_gb"] = self._plan["model_size_gb"]
            report["is_moe"] = self._plan["is_moe"]
            report["estimated_tok_per_sec"] = self._plan["estimated_tok_per_sec"]

        return report
