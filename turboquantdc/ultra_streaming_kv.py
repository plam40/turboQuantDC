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

