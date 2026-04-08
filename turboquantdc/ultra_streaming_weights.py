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

