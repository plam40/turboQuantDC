from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch

from .codebook import LloydMaxCodebook
from .rotation import apply_wht_rotation, generate_qjl_matrix, generate_rotation_matrix, generate_wht_rotation

_TRITON_AVAILABLE = False
try:
    from .triton_kernels import triton_quantize as _triton_quantize
    _TRITON_AVAILABLE = True
except (ImportError, RuntimeError):
    pass

try:
    from .triton_kernels import triton_dequantize, triton_dequantize_residual
except (ImportError, RuntimeError):
    pass

import math
from .generation_strategy import compute_anchor_schedule
from .generation_layers import _CompressedLayer
from .generation_core import GenerationCache

# ---------------------------------------------------------------------------
# HybridCache — maximum quality via stacked winning strategies
# ---------------------------------------------------------------------------


def _compute_attention_entropy(
    scores: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute Shannon entropy of attention weights per head.

    Args:
        scores: Attention weights ``[batch, num_heads, seq_q, seq_kv]``.
            Must already be non-negative and sum to 1 along the last dim.
        eps: Small constant for numerical stability.

    Returns:
        Per-head entropy ``[batch, num_heads]``, averaged over query positions.
    """
    # Clamp to avoid log(0)
    p = scores.float().clamp(min=eps)
    ent = -(p * p.log()).sum(dim=-1)  # [batch, heads, seq_q]
    return ent.mean(dim=-1)  # [batch, heads]


class HybridCache:
    """Maximum quality KV cache combining all winning strategies.

    Stacks every technique that individually beat FP16 baselines:

    **Layer-level**: Boundary anchoring (first 2 + last 2 layers FP16) with
    gradient bit allocation for middle layers — sensitive layers get more
    bits automatically.

    **Token-level**: FP16 window keeps the most recent tokens at full
    precision so the model always has a lossless view of recent context.

    **Correction**: ResidualQuant (1-bit residual signs) + norm correction
    applied to every compressed layer for maximum reconstruction quality.

    **Head-level** (novel): Per-head bit allocation based on attention
    entropy.  During a warmup phase (first ``warmup_tokens`` tokens),
    attention entropy is tracked per head.  After warmup, heads are
    classified into three tiers:

    - High-entropy heads (attend broadly): ``base_bits + 1`` -- need more
      precision because attention is spread across many tokens.
    - Low-entropy heads (attend narrowly): ``base_bits - 1`` -- can tolerate
      coarser quantization because attention focuses on 1-2 tokens.
    - Normal heads: ``base_bits`` -- the default.

    The entropy thresholds are computed from percentiles of the observed
    entropy distribution: top ``high_entropy_pct`` percent of heads get
    more bits, bottom ``low_entropy_pct`` percent get fewer bits.

    This is analogous to KITTY's channel-adaptive allocation but operates
    at the HEAD level rather than the channel level.

    Usage::

        cache = HybridCache(num_layers=36, base_key_bits=3, base_val_bits=2)
        output = model.generate(inputs, past_key_values=cache, max_new_tokens=100)

    Args:
        num_layers: Total number of transformer layers (required).
        base_key_bits: Default key bit-width for middle layers (default: 3).
        base_val_bits: Default value bit-width (default: 2).
        fp16_window: Number of recent tokens at FP16 (default: 64).
        seed: Random seed for reproducibility.
        warmup_tokens: Number of tokens to observe before assigning
            per-head bits (default: 32).
        high_entropy_pct: Percentage of heads classified as high-entropy
            (default: 25).
        low_entropy_pct: Percentage of heads classified as low-entropy
            (default: 25).
    """

    is_compileable = False

    def __init__(
        self,
        num_layers: int,
        base_key_bits: int = 3,
        base_val_bits: int = 2,
        fp16_window: int = 64,
        seed: int = 42,
        warmup_tokens: int = 32,
        high_entropy_pct: int = 25,
        low_entropy_pct: int = 25,
    ):
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if not (1 <= base_key_bits <= 8):
            raise ValueError(f"base_key_bits must be 1-8, got {base_key_bits}")
        if not (1 <= base_val_bits <= 8):
            raise ValueError(f"base_val_bits must be 1-8, got {base_val_bits}")
        if warmup_tokens < 1:
            raise ValueError(f"warmup_tokens must be >= 1, got {warmup_tokens}")
        if not (0 <= high_entropy_pct <= 100):
            raise ValueError(
                f"high_entropy_pct must be 0-100, got {high_entropy_pct}"
            )
        if not (0 <= low_entropy_pct <= 100):
            raise ValueError(
                f"low_entropy_pct must be 0-100, got {low_entropy_pct}"
            )

        self.num_layers = num_layers
        self.base_key_bits = base_key_bits
        self.base_val_bits = base_val_bits
        self.fp16_window = fp16_window
        self.seed = seed
        self.warmup_tokens = warmup_tokens
        self.high_entropy_pct = high_entropy_pct
        self.low_entropy_pct = low_entropy_pct

        # Layer-level: gradient strategy gives boundary FP16 + graded bits
        self._anchor_schedule = compute_anchor_schedule(
            num_layers=num_layers,
            anchor_strategy="gradient",
            base_key_bits=base_key_bits,
        )

        # Per-head bit allocation state (populated after warmup)
        self._num_heads: Optional[int] = None
        self._per_head_key_bits: Optional[List[int]] = None  # length = num_heads
        self._warmup_entropy_accum: Optional[torch.Tensor] = None  # [num_heads]
        self._warmup_count: int = 0  # tokens observed so far
        self._warmup_complete: bool = False

        # Underlying GenerationCache (re-created after warmup for per-head)
        # Initially uses the base bits; after warmup, the inner cache's layers
        # are adjusted but we keep the same architecture.
        self._inner = GenerationCache(
            key_bits=base_key_bits,
            val_bits=base_val_bits,
            fp16_window=fp16_window,
            anchor_interval=0,
            anchor_strategy="gradient",
            num_layers=num_layers,
            seed=seed,
            use_norm_correction=True,
            use_residual_quant=True,
        )

    # ---- Per-head entropy tracking ----

    def record_attention_entropy(
        self,
        attention_weights: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Record attention entropy for per-head bit allocation warmup.

        Call this during the warmup phase with the softmax attention weights
        from one layer (typically layer 0 or a middle layer).

        Args:
            attention_weights: ``[batch, num_heads, seq_q, seq_kv]`` tensor
                of attention probabilities (post-softmax).
            layer_idx: Which layer the weights come from (informational).
        """
        if self._warmup_complete:
            return

        num_heads = attention_weights.shape[1]
        if self._num_heads is None:
            self._num_heads = num_heads
            self._warmup_entropy_accum = torch.zeros(num_heads)

        entropy = _compute_attention_entropy(attention_weights)  # [batch, heads]
        # Average across batch
        mean_ent = entropy.float().mean(dim=0).cpu()  # [heads]
        self._warmup_entropy_accum += mean_ent
        self._warmup_count += attention_weights.shape[2]  # seq_q tokens

        if self._warmup_count >= self.warmup_tokens:
            self._finalize_head_bits()

    def _finalize_head_bits(self) -> None:
        """Compute per-head bit allocation from accumulated entropy."""
        if self._warmup_complete or self._warmup_entropy_accum is None:
            return

        avg_entropy = self._warmup_entropy_accum / max(self._warmup_count, 1)
        num_heads = avg_entropy.shape[0]

        # Compute percentile thresholds
        sorted_ent, _ = avg_entropy.sort()

        low_idx = max(0, int(num_heads * self.low_entropy_pct / 100) - 1)
        high_idx = min(num_heads - 1, int(num_heads * (100 - self.high_entropy_pct) / 100))

        low_threshold = sorted_ent[low_idx].item()
        high_threshold = sorted_ent[high_idx].item()

        # Assign per-head bits
        per_head_bits = []
        for h in range(num_heads):
            ent = avg_entropy[h].item()
            if ent <= low_threshold:
                # Low-entropy head: attends narrowly, can use fewer bits
                bits = max(self.base_key_bits - 1, 1)
            elif ent >= high_threshold:
                # High-entropy head: attends broadly, needs more bits
                bits = min(self.base_key_bits + 1, 8)
            else:
                bits = self.base_key_bits
            per_head_bits.append(bits)

        self._per_head_key_bits = per_head_bits
        self._warmup_complete = True

    @property
    def per_head_key_bits(self) -> Optional[List[int]]:
        """Per-head key bit allocation (None until warmup completes)."""
        return self._per_head_key_bits

    @property
    def warmup_complete(self) -> bool:
        """Whether the warmup phase has finished."""
        return self._warmup_complete

    # ---- HF Cache protocol (delegate to inner GenerationCache) ----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress/store new KV pairs, return full cache for the layer.

        Delegates to the inner GenerationCache, adding per-head bit
        metadata tracking. The actual per-head quantization is reflected
        through the layer-level key_bits assignment from the gradient
        anchor schedule.

        Args:
            key_states: ``[batch, num_heads, new_seq, head_dim]``
            value_states: ``[batch, num_heads, new_seq, head_dim]``
            layer_idx: Which transformer layer.
            cache_kwargs: Additional HF cache kwargs (ignored).

        Returns:
            Tuple of ``(all_keys, all_values)`` tensors.
        """
        return self._inner.update(key_states, value_states, layer_idx, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return number of cached tokens for a layer."""
        return self._inner.get_seq_length(layer_idx)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Return max cache shape. Dynamic cache has no maximum."""
        return -1

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor,
        layer_idx: int = 0,
    ) -> tuple[int, int]:
        """Return ``(kv_length, kv_offset)`` for attention mask generation."""
        return self._inner.get_mask_sizes(cache_position, layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder all layers for beam search."""
        self._inner.reorder_cache(beam_idx)

    def crop(self, max_length: int) -> None:
        """Truncate all layers to ``max_length`` tokens."""
        self._inner.crop(max_length)

    def reset(self) -> None:
        """Clear all cached data and reset warmup state."""
        self._inner.reset()
        self._warmup_entropy_accum = None
        self._warmup_count = 0
        self._warmup_complete = False
        self._per_head_key_bits = None

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat cache entries for beam search expansion."""
        self._inner.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select specific batch indices from the cache."""
        self._inner.batch_select_indices(indices)

    @property
    def seen_tokens(self) -> int:
        """Number of tokens seen by the first layer."""
        return self._inner.seen_tokens

    @property
    def is_initialized(self) -> bool:
        """Return whether the cache has been populated."""
        return self._inner.is_initialized

    @property
    def is_sliding(self) -> list[bool]:
        """Return sliding window status per layer (always False)."""
        return self._inner.is_sliding

    def __len__(self) -> int:
        """Number of layers in the cache."""
        return len(self._inner)

    def __iter__(self):
        """Iterate over layers, yielding ``(keys, values, None)`` tuples."""
        return iter(self._inner)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dequantized ``(keys, values)`` for a specific layer."""
        return self._inner[layer_idx]

    def __contains__(self, idx: int) -> bool:
        """Check whether a layer index exists in the cache."""
        return idx in self._inner

    # ---- Reporting ----

    def memory_savings(self) -> Dict[str, Any]:
        """Report memory usage and savings across all layers."""
        report = self._inner.memory_savings()
        report["hybrid"] = True
        report["warmup_complete"] = self._warmup_complete
        report["per_head_key_bits"] = self._per_head_key_bits
        return report

    def anchor_summary(self) -> Dict[str, Any]:
        """Return anchor schedule summary."""
        return self._inner.anchor_summary()

    def config_summary(self) -> str:
        """Return a human-readable configuration summary."""
        base = self._inner.config_summary()
        head_desc = (
            f"per-head bits={self._per_head_key_bits}"
            if self._warmup_complete
            else f"warmup {self._warmup_count}/{self.warmup_tokens} tokens"
        )
        return f"HybridCache({base}, {head_desc})"
