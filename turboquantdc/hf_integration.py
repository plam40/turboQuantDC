"""HuggingFace transformers integration for TurboQuantDC.

Provides a drop-in KV cache class that compresses key-value pairs
using TurboQuant. Compatible with HuggingFace's generate() and
all attention implementations.

The cache subclasses HF's DynamicCache and overrides the layer-level
storage to compress keys with TurboQuantEstimator (MSE + QJL, two-stage)
and values with PolarQuant (MSE-only, since values need reconstruction,
not inner products).

On retrieval, compressed data is dequantized back to FP16/FP32 tensors so
that standard HF attention can operate without modification. The memory
savings come from the compressed internal representation.

For keys, the dequantized output is the MSE reconstruction (Stage 1 only).
Full unbiased inner-product estimation would require replacing the attention
computation itself; for practical HF integration the MSE reconstruction
(cosine sim > 0.996 at 3-bit, d=128) is close enough.

Usage:
    from turboquantdc.hf_integration import TurboQuantCache
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", ...)
    output = model.generate(
        inputs,
        max_new_tokens=100,
        past_key_values=TurboQuantCache(bits=3),
    )

Reference: TurboQuant paper (arxiv 2504.19874).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import torch

from .estimator import TurboQuantEstimator
from .polarquant import PolarQuant


# ---------------------------------------------------------------------------
# Utility: strip HF-injected internal attributes from parameter dicts
# ---------------------------------------------------------------------------

# Attributes injected by transformers >= 5.3.0 that bitsandbytes param
# constructors (Params4bit, Int8Params) reject as unexpected kwargs.
_HF_INTERNAL_ATTRS: Set[str] = {
    "_is_hf_initialized",
    "_hf_hook",
    "_old_forward",
}


def _clean_param_dict(param: torch.nn.Parameter) -> dict:
    """Return a copy of ``param.__dict__`` without HF-injected internals.

    When HuggingFace transformers >= 5.3.0 initializes a model, it stamps
    parameters with ``_is_hf_initialized = True``.  If ``accelerate`` then
    passes ``param.__dict__`` to a bitsandbytes parameter constructor
    (``Params4bit.__new__()`` or ``Int8Params.__new__()``), the extra key
    causes a ``TypeError``.

    Call this before passing parameter dicts to BnB or any other library
    that does not expect HF-internal keys.
    """
    return {k: v for k, v in param.__dict__.items() if k not in _HF_INTERNAL_ATTRS}


class TurboQuantLayer:
    """A single layer's compressed KV cache storage.

    Stores keys via TurboQuantEstimator (MSE + QJL for unbiased inner products)
    and values via PolarQuant (MSE-only for reconstruction). Tensors are stored
    in compressed form and dequantized on retrieval.

    HF attention expects tensors of shape [batch, num_heads, seq_len, head_dim].
    We compress along the head_dim axis for each (batch, head) independently.

    Args:
        bits: Total bits per coordinate (2, 3, or 4).
        seed: Base random seed for this layer's quantizers.
    """

    def __init__(self, bits: int = 3, seed: int = 42, mse_only: bool = False):
        self.bits = bits
        self.seed = seed
        self.mse_only = mse_only  # Skip QJL, use full b-bit MSE for keys
        self._seq_len: int = 0

        # Offset tracking for streaming / eviction scenarios.
        # When tokens are evicted from the front of the sequence, the mask
        # must know the offset so that it does not attend to dead positions.
        # ``_kv_offset`` counts the total number of evicted tokens so that
        # ``get_mask_sizes`` returns the correct ``(kv_length, kv_offset)``.
        self._kv_offset: int = 0

        # Lazily initialized on first update (need to know head_dim)
        self._key_estimators: Optional[Dict[int, TurboQuantEstimator]] = None
        self._value_quantizers: Optional[Dict[int, PolarQuant]] = None

        # Compressed storage: list of compressed dicts per (batch, head) pair
        # Organized as: _key_compressed[time_step] = {
        #   "mse_indices": (batch, num_heads, new_seq, head_dim),
        #   "qjl_signs":   (batch, num_heads, new_seq, m),
        #   "residual_norm": (batch, num_heads, new_seq),
        #   "vec_norm":     (batch, num_heads, new_seq),
        # }
        self._key_compressed: List[Dict[str, torch.Tensor]] = []
        self._value_compressed: List[Dict[str, torch.Tensor]] = []

        # Cache the original dtype for dequantization output
        self._dtype: Optional[torch.dtype] = None
        self._device: Optional[torch.device] = None

        # Dimensions learned on first update
        self._head_dim: Optional[int] = None
        self._num_heads: Optional[int] = None
        self._batch_size: Optional[int] = None

        # Shared estimator/quantizer (one per layer, shared across heads)
        self._key_est: Optional[TurboQuantEstimator] = None
        self._key_pq: Optional[PolarQuant] = None  # Used when mse_only=True
        self._val_pq: Optional[PolarQuant] = None

    def _lazy_init(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Initialize quantizers from the first observed tensor shapes.

        Supports meta-device tensors: if ``key_states`` is on the ``meta``
        device, we record shapes and dtype but skip allocating real quantizer
        buffers.  Allocation is deferred to the first ``update()`` call with
        a real (non-meta) tensor, enabling ``device_map="auto"`` for 70B+
        models.
        """
        # key_states: [batch, num_heads, seq_len, head_dim]
        self._batch_size = key_states.shape[0]
        self._num_heads = key_states.shape[1]
        self._head_dim = key_states.shape[3]
        self._dtype = key_states.dtype
        self._device = key_states.device

        # Meta tensors are shape-only placeholders used by HF for large
        # models with ``device_map="auto"``.  We cannot allocate real
        # quantizer buffers on ``meta``, so we record the shapes and bail
        # out.  The quantizers will be created on the first real update().
        if key_states.device.type == "meta":
            return

        self._allocate_quantizers()

    def _allocate_quantizers(self) -> None:
        """Allocate quantizer objects on ``self._device``.

        Separated from ``_lazy_init`` so that meta-device layers can defer
        this until the first real (non-meta) update.
        """
        d = self._head_dim
        # Codebook computation (scipy) runs on CPU internally, but the
        # resulting rotation matrices and centroids must live on the same
        # device as the data. PolarQuant and TurboQuantEstimator accept a
        # device argument and move their buffers accordingly.
        device = str(self._device) if self._device is not None else "cpu"

        if self.mse_only:
            # MSE-only mode: use PolarQuant for keys too (full b-bit, no QJL)
            # This gives 8 centroids at 3-bit instead of 4 (2-bit MSE + 1-bit QJL)
            # Better for generation quality where variance > bias matters
            self._key_est = None
            self._key_pq = PolarQuant(
                d=d, bits=self.bits, seed=self.seed, device=device,
            )
        else:
            self._key_est = TurboQuantEstimator(
                d=d, bits=self.bits, seed=self.seed, device=device,
            )
            self._key_pq = None
        self._val_pq = PolarQuant(
            d=d, bits=self.bits, seed=self.seed + 100, device=device,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress and store new key/value states, return dequantized full cache.

        Args:
            key_states: New keys [batch, num_heads, new_seq, head_dim].
            value_states: New values [batch, num_heads, new_seq, head_dim].

        Returns:
            Tuple of (all_keys, all_values) dequantized from cache,
            each of shape [batch, num_heads, total_seq, head_dim].
        """
        if self._key_est is None and self._key_pq is None:
            self._lazy_init(key_states, value_states)

        # Handle meta -> real device transition.  If _lazy_init was called
        # with a meta tensor, quantizers are not yet allocated.  Now that we
        # have a real tensor, allocate them on the real device.
        if self._val_pq is None and self._head_dim is not None:
            # Shapes were recorded from a meta tensor; now we have a real one.
            self._device = key_states.device
            self._dtype = key_states.dtype
            self._allocate_quantizers()

        # Multi-device support: if this layer's quantizers live on a
        # different device than the incoming tensors, move them.
        if self._device is not None and key_states.device != self._device:
            self._device = key_states.device
            self._allocate_quantizers()

        batch, num_heads, new_seq, head_dim = key_states.shape

        # Flatten to (batch * num_heads * new_seq, head_dim) for quantization
        keys_flat = key_states.float().reshape(-1, head_dim)
        vals_flat = value_states.float().reshape(-1, head_dim)

        # Compress keys
        key_norms = keys_flat.norm(dim=-1, keepdim=True)
        keys_normalized = keys_flat / (key_norms + 1e-8)

        if self.mse_only:
            # MSE-only: use PolarQuant with full b-bit codebook (no QJL)
            key_indices = self._key_pq.quantize(keys_normalized)
            key_entry = {
                "mse_indices": key_indices.reshape(batch, num_heads, new_seq, head_dim),
                "vec_norm": key_norms.squeeze(-1).reshape(batch, num_heads, new_seq),
            }
        else:
            # Full TurboQuant (MSE + QJL)
            key_comp = self._key_est.quantize(keys_flat)
            key_entry = {
                "mse_indices": key_comp["mse_indices"].reshape(batch, num_heads, new_seq, head_dim),
                "qjl_signs": key_comp["qjl_signs"].reshape(batch, num_heads, new_seq, -1),
                "residual_norm": key_comp["residual_norm"].reshape(batch, num_heads, new_seq),
                "vec_norm": key_comp["vec_norm"].reshape(batch, num_heads, new_seq),
            }

        # Compress values with MSE-only PolarQuant
        val_norms = vals_flat.norm(dim=-1, keepdim=True)
        vals_normalized = vals_flat / (val_norms + 1e-8)
        val_indices = self._val_pq.quantize(vals_normalized)
        val_entry = {
            "indices": val_indices.reshape(batch, num_heads, new_seq, head_dim),
            "norms": val_norms.squeeze(-1).reshape(batch, num_heads, new_seq),
        }

        self._key_compressed.append(key_entry)
        self._value_compressed.append(val_entry)
        self._seq_len += new_seq

        # Dequantize and return the full cache
        return self._dequantize_all()

    def _dequantize_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize all stored keys and values into full tensors."""
        if self._seq_len == 0:
            empty = torch.zeros(
                self._batch_size or 1, self._num_heads or 1, 0,
                self._head_dim or 1,
                dtype=self._dtype, device=self._device,
            )
            return empty, empty

        batch = self._batch_size
        num_heads = self._num_heads
        head_dim = self._head_dim

        # Concatenate all compressed entries along seq dimension
        all_key_mse = torch.cat([e["mse_indices"] for e in self._key_compressed], dim=2)
        all_key_vec_norm = torch.cat([e["vec_norm"] for e in self._key_compressed], dim=2)
        all_val_indices = torch.cat([e["indices"] for e in self._value_compressed], dim=2)
        all_val_norms = torch.cat([e["norms"] for e in self._value_compressed], dim=2)

        total_seq = all_key_mse.shape[2]

        # Dequantize keys: flatten -> dequantize -> reshape
        key_mse_flat = all_key_mse.reshape(-1, head_dim)
        key_vnorm_flat = all_key_vec_norm.reshape(-1)

        # MSE dequantize: centroid lookup + unrotate
        if self.mse_only:
            key_recon_flat = self._key_pq.dequantize(key_mse_flat)
        else:
            key_recon_flat = self._key_est.polar.dequantize(key_mse_flat)
        # Rescale by original vector norm
        key_recon_flat = key_recon_flat * key_vnorm_flat.unsqueeze(-1)
        keys_out = key_recon_flat.reshape(batch, num_heads, total_seq, head_dim)

        # Dequantize values: flatten -> dequantize -> rescale
        val_idx_flat = all_val_indices.reshape(-1, head_dim)
        val_norms_flat = all_val_norms.reshape(-1)

        val_recon_flat = self._val_pq.dequantize(val_idx_flat)
        val_recon_flat = val_recon_flat * val_norms_flat.unsqueeze(-1)
        vals_out = val_recon_flat.reshape(batch, num_heads, total_seq, head_dim)

        # Cast back to original dtype
        return keys_out.to(self._dtype), vals_out.to(self._dtype)

    def get_seq_length(self) -> int:
        """Return number of cached tokens."""
        return self._seq_len

    def reorder(self, beam_idx: torch.LongTensor) -> None:
        """Reorder cache entries for beam search along the batch dimension."""
        for entry in self._key_compressed:
            for key in entry:
                entry[key] = entry[key].index_select(0, beam_idx)
        for entry in self._value_compressed:
            for key in entry:
                entry[key] = entry[key].index_select(0, beam_idx)

    def evict_tokens(self, n_evict: int) -> None:
        """Evict the oldest ``n_evict`` tokens from the front of the cache.

        Updates ``_kv_offset`` so that ``get_mask_sizes`` can return the
        correct offset for attention mask generation.  Without this, evicted
        positions would still appear in the attention mask, causing the model
        to attend to dead positions and producing garbled output.

        Args:
            n_evict: Number of tokens to evict from the front.
        """
        if n_evict <= 0 or self._seq_len == 0:
            return
        n_evict = min(n_evict, self._seq_len)

        # Walk through compressed entries, dropping from the front
        remaining_to_drop = n_evict
        new_key_comp = []
        new_val_comp = []

        for kc, vc in zip(self._key_compressed, self._value_compressed):
            chunk_seq = kc["mse_indices"].shape[2]
            if remaining_to_drop >= chunk_seq:
                remaining_to_drop -= chunk_seq
                continue  # drop entire chunk
            if remaining_to_drop > 0:
                # Partial drop from front
                kc = {k: v[:, :, remaining_to_drop:] for k, v in kc.items()}
                vc = {k: v[:, :, remaining_to_drop:] for k, v in vc.items()}
                remaining_to_drop = 0
            new_key_comp.append(kc)
            new_val_comp.append(vc)

        self._key_compressed = new_key_comp
        self._value_compressed = new_val_comp
        self._seq_len -= n_evict
        self._kv_offset += n_evict

    def crop(self, max_length: int) -> None:
        """Truncate cached sequence to max_length tokens (from the tail)."""
        if max_length < 0:
            max_length = self._seq_len + max_length
        if self._seq_len <= max_length:
            return

        # Rebuild compressed entries keeping only first max_length tokens
        remaining = max_length
        new_key_comp = []
        new_val_comp = []

        for kc, vc in zip(self._key_compressed, self._value_compressed):
            chunk_seq = kc["mse_indices"].shape[2]
            if remaining <= 0:
                break
            take = min(chunk_seq, remaining)
            new_key_comp.append({k: v[:, :, :take] for k, v in kc.items()})
            new_val_comp.append({k: v[:, :, :take] for k, v in vc.items()})
            remaining -= take

        self._key_compressed = new_key_comp
        self._value_compressed = new_val_comp
        self._seq_len = max_length

    def memory_usage_bits(self) -> Dict[str, int]:
        """Compute memory usage of compressed storage.

        Returns:
            Dict with key_bits, value_bits, total_bits, fp16_baseline_bits,
            and compression_ratio.
        """
        if self._seq_len == 0 or self._head_dim is None:
            return {
                "key_bits": 0,
                "value_bits": 0,
                "total_bits": 0,
                "fp16_baseline_bits": 0,
                "compression_ratio": 0.0,
            }

        d = self._head_dim
        n_heads = self._num_heads
        batch = self._batch_size
        total_tokens = self._seq_len * n_heads * batch

        if self.mse_only:
            # MSE-only mode: keys use full b-bit codebook, no QJL
            # Key: bits*d (MSE indices) + 16 (vec_norm)
            key_bits_per_token = self.bits * d + 16
        else:
            mse_bits = max(self.bits - 1, 1)
            qjl_m = self._key_est.qjl.m
            # Key: mse_bits*d (MSE indices) + qjl_m*1 (signs) + 16 (res_norm) + 16 (vec_norm)
            key_bits_per_token = mse_bits * d + qjl_m + 32
        # Value: bits*d (MSE indices) + 16 (norm)
        val_bits_per_token = self.bits * d + 16

        key_bits = total_tokens * key_bits_per_token
        val_bits = total_tokens * val_bits_per_token
        total = key_bits + val_bits
        fp16_baseline = total_tokens * d * 16 * 2  # keys + values, 16 bits each

        return {
            "key_bits": key_bits,
            "value_bits": val_bits,
            "total_bits": total,
            "fp16_baseline_bits": fp16_baseline,
            "compression_ratio": fp16_baseline / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        """Clear all compressed data and reset the KV offset."""
        self._key_compressed.clear()
        self._value_compressed.clear()
        self._seq_len = 0
        self._kv_offset = 0


class TurboQuantCache:
    """Drop-in TurboQuant KV cache for HuggingFace transformers.

    Subclass-compatible with HF's Cache protocol. Compresses keys with the
    full two-stage TurboQuant estimator and values with MSE-only PolarQuant.

    On retrieval, data is dequantized to FP16/FP32 so standard HF attention
    works unmodified. Memory savings come from compressed internal storage.

    This class implements the interface expected by HuggingFace's generate()
    and attention layers by duck-typing the Cache/DynamicCache protocol rather
    than subclassing (to avoid tight coupling with transformers internals that
    change across versions).

    Args:
        bits: Total bits per coordinate (2, 3, or 4). Default 3.
        seed: Base random seed. Each layer gets seed + layer_idx.
    """

    # Class attribute to signal to HF that this is a valid cache type.
    # Some HF code checks isinstance(past_key_values, Cache) but duck-typing
    # also works for generate().
    is_compileable = False

    def __init__(self, bits: int = 3, seed: int = 42, mse_only: bool = False):
        if not (2 <= bits <= 8):
            raise ValueError(f"bits must be between 2 and 8, got {bits}")
        self.bits = bits
        self.seed = seed
        self.mse_only = mse_only  # Skip QJL, use full b-bit MSE for keys
        self._layers: List[TurboQuantLayer] = []

    # ---- Implement the Cache protocol ----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress and store new KV pairs for a layer, return full dequantized cache.

        This is the main entry point called by HF attention layers.

        Args:
            key_states: [batch, num_heads, new_seq_len, head_dim]
            value_states: [batch, num_heads, new_seq_len, head_dim]
            layer_idx: Which transformer layer this belongs to.
            cache_kwargs: Additional HF cache kwargs (ignored).

        Returns:
            Tuple of (all_keys, all_values) dequantized tensors,
            shape [batch, num_heads, total_seq_len, head_dim].
        """
        # Lazily create layers as needed
        while len(self._layers) <= layer_idx:
            self._layers.append(self._make_layer(len(self._layers)))

        return self._layers[layer_idx].update(key_states, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return number of cached tokens for a layer."""
        if layer_idx >= len(self._layers):
            return 0
        return self._layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Return max cache shape. Dynamic cache has no maximum."""
        return -1

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int
    ) -> tuple[int, int]:
        """Return ``(kv_length, kv_offset)`` for mask generation.

        When tokens have been evicted from the front of the cache, the mask
        must start at the correct offset so that attention does not target
        dead positions.  The offset is tracked per-layer via
        ``TurboQuantLayer._kv_offset``.
        """
        if layer_idx >= len(self._layers):
            return cache_position.shape[0], 0
        layer = self._layers[layer_idx]
        query_length = cache_position.shape[0]
        kv_length = layer.get_seq_length() + query_length + layer._kv_offset
        return kv_length, layer._kv_offset

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder all layers for beam search."""
        for layer in self._layers:
            layer.reorder(beam_idx)

    def crop(self, max_length: int) -> None:
        """Truncate all layers to max_length tokens."""
        for layer in self._layers:
            layer.crop(max_length)

    def reset(self) -> None:
        """Clear all cached data."""
        for layer in self._layers:
            layer.clear()

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat cache entries for beam search expansion.

        Repeats each batch element `repeats` times along the batch dimension
        in all compressed storage tensors.
        """
        for layer in self._layers:
            for entry in layer._key_compressed:
                for key in entry:
                    entry[key] = entry[key].repeat_interleave(repeats, dim=0)
            for entry in layer._value_compressed:
                for key in entry:
                    entry[key] = entry[key].repeat_interleave(repeats, dim=0)
            if layer._batch_size is not None:
                layer._batch_size *= repeats

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select specific batch indices from the cache."""
        for layer in self._layers:
            for entry in layer._key_compressed:
                for key in entry:
                    entry[key] = entry[key][indices]
            for entry in layer._value_compressed:
                for key in entry:
                    entry[key] = entry[key][indices]
            if layer._batch_size is not None:
                layer._batch_size = len(indices)

    @property
    def is_initialized(self) -> bool:
        """Return whether the cache has been populated."""
        return len(self._layers) > 0

    @property
    def is_sliding(self) -> list[bool]:
        """Return sliding window status per layer (always False)."""
        return [False] * len(self._layers)

    def __len__(self) -> int:
        """Number of layers in the cache."""
        return len(self._layers)

    def __iter__(self):
        """Iterate over layers, yielding (keys, values, None) tuples.

        Matches DynamicCache's __iter__ protocol for compatibility.
        """
        for layer in self._layers:
            keys, values = layer._dequantize_all()
            yield keys, values, None

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dequantized (keys, values) for a specific layer."""
        if layer_idx >= len(self._layers):
            raise IndexError(f"Layer {layer_idx} not in cache (have {len(self._layers)} layers)")
        return self._layers[layer_idx]._dequantize_all()

    # ---- Layer factory (used by custom_attention patching) ----

    def _make_layer(self, layer_idx: int) -> TurboQuantLayer:
        """Create a new TurboQuantLayer for the given layer index.

        Used internally by ``update()`` and by ``custom_attention.patch_model_attention()``
        when layers need to be created lazily.
        """
        return TurboQuantLayer(bits=self.bits, seed=self.seed + layer_idx, mse_only=self.mse_only)

    # ---- Custom attention integration ----

    def enable_unbiased_attention(self, model: torch.nn.Module) -> torch.nn.Module:
        """Patch a HuggingFace model to use TurboQuant's unbiased attention.

        This replaces standard Q @ K^T attention with the full two-stage
        unbiased inner product estimator.  At 3-bit, this is the difference
        between garbled output and coherent generation.

        After calling this, ``model.generate()`` will use TurboQuant attention
        transparently.  The cache argument to ``generate()`` should be omitted
        (or set to ``None``) since the patched model uses this cache directly.

        Args:
            model: A HuggingFace CausalLM model.

        Returns:
            The same model, patched in-place.
        """
        from .custom_attention import patch_model_attention
        return patch_model_attention(model, self)

    # ---- TurboQuant-specific methods ----

    def memory_savings(self) -> Dict[str, Any]:
        """Report memory usage and savings across all layers.

        Returns:
            Dict with per_layer list and aggregate totals including
            total_compressed_bits, total_fp16_bits, and overall_compression_ratio.
        """
        per_layer = []
        total_compressed = 0
        total_fp16 = 0

        for i, layer in enumerate(self._layers):
            stats = layer.memory_usage_bits()
            per_layer.append({"layer": i, **stats})
            total_compressed += stats["total_bits"]
            total_fp16 += stats["fp16_baseline_bits"]

        return {
            "per_layer": per_layer,
            "total_compressed_bits": total_compressed,
            "total_fp16_bits": total_fp16,
            "overall_compression_ratio": (
                total_fp16 / total_compressed if total_compressed > 0 else 0.0
            ),
            "bits": self.bits,
            "num_layers": len(self._layers),
        }
