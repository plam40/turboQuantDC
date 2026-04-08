"""Asymmetric KV cache — different bit-widths for keys and values.

Community finding: keys control attention routing via softmax and need higher
precision.  Values just need MSE reconstruction and tolerate aggressive
compression.  K norms are 6-182x larger than V norms in Qwen-family models,
confirming that keys carry the quality-critical signal.

Best known config (TheTom, community): high-bit keys + low-bit values.
Recommended presets:
    "quality":     key_bits=4, val_bits=3  (~3.4x compression, near-lossless)
    "balanced":    key_bits=4, val_bits=2  (~4.5x compression, good quality)
    "aggressive":  key_bits=3, val_bits=2  (~5.8x compression, some artifacts)
    "extreme":     key_bits=2, val_bits=2  (~7.3x compression, degraded)

All keys use MSE-only PolarQuant (no QJL).  Community consensus: QJL hurts
generation quality because the variance it introduces outweighs the bias
correction benefit during autoregressive decoding.

Reference: TurboQuant paper (arxiv 2504.19874), community findings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .polarquant import PolarQuant

# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, int]] = {
    "quality": {"key_bits": 4, "val_bits": 3},
    "balanced": {"key_bits": 4, "val_bits": 2},
    "aggressive": {"key_bits": 3, "val_bits": 2},
    "extreme": {"key_bits": 2, "val_bits": 2},
}


# ---------------------------------------------------------------------------
# AsymmetricKVCache (standalone, lower-level API)
# ---------------------------------------------------------------------------

class AsymmetricKVCache:
    """KV cache with different bit-widths for keys and values.

    Community finding: keys control attention routing (softmax sensitivity)
    and need higher precision.  Values just need MSE reconstruction and
    tolerate aggressive compression.  K norms are 6-182x larger than V norms
    in Qwen models.

    Recommended configs:
        "quality":     key_bits=4, val_bits=3  (3.4x compression, near-lossless)
        "balanced":    key_bits=4, val_bits=2  (4.5x compression, good quality)
        "aggressive":  key_bits=3, val_bits=2  (5.8x compression, some artifacts)
        "extreme":     key_bits=2, val_bits=2  (7.3x compression, degraded)

    All keys use MSE-only (no QJL -- community consensus: QJL hurts generation).

    Args:
        d_key: Key head dimension.
        d_value: Value head dimension.
        key_bits: Bits for key compression (2, 3, 4).
        val_bits: Bits for value compression (2, 3, 4).
        seed: Random seed.
        device: Target device.
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        key_bits: int = 4,
        val_bits: int = 2,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        if key_bits not in (2, 3, 4):
            raise ValueError(f"key_bits must be 2, 3, or 4, got {key_bits}")
        if val_bits not in (2, 3, 4):
            raise ValueError(f"val_bits must be 2, 3, or 4, got {val_bits}")

        self.d_key = d_key
        self.d_value = d_value
        self.key_bits = key_bits
        self.val_bits = val_bits
        self.device = device

        # Keys: MSE-only PolarQuant at key_bits
        self.key_quantizer = PolarQuant(
            d=d_key, bits=key_bits, seed=seed, device=device,
        )

        # Values: MSE-only PolarQuant at val_bits
        self.value_quantizer = PolarQuant(
            d=d_value, bits=val_bits, seed=seed + 100, device=device,
        )

        # Storage
        self._key_indices: List[torch.Tensor] = []
        self._key_norms: List[torch.Tensor] = []
        self._value_indices: List[torch.Tensor] = []
        self._value_norms: List[torch.Tensor] = []

    @property
    def seq_len(self) -> int:
        """Number of tokens in the cache."""
        return len(self._key_indices)

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Compress and store key-value pairs with asymmetric bit-widths.

        Args:
            keys: Key vectors, shape (batch, d_key) or (d_key,).
            values: Value vectors, shape (batch, d_value) or (d_value,).
        """
        squeeze_k = keys.dim() == 1
        squeeze_v = values.dim() == 1
        if squeeze_k:
            keys = keys.unsqueeze(0)
        if squeeze_v:
            values = values.unsqueeze(0)

        keys = keys.float()
        values = values.float()

        # Store key norms and normalize
        key_norms = keys.norm(dim=-1, keepdim=True)  # (batch, 1)
        keys_normalized = keys / (key_norms + 1e-8)
        key_indices = self.key_quantizer.quantize(keys_normalized)

        # Store value norms and normalize
        val_norms = values.norm(dim=-1, keepdim=True)  # (batch, 1)
        vals_normalized = values / (val_norms + 1e-8)
        val_indices = self.value_quantizer.quantize(vals_normalized)

        if squeeze_k:
            key_indices = key_indices.squeeze(0)
            key_norms = key_norms.squeeze(0).squeeze(-1)
        else:
            key_norms = key_norms.squeeze(-1)

        if squeeze_v:
            val_indices = val_indices.squeeze(0)
            val_norms = val_norms.squeeze(0).squeeze(-1)
        else:
            val_norms = val_norms.squeeze(-1)

        self._key_indices.append(key_indices)
        self._key_norms.append(key_norms)
        self._value_indices.append(val_indices)
        self._value_norms.append(val_norms)

    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        """Compute attention scores using MSE-reconstructed keys.

        Uses dequantized keys for standard Q @ K^T attention.
        No QJL correction -- MSE reconstruction is sufficient for keys
        at higher bit-widths (4-bit cosine sim > 0.999).

        Args:
            queries: Query vectors, shape (n_queries, d_key) or (d_key,).

        Returns:
            Attention scores, shape (n_queries, seq_len) or (seq_len,).
        """
        if self.seq_len == 0:
            if queries.dim() == 1:
                return torch.zeros(0, device=queries.device)
            return torch.zeros(queries.shape[0], 0, device=queries.device)

        squeeze_q = queries.dim() == 1
        if squeeze_q:
            queries = queries.unsqueeze(0)

        queries = queries.float()

        # Reconstruct all keys
        all_keys = self._reconstruct_keys()  # (seq_len, d_key)

        # Standard dot product attention: (n_queries, d_key) @ (d_key, seq_len)
        scores = queries @ all_keys.T

        if squeeze_q:
            scores = scores.squeeze(0)

        return scores

    def get_values(self) -> torch.Tensor:
        """Reconstruct all cached values.

        Returns:
            Reconstructed values, shape (seq_len, d_value).
        """
        if self.seq_len == 0:
            return torch.zeros(0, self.d_value, device=self.device)

        all_indices = []
        all_norms = []

        for idx_t, norm_t in zip(self._value_indices, self._value_norms):
            if idx_t.dim() == 1:
                all_indices.append(idx_t.unsqueeze(0))
                all_norms.append(norm_t.unsqueeze(0))
            else:
                all_indices.append(idx_t)
                all_norms.append(norm_t)

        indices_cat = torch.cat(all_indices, dim=0)
        norms_cat = torch.cat(all_norms, dim=0)

        vals_normalized = self.value_quantizer.dequantize(indices_cat)
        if norms_cat.dim() == 1:
            norms_cat = norms_cat.unsqueeze(-1)
        return vals_normalized * norms_cat

    def memory_usage_bits(self) -> Dict[str, int]:
        """Report memory with separate K/V accounting.

        Returns:
            Dict with key_bits_total, value_bits_total, total_bits,
            fp16_baseline_bits, and compression_ratio.
        """
        if self.seq_len == 0:
            return {
                "key_bits_total": 0,
                "value_bits_total": 0,
                "total_bits": 0,
                "fp16_baseline_bits": 0,
                "compression_ratio": 0.0,
            }

        # Count actual tokens
        total_tokens = sum(
            idx.shape[0] if idx.dim() > 1 else 1
            for idx in self._key_indices
        )

        # Key: key_bits * d_key (MSE indices) + 16 (vec_norm)
        key_bits_per_token = self.key_bits * self.d_key + 16
        # Value: val_bits * d_value (MSE indices) + 16 (vec_norm)
        val_bits_per_token = self.val_bits * self.d_value + 16

        key_total = total_tokens * key_bits_per_token
        val_total = total_tokens * val_bits_per_token
        total = key_total + val_total
        fp16_baseline = total_tokens * (self.d_key + self.d_value) * 16

        return {
            "key_bits_total": key_total,
            "value_bits_total": val_total,
            "total_bits": total,
            "fp16_baseline_bits": fp16_baseline,
            "compression_ratio": fp16_baseline / total if total > 0 else 0.0,
        }

    def compression_ratio(self) -> float:
        """Overall compression ratio (FP16 baseline / compressed)."""
        stats = self.memory_usage_bits()
        return stats["compression_ratio"]

    def clear(self) -> None:
        """Clear all cached data."""
        self._key_indices.clear()
        self._key_norms.clear()
        self._value_indices.clear()
        self._value_norms.clear()

    def _reconstruct_keys(self) -> torch.Tensor:
        """Reconstruct all stored keys from compressed representation."""
        all_indices = []
        all_norms = []

        for idx_t, norm_t in zip(self._key_indices, self._key_norms):
            if idx_t.dim() == 1:
                all_indices.append(idx_t.unsqueeze(0))
                all_norms.append(norm_t.unsqueeze(0))
            else:
                all_indices.append(idx_t)
                all_norms.append(norm_t)

        indices_cat = torch.cat(all_indices, dim=0)
        norms_cat = torch.cat(all_norms, dim=0)

        keys_normalized = self.key_quantizer.dequantize(indices_cat)
        if norms_cat.dim() == 1:
            norms_cat = norms_cat.unsqueeze(-1)
        return keys_normalized * norms_cat


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_asymmetric_cache(
    d_key: int,
    d_value: int,
    preset: str = "balanced",
    **kwargs,
) -> AsymmetricKVCache:
    """Create an AsymmetricKVCache from a preset config.

    Args:
        d_key: Key head dimension.
        d_value: Value head dimension.
        preset: One of "quality", "balanced", "aggressive", "extreme".
        **kwargs: Additional kwargs passed to AsymmetricKVCache.

    Returns:
        Configured AsymmetricKVCache.
    """
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Choose from: {list(PRESETS.keys())}"
        )
    config = PRESETS[preset]
    return AsymmetricKVCache(d_key, d_value, **config, **kwargs)


# ---------------------------------------------------------------------------
# AsymmetricTurboQuantLayer (HF-compatible, per-layer)
# ---------------------------------------------------------------------------

class AsymmetricTurboQuantLayer:
    """A single layer's asymmetric compressed KV cache storage.

    Like TurboQuantLayer but uses different bit-widths for keys and values.
    Keys use MSE-only PolarQuant at key_bits, values at val_bits.

    Args:
        key_bits: Bits per key coordinate (2, 3, or 4).
        val_bits: Bits per value coordinate (2, 3, or 4).
        seed: Base random seed for this layer's quantizers.
    """

    def __init__(self, key_bits: int = 4, val_bits: int = 2, seed: int = 42):
        self.key_bits = key_bits
        self.val_bits = val_bits
        self.seed = seed
        self._seq_len: int = 0

        # Lazily initialized on first update
        self._key_pq: Optional[PolarQuant] = None
        self._val_pq: Optional[PolarQuant] = None

        # Compressed storage
        self._key_compressed: List[Dict[str, torch.Tensor]] = []
        self._value_compressed: List[Dict[str, torch.Tensor]] = []

        # Dimensions learned on first update
        self._dtype: Optional[torch.dtype] = None
        self._device: Optional[torch.device] = None
        self._key_head_dim: Optional[int] = None
        self._val_head_dim: Optional[int] = None
        self._num_heads: Optional[int] = None
        self._batch_size: Optional[int] = None

    def _lazy_init(
        self, key_states: torch.Tensor, value_states: torch.Tensor,
    ) -> None:
        """Initialize quantizers from the first observed tensor shapes."""
        self._batch_size = key_states.shape[0]
        self._num_heads = key_states.shape[1]
        self._key_head_dim = key_states.shape[3]
        self._val_head_dim = value_states.shape[3]
        self._dtype = key_states.dtype
        self._device = key_states.device

        device = str(self._device) if self._device is not None else "cpu"

        self._key_pq = PolarQuant(
            d=self._key_head_dim, bits=self.key_bits,
            seed=self.seed, device=device,
        )
        self._val_pq = PolarQuant(
            d=self._val_head_dim, bits=self.val_bits,
            seed=self.seed + 100, device=device,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress and store new KV states, return dequantized full cache.

        Args:
            key_states: [batch, num_heads, new_seq, head_dim].
            value_states: [batch, num_heads, new_seq, head_dim].

        Returns:
            Tuple of (all_keys, all_values) dequantized from cache.
        """
        if self._key_pq is None:
            self._lazy_init(key_states, value_states)

        batch, num_heads, new_seq, key_dim = key_states.shape
        val_dim = value_states.shape[3]

        # Flatten for quantization
        keys_flat = key_states.float().reshape(-1, key_dim)
        vals_flat = value_states.float().reshape(-1, val_dim)

        # Compress keys
        key_norms = keys_flat.norm(dim=-1, keepdim=True)
        keys_normalized = keys_flat / (key_norms + 1e-8)
        key_indices = self._key_pq.quantize(keys_normalized)
        key_entry = {
            "mse_indices": key_indices.reshape(batch, num_heads, new_seq, key_dim),
            "vec_norm": key_norms.squeeze(-1).reshape(batch, num_heads, new_seq),
        }

        # Compress values
        val_norms = vals_flat.norm(dim=-1, keepdim=True)
        vals_normalized = vals_flat / (val_norms + 1e-8)
        val_indices = self._val_pq.quantize(vals_normalized)
        val_entry = {
            "indices": val_indices.reshape(batch, num_heads, new_seq, val_dim),
            "norms": val_norms.squeeze(-1).reshape(batch, num_heads, new_seq),
        }

        self._key_compressed.append(key_entry)
        self._value_compressed.append(val_entry)
        self._seq_len += new_seq

        return self._dequantize_all()

    def _dequantize_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize all stored keys and values into full tensors."""
        if self._seq_len == 0:
            empty_k = torch.zeros(
                self._batch_size or 1, self._num_heads or 1, 0,
                self._key_head_dim or 1,
                dtype=self._dtype, device=self._device,
            )
            empty_v = torch.zeros(
                self._batch_size or 1, self._num_heads or 1, 0,
                self._val_head_dim or 1,
                dtype=self._dtype, device=self._device,
            )
            return empty_k, empty_v

        batch = self._batch_size
        num_heads = self._num_heads
        key_dim = self._key_head_dim
        val_dim = self._val_head_dim

        # Concatenate compressed entries
        all_key_mse = torch.cat(
            [e["mse_indices"] for e in self._key_compressed], dim=2,
        )
        all_key_norms = torch.cat(
            [e["vec_norm"] for e in self._key_compressed], dim=2,
        )
        all_val_idx = torch.cat(
            [e["indices"] for e in self._value_compressed], dim=2,
        )
        all_val_norms = torch.cat(
            [e["norms"] for e in self._value_compressed], dim=2,
        )

        total_seq = all_key_mse.shape[2]

        # Dequantize keys
        key_mse_flat = all_key_mse.reshape(-1, key_dim)
        key_vnorm_flat = all_key_norms.reshape(-1)
        key_recon_flat = self._key_pq.dequantize(key_mse_flat)
        key_recon_flat = key_recon_flat * key_vnorm_flat.unsqueeze(-1)
        keys_out = key_recon_flat.reshape(batch, num_heads, total_seq, key_dim)

        # Dequantize values
        val_idx_flat = all_val_idx.reshape(-1, val_dim)
        val_norms_flat = all_val_norms.reshape(-1)
        val_recon_flat = self._val_pq.dequantize(val_idx_flat)
        val_recon_flat = val_recon_flat * val_norms_flat.unsqueeze(-1)
        vals_out = val_recon_flat.reshape(batch, num_heads, total_seq, val_dim)

        return keys_out.to(self._dtype), vals_out.to(self._dtype)

    def get_seq_length(self) -> int:
        """Return number of cached tokens."""
        return self._seq_len

    def reorder(self, beam_idx: torch.LongTensor) -> None:
        """Reorder cache entries for beam search along batch dimension."""
        for entry in self._key_compressed:
            entry["mse_indices"] = entry["mse_indices"].index_select(0, beam_idx)
            entry["vec_norm"] = entry["vec_norm"].index_select(0, beam_idx)
        for entry in self._value_compressed:
            entry["indices"] = entry["indices"].index_select(0, beam_idx)
            entry["norms"] = entry["norms"].index_select(0, beam_idx)
        if self._batch_size is not None:
            self._batch_size = beam_idx.shape[0]

    def crop(self, max_length: int) -> None:
        """Truncate cached sequence to max_length tokens."""
        if max_length < 0:
            max_length = self._seq_len + max_length
        if self._seq_len <= max_length:
            return

        remaining = max_length
        new_key_comp = []
        new_val_comp = []

        for kc, vc in zip(self._key_compressed, self._value_compressed):
            chunk_seq = kc["mse_indices"].shape[2]
            if remaining <= 0:
                break
            take = min(chunk_seq, remaining)
            new_key_comp.append({
                "mse_indices": kc["mse_indices"][:, :, :take],
                "vec_norm": kc["vec_norm"][:, :, :take],
            })
            new_val_comp.append({
                "indices": vc["indices"][:, :, :take],
                "norms": vc["norms"][:, :, :take],
            })
            remaining -= take

        self._key_compressed = new_key_comp
        self._value_compressed = new_val_comp
        self._seq_len = max_length

    def memory_usage_bits(self) -> Dict[str, Any]:
        """Compute memory usage of compressed storage.

        Returns:
            Dict with key_bits, value_bits, total_bits, fp16_baseline_bits,
            and compression_ratio -- with separate K/V accounting.
        """
        if self._seq_len == 0 or self._key_head_dim is None:
            return {
                "key_bits": 0,
                "value_bits": 0,
                "total_bits": 0,
                "fp16_baseline_bits": 0,
                "compression_ratio": 0.0,
            }

        n_heads = self._num_heads
        batch = self._batch_size
        total_tokens = self._seq_len * n_heads * batch

        # Key: key_bits * d_key + 16 (vec_norm)
        key_bits_per_token = self.key_bits * self._key_head_dim + 16
        # Value: val_bits * d_value + 16 (vec_norm)
        val_bits_per_token = self.val_bits * self._val_head_dim + 16

        key_bits = total_tokens * key_bits_per_token
        val_bits = total_tokens * val_bits_per_token
        total = key_bits + val_bits
        # FP16 baseline: both keys and values at 16 bits
        fp16_baseline = total_tokens * (self._key_head_dim + self._val_head_dim) * 16

        return {
            "key_bits": key_bits,
            "value_bits": val_bits,
            "total_bits": total,
            "fp16_baseline_bits": fp16_baseline,
            "compression_ratio": fp16_baseline / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        """Clear all compressed data."""
        self._key_compressed.clear()
        self._value_compressed.clear()
        self._seq_len = 0


# ---------------------------------------------------------------------------
# AsymmetricTurboQuantCache (HF-compatible, multi-layer)
# ---------------------------------------------------------------------------

class AsymmetricTurboQuantCache:
    """HF-compatible asymmetric KV cache.

    Drop-in replacement for TurboQuantCache with separate K/V bit-widths.
    Duck-types the HF Cache protocol just like TurboQuantCache does.

    Usage:
        cache = AsymmetricTurboQuantCache(key_bits=4, val_bits=2)
        output = model.generate(inputs, past_key_values=cache)

    Args:
        key_bits: Bits for key compression (2, 3, or 4). Default 4.
        val_bits: Bits for value compression (2, 3, or 4). Default 2.
        seed: Base random seed. Each layer gets seed + layer_idx.
    """

    is_compileable = False

    def __init__(
        self,
        key_bits: int = 4,
        val_bits: int = 2,
        seed: int = 42,
    ):
        if key_bits not in (2, 3, 4):
            raise ValueError(f"key_bits must be 2, 3, or 4, got {key_bits}")
        if val_bits not in (2, 3, 4):
            raise ValueError(f"val_bits must be 2, 3, or 4, got {val_bits}")
        self.key_bits = key_bits
        self.val_bits = val_bits
        self.seed = seed
        self._layers: List[AsymmetricTurboQuantLayer] = []

    # ---- Cache protocol ----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress and store new KV pairs for a layer, return dequantized cache.

        Args:
            key_states: [batch, num_heads, new_seq_len, head_dim].
            value_states: [batch, num_heads, new_seq_len, head_dim].
            layer_idx: Which transformer layer this belongs to.
            cache_kwargs: Additional HF cache kwargs (ignored).

        Returns:
            Tuple of (all_keys, all_values) dequantized tensors.
        """
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
        self, cache_position: torch.Tensor, layer_idx: int,
    ) -> tuple[int, int]:
        """Return (kv_length, kv_offset) for mask generation."""
        if layer_idx >= len(self._layers):
            return cache_position.shape[0], 0
        query_length = cache_position.shape[0]
        kv_length = self._layers[layer_idx].get_seq_length() + query_length
        return kv_length, 0

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
        """Repeat cache entries for beam search expansion."""
        for layer in self._layers:
            for entry in layer._key_compressed:
                entry["mse_indices"] = entry["mse_indices"].repeat_interleave(
                    repeats, dim=0,
                )
                entry["vec_norm"] = entry["vec_norm"].repeat_interleave(
                    repeats, dim=0,
                )
            for entry in layer._value_compressed:
                entry["indices"] = entry["indices"].repeat_interleave(
                    repeats, dim=0,
                )
                entry["norms"] = entry["norms"].repeat_interleave(
                    repeats, dim=0,
                )
            if layer._batch_size is not None:
                layer._batch_size *= repeats

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select specific batch indices from the cache."""
        for layer in self._layers:
            for entry in layer._key_compressed:
                entry["mse_indices"] = entry["mse_indices"][indices]
                entry["vec_norm"] = entry["vec_norm"][indices]
            for entry in layer._value_compressed:
                entry["indices"] = entry["indices"][indices]
                entry["norms"] = entry["norms"][indices]
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
        """Iterate over layers, yielding (keys, values, None) tuples."""
        for layer in self._layers:
            keys, values = layer._dequantize_all()
            yield keys, values, None

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dequantized (keys, values) for a specific layer."""
        if layer_idx >= len(self._layers):
            raise IndexError(
                f"Layer {layer_idx} not in cache (have {len(self._layers)} layers)"
            )
        return self._layers[layer_idx]._dequantize_all()

    # ---- Layer factory ----

    def _make_layer(self, layer_idx: int) -> AsymmetricTurboQuantLayer:
        """Create a new AsymmetricTurboQuantLayer for the given layer index."""
        return AsymmetricTurboQuantLayer(
            key_bits=self.key_bits,
            val_bits=self.val_bits,
            seed=self.seed + layer_idx,
        )

    # ---- Reporting ----

    def memory_savings(self) -> Dict[str, Any]:
        """Report memory usage and savings across all layers.

        Returns:
            Dict with per_layer list and aggregate totals including
            total_compressed_bits, total_fp16_bits, overall_compression_ratio,
            and the key/value bit-widths.
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
            "key_bits": self.key_bits,
            "val_bits": self.val_bits,
            "num_layers": len(self._layers),
        }


# ---------------------------------------------------------------------------
# Analysis utility
# ---------------------------------------------------------------------------

def analyze_kv_norms(
    model_name: str,
    prompt: str = "The quick brown fox jumps over the lazy dog",
    device: str = "cuda",
    max_length: int = 50,
) -> Dict[str, Any]:
    """Analyze K vs V norm distributions in a real model.

    Returns per-layer K/V norm statistics showing why asymmetric compression
    works.  The community found 6-182x norm ratios in Qwen models, meaning
    keys carry far more magnitude and are more sensitive to quantization error.

    Args:
        model_name: HuggingFace model name (e.g. "Qwen/Qwen2.5-3B-Instruct").
        prompt: Input text to generate KV activations.
        device: Device for inference ("cuda" or "cpu").
        max_length: Maximum sequence length for generation.

    Returns:
        Dict with:
            - per_layer: list of {layer, k_norm_mean, v_norm_mean, ratio}
            - overall_ratio: mean(K_norm) / mean(V_norm) across all layers
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers is required for analyze_kv_norms. "
            "Install with: pip install transformers"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False, use_cache=True)

    past_kv = outputs.past_key_values

    per_layer = []
    all_k_norms = []
    all_v_norms = []

    # DynamicCache iterates as (key_states, value_states, None) per layer
    for layer_idx, layer_data in enumerate(past_kv):
        k_states = layer_data[0].float()
        v_states = layer_data[1].float()

        # Compute per-head per-token norms then average
        k_norms = k_states.norm(dim=-1)  # [batch, heads, seq]
        v_norms = v_states.norm(dim=-1)

        k_mean = k_norms.mean().item()
        v_mean = v_norms.mean().item()
        ratio = k_mean / (v_mean + 1e-8)

        per_layer.append({
            "layer": layer_idx,
            "k_norm_mean": k_mean,
            "v_norm_mean": v_mean,
            "ratio": ratio,
            "k_norm_std": k_norms.std().item(),
            "v_norm_std": v_norms.std().item(),
        })
        all_k_norms.append(k_mean)
        all_v_norms.append(v_mean)

    overall_k = sum(all_k_norms) / len(all_k_norms)
    overall_v = sum(all_v_norms) / len(all_v_norms)

    return {
        "model": model_name,
        "prompt": prompt,
        "num_layers": len(per_layer),
        "per_layer": per_layer,
        "overall_k_norm_mean": overall_k,
        "overall_v_norm_mean": overall_v,
        "overall_ratio": overall_k / (overall_v + 1e-8),
    }
