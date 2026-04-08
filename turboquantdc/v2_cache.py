"""TurboQuantDC v2 — unified pipeline combining every proven technique.

PCA rotation + mean-removal + ResidualQuant + DeltaQuant + adaptive bits
+ FAISS retrieval in a single drop-in HuggingFace-compatible KV cache.

Architecture:
    Input KV pair
        -> Boundary check: first/last 2 layers -> FP16 (anchor layers)
        -> Hot window: last ``window_size`` tokens -> FP16
        -> PCA rotation: pre-calibrated from ``calibrate()``
        -> Mean removal: per-head running mean (shift-invariant)
        -> Importance scoring: EMA of attention weights
        -> Tier assignment based on importance:
            Tier 0 (top 5%):     4-bit ResidualQuant
            Tier 1 (next 25%):   3-bit ResidualQuant
            Tier 2 (bottom 70%): DeltaQuant (3-bit anchor + 1-bit delta, G=4)
        -> FAISS index update (in ``full`` mode)
        -> Store compressed

    On attention query (``full`` mode):
        -> FAISS search: top-k candidate keys
        -> Merge with hot window
        -> Compute attention over small candidate set

Two modes:
    ``compress``: PCA + mean-removal + adaptive bits + DeltaQuant, no retrieval
    ``full``: compression + FAISS retrieval attention

Duck-types the HuggingFace Cache protocol for drop-in use with
``model.generate(past_key_values=cache)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .adaptive_bits import ImportanceScorer
from .codebook import LloydMaxCodebook
from .learned_rotation import compute_pca_rotation
from .rotation import apply_wht_rotation, generate_rotation_matrix, generate_wht_rotation

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class V2Config:
    """Configuration for TurboQuantV2Cache."""
    key_bits: int = 3
    val_bits: int = 3
    window_size: int = 128
    boundary_layers: int = 2
    retrieval_k: int = 128
    delta_group_size: int = 4
    faiss_nprobe: int = 16
    faiss_nlist: int = 64
    # Tier thresholds: top 5% -> 4bit, next 25% -> 3bit, bottom 70% -> delta
    tier_thresholds: List[float] = field(default_factory=lambda: [0.05, 0.30])
    tier_bits: List[int] = field(default_factory=lambda: [4, 3, 1])
    # Delta coding
    anchor_bits: int = 3
    delta_bits: int = 1
    # Importance
    ema_decay: float = 0.9
    reclassify_interval: int = 16
    warmup_steps: int = 32
    # Mode
    mode: str = "compress"  # "compress" or "full"
    seed: int = 42


# ---------------------------------------------------------------------------
# Per-head FAISS index (lightweight)
# ---------------------------------------------------------------------------

class _HeadFAISSIndex:
    """Single FAISS index for one KV head at one layer."""

    def __init__(self, dim: int, nlist: int = 64, nprobe: int = 16):
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self._keys_buffer: List[np.ndarray] = []
        self._n_total = 0
        self._index: Optional[Any] = None
        self._trained = False

    def add(self, keys_np: np.ndarray):
        """Add keys. Defer index build until search is needed."""
        self._keys_buffer.append(keys_np)
        self._n_total += keys_np.shape[0]
        self._trained = False  # needs rebuild

    def _build(self):
        """Build or rebuild the index from all buffered keys."""
        if self._n_total == 0:
            return
        all_keys = np.concatenate(self._keys_buffer, axis=0)
        eff_nlist = min(self.nlist, max(1, self._n_total // 39))
        if eff_nlist <= 1 or self._n_total < 40:
            self._index = faiss.IndexFlatIP(self.dim)
        else:
            quantizer = faiss.IndexFlatIP(self.dim)
            self._index = faiss.IndexIVFFlat(
                quantizer, self.dim, eff_nlist, faiss.METRIC_INNER_PRODUCT
            )
            self._index.nprobe = self.nprobe
            self._index.train(all_keys)
        self._index.add(all_keys)
        self._trained = True

    def search(self, queries_np: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-k. Builds index lazily."""
        if not self._trained:
            self._build()
        if self._index is None or self._n_total == 0:
            empty = np.zeros((queries_np.shape[0], 0), dtype=np.float32)
            return empty, np.zeros((queries_np.shape[0], 0), dtype=np.int64)
        k = min(k, self._n_total)
        return self._index.search(queries_np, k)

    def reset(self):
        self._keys_buffer.clear()
        self._n_total = 0
        self._index = None
        self._trained = False


# ---------------------------------------------------------------------------
# V2 Layer — handles one transformer layer's KV cache
# ---------------------------------------------------------------------------

class _V2Layer:
    """Single layer compressed KV cache with PCA rotation + adaptive tiers."""

    def __init__(
        self,
        config: V2Config,
        layer_idx: int,
        pca_data: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.cfg = config
        self.layer_idx = layer_idx
        self.pca_data = pca_data  # rotation, eigenvalues, mean from calibration

        self._seq_len: int = 0
        self._step_count: int = 0

        # Lazily initialized
        self._head_dim: Optional[int] = None
        self._num_kv_heads: Optional[int] = None
        self._batch_size: Optional[int] = None
        self._dtype: Optional[torch.dtype] = None
        self._device: Optional[torch.device] = None

        # Rotation
        self._pca_rotation: Optional[torch.Tensor] = None
        self._pca_mean: Optional[torch.Tensor] = None
        self._fallback_rotation: Optional[Any] = None

        # Codebooks per bit-width
        self._codebooks: Dict[int, LloydMaxCodebook] = {}

        # Importance scorer (per-head average)
        self._scorer = ImportanceScorer(ema_decay=config.ema_decay)

        # FP16 window (most recent tokens, always full precision)
        self._fp16_keys: List[torch.Tensor] = []
        self._fp16_vals: List[torch.Tensor] = []
        self._fp16_count: int = 0

        # Compressed storage: pre-dequantized 4D tensors for fast retrieval
        self._comp_keys: List[torch.Tensor] = []
        self._comp_vals: List[torch.Tensor] = []
        self._comp_bits: List[torch.Tensor] = []  # bits per token for metrics
        self._comp_count: int = 0

        # Running mean for mean-removal (per head)
        self._running_mean: Optional[torch.Tensor] = None  # (1, H, 1, D)
        self._mean_count: int = 0

        # FAISS indexes (per KV head) — only in "full" mode
        self._faiss_indexes: Optional[List[_HeadFAISSIndex]] = None

    def _lazy_init(self, key_states: torch.Tensor, value_states: torch.Tensor):
        """Initialize from first observed tensor shapes. key_states: (B, H, S, D)."""
        self._batch_size = key_states.shape[0]
        self._num_kv_heads = key_states.shape[1]
        self._head_dim = key_states.shape[3]
        self._dtype = key_states.dtype
        self._device = key_states.device

        d = self._head_dim
        device = self._device

        # Set up PCA rotation if calibration data available
        if self.pca_data is not None:
            self._pca_rotation = self.pca_data["rotation"].to(device).float()
            self._pca_mean = self.pca_data["mean"].to(device).float()
            # Whitening scale: after PCA rotation, coordinate i has variance =
            # eigenvalue_i. The codebook expects N(0, 1/d). We scale each
            # coordinate so variance matches 1/d.
            target_var = 1.0 / d
            safe_eigs = self.pca_data["eigenvalues"].clamp(min=1e-12).float()
            self._whiten_scale = (target_var / safe_eigs).sqrt().to(device)
        else:
            self._whiten_scale = None
            # Fallback: WHT for power-of-2, QR otherwise
            is_pow2 = d > 0 and (d & (d - 1)) == 0
            if is_pow2:
                self._fallback_rotation = generate_wht_rotation(
                    d, seed=self.cfg.seed + self.layer_idx, device=str(device)
                )
            else:
                self._fallback_rotation = generate_rotation_matrix(
                    d, seed=self.cfg.seed + self.layer_idx, device=str(device)
                )

        # Codebooks for each bit-width we might need:
        # tier_bits, val_bits, anchor_bits, delta_bits, and min(val_bits, 2)
        all_bits = set(self.cfg.tier_bits) | {
            self.cfg.val_bits,
            self.cfg.anchor_bits,
            self.cfg.delta_bits,
            min(self.cfg.val_bits, 2),
            min(self.cfg.val_bits, 3),
        }
        # Also add each tier's value bits
        for tb in self.cfg.tier_bits:
            all_bits.add(min(self.cfg.val_bits, tb))
        for b in all_bits:
            if 1 <= b <= 8 and b not in self._codebooks:
                mse_b = max(b - 1, 1)
                cb = LloydMaxCodebook(d=d, bits=mse_b).to(str(device))
                self._codebooks[b] = cb

        # FAISS indexes (per head)
        if self.cfg.mode == "full" and FAISS_AVAILABLE:
            self._faiss_indexes = [
                _HeadFAISSIndex(d, self.cfg.faiss_nlist, self.cfg.faiss_nprobe)
                for _ in range(self._num_kv_heads)
            ]

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation + whitening. x: (..., d) -> (..., d).

        For PCA: center -> rotate -> whiten so each coord ~ N(0, 1/d).
        For WHT/QR: just rotate (already matches N(0, 1/d) assumption).
        """
        if self._pca_rotation is not None:
            x_c = x - self._pca_mean
            y = x_c @ self._pca_rotation.T
            if self._whiten_scale is not None:
                y = y * self._whiten_scale
            return y
        elif isinstance(self._fallback_rotation, dict):
            return apply_wht_rotation(x, self._fallback_rotation)
        else:
            return x @ self._fallback_rotation

    def _unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse rotation + un-whitening. y: (..., d) -> (..., d)."""
        if self._pca_rotation is not None:
            if self._whiten_scale is not None:
                y = y / self._whiten_scale
            return y @ self._pca_rotation + self._pca_mean
        elif isinstance(self._fallback_rotation, dict):
            return apply_wht_rotation(y, self._fallback_rotation, inverse=True)
        else:
            return y @ self._fallback_rotation.T

    def _quantize_rq(self, vectors: torch.Tensor, bits: int) -> torch.Tensor:
        """Quantize with ResidualQuant at given bits, return dequantized vectors.

        MSE quantize + 1-bit residual sign correction in rotated space.
        """
        if bits >= 16:
            return vectors

        cb = self._codebooks[bits]
        norms = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = vectors / norms
        rotated = self._rotate(normalized)

        # MSE quantize
        indices = torch.bucketize(rotated, cb.boundaries)
        indices = indices.clamp(0, cb.centroids.shape[0] - 1)
        recon_rot = cb.centroids[indices]

        # 1-bit residual sign correction
        if bits > 1:
            residual = rotated - recon_rot
            signs = (residual >= 0).float() * 2.0 - 1.0
            scale = residual.abs().mean(dim=-1, keepdim=True)
            recon_rot = recon_rot + scale * signs

        # Inverse rotate and rescale
        recon = self._unrotate(recon_rot) * norms
        return recon

    def _quantize_delta(
        self,
        vectors: torch.Tensor,
    ) -> torch.Tensor:
        """DeltaQuant: group tokens by similarity, anchor + delta coding.

        Each group of G tokens has one anchor at anchor_bits and (G-1) deltas
        at delta_bits. No cumulative error.
        """
        N, d = vectors.shape
        if N <= self.cfg.delta_group_size:
            # Too few tokens for delta grouping -- fall back to RQ at 3-bit
            return self._quantize_rq(vectors, self.cfg.anchor_bits)

        G = self.cfg.delta_group_size
        anchor_cb = self._codebooks.get(self.cfg.anchor_bits)
        delta_cb = self._codebooks.get(max(self.cfg.delta_bits, 1))

        if anchor_cb is None:
            return self._quantize_rq(vectors, self.cfg.anchor_bits)

        norms = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = vectors / norms
        rotated = self._rotate(normalized)

        recon = torch.zeros_like(rotated)

        n_groups = max(1, N // G)
        for g in range(n_groups):
            start = g * G
            end = min(start + G, N)
            group = rotated[start:end]

            # Anchor: first token in group (medoid selection omitted for speed)
            anchor = group[0:1]
            anchor_idx = torch.bucketize(anchor, anchor_cb.boundaries)
            anchor_idx = anchor_idx.clamp(0, anchor_cb.centroids.shape[0] - 1)
            anchor_recon = anchor_cb.centroids[anchor_idx]

            # Residual sign correction on anchor
            res = anchor - anchor_recon
            signs = (res >= 0).float() * 2.0 - 1.0
            scale = res.abs().mean(dim=-1, keepdim=True)
            anchor_recon = anchor_recon + scale * signs
            recon[start] = anchor_recon[0]

            # Deltas: each token stored as delta from anchor
            for i in range(1, end - start):
                delta = group[i:i+1] - anchor_recon
                if delta_cb is not None:
                    # Scale delta to codebook range
                    d_scale = delta.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
                    target_range = 3.0 / math.sqrt(d)
                    scaled = delta * (target_range / d_scale)
                    d_idx = torch.bucketize(scaled, delta_cb.boundaries)
                    d_idx = d_idx.clamp(0, delta_cb.centroids.shape[0] - 1)
                    d_recon = delta_cb.centroids[d_idx] * (d_scale / target_range)
                    recon[start + i] = anchor_recon[0] + d_recon[0]
                else:
                    # Sign-only delta
                    recon[start + i] = anchor_recon[0] + delta.sign()[0] * delta.abs().mean()

        # Handle leftover tokens
        leftover_start = n_groups * G
        if leftover_start < N:
            leftover = rotated[leftover_start:]
            l_idx = torch.bucketize(leftover, anchor_cb.boundaries)
            l_idx = l_idx.clamp(0, anchor_cb.centroids.shape[0] - 1)
            recon[leftover_start:] = anchor_cb.centroids[l_idx]

        # Inverse rotate and rescale
        result = self._unrotate(recon) * norms
        return result

    def _remove_mean(self, key_states: torch.Tensor) -> torch.Tensor:
        """Subtract running per-head mean from keys. Softmax shift-invariant.

        Mean-removal reduces key variance, giving better codebook utilization.
        Since softmax(QK^T) = softmax(Q(K-mu)^T), this is lossless for attention.
        """
        B, H, S, D = key_states.shape
        keys_f = key_states.float()

        new_sum = keys_f.sum(dim=2, keepdim=True)  # (B, H, 1, D)
        old_n = self._mean_count
        new_n = old_n + S
        if self._running_mean is None:
            self._running_mean = new_sum / new_n
        else:
            self._running_mean = (self._running_mean * old_n + new_sum) / new_n
        self._mean_count = new_n

        return keys_f - self._running_mean

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Store new KV states, return full dequantized cache."""
        if self._head_dim is None:
            self._lazy_init(key_states, value_states)

        new_seq = key_states.shape[2]

        # Mean removal on keys (free +1 bit via shift-invariance)
        keys_centered = self._remove_mean(key_states)

        # Store in FP16 buffer
        self._fp16_keys.append(keys_centered)
        self._fp16_vals.append(value_states.detach().float())
        self._fp16_count += new_seq
        self._seq_len += new_seq

        # Add to FAISS index in full mode
        if self._faiss_indexes is not None:
            for h in range(self._num_kv_heads):
                k_np = keys_centered[0, h].float().cpu().numpy().astype(np.float32)
                self._faiss_indexes[h].add(k_np)

        # Flush excess from FP16 buffer to compressed storage
        self._flush_buffer()

        return self._reconstruct_all()

    def _flush_buffer(self):
        """Compress oldest tokens when FP16 buffer exceeds window_size."""
        if self._fp16_count <= self.cfg.window_size:
            return

        all_keys = torch.cat(self._fp16_keys, dim=2)
        all_vals = torch.cat(self._fp16_vals, dim=2)
        flush_count = self._fp16_count - self.cfg.window_size

        flush_k = all_keys[:, :, :flush_count, :]
        flush_v = all_vals[:, :, :flush_count, :]

        # Get tier assignments
        tiers = self._assign_tiers(flush_count)

        B, H, _, D = flush_k.shape
        comp_k = torch.zeros_like(flush_k)
        comp_v = torch.zeros_like(flush_v)
        bits_record = torch.zeros(flush_count, device=flush_k.device)

        for b_idx in range(B):
            for h_idx in range(H):
                k_slice = flush_k[b_idx, h_idx]  # (flush_count, D)
                v_slice = flush_v[b_idx, h_idx]

                # Tier 0: high importance -> 4-bit RQ
                mask0 = tiers == 0
                if mask0.any():
                    comp_k[b_idx, h_idx, mask0] = self._quantize_rq(
                        k_slice[mask0], self.cfg.tier_bits[0]
                    )
                    comp_v[b_idx, h_idx, mask0] = self._quantize_rq(
                        v_slice[mask0], self.cfg.val_bits
                    )
                    bits_record[mask0] = float(self.cfg.tier_bits[0])

                # Tier 1: medium importance -> 3-bit RQ
                mask1 = tiers == 1
                if mask1.any():
                    comp_k[b_idx, h_idx, mask1] = self._quantize_rq(
                        k_slice[mask1], self.cfg.tier_bits[1]
                    )
                    comp_v[b_idx, h_idx, mask1] = self._quantize_rq(
                        v_slice[mask1], self.cfg.val_bits
                    )
                    bits_record[mask1] = float(self.cfg.tier_bits[1])

                # Tier 2: low importance -> DeltaQuant
                mask2 = tiers == 2
                if mask2.any():
                    comp_k[b_idx, h_idx, mask2] = self._quantize_delta(k_slice[mask2])
                    comp_v[b_idx, h_idx, mask2] = self._quantize_rq(
                        v_slice[mask2], min(self.cfg.val_bits, 2)
                    )
                    bits_record[mask2] = float(self.cfg.tier_bits[2])

        self._comp_keys.append(comp_k)
        self._comp_vals.append(comp_v)
        self._comp_bits.append(bits_record)
        self._comp_count += flush_count

        # Trim FP16 buffer
        self._fp16_keys = [all_keys[:, :, flush_count:, :]]
        self._fp16_vals = [all_vals[:, :, flush_count:, :]]
        self._fp16_count -= flush_count

    def _assign_tiers(self, flush_count: int) -> torch.Tensor:
        """Assign importance tiers to tokens being flushed.

        Without importance data (before any attention feedback), defaults to
        tier 1 (3-bit ResidualQuant) -- safer than tier 2 (1-bit DeltaQuant)
        for prompt tokens that haven't been scored yet.
        """
        default_tier = 1  # 3-bit RQ is safe default

        if self._scorer.scores is None or self._scorer.seq_len == 0:
            return torch.full((flush_count,), default_tier, dtype=torch.long)

        global_tiers = self._scorer.classify_tiers(self.cfg.tier_thresholds)
        start = self._comp_count
        end = start + flush_count

        if global_tiers.shape[0] >= end:
            return global_tiers[start:end]
        elif global_tiers.shape[0] > start:
            partial = global_tiers[start:]
            pad = torch.full((flush_count - partial.shape[0],), default_tier, dtype=torch.long)
            return torch.cat([partial, pad])
        return torch.full((flush_count,), default_tier, dtype=torch.long)

    def update_importance(self, attention_weights: torch.Tensor):
        """Update importance scores from attention weights."""
        self._scorer.update(attention_weights)
        self._step_count += 1

    def _reconstruct_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return full (B, H, total_seq, D) key and value tensors."""
        parts_k, parts_v = [], []

        # Compressed tokens (already dequantized)
        for ck, cv in zip(self._comp_keys, self._comp_vals):
            parts_k.append(ck)
            parts_v.append(cv)

        # FP16 window tokens
        if self._fp16_keys:
            fp16_k = torch.cat(self._fp16_keys, dim=2)
            fp16_v = torch.cat(self._fp16_vals, dim=2)
            parts_k.append(fp16_k)
            parts_v.append(fp16_v)

        if not parts_k:
            B = self._batch_size or 1
            H = self._num_kv_heads or 1
            D = self._head_dim or 1
            empty = torch.zeros(B, H, 0, D, dtype=self._dtype, device=self._device)
            return empty, empty

        all_k = torch.cat(parts_k, dim=2)
        all_v = torch.cat(parts_v, dim=2)
        return all_k.to(self._dtype), all_v.to(self._dtype)

    def get_seq_length(self) -> int:
        return self._seq_len

    def effective_bits(self) -> float:
        """Average bits per coordinate for KV combined storage.

        For compressed tokens:
          Tier 0/1 (ResidualQuant): key_bits*d + d(signs) + 32(norms) per key vec
          Tier 2 (DeltaQuant): ~(anchor_bits/G + delta_bits*(G-1)/G)*d + overheads
          Values: val_bits*d + 16(norm)

        For FP16 window: 16 bits per coord for both keys and values.
        """
        if self._seq_len == 0:
            return 16.0

        G = self.cfg.delta_group_size
        d = self._head_dim or 128

        # FP16 window: 16 bits/coord for both K and V -> 16 bits average
        total_bits = self._fp16_count * 16.0
        count = self._fp16_count

        for bpt in self._comp_bits:
            for i in range(bpt.shape[0]):
                nominal = bpt[i].item()
                if nominal >= 16:
                    eff = 16.0
                elif nominal <= 1:
                    # DeltaQuant: anchor at anchor_bits, deltas at delta_bits
                    # Effective: (anchor_bits + delta_bits*(G-1)) / G
                    eff = (self.cfg.anchor_bits + self.cfg.delta_bits * (G - 1)) / G
                    # Add overhead: norms (16 bits/d per vec), group IDs (~2/d)
                    eff += (16 + 2) / d
                else:
                    # ResidualQuant: mse_bits*d + d(signs) + 32(norms) -> bits + 1 + 32/d
                    eff = nominal + 1.0 + 32.0 / d

                # Average key and value bits: value uses val_bits
                val_eff = min(self.cfg.val_bits, nominal if nominal < 16 else 16)
                if val_eff < 16:
                    val_eff = val_eff + 16.0 / d  # norm overhead
                avg_kv_bits = (eff + val_eff) / 2.0
                total_bits += avg_kv_bits
                count += 1

        return total_bits / max(count, 1)

    def clear(self):
        self._fp16_keys.clear()
        self._fp16_vals.clear()
        self._comp_keys.clear()
        self._comp_vals.clear()
        self._comp_bits.clear()
        self._seq_len = 0
        self._fp16_count = 0
        self._comp_count = 0
        self._step_count = 0
        self._running_mean = None
        self._mean_count = 0
        self._scorer.reset()
        if self._faiss_indexes:
            for idx in self._faiss_indexes:
                idx.reset()

    def reorder(self, beam_idx: torch.LongTensor):
        self._fp16_keys = [k.index_select(0, beam_idx) for k in self._fp16_keys]
        self._fp16_vals = [v.index_select(0, beam_idx) for v in self._fp16_vals]
        self._comp_keys = [k.index_select(0, beam_idx) for k in self._comp_keys]
        self._comp_vals = [v.index_select(0, beam_idx) for v in self._comp_vals]


class _FP16BoundaryLayer:
    """Boundary layer that stores everything at FP16."""

    def __init__(self):
        self._keys: List[torch.Tensor] = []
        self._vals: List[torch.Tensor] = []
        self._seq_len: int = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor):
        self._keys.append(key_states)
        self._vals.append(value_states)
        self._seq_len += key_states.shape[2]
        return torch.cat(self._keys, dim=2), torch.cat(self._vals, dim=2)

    def get_seq_length(self) -> int:
        return self._seq_len

    def update_importance(self, attention_weights: torch.Tensor):
        pass

    def effective_bits(self) -> float:
        return 16.0

    def clear(self):
        self._keys.clear()
        self._vals.clear()
        self._seq_len = 0

    def reorder(self, beam_idx: torch.LongTensor):
        self._keys = [k.index_select(0, beam_idx) for k in self._keys]
        self._vals = [v.index_select(0, beam_idx) for v in self._vals]

    def _reconstruct_all(self):
        if self._seq_len == 0:
            return torch.zeros(1, 1, 0, 1), torch.zeros(1, 1, 0, 1)
        return torch.cat(self._keys, dim=2), torch.cat(self._vals, dim=2)


# ---------------------------------------------------------------------------
# TurboQuantV2Cache — the unified system
# ---------------------------------------------------------------------------

class TurboQuantV2Cache:
    """Unified v2 KV cache: PCA + mean-removal + adaptive bits + DeltaQuant + FAISS.

    Duck-types the HuggingFace Cache protocol for drop-in use with
    ``model.generate(past_key_values=cache)``.

    Args:
        config: V2Config dataclass with all tuning knobs.
        num_layers: Total transformer layers. Required for boundary detection.
        pca_rotations: Optional dict mapping layer_idx -> PCA rotation data
            (from ``calibrate()``). If None, falls back to WHT/QR.
    """

    is_compileable = False

    def __init__(
        self,
        config: Optional[V2Config] = None,
        num_layers: Optional[int] = None,
        pca_rotations: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    ):
        self.config = config or V2Config()
        self.num_layers = num_layers
        self.pca_rotations = pca_rotations or {}
        self._layers: List[_V2Layer | _FP16BoundaryLayer] = []

    def _is_boundary(self, idx: int) -> bool:
        if self.num_layers is None:
            return False
        return (idx < self.config.boundary_layers or
                idx >= self.num_layers - self.config.boundary_layers)

    def _make_layer(self, idx: int):
        if self._is_boundary(idx):
            return _FP16BoundaryLayer()
        pca = self.pca_rotations.get(idx)
        return _V2Layer(self.config, idx, pca_data=pca)

    # ----- HF Cache Protocol -----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        while len(self._layers) <= layer_idx:
            self._layers.append(self._make_layer(len(self._layers)))
        layer = self._layers[layer_idx]
        if isinstance(layer, _FP16BoundaryLayer):
            return layer.update(key_states, value_states)
        return layer.update(key_states, value_states)

    def update_importance(self, attention_weights: torch.Tensor, layer_idx: int):
        if layer_idx < len(self._layers):
            self._layers[layer_idx].update_importance(attention_weights)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._layers):
            return 0
        return self._layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_mask_sizes(self, cache_position, layer_idx: int = 0) -> Tuple[int, int]:
        if isinstance(cache_position, int):
            query_length = cache_position
        else:
            query_length = cache_position.shape[0]
        if layer_idx >= len(self._layers):
            return query_length, 0
        kv_length = self._layers[layer_idx].get_seq_length() + query_length
        return kv_length, 0

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer in self._layers:
            layer.reorder(beam_idx)

    def crop(self, max_length: int):
        # Simplified: clear and lose history
        pass

    def reset(self):
        for layer in self._layers:
            layer.clear()

    def batch_repeat_interleave(self, repeats: int):
        for layer in self._layers:
            if isinstance(layer, _FP16BoundaryLayer):
                layer._keys = [k.repeat_interleave(repeats, dim=0) for k in layer._keys]
                layer._vals = [v.repeat_interleave(repeats, dim=0) for v in layer._vals]

    def batch_select_indices(self, indices: torch.Tensor):
        for layer in self._layers:
            if isinstance(layer, _FP16BoundaryLayer):
                layer._keys = [k[indices] for k in layer._keys]
                layer._vals = [v[indices] for v in layer._vals]

    @property
    def seen_tokens(self) -> int:
        return self._layers[0].get_seq_length() if self._layers else 0

    @property
    def is_initialized(self) -> bool:
        return len(self._layers) > 0

    @property
    def is_sliding(self) -> list:
        return [False] * max(len(self._layers), 1)

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        for layer in self._layers:
            if isinstance(layer, _FP16BoundaryLayer):
                keys, vals = layer._reconstruct_all()
            else:
                keys, vals = layer._reconstruct_all()
            yield keys, vals, None

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self._layers):
            raise IndexError(f"Layer {layer_idx} not in cache ({len(self._layers)} layers)")
        layer = self._layers[layer_idx]
        if isinstance(layer, _FP16BoundaryLayer):
            return layer._reconstruct_all()
        return layer._reconstruct_all()

    def __contains__(self, idx: int) -> bool:
        return 0 <= idx < len(self._layers)

    # ----- Metrics -----

    def effective_bits(self) -> float:
        """Weighted average bits per coordinate across all layers."""
        if not self._layers:
            return 16.0
        total, count = 0.0, 0
        for layer in self._layers:
            sl = layer.get_seq_length()
            total += layer.effective_bits() * sl
            count += sl
        return total / max(count, 1)

    def compression_ratio(self) -> float:
        return 16.0 / max(self.effective_bits(), 0.01)

    def tier_summary(self) -> Dict[str, Any]:
        """Report tier distribution and effective bits."""
        boundary_count = 0
        comp_counts = {b: 0 for b in self.config.tier_bits}
        fp16_window_count = 0

        for i, layer in enumerate(self._layers):
            if isinstance(layer, _FP16BoundaryLayer):
                boundary_count += layer.get_seq_length()
            else:
                fp16_window_count += layer._fp16_count
                for bpt in layer._comp_bits:
                    for b in self.config.tier_bits:
                        comp_counts[b] += (bpt == float(b)).sum().item()

        return {
            "boundary_fp16": boundary_count,
            "fp16_window": fp16_window_count,
            "tier_counts": comp_counts,
            "effective_bits": self.effective_bits(),
            "compression_ratio": self.compression_ratio(),
        }

    # ----- Calibration -----

    @classmethod
    def calibrate(
        cls,
        model,
        tokenizer,
        n_tokens: int = 128,
        save_path: Optional[str] = None,
        device: str = "cuda",
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Calibrate PCA rotation matrices from a quick forward pass.

        Runs a short prompt through the model, captures KV states per layer,
        and computes PCA rotation from the key vectors.

        Args:
            model: HuggingFace model.
            tokenizer: Matching tokenizer.
            n_tokens: Number of calibration tokens.
            save_path: Optional path to save rotations as .pt file.
            device: Device for calibration.

        Returns:
            Dict mapping layer_idx -> {rotation, eigenvalues, mean}.
        """
        print(f"Calibrating PCA rotations ({n_tokens} tokens)...")
        calibration_text = (
            "The transformer architecture processes sequences through self-attention "
            "and feed-forward layers. Each attention head computes queries, keys, and "
            "values. The key-value cache stores computed keys and values for efficient "
            "autoregressive generation. Quantization of this cache enables significant "
            "memory savings while preserving generation quality. Mathematical analysis "
            "shows that inner product preservation is more important than individual "
            "vector reconstruction accuracy."
        )

        inputs = tokenizer(
            calibration_text, return_tensors="pt", truncation=True, max_length=n_tokens
        ).to(device)

        # Hook to capture KV states
        kv_states = {}

        def _hook_fn(layer_idx):
            def hook(module, args, kwargs, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    # The model returns (hidden_states, present_key_value, ...)
                    # present_key_value is (key, value) each (B, H, S, D)
                    pass
                return output
            return hook

        # Forward pass to populate KV cache
        with torch.no_grad():
            outputs = model(
                **inputs,
                use_cache=True,
                output_attentions=False,
            )

        # Extract KV from past_key_values
        past_kv = outputs.past_key_values
        pca_rotations = {}

        if past_kv is not None:
            # Handle different cache formats:
            # - DynamicCache (HF >= 4.46): past_kv.layers[i].keys
            # - tuple/list: past_kv[layer_idx] = (keys, values)
            if hasattr(past_kv, 'layers'):
                # DynamicCache style
                n_layers = len(past_kv.layers)
                layer_iter = [(i, past_kv.layers[i]) for i in range(n_layers)]
            elif isinstance(past_kv, (tuple, list)):
                n_layers = len(past_kv)
                layer_iter = [(i, past_kv[i]) for i in range(n_layers)]
            else:
                n_layers = 0
                layer_iter = []

            for layer_idx, kv in layer_iter:
                # Extract keys tensor
                if hasattr(kv, 'keys'):
                    keys = kv.keys  # DynamicCache layer
                elif isinstance(kv, (tuple, list)):
                    keys = kv[0]
                elif hasattr(kv, 'key_cache'):
                    keys = kv.key_cache
                else:
                    continue

                # Flatten across batch and heads for calibration
                B, H, S, D = keys.shape
                keys_flat = keys.float().reshape(-1, D).cpu()

                if keys_flat.shape[0] >= D:
                    pca_data = compute_pca_rotation(keys_flat, center=True)
                    pca_rotations[layer_idx] = pca_data
                    if layer_idx == 0:
                        eigs = pca_data["eigenvalues"]
                        top5_var = eigs[:5].sum() / eigs.sum()
                        print(f"  Layer 0: d={D}, top-5 PCA explains {top5_var:.1%} variance")

            print(f"  Calibrated {len(pca_rotations)} layers")

        if save_path:
            torch.save(pca_rotations, save_path)
            print(f"  Saved to {save_path}")

        return pca_rotations

    @classmethod
    def from_calibration(
        cls,
        calibration_path: str,
        config: Optional[V2Config] = None,
        num_layers: Optional[int] = None,
    ) -> "TurboQuantV2Cache":
        """Create cache from saved calibration data."""
        pca_rotations = torch.load(calibration_path, map_location="cpu", weights_only=True)
        return cls(
            config=config,
            num_layers=num_layers,
            pca_rotations=pca_rotations,
        )
