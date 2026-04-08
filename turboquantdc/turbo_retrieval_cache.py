from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from transformers.cache_utils import Cache
from .retrieval_cache import FAISSIndex, FAISSRetrievalResult
from .generation_layers import _CompressedLayer

class TurboRetrievalCache(Cache):
    """FAISS-backed KV cache that stores values in TurboQuant 2/3-bit format.
    
    This merges the O(log n) approximate attention of RetrievalKVCache with
    the massive VRAM savings of GenerationCache.
    """

    def __init__(
        self,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        index_type: Literal["flat", "ivf_flat", "ivf_pq"] = "ivf_pq",
        nlist: int = 64,
        nprobe: int = 8,
        window_size: int = 64,
        k: int = 128,
        m_subquantizers: int = 16,
        key_bits: int = 3,
        val_bits: int = 2,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.window_size = window_size
        self.k = k
        self.m_subquantizers = m_subquantizers
        self.key_bits = key_bits
        self.val_bits = val_bits

        # FAISS indexes: one per layer per KV head
        self.indexes: List[List[Optional[FAISSIndex]]] = [
            [None for _ in range(num_kv_heads)]
            for _ in range(num_layers)
        ]
        
        # TurboQuant Cache: one CompressedLayer per transformer layer
        self.layers: List[_CompressedLayer] = []
        for i in range(num_layers):
            self.layers.append(_CompressedLayer(
                key_bits=key_bits,
                val_bits=val_bits,
                fp16_window=window_size,  # match recent window
                use_residual_quant=True,
                use_norm_correction=True,
            ))

    @property
    def is_compileable(self) -> bool:
        return False

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add new tokens to FAISS and TurboQuant layer."""
        # 1. Compress without full dequantization.
        # TurboRetrievalCache uses dequantize_selected() (on-demand) rather than
        # the full dequantized cache that _CompressedLayer.update() builds.
        self.layers[layer_idx].compress_only(key_states, value_states)

        # 2. Add to FAISS index (assuming batch=1 for simplicity)
        num_heads = key_states.shape[1]
        for h in range(min(num_heads, self.num_kv_heads)):
            k_np = key_states[0, h].float().cpu().numpy().astype(np.float32)

            if self.indexes[layer_idx][h] is None:
                # Use flat index for small contexts, IVF for large
                n_vectors = k_np.shape[0]
                effective_type = "flat" if n_vectors < self.nlist * 4 else self.index_type
                idx = FAISSIndex(
                    dim=self.head_dim,
                    index_type=effective_type,
                    nlist=min(self.nlist, max(1, n_vectors // 4)),
                    nprobe=self.nprobe,
                    m_subquantizers=self.m_subquantizers,
                )
                idx.build(k_np)
                self.indexes[layer_idx][h] = idx
            else:
                self.indexes[layer_idx][h].add(k_np)

        # 3. During prefill, return a sliding window of the ORIGINAL (uncompressed)
        # key/value states. This avoids expensive dequantize_selected calls during
        # prefill. The compressed indices are stored on CPU for decode-time retrieval.
        seq_len = self.layers[layer_idx].get_seq_length()
        window_size = 2048
        new_seq = key_states.shape[2]

        if new_seq > 1:
            # PREFILL: Use original key/value states (already on GPU, no decompression needed)
            # Combine with previously stored FP16 window if needed
            layer = self.layers[layer_idx]
            if layer._raw_keys:
                raw_k = torch.cat(layer._raw_keys, dim=2)
                raw_v = torch.cat(layer._raw_vals, dim=2)
                win = min(window_size, raw_k.shape[2])
                return raw_k[:, :, -win:, :], raw_v[:, :, -win:, :]
            return key_states, value_states
        else:
            # DECODE (seq=1): Use dequantize_selected for the sliding window
            start_idx = max(0, seq_len - window_size)
            indices = torch.arange(start_idx, seq_len, device=key_states.device)
            return self.layers[layer_idx].dequantize_selected(indices)

    def retrieve_and_attend(
        self,
        layer_idx: int,
        head_idx: int,
        queries: torch.Tensor,
        k: Optional[int] = None,
        scale: Optional[float] = None,
    ) -> FAISSRetrievalResult:
        """Fetch subset from FAISS, decompress strictly that subset, and attend."""
        k = k or self.k
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        idx = self.indexes[layer_idx][head_idx]
        assert idx is not None, f"No index built for layer {layer_idx} head {head_idx}"

        seq_len = self.layers[layer_idx].get_seq_length()
        device = queries.device
        
        # FAISS search
        q_np = queries.float().cpu().numpy().astype(np.float32)
        t0 = time.perf_counter()
        distances, I = idx.search(q_np, min(k, seq_len))
        search_time = (time.perf_counter() - t0) * 1000

        window_start = max(0, seq_len - self.window_size)
        window_indices = torch.arange(window_start, seq_len, device=device)

        n_queries = queries.shape[0]
        all_outputs = []
        all_idx_lists = []

        t1 = time.perf_counter()
        for qi in range(n_queries):
            faiss_indices = torch.from_numpy(I[qi]).long().to(device)
            faiss_indices = faiss_indices[faiss_indices >= 0]

            combined = torch.cat([faiss_indices, window_indices])
            unique_indices = torch.unique(combined)
            unique_indices, _ = torch.sort(unique_indices)

            # --- TURBO QUANT DEQUANTIZE SELECTED ---
            # Returns [batch, num_heads, len(unique_indices), head_dim]
            sel_keys, sel_vals = self.layers[layer_idx].dequantize_selected(unique_indices)
            
            # Select specific head
            sel_k_head = sel_keys[0, head_idx]
            sel_v_head = sel_vals[0, head_idx]

            q = queries[qi:qi+1]
            scores = (q @ sel_k_head.T) * scale
            weights = F.softmax(scores, dim=-1)
            output = weights @ sel_v_head

            all_outputs.append(output.squeeze(0))
            all_idx_lists.append(unique_indices)

        attn_time = (time.perf_counter() - t1) * 1000
        outputs = torch.stack(all_outputs)
        k_effective = max(idx_t.shape[0] for idx_t in all_idx_lists)

        return FAISSRetrievalResult(
            output=outputs,
            retrieved_indices=all_idx_lists,
            k_effective=k_effective,
            search_time_ms=search_time,
            attn_time_ms=attn_time,
        )

    # Required HF methods
    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers): return 0
        return self.layers[layer_idx].get_seq_length()
        
    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_mask_sizes(self, cache_position, layer_idx: int = 0) -> tuple[int, int]:
        if isinstance(cache_position, int):
            query_length = cache_position
        else:
            query_length = cache_position.shape[0]
        if layer_idx >= len(self.layers):
            return query_length, 0
        kv_length = self.layers[layer_idx].get_seq_length() + query_length
        return kv_length, 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        pass
        
    def __len__(self) -> int:
        return self.num_layers
