"""FAISS-based retrieval KV cache for O(log n) approximate attention.

Previous LSH implementation achieved only 62% top-1 recall. FAISS IVF-Flat
and IVF-PQ provide much better recall with tunable nprobe parameter.

Architecture:
    - One FAISS index per layer per KV head (not per query head — GQA shares)
    - Keys stored in FAISS index in FP32 for search accuracy
    - Values stored in a flat buffer (FP16 or compressed)
    - On query: search FAISS for top-k keys, fetch corresponding values
    - Combine with recent window tokens for safety
    - Compute standard scaled dot-product attention over the small set

Index types:
    - Flat: exact inner product (oracle, still O(n) but with FAISS BLAS)
    - IVFFlat: inverted file with exact storage (O(sqrt(n)) with nprobe)
    - IVFPQ: inverted file with product quantization (~4 bytes/key)

The critical insight: FAISS indexes are designed for MIPS (Maximum Inner
Product Search), which is exactly what attention needs (q @ k^T ranking).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class FAISSRetrievalResult:
    """Result from a FAISS retrieval attention computation."""
    # Attention output: (n_queries, head_dim)
    output: torch.Tensor
    # Indices of retrieved+window tokens per query: list of tensors
    retrieved_indices: List[torch.Tensor]
    # Number of tokens attended to (max across queries)
    k_effective: int
    # FAISS search time in ms
    search_time_ms: float = 0.0
    # Attention compute time in ms
    attn_time_ms: float = 0.0


@dataclass
class FAISSQualityMetrics:
    """Quality comparison between FAISS retrieval and full attention."""
    top1_match: float = 0.0
    top5_match: float = 0.0
    top10_match: float = 0.0
    output_cosine_sim: float = 0.0
    recall_at_k: float = 0.0
    k_effective: float = 0.0
    search_time_ms: float = 0.0
    attn_time_ms: float = 0.0
    total_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# FAISS Index Wrapper
# ---------------------------------------------------------------------------
class FAISSIndex:
    """Wrapper around FAISS index for key retrieval.

    Supports three index types:
        - "flat": IndexFlatIP (exact, O(n), BLAS-optimized)
        - "ivf_flat": IndexIVFFlat (approximate, O(sqrt(n)))
        - "ivf_pq": IndexIVFPQ (approximate, compressed, O(sqrt(n)))

    All use METRIC_INNER_PRODUCT for attention-compatible search.

    Args:
        dim: Key vector dimension (e.g. 128).
        index_type: One of "flat", "ivf_flat", "ivf_pq".
        nlist: Number of IVF clusters (only for ivf_* types).
        nprobe: Number of clusters to search (only for ivf_* types).
        m_subquantizers: Number of PQ subquantizers (only for ivf_pq).
        nbits_per_code: Bits per PQ code (only for ivf_pq).
    """

    def __init__(
        self,
        dim: int = 128,
        index_type: Literal["flat", "ivf_flat", "ivf_pq"] = "ivf_flat",
        nlist: int = 64,
        nprobe: int = 8,
        m_subquantizers: int = 16,
        nbits_per_code: int = 8,
    ):
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not installed. Run: pip install faiss-cpu or faiss-gpu")

        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.m_subquantizers = m_subquantizers
        self.nbits_per_code = nbits_per_code

        self.index: Optional[faiss.Index] = None
        self.is_trained = False
        self.n_vectors = 0

    def _create_index(self, n_train: int) -> faiss.Index:
        """Create the FAISS index based on the configured type."""
        # Adjust nlist if we don't have enough training data
        effective_nlist = min(self.nlist, max(1, n_train // 39))

        if self.index_type == "flat":
            return faiss.IndexFlatIP(self.dim)

        elif self.index_type == "ivf_flat":
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(
                quantizer, self.dim, effective_nlist,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.nprobe = self.nprobe
            return index

        elif self.index_type == "ivf_pq":
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFPQ(
                quantizer, self.dim, effective_nlist,
                self.m_subquantizers, self.nbits_per_code,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.nprobe = self.nprobe
            return index

        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

    def build(self, keys: np.ndarray):
        """Build the index from a batch of key vectors.

        Args:
            keys: (n, dim) float32 array of key vectors.
        """
        assert keys.dtype == np.float32, f"Keys must be float32, got {keys.dtype}"
        assert keys.shape[1] == self.dim, f"Key dim {keys.shape[1]} != {self.dim}"

        n = keys.shape[0]
        self.index = self._create_index(n)

        if self.index_type != "flat":
            self.index.train(keys)

        self.index.add(keys)
        self.is_trained = True
        self.n_vectors = n

    def add(self, keys: np.ndarray):
        """Add new keys to an existing index.

        For IVF indexes, the index must already be trained.
        """
        if self.index is None:
            self.build(keys)
            return

        self.index.add(keys)
        self.n_vectors += keys.shape[0]

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-k nearest keys by inner product.

        Args:
            queries: (n_queries, dim) float32 array.
            k: Number of neighbors to retrieve.

        Returns:
            (distances, indices): each (n_queries, k) arrays.
        """
        assert self.index is not None, "Index not built. Call build() first."
        k = min(k, self.n_vectors)
        return self.index.search(queries, k)

    def set_nprobe(self, nprobe: int):
        """Update the number of probes for IVF indexes."""
        self.nprobe = nprobe
        if self.index is not None and hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe

    @property
    def memory_bytes(self) -> int:
        """Estimate memory usage of the index."""
        if self.n_vectors == 0:
            return 0
        if self.index_type == "flat":
            return self.n_vectors * self.dim * 4  # float32
        elif self.index_type == "ivf_flat":
            return self.n_vectors * self.dim * 4 + self.nlist * self.dim * 4
        elif self.index_type == "ivf_pq":
            # PQ codes: n * m_sub * nbits/8 + centroids
            code_bytes = self.n_vectors * self.m_subquantizers * self.nbits_per_code // 8
            centroid_bytes = self.nlist * self.dim * 4
            pq_table_bytes = self.m_subquantizers * (2 ** self.nbits_per_code) * (self.dim // self.m_subquantizers) * 4
            return code_bytes + centroid_bytes + pq_table_bytes
        return 0


# ---------------------------------------------------------------------------
# Retrieval KV Cache
# ---------------------------------------------------------------------------
class RetrievalKVCache:
    """FAISS-backed KV cache with O(log n) retrieval attention.

    For each layer and KV head, maintains:
        - A FAISS index over all keys (for fast top-k search)
        - A flat buffer of all values (for gathering after search)
        - A window of recent key-value pairs (always included in attention)

    On update(): adds new key-value pairs to the index and buffers.
    On retrieve(): searches for top-k keys per query, gathers values,
                   concatenates with window, computes attention.

    Args:
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV heads per layer.
        head_dim: Dimension per head.
        index_type: FAISS index type.
        nlist: Number of IVF clusters.
        nprobe: Default search probe count.
        window_size: Recent tokens always included in attention.
        k: Number of tokens to retrieve from index.
    """

    def __init__(
        self,
        num_layers: int = 36,
        num_kv_heads: int = 2,
        head_dim: int = 128,
        index_type: Literal["flat", "ivf_flat", "ivf_pq"] = "ivf_flat",
        nlist: int = 64,
        nprobe: int = 8,
        window_size: int = 64,
        k: int = 128,
        m_subquantizers: int = 16,
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

        # Per-layer, per-head storage
        # indexes[layer][head] = FAISSIndex
        self.indexes: List[List[Optional[FAISSIndex]]] = [
            [None for _ in range(num_kv_heads)]
            for _ in range(num_layers)
        ]
        # key_buffers[layer][head] = list of key tensors (for rebuilding)
        self.key_buffers: List[List[List[torch.Tensor]]] = [
            [[] for _ in range(num_kv_heads)]
            for _ in range(num_layers)
        ]
        # value_buffers[layer][head] = list of value tensors
        self.value_buffers: List[List[List[torch.Tensor]]] = [
            [[] for _ in range(num_kv_heads)]
            for _ in range(num_layers)
        ]
        self.seq_lengths: List[int] = [0] * num_layers

    def build_from_tensors(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """Build index from pre-existing key-value tensors.

        Args:
            layer_idx: Layer index.
            keys: (num_kv_heads, seq_len, head_dim) or (seq_len, head_dim) for single head.
            values: Same shape as keys.
        """
        if keys.dim() == 2:
            keys = keys.unsqueeze(0)
            values = values.unsqueeze(0)

        num_heads = keys.shape[0]
        seq_len = keys.shape[1]

        for h in range(min(num_heads, self.num_kv_heads)):
            k_np = keys[h].float().cpu().numpy().astype(np.float32)

            idx = FAISSIndex(
                dim=self.head_dim,
                index_type=self.index_type,
                nlist=self.nlist,
                nprobe=self.nprobe,
                m_subquantizers=self.m_subquantizers,
            )
            idx.build(k_np)
            self.indexes[layer_idx][h] = idx
            self.key_buffers[layer_idx][h] = [keys[h]]
            self.value_buffers[layer_idx][h] = [values[h]]

        self.seq_lengths[layer_idx] = seq_len

    def set_nprobe(self, nprobe: int):
        """Update nprobe for all indexes."""
        self.nprobe = nprobe
        for layer_idxs in self.indexes:
            for idx in layer_idxs:
                if idx is not None:
                    idx.set_nprobe(nprobe)

    def retrieve_and_attend(
        self,
        layer_idx: int,
        head_idx: int,
        queries: torch.Tensor,
        k: Optional[int] = None,
        scale: Optional[float] = None,
    ) -> FAISSRetrievalResult:
        """Retrieve top-k keys and compute attention.

        Args:
            layer_idx: Layer index.
            head_idx: KV head index (for GQA, map query head to KV head).
            queries: (n_queries, head_dim) query vectors.
            k: Override default k.
            scale: Attention scale (default: 1/sqrt(d)).

        Returns:
            FAISSRetrievalResult with output and metadata.
        """
        k = k or self.k
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        idx = self.indexes[layer_idx][head_idx]
        assert idx is not None, f"No index built for layer {layer_idx} head {head_idx}"

        # Get full key and value tensors
        all_keys = torch.cat(self.key_buffers[layer_idx][head_idx], dim=0)
        all_values = torch.cat(self.value_buffers[layer_idx][head_idx], dim=0)
        seq_len = all_keys.shape[0]
        device = queries.device

        # FAISS search
        q_np = queries.float().cpu().numpy().astype(np.float32)
        t0 = time.perf_counter()
        _, I = idx.search(q_np, min(k, seq_len))
        search_time = (time.perf_counter() - t0) * 1000

        # Window indices
        window_start = max(0, seq_len - self.window_size)
        window_indices = torch.arange(window_start, seq_len, device=device)

        # Compute attention per query
        n_queries = queries.shape[0]
        all_outputs = []
        all_idx_lists = []

        t1 = time.perf_counter()
        for qi in range(n_queries):
            # Merge FAISS results with window
            faiss_indices = torch.from_numpy(I[qi]).long().to(device)
            # Filter out -1 (FAISS returns -1 for missing results)
            faiss_indices = faiss_indices[faiss_indices >= 0]

            combined = torch.cat([faiss_indices, window_indices])
            unique_indices = torch.unique(combined)
            unique_indices, _ = torch.sort(unique_indices)

            # Gather and compute attention
            sel_keys = all_keys[unique_indices].to(device)
            sel_values = all_values[unique_indices].to(device)

            q = queries[qi:qi+1]
            scores = (q @ sel_keys.T) * scale
            weights = F.softmax(scores, dim=-1)
            output = weights @ sel_values

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


# ---------------------------------------------------------------------------
# Standalone retrieval attention function
# ---------------------------------------------------------------------------
def retrieval_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    k: int = 128,
    window: int = 64,
    index_type: str = "ivf_flat",
    nlist: int = 64,
    nprobe: int = 8,
    scale: Optional[float] = None,
    m_subquantizers: int = 16,
) -> FAISSRetrievalResult:
    """One-shot FAISS retrieval attention.

    Builds a FAISS index on the fly, retrieves top-k keys per query,
    concatenates with the recent window, and computes attention.

    Args:
        queries: (n_queries, head_dim)
        keys: (seq_len, head_dim)
        values: (seq_len, head_dim)
        k: Number of keys to retrieve from FAISS.
        window: Number of recent tokens to always include.
        index_type: "flat", "ivf_flat", or "ivf_pq".
        nlist: Number of IVF clusters.
        nprobe: Number of clusters to search.
        scale: Attention scale (default: 1/sqrt(d)).
        m_subquantizers: PQ subquantizers (for ivf_pq).

    Returns:
        FAISSRetrievalResult
    """
    head_dim = keys.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    seq_len = keys.shape[0]
    device = queries.device

    # Build FAISS index
    k_np = keys.float().cpu().numpy().astype(np.float32)

    idx = FAISSIndex(
        dim=head_dim,
        index_type=index_type,
        nlist=nlist,
        nprobe=nprobe,
        m_subquantizers=m_subquantizers,
    )
    idx.build(k_np)

    # Search
    q_np = queries.float().cpu().numpy().astype(np.float32)
    t0 = time.perf_counter()
    _, I = idx.search(q_np, min(k, seq_len))
    search_time = (time.perf_counter() - t0) * 1000

    # Window
    window_start = max(0, seq_len - window)
    window_indices = torch.arange(window_start, seq_len, device=device)

    # Attention per query
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

        sel_keys = keys[unique_indices]
        sel_values = values[unique_indices]

        q = queries[qi:qi+1]
        scores = (q @ sel_keys.T) * scale
        weights = F.softmax(scores, dim=-1)
        output = weights @ sel_values

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


# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------
def compute_full_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard full attention (ground truth)."""
    head_dim = keys.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    scores = (queries @ keys.T) * scale
    weights = F.softmax(scores, dim=-1)
    output = weights @ values
    return output, weights


def evaluate_faiss_quality(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    k: int = 128,
    window: int = 64,
    index_type: str = "ivf_flat",
    nlist: int = 64,
    nprobe: int = 8,
    scale: Optional[float] = None,
    m_subquantizers: int = 16,
) -> FAISSQualityMetrics:
    """Compare FAISS retrieval attention against full attention."""
    head_dim = keys.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Ground truth
    full_output, full_weights = compute_full_attention(queries, keys, values, scale)

    # FAISS retrieval
    result = retrieval_attention(
        queries, keys, values,
        k=k, window=window,
        index_type=index_type, nlist=nlist, nprobe=nprobe,
        scale=scale, m_subquantizers=m_subquantizers,
    )

    n_queries = queries.shape[0]
    seq_len = keys.shape[0]

    top1_matches = 0
    top5_matches = 0
    top10_matches = 0
    recall_sum = 0.0

    for qi in range(n_queries):
        fw = full_weights[qi]

        full_top1 = torch.topk(fw, k=min(1, seq_len)).indices
        full_top5 = set(torch.topk(fw, k=min(5, seq_len)).indices.tolist())
        full_top10 = set(torch.topk(fw, k=min(10, seq_len)).indices.tolist())

        ret_set = set(result.retrieved_indices[qi].tolist())

        if full_top1[0].item() in ret_set:
            top1_matches += 1
        top5_matches += len(full_top5 & ret_set) / max(len(full_top5), 1)
        top10_matches += len(full_top10 & ret_set) / max(len(full_top10), 1)

        significant = set((fw > 0.01).nonzero(as_tuple=True)[0].tolist())
        if len(significant) > 0:
            recall_sum += len(significant & ret_set) / len(significant)
        else:
            recall_sum += 1.0

    cos_sim = F.cosine_similarity(
        full_output, result.output, dim=-1
    ).mean().item()

    return FAISSQualityMetrics(
        top1_match=top1_matches / n_queries,
        top5_match=top5_matches / n_queries,
        top10_match=top10_matches / n_queries,
        output_cosine_sim=cos_sim,
        recall_at_k=recall_sum / n_queries,
        k_effective=result.k_effective,
        search_time_ms=result.search_time_ms,
        attn_time_ms=result.attn_time_ms,
        total_time_ms=result.search_time_ms + result.attn_time_ms,
    )
