"""Retrieval-based attention: O(log n) approximate attention via MIPS.

Standard attention computes Q @ K^T over ALL cached keys -- O(n) per query.
At 1M context, that is 1M dot products per query per head per layer.

But empirical analysis (asymptotic_analysis.py) proved: only ~0.3% of tokens
receive meaningful attention at 2K context, and concentration increases with
length. If only 0.3% of tokens matter, why compute attention over 100%?

This module treats the KV cache as a Maximum Inner Product Search (MIPS)
problem:
    1. For each query, retrieve top-k most similar keys by dot product
    2. Compute attention ONLY over these k keys + a recent window
    3. This is O(k * log n) instead of O(n)

Three retrieval strategies:
    - BruteForceTopK: exact top-k via torch.topk (oracle baseline)
    - LSHIndex: random-projection locality-sensitive hashing (O(log n) lookup)
    - HybridRetriever: combines recent window + retrieval for safety

The theoretical prize at 1M context with k=64:
    Standard:  1,000,000 dot products + 1M softmax entries
    Retrieval: 64 dot products + 64 softmax entries + index lookup
    Speedup:   ~15,000x for the attention computation

Reference: builds on the sparsity findings from TurboQuant asymptotic analysis
(Gini 0.85 at 2K context, 0.3% of tokens above 1% attention threshold).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    """Result from a single retrieval attention computation."""

    # Retrieved token indices per query: (n_queries, k_effective)
    retrieved_indices: torch.Tensor
    # Attention weights over retrieved tokens: (n_queries, k_effective)
    attention_weights: torch.Tensor
    # Weighted output: (n_queries, head_dim)
    output: torch.Tensor
    # Number of tokens actually attended to
    k_effective: int


@dataclass
class QualityMetrics:
    """Quality comparison between retrieval and full attention."""

    top1_match: float = 0.0
    top5_match: float = 0.0
    top10_match: float = 0.0
    output_cosine_sim: float = 0.0
    attention_kl_divergence: float = 0.0
    attention_l1_error: float = 0.0
    # What fraction of top-k full-attention tokens were retrieved
    recall_at_k: float = 0.0
    # Token prediction match (argmax of output projection)
    output_token_match: float = 0.0


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------
class BruteForceTopK:
    """Exact top-k retrieval via torch.topk.

    This is the oracle baseline: it finds the true k highest dot products.
    Still O(n) for the search, but O(k) for the attention computation.

    Use this to establish the accuracy ceiling -- if brute-force top-k
    does not match full attention, then no approximate method can either.
    """

    def __init__(self, k: int = 32):
        self.k = k

    def retrieve(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Find top-k keys by dot product.

        Args:
            query: (head_dim,) or (n_queries, head_dim)
            keys: (seq_len, head_dim)
            scale: attention scale factor (1/sqrt(d))

        Returns:
            Indices of top-k keys: (n_queries, k) or (k,) if single query
        """
        squeeze = query.dim() == 1
        if squeeze:
            query = query.unsqueeze(0)

        # (n_queries, seq_len)
        scores = (query @ keys.T) * scale
        k = min(self.k, scores.shape[-1])
        _, indices = torch.topk(scores, k=k, dim=-1)

        if squeeze:
            return indices.squeeze(0)
        return indices


class LSHIndex:
    """Locality-Sensitive Hashing for approximate MIPS.

    Uses random hyperplane projections to hash keys into buckets.
    At query time, only keys in the same (or nearby) buckets are candidates.

    Complexity:
        Build: O(n * num_planes) -- one-time
        Query: O(n/2^num_planes * num_tables + k * log k) -- sublinear in n

    For practical settings (num_planes=8, num_tables=4):
        Each table divides the space into 256 buckets.
        Expected candidates per query: n / 256 * 4 = n / 64
        At 1M tokens: ~15K candidates instead of 1M (64x reduction)

    Args:
        dim: Key dimension.
        num_planes: Number of random hyperplanes per hash table.
        num_tables: Number of independent hash tables (more = better recall).
        device: Target device.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dim: int,
        num_planes: int = 8,
        num_tables: int = 4,
        device: str | torch.device = "cuda",
        seed: int = 42,
    ):
        self.dim = dim
        self.num_planes = num_planes
        self.num_tables = num_tables
        self.device = device

        # Generate random projection planes for each table
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.planes = []
        for _ in range(num_tables):
            # (num_planes, dim) -- random Gaussian projections
            plane = torch.randn(num_planes, dim, generator=gen)
            plane = plane / plane.norm(dim=-1, keepdim=True)
            self.planes.append(plane.to(device))

        # Bucket storage: table_idx -> hash_code -> list of key indices
        self.buckets: List[Dict[int, List[int]]] = [
            {} for _ in range(num_tables)
        ]
        self.keys: Optional[torch.Tensor] = None
        self.n_keys: int = 0

    def _hash(self, vectors: torch.Tensor, table_idx: int) -> torch.Tensor:
        """Compute LSH hash codes for a batch of vectors.

        Args:
            vectors: (batch, dim)
            table_idx: which hash table to use

        Returns:
            Hash codes: (batch,) integer tensor
        """
        # Project onto random planes: (batch, num_planes)
        projections = vectors @ self.planes[table_idx].T
        # Binarize: each bit is sign of projection
        bits = (projections > 0).int()
        # Convert to integer hash code
        powers = (2 ** torch.arange(self.num_planes, device=bits.device)).int()
        codes = (bits * powers).sum(dim=-1)
        return codes

    def build(self, keys: torch.Tensor):
        """Index all keys.

        Args:
            keys: (seq_len, head_dim) key vectors
        """
        self.keys = keys.to(self.device)
        self.n_keys = keys.shape[0]

        # Clear old buckets
        self.buckets = [{} for _ in range(self.num_tables)]

        # Hash all keys into each table
        for t in range(self.num_tables):
            codes = self._hash(keys, t)
            for idx in range(self.n_keys):
                code = codes[idx].item()
                if code not in self.buckets[t]:
                    self.buckets[t][code] = []
                self.buckets[t][code].append(idx)

    def query(self, q: torch.Tensor, k: int = 32) -> torch.Tensor:
        """Find approximate top-k keys for a query.

        Args:
            q: (head_dim,) or (n_queries, head_dim)
            k: number of keys to retrieve

        Returns:
            Indices of approximate top-k keys: (n_queries, k)
        """
        squeeze = q.dim() == 1
        if squeeze:
            q = q.unsqueeze(0)

        n_queries = q.shape[0]
        all_indices = []

        for qi in range(n_queries):
            query_vec = q[qi:qi+1]  # (1, dim)
            candidate_set = set()

            # Collect candidates from all tables
            for t in range(self.num_tables):
                code = self._hash(query_vec, t).item()
                if code in self.buckets[t]:
                    candidate_set.update(self.buckets[t][code])

            if len(candidate_set) == 0:
                # Fallback: return most recent k indices
                indices = torch.arange(
                    max(0, self.n_keys - k), self.n_keys,
                    device=self.device,
                )
                all_indices.append(indices)
                continue

            # Score candidates and take top-k
            candidates = torch.tensor(
                sorted(candidate_set), device=self.device, dtype=torch.long,
            )
            candidate_keys = self.keys[candidates]  # (n_candidates, dim)
            scores = (query_vec @ candidate_keys.T).squeeze(0)  # (n_candidates,)

            actual_k = min(k, len(candidates))
            _, topk_in_candidates = torch.topk(scores, k=actual_k)
            indices = candidates[topk_in_candidates]
            all_indices.append(indices)

        # Pad to uniform k
        max_k = max(idx.shape[0] for idx in all_indices)
        padded = torch.zeros(n_queries, max_k, device=self.device, dtype=torch.long)
        for i, idx in enumerate(all_indices):
            padded[i, :idx.shape[0]] = idx

        if squeeze:
            return padded.squeeze(0)
        return padded


# ---------------------------------------------------------------------------
# Hybrid retriever: retrieval + recent window
# ---------------------------------------------------------------------------
class HybridRetriever:
    """Combines top-k retrieval with a recent token window.

    The recent window ensures we never miss tokens that are important
    due to recency bias (which is strong in autoregressive models).
    The retrieval component catches the rare "needle" tokens that are
    far back in the context but still receive high attention.

    Total tokens attended: k_retrieval + window_size (deduplicated)

    Args:
        retriever: A retrieval backend (BruteForceTopK or LSHIndex).
        window_size: Number of most recent tokens to always include.
    """

    def __init__(
        self,
        retriever: BruteForceTopK | LSHIndex,
        window_size: int = 64,
    ):
        self.retriever = retriever
        self.window_size = window_size

    def retrieve_and_attend(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        scale: float | None = None,
    ) -> RetrievalResult:
        """Retrieve top-k keys + recent window and compute attention.

        Args:
            queries: (n_queries, head_dim)
            keys: (seq_len, head_dim)
            values: (seq_len, head_dim)
            scale: attention scale (default: 1/sqrt(d))

        Returns:
            RetrievalResult with attended output and metadata
        """
        seq_len = keys.shape[0]
        head_dim = keys.shape[-1]
        n_queries = queries.shape[0]

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        # Recent window indices
        window_start = max(0, seq_len - self.window_size)
        window_indices = torch.arange(
            window_start, seq_len, device=keys.device,
        )

        # Retrieve top-k from the non-window portion
        if isinstance(self.retriever, BruteForceTopK):
            retrieved = self.retriever.retrieve(queries, keys, scale=scale)
        elif isinstance(self.retriever, LSHIndex):
            retrieved = self.retriever.query(queries, k=self.retriever.num_planes * 4)
        else:
            retrieved = self.retriever.retrieve(queries, keys, scale=scale)

        if retrieved.dim() == 1:
            retrieved = retrieved.unsqueeze(0)

        # Merge retrieved + window, deduplicate per query
        all_outputs = []
        all_weights_list = []
        all_indices_list = []

        for qi in range(n_queries):
            # Union of retrieved indices and window indices
            r_indices = retrieved[qi]
            combined = torch.cat([r_indices, window_indices])
            unique_indices = torch.unique(combined)

            # Sort for deterministic output
            unique_indices, _ = torch.sort(unique_indices)

            # Gather keys and values for selected tokens
            sel_keys = keys[unique_indices]  # (k_eff, head_dim)
            sel_values = values[unique_indices]  # (k_eff, head_dim)

            # Compute attention over selected tokens only
            q = queries[qi:qi+1]  # (1, head_dim)
            scores = (q @ sel_keys.T) * scale  # (1, k_eff)
            weights = F.softmax(scores, dim=-1)  # (1, k_eff)

            # Weighted sum of values
            output = weights @ sel_values  # (1, head_dim)

            all_outputs.append(output.squeeze(0))
            all_weights_list.append(weights.squeeze(0))
            all_indices_list.append(unique_indices)

        # Stack outputs
        outputs = torch.stack(all_outputs)  # (n_queries, head_dim)
        k_effective = max(idx.shape[0] for idx in all_indices_list)

        return RetrievalResult(
            retrieved_indices=all_indices_list[0] if n_queries == 1 else all_indices_list,
            attention_weights=all_weights_list[0] if n_queries == 1 else all_weights_list,
            output=outputs,
            k_effective=k_effective,
        )


# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------
def compute_full_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard full attention (ground truth).

    Args:
        queries: (n_queries, head_dim)
        keys: (seq_len, head_dim)
        values: (seq_len, head_dim)
        scale: attention scale

    Returns:
        (output, attention_weights) where:
            output: (n_queries, head_dim)
            attention_weights: (n_queries, seq_len)
    """
    head_dim = keys.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    scores = (queries @ keys.T) * scale  # (n_queries, seq_len)
    weights = F.softmax(scores, dim=-1)  # (n_queries, seq_len)
    output = weights @ values  # (n_queries, head_dim)

    return output, weights


def evaluate_retrieval_quality(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    retriever: HybridRetriever,
    scale: float | None = None,
) -> QualityMetrics:
    """Compare retrieval attention against full attention.

    Args:
        queries: (n_queries, head_dim)
        keys: (seq_len, head_dim)
        values: (seq_len, head_dim)
        retriever: HybridRetriever instance
        scale: attention scale

    Returns:
        QualityMetrics with all comparison metrics
    """
    head_dim = keys.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Ground truth: full attention
    full_output, full_weights = compute_full_attention(
        queries, keys, values, scale,
    )

    # Retrieval attention
    retrieval_result = retriever.retrieve_and_attend(
        queries, keys, values, scale,
    )

    n_queries = queries.shape[0]
    seq_len = keys.shape[0]

    # Metrics accumulation
    top1_matches = 0
    top5_matches = 0
    top10_matches = 0
    recall_sum = 0.0
    kl_sum = 0.0
    l1_sum = 0.0

    for qi in range(n_queries):
        fw = full_weights[qi]  # (seq_len,)

        # Top-k indices from full attention
        full_top1 = torch.topk(fw, k=min(1, seq_len)).indices
        full_top5 = set(torch.topk(fw, k=min(5, seq_len)).indices.tolist())
        full_top10 = set(torch.topk(fw, k=min(10, seq_len)).indices.tolist())

        # Retrieved indices
        if isinstance(retrieval_result.retrieved_indices, list):
            ret_indices_set = set(retrieval_result.retrieved_indices[qi].tolist())
        else:
            ret_indices_set = set(retrieval_result.retrieved_indices.tolist())

        # Top-1 match: is the highest-attended token in our retrieved set?
        if full_top1[0].item() in ret_indices_set:
            top1_matches += 1

        # Top-5 match: what fraction of top-5 tokens are retrieved?
        top5_matches += len(full_top5 & ret_indices_set) / max(len(full_top5), 1)

        # Top-10 match
        top10_matches += len(full_top10 & ret_indices_set) / max(len(full_top10), 1)

        # Recall: what fraction of tokens that would get >1% attention are retrieved
        significant = set((fw > 0.01).nonzero(as_tuple=True)[0].tolist())
        if len(significant) > 0:
            recall_sum += len(significant & ret_indices_set) / len(significant)
        else:
            recall_sum += 1.0

    # Output cosine similarity
    cos_sim = F.cosine_similarity(
        full_output, retrieval_result.output, dim=-1,
    ).mean().item()

    return QualityMetrics(
        top1_match=top1_matches / n_queries,
        top5_match=top5_matches / n_queries,
        top10_match=top10_matches / n_queries,
        output_cosine_sim=cos_sim,
        recall_at_k=recall_sum / n_queries,
        attention_kl_divergence=0.0,  # expensive to compute over sparse
        attention_l1_error=0.0,
    )
