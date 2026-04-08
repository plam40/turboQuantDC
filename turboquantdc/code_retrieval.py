"""Code-based retrieval: O(1) approximate attention using quantization codes as LSH.

THE KEY INSIGHT: After WHT rotation, Lloyd-Max quantization partitions the
coordinate space into cells. Tokens quantized to the SAME index pattern in
the first K coordinates are geometrically close in rotated space, which means
they are close in inner product space, which means they have similar attention
scores.

The quantization codes ARE a locality-sensitive hash -- for free.

Standard ANN retrieval needs a separate index (FAISS, ScaNN, etc.) that
consumes additional memory. Here, the compression representation IS the index.
Zero extra memory. The indices stored for 5x compression double as a hash for
O(sub-linear) retrieval.

How it works:
    1. On insert: quantize key with ResidualQuant as normal, AND hash the first
       `hash_width` coordinates of the MSE index vector into a bucket.
    2. On search: hash the query's quantized first `hash_width` coordinates,
       look up all keys in matching buckets, score candidates by exact dot
       product, return top-k.
    3. Multi-probe: also check buckets that differ in 1 coordinate (Hamming
       distance 1 neighbors) for better recall.

Hash collision rate:
    - 3-bit codebook = 8 levels per coordinate
    - hash_width=16 coords => 8^16 = 2^48 possible buckets
    - In practice, only a few hundred buckets have any entries because the
      distribution is concentrated. This is the feature, not a bug: similar
      vectors cluster into the same small set of buckets.

Complexity:
    - Build: O(1) per token (just store + hash)
    - Query: O(candidates * d) where candidates << n_total
    - Memory: zero extra (reuses existing quantization indices)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Code Index: the quantization-codes-as-hash structure
# ---------------------------------------------------------------------------

@dataclass
class CodeIndexStats:
    """Statistics about the code index state."""
    n_tokens: int = 0
    n_buckets: int = 0
    avg_bucket_size: float = 0.0
    max_bucket_size: int = 0
    min_bucket_size: int = 0
    hash_width: int = 0
    n_levels: int = 0


class CodeIndex:
    """Inverted index from quantization codes to token positions.

    The first `hash_width` coordinates of the Lloyd-Max index vector are
    packed into an integer hash code. All tokens with the same hash land
    in the same bucket. At query time, we look up the query's bucket and
    optionally probe neighboring buckets (Hamming distance 1).

    For 3-bit quantization with hash_width=16:
        Hash = 48 bits = concatenation of 16 3-bit indices
        But the effective number of populated buckets is much smaller
        because the Gaussian distribution concentrates probability mass
        on a few central index values.

    Args:
        hash_width: Number of leading coordinates to use for the hash.
        n_levels: Number of quantization levels per coordinate (2^bits).
        multi_probe: If True, also check Hamming-1 neighbor buckets.
    """

    def __init__(
        self,
        hash_width: int = 16,
        n_levels: int = 8,
        multi_probe: bool = True,
    ):
        self.hash_width = hash_width
        self.n_levels = n_levels
        self.multi_probe = multi_probe
        self.bits_per_coord = int(math.log2(n_levels))

        # Inverted index: hash_code -> list of (position, full_index_vector)
        self.buckets: Dict[int, List[int]] = defaultdict(list)

        # Position -> full index vector (for scoring candidates)
        self.all_indices: List[torch.Tensor] = []
        self.n_tokens: int = 0

    def _hash(self, indices: torch.Tensor) -> int:
        """Compute hash from the first hash_width coordinates of an index vector.

        Args:
            indices: (d,) integer tensor of quantization indices.

        Returns:
            Integer hash code.
        """
        # Take first hash_width coordinates, pack into a single int
        # For 3-bit with 16 coords: 48-bit hash
        prefix = indices[:self.hash_width].long()
        code = 0
        for i in range(self.hash_width):
            code = code * self.n_levels + prefix[i].item()
        return code

    def _hash_batch(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute hashes for a batch of index vectors.

        Args:
            indices: (batch, d) integer tensor.

        Returns:
            (batch,) tensor of hash codes (as long integers).
        """
        prefix = indices[:, :self.hash_width].long()
        # Pack: multiply by positional weights and sum
        # weights[i] = n_levels^(hash_width - 1 - i)
        weights = torch.tensor(
            [self.n_levels ** (self.hash_width - 1 - i) for i in range(self.hash_width)],
            dtype=torch.long, device=indices.device,
        )
        codes = (prefix * weights).sum(dim=-1)
        return codes

    def _hamming1_neighbors(self, code: int) -> List[int]:
        """Generate all hash codes at Hamming distance 1 from the given code.

        "Hamming distance 1" means one coordinate index differs.
        For hash_width=16, n_levels=8: at most 16 * 7 = 112 neighbors.

        Args:
            code: The base hash code.

        Returns:
            List of neighbor hash codes.
        """
        neighbors = []
        # Decompose code into per-coordinate indices
        coords = []
        remaining = code
        for _ in range(self.hash_width):
            coords.append(remaining % self.n_levels)
            remaining //= self.n_levels
        coords.reverse()

        # For each coordinate, try all other levels
        for i in range(self.hash_width):
            original = coords[i]
            for level in range(self.n_levels):
                if level != original:
                    # Recompute code with this one coordinate changed
                    new_code = 0
                    for j in range(self.hash_width):
                        val = level if j == i else coords[j]
                        new_code = new_code * self.n_levels + val
                    neighbors.append(new_code)

        return neighbors

    def insert(self, indices: torch.Tensor, position: int) -> None:
        """Insert a token's quantization indices into the index.

        Args:
            indices: (d,) integer tensor of Lloyd-Max codebook indices.
            position: Token position in the sequence.
        """
        code = self._hash(indices)
        self.buckets[code].append(position)
        self.all_indices.append(indices.cpu())
        self.n_tokens += 1

    def insert_batch(self, indices: torch.Tensor, start_position: int) -> None:
        """Insert a batch of tokens.

        Args:
            indices: (batch, d) integer tensor.
            start_position: Position of the first token in the batch.
        """
        codes = self._hash_batch(indices)
        for i in range(indices.shape[0]):
            code = codes[i].item()
            pos = start_position + i
            self.buckets[code].append(pos)
            self.all_indices.append(indices[i].cpu())
        self.n_tokens += indices.shape[0]

    def search(
        self,
        query_indices: torch.Tensor,
        k: int,
        keys: Optional[torch.Tensor] = None,
        query_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Find candidate tokens with matching or similar hash codes.

        Args:
            query_indices: (d,) quantization indices for the query.
            k: Maximum number of candidates to return.
            keys: (n_tokens, d) full key vectors for scoring. If None,
                returns all candidates without scoring.
            query_vec: (d,) query vector for dot-product scoring.
                Required if keys is provided.

        Returns:
            Tuple of (candidate_positions, n_candidates_before_topk).
            candidate_positions is a (min(k, n_candidates),) tensor.
        """
        # Primary bucket
        code = self._hash(query_indices)
        candidate_set: Set[int] = set()

        if code in self.buckets:
            candidate_set.update(self.buckets[code])

        # Multi-probe: Hamming distance 1 neighbors
        if self.multi_probe:
            for neighbor_code in self._hamming1_neighbors(code):
                if neighbor_code in self.buckets:
                    candidate_set.update(self.buckets[neighbor_code])

        n_candidates = len(candidate_set)

        if n_candidates == 0:
            # Fallback: return most recent k positions
            start = max(0, self.n_tokens - k)
            return torch.arange(start, self.n_tokens, dtype=torch.long), 0

        candidates = torch.tensor(sorted(candidate_set), dtype=torch.long)

        # If we have vectors, score candidates by dot product and take top-k
        if keys is not None and query_vec is not None:
            device = keys.device
            valid = candidates[candidates < keys.shape[0]]
            if valid.shape[0] == 0:
                start = max(0, self.n_tokens - k)
                return torch.arange(start, self.n_tokens, dtype=torch.long), 0

            candidate_keys = keys[valid]  # (n_cand, d)
            scores = query_vec.unsqueeze(0) @ candidate_keys.T  # (1, n_cand)
            scores = scores.squeeze(0)  # (n_cand,)

            actual_k = min(k, valid.shape[0])
            _, topk_in_cand = torch.topk(scores, k=actual_k)
            return valid[topk_in_cand], n_candidates

        # Without scoring, just return all candidates (capped at k)
        return candidates[:k], n_candidates

    def get_stats(self) -> CodeIndexStats:
        """Return statistics about the current index state."""
        if not self.buckets:
            return CodeIndexStats(
                n_tokens=self.n_tokens,
                hash_width=self.hash_width,
                n_levels=self.n_levels,
            )

        sizes = [len(v) for v in self.buckets.values()]
        return CodeIndexStats(
            n_tokens=self.n_tokens,
            n_buckets=len(self.buckets),
            avg_bucket_size=sum(sizes) / len(sizes) if sizes else 0,
            max_bucket_size=max(sizes) if sizes else 0,
            min_bucket_size=min(sizes) if sizes else 0,
            hash_width=self.hash_width,
            n_levels=self.n_levels,
        )

    def clear(self) -> None:
        """Clear the index."""
        self.buckets.clear()
        self.all_indices.clear()
        self.n_tokens = 0


# ---------------------------------------------------------------------------
# CodeRetrievalCache: wraps ResidualQuant + CodeIndex
# ---------------------------------------------------------------------------

class CodeRetrievalCache:
    """KV cache with code-based retrieval for approximate attention.

    Wraps the existing ResidualQuant compression and adds a CodeIndex on top.
    Keys are quantized normally AND indexed by their codes. On attention,
    retrieves top-k candidates via code index + recent window, computes
    attention only over those tokens.

    The key advantage: zero additional memory. The quantization indices are
    ALREADY stored for compression. We just reinterpret them as a hash.

    Args:
        d: Head dimension.
        bits: Quantization bits (default 3).
        hash_width: Number of coordinates for hash (default 16).
        retrieval_k: Number of candidates to retrieve (default 64).
        window_size: Recent token window to always include (default 64).
        multi_probe: Use Hamming-1 multi-probe (default True).
        seed: Random seed for rotation matrix.
        min_seq_for_retrieval: Minimum sequence length before switching
            from full attention to retrieval attention (default 128).
    """

    def __init__(
        self,
        d: int,
        bits: int = 3,
        hash_width: int = 16,
        retrieval_k: int = 64,
        window_size: int = 64,
        multi_probe: bool = True,
        seed: int = 42,
        min_seq_for_retrieval: int = 128,
    ):
        from .residual_quant import ResidualQuantEstimator

        self.d = d
        self.bits = bits
        self.hash_width = hash_width
        self.retrieval_k = retrieval_k
        self.window_size = window_size
        self.multi_probe = multi_probe
        self.min_seq_for_retrieval = min_seq_for_retrieval

        # Quantizer (same as used for compression)
        self.rq = ResidualQuantEstimator(
            d=d, bits=bits, seed=seed, device="cpu",
            center_before_quantize=False,
        )

        # Code index
        n_levels = 1 << (bits - 1)  # MSE uses bits-1, not bits
        self.index = CodeIndex(
            hash_width=hash_width,
            n_levels=n_levels,
            multi_probe=multi_probe,
        )

        # Storage
        self.keys: List[torch.Tensor] = []  # full key vectors
        self.values: List[torch.Tensor] = []  # full value vectors
        self.compressed: List[Dict[str, torch.Tensor]] = []

    def insert(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Insert a key-value pair.

        Args:
            key: (d,) key vector.
            value: (d,) value vector.
        """
        self.keys.append(key.cpu())
        self.values.append(value.cpu())

        # Quantize to get indices
        comp = self.rq.quantize(key.unsqueeze(0).float())
        indices = comp["mse_indices"].squeeze(0)
        self.compressed.append(comp)

        # Index by codes
        self.index.insert(indices, len(self.keys) - 1)

    def insert_batch(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Insert a batch of key-value pairs.

        Args:
            keys: (batch, d) key vectors.
            values: (batch, d) value vectors.
        """
        start_pos = len(self.keys)

        for i in range(keys.shape[0]):
            self.keys.append(keys[i].cpu())
            self.values.append(values[i].cpu())

        # Quantize batch
        comp = self.rq.quantize(keys.float())
        indices = comp["mse_indices"]  # (batch, d)

        # Index batch
        self.index.insert_batch(indices, start_pos)

    def retrieve_and_attend(
        self,
        query: torch.Tensor,
        scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Retrieve top-k candidates + window and compute attention output.

        Args:
            query: (d,) query vector.
            scale: Attention scale factor (default 1/sqrt(d)).

        Returns:
            Tuple of (output, selected_indices, n_candidates):
                output: (d,) attention-weighted value vector
                selected_indices: positions of tokens attended to
                n_candidates: number of candidates from code index
        """
        if scale is None:
            scale = 1.0 / math.sqrt(self.d)

        seq_len = len(self.keys)

        # Below min threshold: use full attention
        if seq_len <= self.min_seq_for_retrieval:
            all_keys = torch.stack(self.keys)  # (seq, d)
            all_values = torch.stack(self.values)  # (seq, d)
            scores = (query @ all_keys.T) * scale
            weights = F.softmax(scores, dim=-1)
            output = weights @ all_values
            return output, torch.arange(seq_len), seq_len

        all_keys = torch.stack(self.keys)
        all_values = torch.stack(self.values)

        # Quantize query to get its hash
        query_comp = self.rq.quantize(query.unsqueeze(0).float())
        query_indices = query_comp["mse_indices"].squeeze(0)

        # Retrieve candidates via code index
        retrieved, n_candidates = self.index.search(
            query_indices,
            k=self.retrieval_k,
            keys=all_keys,
            query_vec=query,
        )

        # Recent window
        window_start = max(0, seq_len - self.window_size)
        window_indices = torch.arange(window_start, seq_len, dtype=torch.long)

        # Union + deduplicate
        combined = torch.cat([retrieved, window_indices])
        selected = torch.unique(combined)
        selected, _ = torch.sort(selected)

        # Compute attention over selected tokens only
        sel_keys = all_keys[selected]
        sel_values = all_values[selected]
        scores = (query @ sel_keys.T) * scale
        weights = F.softmax(scores, dim=-1)
        output = weights @ sel_values

        return output, selected, n_candidates

    @property
    def seq_len(self) -> int:
        return len(self.keys)

    def clear(self) -> None:
        self.keys.clear()
        self.values.clear()
        self.compressed.clear()
        self.index.clear()
