"""Cross-token delta coding in WHT-rotated space for KV cache compression.

Exploits the known i.i.d. Gaussian structure of WHT-rotated coordinates:
after rotation, each coordinate ~ N(0, 1/d). If two tokens have cosine
similarity c in original space, their WHT deltas have variance proportional
to (1 - c^2), which is much tighter than 1/d for similar tokens.

**Key difference from temporal delta coding (which failed):**
- Temporal: token[t] = anchor + cumsum(deltas[0:t]) -- ERROR ACCUMULATES
- DeltaQuant: token[t] = anchor[group[t]] + delta[t] -- NO ACCUMULATION
Each token independently references its group anchor (the medoid).

Storage model:
    anchors:    G tokens at b bits each (Lloyd-Max in WHT space)
    deltas:     (N-G) tokens at d bits each (tighter codebook)
    assignment: log2(G) bits per token for group ID
    norms:      16 bits per token (FP16 for reconstruction)

Effective rate = (G*b + (N-G)*d_bits) / N + assignment_overhead bits/token

For G=4 groups per 16 tokens, delta=1bit: (4*3 + 12*1)/16 + 0.25 = 1.75 bits
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .codebook import LloydMaxCodebook
from .rotation import apply_wht_rotation, generate_wht_rotation


# ---------------------------------------------------------------------------
# Clustering utilities
# ---------------------------------------------------------------------------

def greedy_group_by_similarity(
    vectors: torch.Tensor,
    group_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cluster vectors into groups of fixed size by greedy nearest-neighbor.

    Algorithm:
        1. Pick a random ungrouped vector as seed
        2. Find its (group_size - 1) nearest neighbors among ungrouped vectors
        3. Form a group; repeat until all vectors assigned

    This is O(N^2 * d) but N is seq_len (hundreds to thousands), not huge.

    Args:
        vectors: (N, d) normalized vectors for similarity computation.
        group_size: Target group size (last group may be smaller).

    Returns:
        group_ids: (N,) integer tensor, group assignment per token.
        medoid_mask: (N,) bool tensor, True for medoid (anchor) tokens.
    """
    N, d = vectors.shape
    device = vectors.device

    # Precompute cosine similarity matrix
    norms = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized = vectors / norms
    sim_matrix = normalized @ normalized.T  # (N, N)

    assigned = torch.zeros(N, dtype=torch.bool, device=device)
    group_ids = torch.full((N,), -1, dtype=torch.long, device=device)
    medoid_mask = torch.zeros(N, dtype=torch.bool, device=device)

    group_id = 0
    remaining_indices = torch.arange(N, device=device)

    while not assigned.all():
        unassigned = (~assigned).nonzero(as_tuple=True)[0]
        if len(unassigned) == 0:
            break

        # Pick a random seed from unassigned
        seed_local = torch.randint(len(unassigned), (1,)).item()
        seed_idx = unassigned[seed_local].item()

        # Find nearest neighbors among unassigned
        sims_to_seed = sim_matrix[seed_idx]
        sims_to_seed[assigned] = -2.0  # exclude assigned

        # Top group_size most similar (including seed itself)
        actual_size = min(group_size, len(unassigned))
        _, topk_indices = sims_to_seed.topk(actual_size)

        # Find medoid: the vector with highest average similarity to group
        group_sims = sim_matrix[topk_indices][:, topk_indices]
        avg_sims = group_sims.mean(dim=1)
        medoid_local = avg_sims.argmax().item()
        medoid_global = topk_indices[medoid_local].item()

        # Assign
        group_ids[topk_indices] = group_id
        assigned[topk_indices] = True
        medoid_mask[medoid_global] = True
        group_id += 1

    return group_ids, medoid_mask


def kmeans_grouping(
    vectors: torch.Tensor,
    n_groups: int,
    max_iter: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """K-means clustering for grouping tokens.

    Args:
        vectors: (N, d) vectors.
        n_groups: Number of groups.
        max_iter: Max iterations.

    Returns:
        group_ids: (N,) group assignment.
        medoid_mask: (N,) bool, True for closest-to-centroid token per group.
    """
    N, d = vectors.shape
    device = vectors.device

    # Initialize with k-means++ style: pick diverse seeds
    indices = torch.zeros(n_groups, dtype=torch.long, device=device)
    indices[0] = torch.randint(N, (1,)).item()

    for k in range(1, n_groups):
        # Distance to nearest existing centroid
        centroids_so_far = vectors[indices[:k]]  # (k, d)
        dists = torch.cdist(vectors, centroids_so_far)  # (N, k)
        min_dists = dists.min(dim=1).values  # (N,)
        # Probability proportional to squared distance
        probs = min_dists ** 2
        probs /= probs.sum() + 1e-10
        indices[k] = torch.multinomial(probs, 1).item()

    centroids = vectors[indices].clone()  # (n_groups, d)

    for _ in range(max_iter):
        # Assignment
        dists = torch.cdist(vectors, centroids)  # (N, n_groups)
        group_ids = dists.argmin(dim=1)  # (N,)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(n_groups, device=device)
        for g in range(n_groups):
            mask = group_ids == g
            if mask.any():
                new_centroids[g] = vectors[mask].mean(dim=0)
                counts[g] = mask.sum()
            else:
                new_centroids[g] = centroids[g]

        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    # Find medoid per group (closest actual token to centroid)
    medoid_mask = torch.zeros(N, dtype=torch.bool, device=device)
    for g in range(n_groups):
        mask = group_ids == g
        if mask.any():
            group_vecs = vectors[mask]
            group_indices = mask.nonzero(as_tuple=True)[0]
            dists_to_centroid = (group_vecs - centroids[g]).norm(dim=-1)
            medoid_local = dists_to_centroid.argmin()
            medoid_mask[group_indices[medoid_local]] = True

    return group_ids, medoid_mask


# ---------------------------------------------------------------------------
# DeltaQuant Encoder / Decoder
# ---------------------------------------------------------------------------

class DeltaQuantEncoder:
    """Encode tokens as anchors + deltas in WHT-rotated space.

    Tokens are grouped by key similarity. Each group has one anchor
    (the medoid) quantized at full bit-width, and remaining tokens
    stored as deltas from the anchor, quantized at fewer bits.

    NO cumulative error: each token reconstructs independently from
    its group anchor.

    Args:
        d: Head dimension (must be power of 2 for WHT).
        anchor_bits: Bits for anchor tokens (default 3).
        delta_bits: Bits for delta tokens (default 1 or 2).
        group_size: Tokens per group (default 4).
        seed: Random seed for WHT rotation.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        anchor_bits: int = 3,
        delta_bits: int = 1,
        group_size: int = 4,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        self.d = d
        self.anchor_bits = anchor_bits
        self.delta_bits = delta_bits
        self.group_size = group_size
        self.device = device

        # WHT rotation
        self.wht_params = generate_wht_rotation(d, seed=seed, device=device)

        # Anchor codebook: standard Lloyd-Max for N(0, 1/d)
        self.anchor_codebook = LloydMaxCodebook(d, anchor_bits)

        # Delta codebook: Lloyd-Max for a TIGHTER distribution
        # Delta variance is (1 - cos^2) * (1/d) for pairs with cosine c
        # For cos=0.9, variance = 0.19/d. We use a smaller "effective d"
        # to get tighter centroids. d_eff = d / (1 - cos^2_target)
        # For cos=0.9: d_eff = d / 0.19 ~ 5.3*d
        # This gives centroids spaced for the tighter distribution.
        self.delta_codebook = LloydMaxCodebook(d * 5, delta_bits)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply WHT rotation to vectors."""
        return apply_wht_rotation(x, self.wht_params)

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse WHT rotation."""
        return apply_wht_rotation(y, self.wht_params, inverse=True)

    def encode(
        self,
        keys: torch.Tensor,
        clustering: str = "greedy",
    ) -> Dict[str, Any]:
        """Encode a batch of key vectors using delta coding in WHT space.

        Args:
            keys: (N, d) key vectors (original space, not yet rotated).
            clustering: "greedy" or "kmeans".

        Returns:
            Dict with:
                - anchor_indices: (n_anchors, d) codebook indices for anchors
                - delta_indices: (n_deltas, d) codebook indices for deltas
                - delta_scales: (n_deltas, 1) per-vector scale for delta
                - group_ids: (N,) group assignment
                - medoid_mask: (N,) bool mask for anchors
                - norms: (N,) original vector norms (FP16)
                - group_medoid_map: (n_groups,) index of medoid for each group
                - wht_params: rotation parameters (for decoding)
        """
        N, d = keys.shape
        assert d == self.d

        # Step 1: Rotate all keys into WHT space
        norms = keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        keys_normalized = keys / norms
        keys_rotated = self.rotate(keys_normalized)  # (N, d)

        # Step 2: Cluster tokens into groups
        n_groups = max(1, N // self.group_size)
        if clustering == "kmeans":
            group_ids, medoid_mask = kmeans_grouping(keys_normalized, n_groups)
        else:
            group_ids, medoid_mask = greedy_group_by_similarity(
                keys_normalized, self.group_size
            )
            n_groups = group_ids.max().item() + 1

        # Step 3: Build medoid map (group_id -> token index)
        group_medoid_map = torch.full((n_groups,), -1, dtype=torch.long)
        medoid_indices = medoid_mask.nonzero(as_tuple=True)[0]
        for idx in medoid_indices:
            gid = group_ids[idx].item()
            group_medoid_map[gid] = idx.item()

        # Step 4: Quantize anchors at full bits in WHT space
        anchor_rotated = keys_rotated[medoid_mask]  # (n_anchors, d)
        anchor_indices = self.anchor_codebook.quantize(anchor_rotated)
        anchor_reconstructed = self.anchor_codebook.dequantize(anchor_indices)

        # Step 5: Compute and quantize deltas for non-anchor tokens
        non_anchor_mask = ~medoid_mask
        non_anchor_indices_list = non_anchor_mask.nonzero(as_tuple=True)[0]

        if len(non_anchor_indices_list) > 0:
            non_anchor_rotated = keys_rotated[non_anchor_indices_list]  # (n_deltas, d)

            # Each non-anchor's anchor reconstruction
            non_anchor_groups = group_ids[non_anchor_indices_list]
            anchor_for_each = torch.zeros_like(non_anchor_rotated)
            for i, tok_idx in enumerate(non_anchor_indices_list):
                gid = group_ids[tok_idx].item()
                # Find which position in anchor array this group's medoid is
                medoid_tok = group_medoid_map[gid].item()
                medoid_pos = (medoid_mask[:medoid_tok + 1]).sum().item() - 1
                anchor_for_each[i] = anchor_reconstructed[medoid_pos]

            # Delta = rotated_token - anchor_reconstruction
            deltas = non_anchor_rotated - anchor_for_each

            # Per-vector scale for delta quantization
            delta_abs_max = deltas.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)

            # Scale deltas to match the delta codebook's expected range
            # Delta codebook is built for N(0, 1/(d*5)), so sigma = 1/sqrt(d*5)
            # Our actual deltas have some per-vector scale.
            # We rescale each delta to unit variance, quantize, then store the scale.
            delta_sigma = deltas.std(dim=-1, keepdim=True).clamp(min=1e-10)
            target_sigma = 1.0 / math.sqrt(self.d * 5)
            scaled_deltas = deltas * (target_sigma / delta_sigma)

            delta_indices = self.delta_codebook.quantize(scaled_deltas)
        else:
            delta_indices = torch.empty(0, d, dtype=torch.long)
            delta_sigma = torch.empty(0, 1)

        return {
            "anchor_indices": anchor_indices,
            "delta_indices": delta_indices,
            "delta_sigma": delta_sigma if len(non_anchor_indices_list) > 0 else torch.empty(0, 1),
            "group_ids": group_ids,
            "medoid_mask": medoid_mask,
            "norms": norms.squeeze(-1).half(),
            "group_medoid_map": group_medoid_map,
            "n_groups": n_groups,
            "non_anchor_indices": non_anchor_indices_list,
        }

    def decode(self, encoded: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct all key vectors from encoded representation.

        Each token is INDEPENDENTLY decoded:
            anchor tokens: dequantize(anchor_indices) -> unrotate -> rescale
            delta tokens:  dequantize(anchor) + rescale(dequantize(delta)) -> unrotate -> rescale

        NO cumulative sum. No error accumulation.

        Args:
            encoded: Output from encode().

        Returns:
            Reconstructed keys (N, d) in original space.
        """
        anchor_indices = encoded["anchor_indices"]
        delta_indices = encoded["delta_indices"]
        delta_sigma = encoded["delta_sigma"]
        group_ids = encoded["group_ids"]
        medoid_mask = encoded["medoid_mask"]
        norms = encoded["norms"].float()
        group_medoid_map = encoded["group_medoid_map"]
        non_anchor_indices = encoded["non_anchor_indices"]

        N = len(group_ids)
        d = self.d
        target_sigma = 1.0 / math.sqrt(d * 5)

        # Reconstruct anchor WHT vectors
        anchor_reconstructed = self.anchor_codebook.dequantize(anchor_indices)  # (n_anchors, d)

        # Reconstruct all tokens in WHT space
        reconstructed_rotated = torch.zeros(N, d, dtype=torch.float32)

        # Place anchors
        anchor_positions = medoid_mask.nonzero(as_tuple=True)[0]
        reconstructed_rotated[anchor_positions] = anchor_reconstructed

        # Place deltas
        if len(non_anchor_indices) > 0:
            delta_reconstructed = self.delta_codebook.dequantize(delta_indices)  # (n_deltas, d)

            # Rescale deltas back to original scale
            delta_rescaled = delta_reconstructed * (delta_sigma / target_sigma)

            for i, tok_idx in enumerate(non_anchor_indices):
                gid = group_ids[tok_idx].item()
                medoid_tok = group_medoid_map[gid].item()
                medoid_pos = (medoid_mask[:medoid_tok + 1]).sum().item() - 1
                reconstructed_rotated[tok_idx] = anchor_reconstructed[medoid_pos] + delta_rescaled[i]

        # Unrotate back to original space
        reconstructed = self.unrotate(reconstructed_rotated)

        # Rescale by original norms
        reconstructed = reconstructed * norms.unsqueeze(-1)

        return reconstructed

    def compute_effective_bits(self, encoded: Dict[str, Any]) -> Dict[str, float]:
        """Compute the effective bits per token for this encoding.

        Storage breakdown:
            anchors: n_anchors * d * anchor_bits
            deltas:  n_deltas * d * delta_bits
            scales:  n_deltas * 32 bits (float32 sigma per vector)
            norms:   N * 16 bits (FP16)
            groups:  N * ceil(log2(n_groups)) bits

        Returns:
            Dict with per-component and total effective bits per dimension.
        """
        N = len(encoded["group_ids"])
        n_anchors = encoded["medoid_mask"].sum().item()
        n_deltas = N - n_anchors
        n_groups = encoded["n_groups"]
        d = self.d

        anchor_total = n_anchors * d * self.anchor_bits
        delta_total = n_deltas * d * self.delta_bits
        scale_total = n_deltas * 32  # per-vector sigma
        norm_total = N * 16  # FP16 norms
        group_total = N * max(1, math.ceil(math.log2(max(n_groups, 2))))

        total_bits = anchor_total + delta_total + scale_total + norm_total + group_total

        return {
            "anchor_bits_per_dim": anchor_total / (N * d),
            "delta_bits_per_dim": delta_total / (N * d),
            "scale_overhead_per_dim": scale_total / (N * d),
            "norm_overhead_per_dim": norm_total / (N * d),
            "group_overhead_per_dim": group_total / (N * d),
            "total_bits_per_dim": total_bits / (N * d),
            "compression_ratio": 16.0 / (total_bits / (N * d)),
            "n_anchors": n_anchors,
            "n_deltas": n_deltas,
            "n_groups": n_groups,
            "anchor_fraction": n_anchors / N,
        }


# ---------------------------------------------------------------------------
# DeltaQuant with entropy coding analysis
# ---------------------------------------------------------------------------

def analyze_delta_entropy(delta_indices: torch.Tensor, bits: int) -> Dict[str, float]:
    """Analyze the entropy of delta indices (should be very non-uniform).

    If deltas are tight Gaussians centered at 0, most indices should map
    to the central centroid. The entropy should be well below the uniform
    maximum of `bits`.

    Args:
        delta_indices: (n_deltas, d) integer indices.
        bits: Nominal bits per index.

    Returns:
        Dict with entropy stats.
    """
    n_levels = 1 << bits
    flat = delta_indices.flatten().long()

    # Count occurrences of each level
    counts = torch.bincount(flat, minlength=n_levels).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # avoid log(0)

    entropy = -(probs * probs.log2()).sum().item()
    max_entropy = bits  # uniform distribution

    # Most frequent index (should be center for tight deltas)
    mode_idx = counts.argmax().item()
    mode_prob = counts[mode_idx].item() / flat.numel()

    return {
        "entropy": entropy,
        "max_entropy": max_entropy,
        "entropy_ratio": entropy / max_entropy,
        "mode_index": mode_idx,
        "mode_probability": mode_prob,
        "effective_bits_with_entropy": entropy,  # theoretical lower bound
    }
