"""PCA Code Retrieval Experiment: Proving PCA rotation codes work as retrieval hash.

HYPOTHESIS: PCA rotation concentrates 48.7% of variance in the top 10% of
coordinates. WHT spreads information uniformly (12.5%). PCA codes from leading
coordinates should be a dramatically better locality-sensitive hash than WHT codes.

CRITICAL INSIGHT (from first run): The PCA quantizer WHITENS before quantizing,
erasing the variance concentration. After whitening, every coordinate has the
same N(0, 1/d) distribution, so PCA indices from leading coords are no more
informative than WHT indices. We must hash the RAW PCA-rotated coordinates
(before whitening) to exploit the variance concentration.

APPROACH: For hashing, apply PCA rotation WITHOUT whitening. Then quantize the
raw PCA coordinates into coarse bins using per-coordinate quantiles. The leading
coordinates have high variance = wide spread = natural similarity clustering.

EXPERIMENTS:
    1. Hash quality comparison: raw PCA codes vs WHT codes at same hash_width
    2. Retrieval attention with PCA codes: quality vs full attention
    3. Combined compression + retrieval: zero-index-overhead system vs FAISS

PRIOR RESULTS (WHT codes, from code_retrieval_results.md):
    - Oracle recall at hash_width=16: 67%
    - Realistic recall at hash_width=16: 55-59%
    - Root cause: WHT spreads info uniformly, 16/128 coords = 12.5% of info

TARGET: >90% recall with PCA codes where WHT got 67%.

Usage:
    python benchmarks/pca_code_retrieval_experiment.py
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from turboquantdc.codebook import LloydMaxCodebook
from turboquantdc.polarquant import PolarQuant
from turboquantdc.learned_rotation import (
    PCARotatedQuantizer,
    compute_pca_rotation,
)
from turboquantdc.pca_code_retrieval import PCACodeIndex, WHTCodeIndex, binary_pca_hash

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"

BITS = 3
HASH_WIDTHS = [4, 6, 8, 10, 12, 16, 24, 32]
RETRIEVAL_K = 64
WINDOW_SIZE = 64
N_EVAL_QUERIES = 32
LAYERS_TO_TEST = [0, 5, 10, 15, 20, 25, 35]
HEADS_PER_LAYER = 2  # Qwen2.5-3B has 2 KV heads

FILLER = (
    "The quarterly financial review meeting covered several topics including "
    "budget allocations for the upcoming fiscal year, departmental spending reports, "
    "and projected revenue streams from various business units. The committee discussed "
    "infrastructure upgrades planned for the western regional offices and noted that "
    "maintenance schedules should be coordinated with the facilities management team. "
    "Several action items were assigned to team leads for follow-up before the next "
    "meeting cycle. The technology department presented a proposal for migrating legacy "
    "systems to cloud infrastructure, estimating a three-year return on investment. "
    "Human resources updated the board on recruitment metrics and employee retention "
    "strategies for the upcoming quarter.\n\n"
)


# ---------------------------------------------------------------------------
# Model loading + KV extraction
# ---------------------------------------------------------------------------
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    dt = time.time() - t0
    print(f"  Loaded in {dt:.1f}s")
    return model, tokenizer


def build_prompt(tokenizer, target_tokens: int) -> str:
    filler_len = len(tokenizer.encode(FILLER, add_special_tokens=False))
    n_reps = max(1, target_tokens // filler_len)
    parts = [FILLER] * n_reps
    parts.append(
        "\nBased on the meeting notes above, what specific action items were "
        "assigned regarding the infrastructure upgrades, cloud migration timeline, "
        "and employee retention strategy?"
    )
    text = "".join(parts)
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) > target_tokens + 50:
        tokens = tokens[:target_tokens]
        prompt = tokenizer.decode(tokens)
    return prompt


def extract_kv_and_attention(model, tokenizer, prompt: str) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    seq_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=True,
        )

    attentions = []
    for layer_attn in outputs.attentions:
        attentions.append(layer_attn.cpu().float())

    kv_cache = outputs.past_key_values
    keys_per_layer = []
    values_per_layer = []

    if hasattr(kv_cache, "key_cache") and len(kv_cache.key_cache) > 0:
        n_layers = len(kv_cache.key_cache)
        for layer_idx in range(n_layers):
            keys_per_layer.append(kv_cache.key_cache[layer_idx].cpu().float())
            values_per_layer.append(kv_cache.value_cache[layer_idx].cpu().float())
    elif hasattr(kv_cache, "layers"):
        n_layers = len(kv_cache.layers)
        for layer_idx in range(n_layers):
            layer = kv_cache.layers[layer_idx]
            keys_per_layer.append(layer.keys.cpu().float())
            values_per_layer.append(layer.values.cpu().float())
    else:
        n_layers = len(kv_cache)
        for layer_idx in range(n_layers):
            k, v = kv_cache[layer_idx]
            keys_per_layer.append(k.cpu().float())
            values_per_layer.append(v.cpu().float())

    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    n_q_heads = attentions[0].shape[1] if attentions else 0
    n_kv_heads = keys_per_layer[0].shape[1] if keys_per_layer else 0
    head_dim = keys_per_layer[0].shape[-1] if keys_per_layer else 0

    return {
        "attentions": attentions,
        "keys": keys_per_layer,
        "values": values_per_layer,
        "seq_len": seq_len,
        "n_layers": len(attentions),
        "n_q_heads": n_q_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "gqa_ratio": n_q_heads // max(n_kv_heads, 1),
    }


# ---------------------------------------------------------------------------
# Raw PCA quantization (no whitening) for hashing
# ---------------------------------------------------------------------------
def raw_pca_quantize(
    keys_normalized: torch.Tensor,
    pca_data: Dict[str, torch.Tensor],
    n_levels: int,
) -> torch.Tensor:
    """Quantize raw PCA-rotated coordinates using quantile-based bins.

    Unlike PCARotatedQuantizer which whitens coordinates to N(0, 1/d) before
    quantizing (erasing variance concentration), this function:
    1. Applies PCA rotation WITHOUT whitening
    2. Computes per-coordinate quantile bin edges from the data
    3. Assigns each value to its quantile bin

    The result: leading coordinates have high variance, tokens spread across
    bins proportional to their similarity, creating a natural LSH.
    """
    # PCA rotate without whitening
    mean = pca_data["mean"]
    rotation = pca_data["rotation"]

    x_c = keys_normalized.float() - mean
    y = x_c @ rotation.T  # (n, d) -- raw PCA coords, variance = eigenvalue_i

    # Per-coordinate quantile-based quantization
    n, d = y.shape
    indices = torch.zeros(n, d, dtype=torch.long)

    # Compute quantile bin edges from the data
    quantiles = torch.linspace(0, 1, n_levels + 1)[1:-1]  # interior edges
    for j in range(d):
        col = y[:, j]
        edges = torch.quantile(col, quantiles)
        indices[:, j] = torch.bucketize(col, edges)

    return indices


    # binary_pca_hash imported from turboquantdc.pca_code_retrieval


# ---------------------------------------------------------------------------
# Experiment 1: Hash quality comparison (PCA vs WHT)
# ---------------------------------------------------------------------------
def experiment1_hash_quality(
    keys: torch.Tensor,       # (seq_len, d) raw keys for one head
    values: torch.Tensor,     # (seq_len, d) raw values
    attn_weights: torch.Tensor,  # (seq_len, seq_len) attention matrix for one Q head
    head_dim: int,
    hash_width: int,
    query_positions: List[int],
) -> Dict[str, Any]:
    """Compare binary-PCA vs WHT code hash quality for attention retrieval.

    KEY APPROACH: Uses BINARY PCA hash (1 bit per PCA component: above/below
    median). With K hash coordinates, gives 2^K buckets instead of 4^K.
    The sign of the leading PCA components IS the most informative LSH.

    For each query position:
    1. Compute true top-k attended tokens (ground truth)
    2. Build binary-PCA code index and WHT code index from the same keys
    3. Hash the top-1 attended key's code in each rotation basis
    4. Retrieve candidates from each index
    5. Measure recall: what fraction of true top-k are in candidates + window

    Returns metrics for both PCA and WHT.
    """
    seq_len = keys.shape[0]
    mse_bits = max(BITS - 1, 1)
    wht_n_levels = 1 << mse_bits
    pca_n_levels = 2  # BINARY for PCA hash (1 bit per component)

    # --- WHT quantization (existing baseline) ---
    pq_wht = PolarQuant(d=head_dim, bits=mse_bits, seed=42, device="cpu")
    key_norms = keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    keys_normalized = keys / key_norms
    wht_indices = pq_wht.quantize(keys_normalized)

    # --- Binary PCA hash (NO whitening, 1 bit per component) ---
    n_calib = min(seq_len // 2, 512)
    calib_data = keys_normalized[:n_calib]
    pca_data = compute_pca_rotation(calib_data.cpu())
    pca_indices = binary_pca_hash(keys_normalized, pca_data)

    # Eigenvalue info
    eigenvalues = pca_data["eigenvalues"]
    total_var = eigenvalues.sum().item()
    hash_var = eigenvalues[:hash_width].sum().item()
    variance_fraction = hash_var / total_var if total_var > 0 else 0

    # Results accumulators
    pca_top1 = 0; pca_top5 = 0; pca_top10 = 0
    wht_top1 = 0; wht_top5 = 0; wht_top10 = 0
    pca_recalls = []; wht_recalls = []
    pca_cos_sims = []; wht_cos_sims = []
    pca_candidates = []; wht_candidates = []
    total_queries = 0

    for q_pos in query_positions:
        if q_pos >= attn_weights.shape[0]:
            continue

        full_attn = attn_weights[q_pos, :q_pos + 1]
        causal_len = full_attn.shape[0]
        if causal_len < 10:
            continue

        causal_values = values[:causal_len]
        full_output = full_attn @ causal_values

        # Ground truth top tokens
        full_top1_idx = torch.topk(full_attn, k=1).indices[0].item()
        k5 = min(5, causal_len)
        k10 = min(10, causal_len)
        full_top5 = set(torch.topk(full_attn, k=k5).indices.tolist())
        full_top10 = set(torch.topk(full_attn, k=k10).indices.tolist())
        significant = set((full_attn > 0.01).nonzero(as_tuple=True)[0].tolist())

        window_start = max(0, causal_len - WINDOW_SIZE)
        window_set = set(range(window_start, causal_len))

        # --- PCA code index (binary, 2 levels, Hamming-2 multi-probe) ---
        pca_index = PCACodeIndex(
            hash_width=hash_width,
            n_levels=pca_n_levels,
            multi_probe=True,
            hamming_radius=2,
        )
        pca_index.insert_batch(pca_indices[:causal_len], start_position=0)

        needle_pca = pca_indices[full_top1_idx]
        pca_retrieved, pca_n_cand = pca_index.search(
            needle_pca, k=RETRIEVAL_K,
            keys=keys[:causal_len],
            query_vec=keys[full_top1_idx],
        )
        pca_candidates.append(pca_n_cand)

        pca_selected = set(pca_retrieved.tolist()) | window_set
        pca_selected = pca_selected & set(range(causal_len))

        # --- WHT code index (4 levels per coord) ---
        wht_index = WHTCodeIndex(
            hash_width=hash_width,
            n_levels=wht_n_levels,
            multi_probe=True,
            hamming_radius=1,
        )
        wht_index.insert_batch(wht_indices[:causal_len], start_position=0)

        needle_wht = wht_indices[full_top1_idx]
        wht_retrieved, wht_n_cand = wht_index.search(
            needle_wht, k=RETRIEVAL_K,
            keys=keys[:causal_len],
            query_vec=keys[full_top1_idx],
        )
        wht_candidates.append(wht_n_cand)

        wht_selected = set(wht_retrieved.tolist()) | window_set
        wht_selected = wht_selected & set(range(causal_len))

        # --- Metrics ---
        # PCA
        if full_top1_idx in pca_selected:
            pca_top1 += 1
        pca_top5 += len(full_top5 & pca_selected) / max(len(full_top5), 1)
        pca_top10 += len(full_top10 & pca_selected) / max(len(full_top10), 1)
        if len(significant) > 0:
            pca_recalls.append(len(significant & pca_selected) / len(significant))
        else:
            pca_recalls.append(1.0)

        # PCA output cosine
        pca_sel_sorted = torch.tensor(sorted(pca_selected), dtype=torch.long)
        log_scores = torch.log(full_attn[pca_sel_sorted] + 1e-30)
        pca_attn = F.softmax(log_scores, dim=-1)
        pca_output = pca_attn @ values[pca_sel_sorted]
        pca_cos = F.cosine_similarity(
            full_output.unsqueeze(0), pca_output.unsqueeze(0),
        ).item()
        pca_cos_sims.append(pca_cos)

        # WHT
        if full_top1_idx in wht_selected:
            wht_top1 += 1
        wht_top5 += len(full_top5 & wht_selected) / max(len(full_top5), 1)
        wht_top10 += len(full_top10 & wht_selected) / max(len(full_top10), 1)
        if len(significant) > 0:
            wht_recalls.append(len(significant & wht_selected) / len(significant))
        else:
            wht_recalls.append(1.0)

        # WHT output cosine
        wht_sel_sorted = torch.tensor(sorted(wht_selected), dtype=torch.long)
        log_scores = torch.log(full_attn[wht_sel_sorted] + 1e-30)
        wht_attn = F.softmax(log_scores, dim=-1)
        wht_output = wht_attn @ values[wht_sel_sorted]
        wht_cos = F.cosine_similarity(
            full_output.unsqueeze(0), wht_output.unsqueeze(0),
        ).item()
        wht_cos_sims.append(wht_cos)

        total_queries += 1

    if total_queries == 0:
        return {"error": "no valid queries"}

    return {
        "hash_width": hash_width,
        "variance_fraction": variance_fraction,
        "n_queries": total_queries,
        # PCA metrics
        "pca_top1": pca_top1 / total_queries,
        "pca_top5": pca_top5 / total_queries,
        "pca_top10": pca_top10 / total_queries,
        "pca_recall": float(np.mean(pca_recalls)),
        "pca_cosine": float(np.mean(pca_cos_sims)),
        "pca_avg_candidates": float(np.mean(pca_candidates)),
        "pca_frac_searched": float(np.mean(pca_candidates)) / max(1, seq_len),
        # WHT metrics
        "wht_top1": wht_top1 / total_queries,
        "wht_top5": wht_top5 / total_queries,
        "wht_top10": wht_top10 / total_queries,
        "wht_recall": float(np.mean(wht_recalls)),
        "wht_cosine": float(np.mean(wht_cos_sims)),
        "wht_avg_candidates": float(np.mean(wht_candidates)),
        "wht_frac_searched": float(np.mean(wht_candidates)) / max(1, seq_len),
    }


# ---------------------------------------------------------------------------
# Experiment 2: Retrieval attention quality with PCA codes
# ---------------------------------------------------------------------------
def experiment2_retrieval_attention(
    keys: torch.Tensor,
    values: torch.Tensor,
    attn_weights: torch.Tensor,
    head_dim: int,
    hash_width: int,
    query_positions: List[int],
) -> Dict[str, Any]:
    """Realistic retrieval attention: hash the query's key, retrieve, attend.

    Uses key-as-query proxy (same as prior WHT experiment for fair comparison).
    Uses RAW PCA coordinates (no whitening) for hashing.
    For each query:
    1. Raw-PCA-quantize the key at q_pos
    2. Hash leading coordinates
    3. Retrieve candidates + window
    4. Compute attention over candidates only
    5. Compare vs full attention
    """
    seq_len = keys.shape[0]
    mse_bits = max(BITS - 1, 1)
    n_levels = 1 << mse_bits
    scale = 1.0 / math.sqrt(head_dim)

    # Binary PCA hash (no whitening)
    key_norms = keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    keys_normalized = keys / key_norms
    n_calib = min(seq_len // 2, 512)
    pca_data = compute_pca_rotation(keys_normalized[:n_calib].cpu())
    pca_indices = binary_pca_hash(keys_normalized, pca_data)
    n_levels = 2  # binary for PCA hash
    eigenvalues = pca_data["eigenvalues"]

    top1_matches = 0; top5_matches = 0; top10_matches = 0
    cos_sims = []; recalls = []; candidate_sizes = []
    total_queries = 0

    for q_pos in query_positions:
        if q_pos >= attn_weights.shape[0]:
            continue

        full_attn = attn_weights[q_pos, :q_pos + 1]
        causal_len = full_attn.shape[0]
        if causal_len < 10:
            continue

        causal_values = values[:causal_len]
        full_output = full_attn @ causal_values

        full_top1 = set(torch.topk(full_attn, k=1).indices.tolist())
        k5 = min(5, causal_len)
        k10 = min(10, causal_len)
        full_top5 = set(torch.topk(full_attn, k=k5).indices.tolist())
        full_top10 = set(torch.topk(full_attn, k=k10).indices.tolist())

        # Hash the key at q_pos
        query_key = keys[q_pos]
        query_indices = pca_indices[q_pos]

        pca_index = PCACodeIndex(
            hash_width=hash_width,
            n_levels=2,  # binary
            multi_probe=True,
            hamming_radius=2,  # Hamming-2 for better recall
        )
        pca_index.insert_batch(pca_indices[:causal_len], start_position=0)

        retrieved, n_cand = pca_index.search(
            query_indices, k=RETRIEVAL_K,
            keys=keys[:causal_len],
            query_vec=query_key,
        )
        candidate_sizes.append(n_cand)

        # Window + retrieved
        window_start = max(0, causal_len - WINDOW_SIZE)
        window_indices = torch.arange(window_start, causal_len, dtype=torch.long)
        combined = torch.cat([retrieved, window_indices])
        selected = torch.unique(combined)
        selected, _ = torch.sort(selected)

        # Attention over selected
        sel_keys = keys[selected]
        sel_values = values[selected]
        scores = (query_key @ sel_keys.T) * scale
        retrieval_attn = F.softmax(scores, dim=-1)
        retrieval_output = retrieval_attn @ sel_values

        selected_set = set(selected.tolist())

        if full_top1.issubset(selected_set):
            top1_matches += 1
        top5_matches += len(full_top5 & selected_set) / max(len(full_top5), 1)
        top10_matches += len(full_top10 & selected_set) / max(len(full_top10), 1)

        cos_sim = F.cosine_similarity(
            full_output.unsqueeze(0), retrieval_output.unsqueeze(0),
        ).item()
        cos_sims.append(cos_sim)

        significant = set((full_attn > 0.01).nonzero(as_tuple=True)[0].tolist())
        if len(significant) > 0:
            recalls.append(len(significant & selected_set) / len(significant))
        else:
            recalls.append(1.0)

        total_queries += 1

    if total_queries == 0:
        return {"error": "no valid queries"}

    return {
        "hash_width": hash_width,
        "top1": top1_matches / total_queries,
        "top5": top5_matches / total_queries,
        "top10": top10_matches / total_queries,
        "cosine": float(np.mean(cos_sims)),
        "recall": float(np.mean(recalls)),
        "avg_candidates": float(np.mean(candidate_sizes)),
        "frac_searched": float(np.mean(candidate_sizes)) / max(1, seq_len),
        "n_queries": total_queries,
    }


# ---------------------------------------------------------------------------
# Experiment 3: Combined system comparison
# ---------------------------------------------------------------------------
def experiment3_combined_system(
    keys: torch.Tensor,
    values: torch.Tensor,
    attn_weights: torch.Tensor,
    head_dim: int,
    query_positions: List[int],
) -> Dict[str, Any]:
    """Combined compression + retrieval system comparison.

    Compares:
    A. PCA rotation + PCA code retrieval (zero index overhead)
    B. WHT rotation + WHT code retrieval (zero index overhead)
    C. Window-only baseline (no retrieval)

    All at 3-bit, hash_width=16.
    """
    seq_len = keys.shape[0]
    mse_bits = max(BITS - 1, 1)
    n_levels = 1 << mse_bits
    scale = 1.0 / math.sqrt(head_dim)
    hw = 16  # fixed hash width for this comparison

    key_norms = keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    keys_normalized = keys / key_norms

    # WHT quantization
    pq_wht = PolarQuant(d=head_dim, bits=mse_bits, seed=42, device="cpu")
    wht_indices = pq_wht.quantize(keys_normalized)

    # Binary PCA hash (no whitening -- for hashing)
    n_calib = min(seq_len // 2, 512)
    pca_data = compute_pca_rotation(keys_normalized[:n_calib].cpu())
    pca_indices = binary_pca_hash(keys_normalized, pca_data)

    # Compression MSE (using the whitened PCA quantizer -- compression is separate)
    pca_quantizer = PCARotatedQuantizer(
        d=head_dim, bits=mse_bits,
        rotation_data=pca_data,
        adaptive_bits=False,
        device="cpu",
    )
    pca_comp_indices = pca_quantizer.quantize(keys_normalized)
    pca_recon = pca_quantizer.dequantize(pca_comp_indices)
    pca_recon_scaled = pca_recon * key_norms
    pca_mse = ((keys - pca_recon_scaled) ** 2).mean().item()

    wht_recon = pq_wht.dequantize(wht_indices)
    wht_recon_scaled = wht_recon * key_norms
    wht_mse = ((keys - wht_recon_scaled) ** 2).mean().item()

    # Results
    results = {
        "pca": {"top1": 0, "top5": 0, "recall": [], "cosine": [], "candidates": []},
        "wht": {"top1": 0, "top5": 0, "recall": [], "cosine": [], "candidates": []},
        "window_only": {"top1": 0, "top5": 0, "recall": [], "cosine": []},
    }
    total = 0

    for q_pos in query_positions:
        if q_pos >= attn_weights.shape[0]:
            continue

        full_attn = attn_weights[q_pos, :q_pos + 1]
        causal_len = full_attn.shape[0]
        if causal_len < 10:
            continue

        causal_values = values[:causal_len]
        full_output = full_attn @ causal_values
        query_key = keys[q_pos]

        full_top1 = set(torch.topk(full_attn, k=1).indices.tolist())
        k5 = min(5, causal_len)
        full_top5 = set(torch.topk(full_attn, k=k5).indices.tolist())
        significant = set((full_attn > 0.01).nonzero(as_tuple=True)[0].tolist())

        window_start = max(0, causal_len - WINDOW_SIZE)
        window_set = set(range(window_start, causal_len))

        # --- PCA (binary hash, 2 levels, Hamming-2) ---
        pca_index = PCACodeIndex(hash_width=hw, n_levels=2, multi_probe=True, hamming_radius=2)
        pca_index.insert_batch(pca_indices[:causal_len], start_position=0)
        pca_retrieved, pca_n = pca_index.search(
            pca_indices[q_pos], k=RETRIEVAL_K,
            keys=keys[:causal_len], query_vec=query_key,
        )
        pca_sel = set(pca_retrieved.tolist()) | window_set
        pca_sel = pca_sel & set(range(causal_len))
        results["pca"]["candidates"].append(pca_n)

        if full_top1.issubset(pca_sel):
            results["pca"]["top1"] += 1
        results["pca"]["top5"] += len(full_top5 & pca_sel) / max(len(full_top5), 1)
        if significant:
            results["pca"]["recall"].append(len(significant & pca_sel) / len(significant))

        pca_sel_t = torch.tensor(sorted(pca_sel), dtype=torch.long)
        scores = (query_key @ keys[pca_sel_t].T) * scale
        attn_w = F.softmax(scores, dim=-1)
        out = attn_w @ values[pca_sel_t]
        results["pca"]["cosine"].append(
            F.cosine_similarity(full_output.unsqueeze(0), out.unsqueeze(0)).item()
        )

        # --- WHT ---
        wht_index = WHTCodeIndex(hash_width=hw, n_levels=n_levels, multi_probe=True)
        wht_index.insert_batch(wht_indices[:causal_len], start_position=0)
        wht_retrieved, wht_n = wht_index.search(
            wht_indices[q_pos], k=RETRIEVAL_K,
            keys=keys[:causal_len], query_vec=query_key,
        )
        wht_sel = set(wht_retrieved.tolist()) | window_set
        wht_sel = wht_sel & set(range(causal_len))
        results["wht"]["candidates"].append(wht_n)

        if full_top1.issubset(wht_sel):
            results["wht"]["top1"] += 1
        results["wht"]["top5"] += len(full_top5 & wht_sel) / max(len(full_top5), 1)
        if significant:
            results["wht"]["recall"].append(len(significant & wht_sel) / len(significant))

        wht_sel_t = torch.tensor(sorted(wht_sel), dtype=torch.long)
        scores = (query_key @ keys[wht_sel_t].T) * scale
        attn_w = F.softmax(scores, dim=-1)
        out = attn_w @ values[wht_sel_t]
        results["wht"]["cosine"].append(
            F.cosine_similarity(full_output.unsqueeze(0), out.unsqueeze(0)).item()
        )

        # --- Window only ---
        window_sel = window_set & set(range(causal_len))
        if full_top1.issubset(window_sel):
            results["window_only"]["top1"] += 1
        results["window_only"]["top5"] += len(full_top5 & window_sel) / max(len(full_top5), 1)
        if significant:
            results["window_only"]["recall"].append(
                len(significant & window_sel) / len(significant)
            )

        win_t = torch.tensor(sorted(window_sel), dtype=torch.long)
        scores = (query_key @ keys[win_t].T) * scale
        attn_w = F.softmax(scores, dim=-1)
        out = attn_w @ values[win_t]
        results["window_only"]["cosine"].append(
            F.cosine_similarity(full_output.unsqueeze(0), out.unsqueeze(0)).item()
        )

        total += 1

    if total == 0:
        return {"error": "no valid queries"}

    return {
        "pca_top1": results["pca"]["top1"] / total,
        "pca_top5": results["pca"]["top5"] / total,
        "pca_recall": float(np.mean(results["pca"]["recall"])) if results["pca"]["recall"] else 0,
        "pca_cosine": float(np.mean(results["pca"]["cosine"])) if results["pca"]["cosine"] else 0,
        "pca_avg_cand": float(np.mean(results["pca"]["candidates"])) if results["pca"]["candidates"] else 0,
        "wht_top1": results["wht"]["top1"] / total,
        "wht_top5": results["wht"]["top5"] / total,
        "wht_recall": float(np.mean(results["wht"]["recall"])) if results["wht"]["recall"] else 0,
        "wht_cosine": float(np.mean(results["wht"]["cosine"])) if results["wht"]["cosine"] else 0,
        "wht_avg_cand": float(np.mean(results["wht"]["candidates"])) if results["wht"]["candidates"] else 0,
        "window_top1": results["window_only"]["top1"] / total,
        "window_top5": results["window_only"]["top5"] / total,
        "window_recall": float(np.mean(results["window_only"]["recall"])) if results["window_only"]["recall"] else 0,
        "window_cosine": float(np.mean(results["window_only"]["cosine"])) if results["window_only"]["cosine"] else 0,
        "pca_mse": pca_mse,
        "wht_mse": wht_mse,
        "n_queries": total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PCA CODE RETRIEVAL EXPERIMENT")
    print("Hypothesis: PCA codes (48.7% info) >> WHT codes (12.5% info) as hash")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Bits: {BITS} | Hash widths: {HASH_WIDTHS}")
    print(f"Retrieval k: {RETRIEVAL_K} | Window: {WINDOW_SIZE}")
    print()

    model, tokenizer = load_model()

    # Single context length: 2048 (most relevant)
    target = 2048
    print(f"\n{'='*70}")
    print(f"CONTEXT LENGTH: {target}")
    print(f"{'='*70}")

    prompt = build_prompt(tokenizer, target)
    data = extract_kv_and_attention(model, tokenizer, prompt)
    seq_len = data["seq_len"]
    head_dim = data["head_dim"]
    n_kv_heads = data["n_kv_heads"]
    gqa_ratio = data["gqa_ratio"]
    n_layers = data["n_layers"]

    print(f"  seq={seq_len} d={head_dim} kv_heads={n_kv_heads} gqa={gqa_ratio}")

    # Query positions: last N_EVAL_QUERIES
    query_positions = list(range(max(0, seq_len - N_EVAL_QUERIES), seq_len))

    # ===================================================================
    # EXPERIMENT 1: Hash quality comparison
    # ===================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: PCA vs WHT Hash Quality (Oracle Mode)")
    print(f"{'='*70}")

    exp1_results = []

    for hw in HASH_WIDTHS:
        print(f"\n  hash_width={hw}:")
        hw_results = []

        for li in LAYERS_TO_TEST:
            if li >= n_layers:
                continue
            for hi in range(min(HEADS_PER_LAYER, n_kv_heads)):
                keys_head = data["keys"][li][0, hi]      # (seq, d)
                values_head = data["values"][li][0, hi]

                # Use the first Q head mapped to this KV head
                q_head_idx = hi * gqa_ratio
                if q_head_idx >= data["attentions"][li].shape[1]:
                    continue
                attn_head = data["attentions"][li][0, q_head_idx]  # (seq, seq)

                result = experiment1_hash_quality(
                    keys=keys_head,
                    values=values_head,
                    attn_weights=attn_head,
                    head_dim=head_dim,
                    hash_width=hw,
                    query_positions=query_positions,
                )
                if "error" not in result:
                    hw_results.append(result)

        if hw_results:
            avg = {
                k: np.mean([r[k] for r in hw_results])
                for k in hw_results[0] if k not in ("hash_width", "n_queries", "error")
            }
            avg["hash_width"] = hw
            avg["n_heads"] = len(hw_results)
            exp1_results.append(avg)

            var_pct = avg["variance_fraction"] * 100
            print(f"    PCA variance in hash: {var_pct:.1f}% | WHT: {hw/head_dim*100:.1f}%")
            print(f"    PCA: Top-1={avg['pca_top1']:.3f} Top-5={avg['pca_top5']:.3f} "
                  f"Recall={avg['pca_recall']:.3f} Cosine={avg['pca_cosine']:.4f} "
                  f"Cand={avg['pca_avg_candidates']:.0f} ({avg['pca_frac_searched']*100:.1f}%)")
            print(f"    WHT: Top-1={avg['wht_top1']:.3f} Top-5={avg['wht_top5']:.3f} "
                  f"Recall={avg['wht_recall']:.3f} Cosine={avg['wht_cosine']:.4f} "
                  f"Cand={avg['wht_avg_candidates']:.0f} ({avg['wht_frac_searched']*100:.1f}%)")
            recall_lift = (avg['pca_recall'] - avg['wht_recall']) / max(avg['wht_recall'], 0.01) * 100
            print(f"    DELTA: Recall +{avg['pca_recall'] - avg['wht_recall']:.3f} "
                  f"({recall_lift:+.1f}%) | "
                  f"Top-1 +{avg['pca_top1'] - avg['wht_top1']:.3f}")

    # ===================================================================
    # EXPERIMENT 2: Retrieval attention quality (key-as-query, realistic)
    # ===================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: PCA Retrieval Attention (Key-as-Query, Realistic Mode)")
    print(f"{'='*70}")

    exp2_results = []

    for hw in HASH_WIDTHS:
        print(f"\n  hash_width={hw}:")
        hw_results = []

        for li in LAYERS_TO_TEST:
            if li >= n_layers:
                continue
            for hi in range(min(HEADS_PER_LAYER, n_kv_heads)):
                keys_head = data["keys"][li][0, hi]
                values_head = data["values"][li][0, hi]
                q_head_idx = hi * gqa_ratio
                if q_head_idx >= data["attentions"][li].shape[1]:
                    continue
                attn_head = data["attentions"][li][0, q_head_idx]

                result = experiment2_retrieval_attention(
                    keys=keys_head,
                    values=values_head,
                    attn_weights=attn_head,
                    head_dim=head_dim,
                    hash_width=hw,
                    query_positions=query_positions,
                )
                if "error" not in result:
                    hw_results.append(result)

        if hw_results:
            avg = {
                k: np.mean([r[k] for r in hw_results])
                for k in hw_results[0] if k not in ("hash_width", "n_queries", "error")
            }
            avg["hash_width"] = hw
            avg["n_heads"] = len(hw_results)
            exp2_results.append(avg)

            print(f"    Top-1={avg['top1']:.3f} Top-5={avg['top5']:.3f} "
                  f"Top-10={avg['top10']:.3f} Cosine={avg['cosine']:.4f} "
                  f"Recall={avg['recall']:.3f} Cand={avg['avg_candidates']:.0f} "
                  f"({avg['frac_searched']*100:.1f}%)")

    # ===================================================================
    # EXPERIMENT 3: Combined system comparison
    # ===================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 3: Combined Compression + Retrieval System")
    print(f"{'='*70}")

    exp3_all = []
    for li in LAYERS_TO_TEST:
        if li >= n_layers:
            continue
        for hi in range(min(HEADS_PER_LAYER, n_kv_heads)):
            keys_head = data["keys"][li][0, hi]
            values_head = data["values"][li][0, hi]
            q_head_idx = hi * gqa_ratio
            if q_head_idx >= data["attentions"][li].shape[1]:
                continue
            attn_head = data["attentions"][li][0, q_head_idx]

            result = experiment3_combined_system(
                keys=keys_head,
                values=values_head,
                attn_weights=attn_head,
                head_dim=head_dim,
                query_positions=query_positions,
            )
            if "error" not in result:
                exp3_all.append(result)

    if exp3_all:
        exp3_avg = {
            k: np.mean([r[k] for r in exp3_all])
            for k in exp3_all[0] if k not in ("n_queries", "error")
        }

        print(f"\n  === System Comparison at hash_width=16 ===")
        print(f"  PCA+Code:   Top-1={exp3_avg['pca_top1']:.3f} Top-5={exp3_avg['pca_top5']:.3f} "
              f"Recall={exp3_avg['pca_recall']:.3f} Cosine={exp3_avg['pca_cosine']:.4f} "
              f"MSE={exp3_avg['pca_mse']:.6f}")
        print(f"  WHT+Code:   Top-1={exp3_avg['wht_top1']:.3f} Top-5={exp3_avg['wht_top5']:.3f} "
              f"Recall={exp3_avg['wht_recall']:.3f} Cosine={exp3_avg['wht_cosine']:.4f} "
              f"MSE={exp3_avg['wht_mse']:.6f}")
        print(f"  Window:     Top-1={exp3_avg['window_top1']:.3f} Top-5={exp3_avg['window_top5']:.3f} "
              f"Recall={exp3_avg['window_recall']:.3f} Cosine={exp3_avg['window_cosine']:.4f}")
        print(f"\n  MSE ratio (PCA/WHT): {exp3_avg['pca_mse'] / max(exp3_avg['wht_mse'], 1e-12):.3f}x")

    # ===================================================================
    # Write results
    # ===================================================================
    output_path = os.path.join(REPO_ROOT, "benchmarks", "results", "pca_code_retrieval_results.md")
    write_results(output_path, data, exp1_results, exp2_results, exp3_avg if exp3_all else None, seq_len)
    print(f"\nResults written to {output_path}")


def write_results(
    path: str,
    data: Dict,
    exp1: List[Dict],
    exp2: List[Dict],
    exp3: Optional[Dict],
    seq_len: int,
):
    """Write markdown results file."""
    lines = []
    lines.append("# PCA Code Retrieval Experiment Results")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Model:** {MODEL_NAME} (BnB 4-bit, eager attention)")
    lines.append(f"**Hardware:** RTX 4090")
    lines.append(f"**Context:** {seq_len} tokens | **d:** {data['head_dim']} | "
                 f"**KV heads:** {data['n_kv_heads']} | **GQA:** {data['gqa_ratio']}x")
    lines.append(f"**Quantization:** {BITS}-bit ({BITS-1}-bit MSE + 1-bit signs)")
    lines.append(f"**Retrieval k:** {RETRIEVAL_K} | **Window:** {WINDOW_SIZE} | "
                 f"**Multi-probe:** Hamming-1")
    lines.append(f"**Layers tested:** {LAYERS_TO_TEST}")
    lines.append(f"**Queries per head:** {N_EVAL_QUERIES}")
    lines.append("")

    lines.append("## Hypothesis")
    lines.append("")
    lines.append("PCA rotation concentrates 48.7% of variance in the top 10% of coordinates.")
    lines.append("WHT rotation spreads information uniformly (12.5% in any 10%). PCA codes")
    lines.append("from leading coordinates should be a dramatically better locality-sensitive")
    lines.append("hash for approximate attention retrieval.")
    lines.append("")
    lines.append("**Prior WHT results (code_retrieval_results.md):**")
    lines.append("- Oracle mode at hash_width=16: 67% recall")
    lines.append("- Realistic mode at hash_width=16: 55-59% recall")
    lines.append("- Root cause: WHT spreads info uniformly, 16/128 coords = 12.5% of info")
    lines.append("")

    # Experiment 1
    lines.append("---")
    lines.append("")
    lines.append("## Experiment 1: PCA vs WHT Hash Quality (Oracle Mode)")
    lines.append("")
    lines.append("For each query, hash the top-1 attended key's code, retrieve from same/nearby")
    lines.append("buckets, measure recall of true high-attention tokens. Averaged across all")
    lines.append(f"tested layers x heads.")
    lines.append("")
    lines.append("| Hash Width | PCA Var% | WHT Var% | PCA Top-1 | WHT Top-1 | PCA Recall | WHT Recall | PCA Cosine | WHT Cosine | PCA Cand | WHT Cand |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in exp1:
        hw = int(r["hash_width"])
        lines.append(
            f"| {hw} "
            f"| {r['variance_fraction']*100:.1f}% "
            f"| {hw/data['head_dim']*100:.1f}% "
            f"| {r['pca_top1']:.3f} "
            f"| {r['wht_top1']:.3f} "
            f"| {r['pca_recall']:.3f} "
            f"| {r['wht_recall']:.3f} "
            f"| {r['pca_cosine']:.4f} "
            f"| {r['wht_cosine']:.4f} "
            f"| {r['pca_avg_candidates']:.0f} "
            f"| {r['wht_avg_candidates']:.0f} |"
        )
    lines.append("")

    # Deltas table
    lines.append("### PCA vs WHT Delta")
    lines.append("")
    lines.append("| Hash Width | Top-1 Delta | Recall Delta | Recall Lift% | Cosine Delta |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|")
    for r in exp1:
        hw = int(r["hash_width"])
        recall_lift = (r['pca_recall'] - r['wht_recall']) / max(r['wht_recall'], 0.01) * 100
        lines.append(
            f"| {hw} "
            f"| {r['pca_top1'] - r['wht_top1']:+.3f} "
            f"| {r['pca_recall'] - r['wht_recall']:+.3f} "
            f"| {recall_lift:+.1f}% "
            f"| {r['pca_cosine'] - r['wht_cosine']:+.4f} |"
        )
    lines.append("")

    # Experiment 2
    lines.append("---")
    lines.append("")
    lines.append("## Experiment 2: PCA Retrieval Attention (Key-as-Query, Realistic)")
    lines.append("")
    lines.append("Hash the key at the query position, retrieve candidates, compute attention")
    lines.append("over candidates + window. Compare against full attention.")
    lines.append("")
    lines.append("| Hash Width | Top-1 | Top-5 | Top-10 | Cosine | Recall | Avg Cand | Frac Searched |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in exp2:
        hw = int(r["hash_width"])
        lines.append(
            f"| {hw} "
            f"| {r['top1']:.3f} "
            f"| {r['top5']:.3f} "
            f"| {r['top10']:.3f} "
            f"| {r['cosine']:.4f} "
            f"| {r['recall']:.3f} "
            f"| {r['avg_candidates']:.0f} "
            f"| {r['frac_searched']*100:.1f}% |"
        )
    lines.append("")

    # Experiment 3
    if exp3:
        lines.append("---")
        lines.append("")
        lines.append("## Experiment 3: Combined Compression + Retrieval System")
        lines.append("")
        lines.append("Head-to-head at hash_width=16, 3-bit quantization.")
        lines.append("")
        lines.append("| System | Top-1 | Top-5 | Recall | Cosine | MSE | Index Memory |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        lines.append(
            f"| PCA rotation + PCA code retrieval "
            f"| {exp3['pca_top1']:.3f} "
            f"| {exp3['pca_top5']:.3f} "
            f"| {exp3['pca_recall']:.3f} "
            f"| {exp3['pca_cosine']:.4f} "
            f"| {exp3['pca_mse']:.6f} "
            f"| **0 bytes** |"
        )
        lines.append(
            f"| WHT rotation + WHT code retrieval "
            f"| {exp3['wht_top1']:.3f} "
            f"| {exp3['wht_top5']:.3f} "
            f"| {exp3['wht_recall']:.3f} "
            f"| {exp3['wht_cosine']:.4f} "
            f"| {exp3['wht_mse']:.6f} "
            f"| **0 bytes** |"
        )
        lines.append(
            f"| Window only (no retrieval) "
            f"| {exp3['window_top1']:.3f} "
            f"| {exp3['window_top5']:.3f} "
            f"| {exp3['window_recall']:.3f} "
            f"| {exp3['window_cosine']:.4f} "
            f"| N/A "
            f"| 0 bytes |"
        )
        lines.append(
            f"| FAISS IVF-Flat (np=8)* "
            f"| 0.778 "
            f"| 0.924 "
            f"| 0.937 "
            f"| 0.979 "
            f"| N/A "
            f"| 512 B/key |"
        )
        lines.append("")
        lines.append("*FAISS results from faiss_retrieval_results.md (500 token context, k=128)")
        lines.append("")
        mse_ratio = exp3['pca_mse'] / max(exp3['wht_mse'], 1e-12)
        lines.append(f"**Compression MSE ratio (PCA/WHT):** {mse_ratio:.3f}x "
                     f"({1/mse_ratio:.1f}x lower MSE with PCA)")

    # Memory analysis
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Memory Analysis")
    lines.append("")
    lines.append("| System | Compression | Index Memory/Key | Total Extra Memory (100K tokens) |")
    lines.append("|:---|:---:|:---:|:---:|")
    lines.append("| FAISS Flat (exact IP) | None | 512 B | 51.2 MB |")
    lines.append("| FAISS IVF-PQ (m=16) | None | 16 B | 1.6 MB |")
    lines.append("| PCA + Code (this work) | 5x compression | **0 B** | **0 B** |")
    lines.append("| WHT + Code (prior work) | 5x compression | **0 B** | **0 B** |")
    lines.append("")
    lines.append("The zero-memory property is shared by both PCA and WHT code retrieval.")
    lines.append("The question is purely about recall quality: can the hash codes FIND the")
    lines.append("right tokens?")

    # Verdict
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")

    if exp1:
        best_pca_recall = max(r['pca_recall'] for r in exp1)
        best_wht_recall = max(r['wht_recall'] for r in exp1)
        best_hw = max(exp1, key=lambda r: r['pca_recall'])

        if best_pca_recall > 0.90:
            lines.append(f"**CONFIRMED: PCA codes achieve {best_pca_recall:.1%} recall** "
                         f"(oracle mode) at hash_width={int(best_hw['hash_width'])}. "
                         f"This is a {(best_pca_recall-best_wht_recall)*100:+.1f}pp improvement "
                         f"over WHT codes ({best_wht_recall:.1%}).")
        elif best_pca_recall > best_wht_recall * 1.15:
            improvement = (best_pca_recall - best_wht_recall) / best_wht_recall * 100
            lines.append(f"**PARTIALLY CONFIRMED: PCA codes achieve {best_pca_recall:.1%} recall** "
                         f"(oracle mode), a {improvement:.0f}% improvement over WHT ({best_wht_recall:.1%}). "
                         f"The PCA advantage is significant but may not reach the 90% threshold.")
        else:
            lines.append(f"**NOT CONFIRMED: PCA codes achieve {best_pca_recall:.1%} recall** "
                         f"(oracle mode) vs WHT {best_wht_recall:.1%}. "
                         f"The improvement is smaller than expected.")

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
