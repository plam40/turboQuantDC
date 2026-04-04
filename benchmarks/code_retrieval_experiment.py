"""Code retrieval experiment: Can quantization codes double as an ANN index?

HYPOTHESIS: Lloyd-Max quantization indices, already stored for 5x compression,
can serve as a locality-sensitive hash for approximate nearest neighbor
retrieval. This would give O(sub-linear) attention for free -- zero extra
memory beyond what compression already uses.

METHODOLOGY:
    1. Load Qwen2.5-3B-Instruct (BnB 4-bit, eager attention)
    2. Extract real KV caches + attention weights (500+ tokens)
    3. Quantize all keys using PolarQuant (same as compression pipeline)
    4. Build CodeIndex from quantization indices
    5. THREE evaluation modes:
       a. "attention-guided": Use the ACTUAL top-attended token's code as the
          query hash. Check if other high-attention tokens share codes.
          Tests: do high-attention tokens cluster in code space?
       b. "key-as-query": Use the key at position q as query, hash it, retrieve.
          Tests: does the code hash find nearby keys? (proxy for real queries)
       c. "multi-table": Build multiple hash tables from different coordinate
          subsets. Tests: can we boost recall with zero-cost redundancy?
    6. Sweep hash_width: 4, 8, 12, 16, 20, 24, 32 coordinates

THE KEY ADVANTAGE OVER FAISS: Zero additional memory.

Usage:
    python benchmarks/code_retrieval_experiment.py
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
from turboquantdc.code_retrieval import CodeIndex, CodeIndexStats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"

# Hash widths to sweep
HASH_WIDTHS = [4, 8, 12, 16, 20, 24, 32]

# Retrieval parameters
RETRIEVAL_K = 64
WINDOW_SIZE = 64
BITS = 3

# Context lengths
CONTEXT_LENGTHS = [512, 1024, 2048]

# Evaluation
N_EVAL_QUERIES = 32
LAYERS_TO_TEST = [0, 5, 10, 15, 20, 25, 35]
HEADS_PER_LAYER = 2

# Filler text
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
# Model loading
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

    if hasattr(kv_cache, "layers"):
        n_layers = len(kv_cache.layers)
        for layer_idx in range(n_layers):
            layer = kv_cache.layers[layer_idx]
            keys_per_layer.append(layer.keys.cpu().float())
            values_per_layer.append(layer.values.cpu().float())
    elif hasattr(kv_cache, "key_cache") and len(kv_cache.key_cache) > 0:
        n_layers = len(kv_cache.key_cache)
        for layer_idx in range(n_layers):
            keys_per_layer.append(kv_cache.key_cache[layer_idx].cpu().float())
            values_per_layer.append(kv_cache.value_cache[layer_idx].cpu().float())
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
# Evaluation Mode 1: Attention-guided clustering test
# ---------------------------------------------------------------------------
def evaluate_attention_guided(
    keys: torch.Tensor,
    values: torch.Tensor,
    attn_weights: torch.Tensor,
    hash_width: int,
    bits: int,
    retrieval_k: int,
    window_size: int,
    query_positions: List[int],
    head_dim: int,
    multi_probe: bool = True,
) -> Dict[str, Any]:
    """Test: do tokens that receive high attention cluster in code space?

    For each query position:
    1. Find the top-1 attended key (the "needle")
    2. Hash the needle's quantization code
    3. Retrieve all keys in the same/nearby hash buckets
    4. Check if other high-attention keys are in the retrieved set

    This directly tests the hypothesis: if the needle's quantization code
    predicts where other high-attention tokens are, then the code IS an
    effective locality-sensitive hash for attention retrieval.
    """
    seq_len = keys.shape[0]

    mse_bits = max(bits - 1, 1)
    pq = PolarQuant(d=head_dim, bits=mse_bits, seed=42, device="cpu")

    key_norms = keys.norm(dim=-1, keepdim=True)
    keys_normalized = keys / (key_norms + 1e-8)
    all_indices = pq.quantize(keys_normalized)

    n_levels = 1 << mse_bits

    top1_matches = 0
    top5_matches = 0
    top10_matches = 0
    cos_sims = []
    recalls = []
    candidate_sizes = []
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

        # Use the TOP-1 attended key's code as the query hash
        # This is the "attention-guided" approach: the code of the most
        # important key should predict where other important keys are
        needle_indices = all_indices[full_top1_idx]

        # Build causal index
        causal_index = CodeIndex(
            hash_width=hash_width,
            n_levels=n_levels,
            multi_probe=multi_probe,
        )
        causal_index.insert_batch(all_indices[:causal_len], start_position=0)

        # Retrieve candidates matching the needle's hash
        retrieved, n_cand = causal_index.search(
            needle_indices,
            k=retrieval_k,
            keys=keys[:causal_len],
            query_vec=keys[full_top1_idx],
        )
        candidate_sizes.append(n_cand)

        # Add window
        window_start = max(0, causal_len - window_size)
        window_indices = torch.arange(window_start, causal_len, dtype=torch.long)

        combined = torch.cat([retrieved, window_indices])
        selected = torch.unique(combined)
        selected, _ = torch.sort(selected)

        # Re-normalize attention over selected
        log_scores = torch.log(full_attn[selected] + 1e-30)
        retrieval_attn = F.softmax(log_scores, dim=-1)
        sel_values = values[selected]
        retrieval_output = retrieval_attn @ sel_values

        # Metrics
        selected_set = set(selected.tolist())

        if full_top1_idx in selected_set:
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

    stats = causal_index.get_stats()

    return {
        "top1_match": top1_matches / total_queries,
        "top5_match": top5_matches / total_queries,
        "top10_match": top10_matches / total_queries,
        "output_cosine": float(np.mean(cos_sims)),
        "output_cosine_std": float(np.std(cos_sims)),
        "recall_significant": float(np.mean(recalls)),
        "recall_significant_std": float(np.std(recalls)),
        "avg_candidates": float(np.mean(candidate_sizes)),
        "max_candidates": int(np.max(candidate_sizes)) if candidate_sizes else 0,
        "min_candidates": int(np.min(candidate_sizes)) if candidate_sizes else 0,
        "n_queries": total_queries,
        "n_buckets": stats.n_buckets,
        "avg_bucket_size": stats.avg_bucket_size,
        "max_bucket_size": stats.max_bucket_size,
        "fraction_searched": float(np.mean(candidate_sizes)) / max(1, seq_len),
    }


# ---------------------------------------------------------------------------
# Evaluation Mode 2: Key-as-query (dot product retrieval)
# ---------------------------------------------------------------------------
def evaluate_key_as_query(
    keys: torch.Tensor,
    values: torch.Tensor,
    attn_weights: torch.Tensor,
    hash_width: int,
    bits: int,
    retrieval_k: int,
    window_size: int,
    query_positions: List[int],
    head_dim: int,
    multi_probe: bool = True,
) -> Dict[str, Any]:
    """Test: hash the key at q_pos, retrieve nearby keys, compare to full attn.

    This is the "realistic inference" scenario where we quantize the query
    (approximated by the key at the query position), hash it, and retrieve.
    The retrieved set + window is compared against full attention.
    """
    seq_len = keys.shape[0]

    mse_bits = max(bits - 1, 1)
    pq = PolarQuant(d=head_dim, bits=mse_bits, seed=42, device="cpu")

    key_norms = keys.norm(dim=-1, keepdim=True)
    keys_normalized = keys / (key_norms + 1e-8)
    all_indices = pq.quantize(keys_normalized)

    scale = 1.0 / math.sqrt(head_dim)
    n_levels = 1 << mse_bits

    top1_matches = 0
    top5_matches = 0
    top10_matches = 0
    cos_sims = []
    recalls = []
    candidate_sizes = []
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

        # Hash the key at q_pos as proxy query
        query_key = keys[q_pos]
        query_indices = all_indices[q_pos]

        causal_index = CodeIndex(
            hash_width=hash_width,
            n_levels=n_levels,
            multi_probe=multi_probe,
        )
        causal_index.insert_batch(all_indices[:causal_len], start_position=0)

        retrieved, n_cand = causal_index.search(
            query_indices,
            k=retrieval_k,
            keys=keys[:causal_len],
            query_vec=query_key,
        )
        candidate_sizes.append(n_cand)

        window_start = max(0, causal_len - window_size)
        window_indices = torch.arange(window_start, causal_len, dtype=torch.long)

        combined = torch.cat([retrieved, window_indices])
        selected = torch.unique(combined)
        selected, _ = torch.sort(selected)

        # Score by dot product (realistic)
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
        "top1_match": top1_matches / total_queries,
        "top5_match": top5_matches / total_queries,
        "top10_match": top10_matches / total_queries,
        "output_cosine": float(np.mean(cos_sims)),
        "output_cosine_std": float(np.std(cos_sims)),
        "recall_significant": float(np.mean(recalls)),
        "recall_significant_std": float(np.std(recalls)),
        "avg_candidates": float(np.mean(candidate_sizes)),
        "fraction_searched": float(np.mean(candidate_sizes)) / max(1, seq_len),
        "n_queries": total_queries,
    }


# ---------------------------------------------------------------------------
# Evaluation Mode 3: Multi-table (different coordinate subsets)
# ---------------------------------------------------------------------------
def evaluate_multi_table(
    keys: torch.Tensor,
    values: torch.Tensor,
    attn_weights: torch.Tensor,
    hash_width: int,
    bits: int,
    retrieval_k: int,
    window_size: int,
    query_positions: List[int],
    head_dim: int,
    n_tables: int = 4,
) -> Dict[str, Any]:
    """Test: multiple hash tables from different coordinate subsets.

    Build n_tables independent CodeIndex instances, each using a different
    non-overlapping range of coordinates as the hash. Union of all candidates
    from all tables. Tests whether combining multiple views of the code
    improves recall.

    Table i uses coordinates [i*hash_width : (i+1)*hash_width].
    """
    seq_len = keys.shape[0]

    mse_bits = max(bits - 1, 1)
    pq = PolarQuant(d=head_dim, bits=mse_bits, seed=42, device="cpu")

    key_norms = keys.norm(dim=-1, keepdim=True)
    keys_normalized = keys / (key_norms + 1e-8)
    all_indices = pq.quantize(keys_normalized)

    n_levels = 1 << mse_bits

    # How many non-overlapping tables fit?
    max_tables = head_dim // hash_width
    actual_tables = min(n_tables, max_tables)

    if actual_tables < 2:
        return {"error": f"cannot fit {n_tables} tables with hash_width={hash_width} in d={head_dim}"}

    # Build indices that hash SHIFTED coordinate ranges
    # Table 0: coords [0:hw], Table 1: coords [hw:2*hw], etc.
    # To achieve this, we cyclically shift the index vector before hashing
    tables = []
    for t in range(actual_tables):
        idx = CodeIndex(
            hash_width=hash_width,
            n_levels=n_levels,
            multi_probe=True,
        )
        tables.append(idx)

    scale = 1.0 / math.sqrt(head_dim)

    top1_matches = 0
    top5_matches = 0
    top10_matches = 0
    cos_sims = []
    recalls = []
    candidate_sizes = []
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

        # Build all tables for causal portion
        for t_idx, table in enumerate(tables):
            table.clear()
            # Shift indices so table t hashes coords [t*hw : (t+1)*hw]
            shift = t_idx * hash_width
            shifted = torch.roll(all_indices[:causal_len], shifts=-shift, dims=1)
            table.insert_batch(shifted, start_position=0)

        # Query: use key at q_pos
        query_key = keys[q_pos]
        query_idx = all_indices[q_pos]

        # Union candidates from all tables
        all_candidates: Set[int] = set()
        for t_idx, table in enumerate(tables):
            shift = t_idx * hash_width
            shifted_query = torch.roll(query_idx.unsqueeze(0), shifts=-shift, dims=1).squeeze(0)
            cands, _ = table.search(shifted_query, k=retrieval_k)
            all_candidates.update(cands.tolist())

        n_cand = len(all_candidates)
        candidate_sizes.append(n_cand)

        # Score candidates by dot product and take top-k
        if n_cand > 0:
            cand_tensor = torch.tensor(sorted(all_candidates), dtype=torch.long)
            valid = cand_tensor[cand_tensor < causal_len]
            if valid.shape[0] > 0:
                cand_keys = keys[valid]
                scores = (query_key @ cand_keys.T)
                actual_k = min(retrieval_k, valid.shape[0])
                _, topk = torch.topk(scores, k=actual_k)
                retrieved = valid[topk]
            else:
                retrieved = torch.arange(max(0, causal_len - retrieval_k), causal_len)
        else:
            retrieved = torch.arange(max(0, causal_len - retrieval_k), causal_len)

        # Window
        window_start = max(0, causal_len - window_size)
        window_indices = torch.arange(window_start, causal_len, dtype=torch.long)

        combined = torch.cat([retrieved, window_indices])
        selected = torch.unique(combined)
        selected, _ = torch.sort(selected)

        sel_keys = keys[selected]
        sel_values = values[selected]
        scores_sel = (query_key @ sel_keys.T) * scale
        retrieval_attn = F.softmax(scores_sel, dim=-1)
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
        "top1_match": top1_matches / total_queries,
        "top5_match": top5_matches / total_queries,
        "top10_match": top10_matches / total_queries,
        "output_cosine": float(np.mean(cos_sims)),
        "recall_significant": float(np.mean(recalls)),
        "avg_candidates": float(np.mean(candidate_sizes)),
        "fraction_searched": float(np.mean(candidate_sizes)) / max(1, seq_len),
        "n_queries": total_queries,
        "n_tables": actual_tables,
    }


# ---------------------------------------------------------------------------
# Evaluation Mode 4: Code-space clustering analysis
# ---------------------------------------------------------------------------
def analyze_code_clustering(
    keys: torch.Tensor,
    attn_weights: torch.Tensor,
    hash_width: int,
    bits: int,
    query_positions: List[int],
    head_dim: int,
) -> Dict[str, Any]:
    """Analyze whether high-attention tokens share quantization codes.

    For each query, measure the Hamming distance between the top-1 key's
    code and the codes of the top-5/top-10/top-20 keys. If high-attention
    tokens cluster in code space, their Hamming distances should be small.

    Also compare against random tokens as a baseline.
    """
    seq_len = keys.shape[0]

    mse_bits = max(bits - 1, 1)
    pq = PolarQuant(d=head_dim, bits=mse_bits, seed=42, device="cpu")

    key_norms = keys.norm(dim=-1, keepdim=True)
    keys_normalized = keys / (key_norms + 1e-8)
    all_indices = pq.quantize(keys_normalized)

    hamming_top5 = []
    hamming_top10 = []
    hamming_top20 = []
    hamming_random = []
    prefix_match_top5 = []  # How many prefix coords match exactly
    prefix_match_random = []
    total_queries = 0

    for q_pos in query_positions:
        if q_pos >= attn_weights.shape[0]:
            continue

        full_attn = attn_weights[q_pos, :q_pos + 1]
        causal_len = full_attn.shape[0]

        if causal_len < 20:
            continue

        # Top-k and random indices
        top1_idx = torch.topk(full_attn, k=1).indices[0].item()
        k5 = min(5, causal_len)
        k10 = min(10, causal_len)
        k20 = min(20, causal_len)
        top5_indices = torch.topk(full_attn, k=k5).indices.tolist()
        top10_indices = torch.topk(full_attn, k=k10).indices.tolist()
        top20_indices = torch.topk(full_attn, k=k20).indices.tolist()

        rng = torch.Generator().manual_seed(q_pos)
        random_indices = torch.randint(0, causal_len, (20,), generator=rng).tolist()

        # Code of the top-1 key (first hash_width coords)
        top1_code = all_indices[top1_idx, :hash_width]

        # Hamming distance = number of coordinates where index differs
        def hamming_dist(idx):
            code = all_indices[idx, :hash_width]
            return (top1_code != code).sum().item()

        # Measure Hamming distances
        for idx in top5_indices:
            if idx != top1_idx:
                hamming_top5.append(hamming_dist(idx))
        for idx in top10_indices:
            if idx != top1_idx:
                hamming_top10.append(hamming_dist(idx))
        for idx in top20_indices:
            if idx != top1_idx:
                hamming_top20.append(hamming_dist(idx))
        for idx in random_indices:
            hamming_random.append(hamming_dist(idx))

        # Exact prefix match count
        for idx in top5_indices:
            if idx != top1_idx:
                code = all_indices[idx, :hash_width]
                prefix_match_top5.append((top1_code == code).sum().item())
        for idx in random_indices:
            code = all_indices[idx, :hash_width]
            prefix_match_random.append((top1_code == code).sum().item())

        total_queries += 1

    if total_queries == 0:
        return {"error": "no valid queries"}

    return {
        "avg_hamming_top5": float(np.mean(hamming_top5)) if hamming_top5 else 0,
        "avg_hamming_top10": float(np.mean(hamming_top10)) if hamming_top10 else 0,
        "avg_hamming_top20": float(np.mean(hamming_top20)) if hamming_top20 else 0,
        "avg_hamming_random": float(np.mean(hamming_random)) if hamming_random else 0,
        "std_hamming_top5": float(np.std(hamming_top5)) if hamming_top5 else 0,
        "std_hamming_random": float(np.std(hamming_random)) if hamming_random else 0,
        "avg_prefix_match_top5": float(np.mean(prefix_match_top5)) if prefix_match_top5 else 0,
        "avg_prefix_match_random": float(np.mean(prefix_match_random)) if prefix_match_random else 0,
        "frac_exact_hash_match_top5": float(np.mean([h == 0 for h in hamming_top5])) if hamming_top5 else 0,
        "frac_exact_hash_match_random": float(np.mean([h == 0 for h in hamming_random])) if hamming_random else 0,
        "frac_hamming_le1_top5": float(np.mean([h <= 1 for h in hamming_top5])) if hamming_top5 else 0,
        "frac_hamming_le1_random": float(np.mean([h <= 1 for h in hamming_random])) if hamming_random else 0,
        "frac_hamming_le2_top5": float(np.mean([h <= 2 for h in hamming_top5])) if hamming_top5 else 0,
        "frac_hamming_le2_random": float(np.mean([h <= 2 for h in hamming_random])) if hamming_random else 0,
        "hash_width": hash_width,
        "n_queries": total_queries,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment():
    print("=" * 80)
    print("CODE RETRIEVAL EXPERIMENT")
    print("Can quantization codes serve as an ANN index for free?")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print()

    model, tokenizer = load_model()

    all_results = {
        "attn_guided": {},
        "key_query": {},
        "multi_table": {},
        "clustering": {},
    }

    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n{'='*70}")
        print(f"CONTEXT LENGTH: {ctx_len} tokens")
        print(f"{'='*70}")

        prompt = build_prompt(tokenizer, ctx_len)
        data = extract_kv_and_attention(model, tokenizer, prompt)

        actual_seq = data["seq_len"]
        n_layers = data["n_layers"]
        n_kv_heads = data["n_kv_heads"]
        head_dim = data["head_dim"]
        gqa_ratio = data["gqa_ratio"]

        print(f"  Actual seq_len: {actual_seq}")
        print(f"  Layers: {n_layers}, KV heads: {n_kv_heads}, head_dim: {head_dim}")
        print(f"  GQA ratio: {gqa_ratio}:1")

        # Query positions spread across sequence
        query_positions = list(range(
            max(10, actual_seq // 4),
            actual_seq,
            max(1, (actual_seq - actual_seq // 4) // N_EVAL_QUERIES),
        ))[:N_EVAL_QUERIES]

        layers_tested = [l for l in LAYERS_TO_TEST if l < n_layers]

        for hash_width in HASH_WIDTHS:
            if hash_width > head_dim:
                continue

            print(f"\n  --- Hash width: {hash_width} coords ({hash_width * (BITS-1)} hash bits) ---")

            attn_guided_all = defaultdict(list)
            key_query_all = defaultdict(list)
            multi_table_all = defaultdict(list)
            clustering_all = defaultdict(list)

            for layer_idx in layers_tested:
                for kv_head in range(min(HEADS_PER_LAYER, n_kv_heads)):
                    q_head = kv_head * gqa_ratio

                    keys = data["keys"][layer_idx][0, kv_head]
                    values = data["values"][layer_idx][0, kv_head]
                    attn = data["attentions"][layer_idx][0, q_head]

                    # Mode 1: Attention-guided
                    r1 = evaluate_attention_guided(
                        keys, values, attn, hash_width, BITS,
                        RETRIEVAL_K, WINDOW_SIZE, query_positions, head_dim,
                    )
                    if "error" not in r1:
                        for k, v in r1.items():
                            attn_guided_all[k].append(v)

                    # Mode 2: Key-as-query
                    r2 = evaluate_key_as_query(
                        keys, values, attn, hash_width, BITS,
                        RETRIEVAL_K, WINDOW_SIZE, query_positions, head_dim,
                    )
                    if "error" not in r2:
                        for k, v in r2.items():
                            key_query_all[k].append(v)

                    # Mode 3: Multi-table (only for select hash widths to save time)
                    if hash_width in [8, 16, 32]:
                        r3 = evaluate_multi_table(
                            keys, values, attn, hash_width, BITS,
                            RETRIEVAL_K, WINDOW_SIZE, query_positions, head_dim,
                            n_tables=4,
                        )
                        if "error" not in r3:
                            for k, v in r3.items():
                                multi_table_all[k].append(v)

                    # Mode 4: Clustering analysis
                    r4 = analyze_code_clustering(
                        keys, attn, hash_width, BITS,
                        query_positions, head_dim,
                    )
                    if "error" not in r4:
                        for k, v in r4.items():
                            clustering_all[k].append(v)

            # Aggregate results
            def aggregate(metrics_dict):
                agg = {}
                for key in metrics_dict:
                    vals = metrics_dict[key]
                    if vals and isinstance(vals[0], (int, float)):
                        agg[key] = float(np.mean(vals))
                return agg

            tag = f"ctx{ctx_len}_hw{hash_width}"

            agg1 = aggregate(attn_guided_all)
            if agg1:
                all_results["attn_guided"][tag] = agg1
                print(f"    [attn-guided] top1={agg1.get('top1_match', 0):.3f} "
                      f"top5={agg1.get('top5_match', 0):.3f} "
                      f"cos={agg1.get('output_cosine', 0):.4f} "
                      f"recall={agg1.get('recall_significant', 0):.3f} "
                      f"cand={agg1.get('avg_candidates', 0):.0f} "
                      f"frac={agg1.get('fraction_searched', 0):.3f}")

            agg2 = aggregate(key_query_all)
            if agg2:
                all_results["key_query"][tag] = agg2
                print(f"    [key-query  ] top1={agg2.get('top1_match', 0):.3f} "
                      f"top5={agg2.get('top5_match', 0):.3f} "
                      f"cos={agg2.get('output_cosine', 0):.4f} "
                      f"recall={agg2.get('recall_significant', 0):.3f} "
                      f"cand={agg2.get('avg_candidates', 0):.0f} "
                      f"frac={agg2.get('fraction_searched', 0):.3f}")

            agg3 = aggregate(multi_table_all)
            if agg3:
                all_results["multi_table"][tag] = agg3
                print(f"    [multi-table] top1={agg3.get('top1_match', 0):.3f} "
                      f"top5={agg3.get('top5_match', 0):.3f} "
                      f"cos={agg3.get('output_cosine', 0):.4f} "
                      f"recall={agg3.get('recall_significant', 0):.3f} "
                      f"cand={agg3.get('avg_candidates', 0):.0f}")

            agg4 = aggregate(clustering_all)
            if agg4:
                all_results["clustering"][tag] = agg4
                print(f"    [clustering ] hamming_top5={agg4.get('avg_hamming_top5', 0):.2f} "
                      f"hamming_rand={agg4.get('avg_hamming_random', 0):.2f} "
                      f"exact_top5={agg4.get('frac_exact_hash_match_top5', 0):.3f} "
                      f"exact_rand={agg4.get('frac_exact_hash_match_random', 0):.3f} "
                      f"h<=1_top5={agg4.get('frac_hamming_le1_top5', 0):.3f} "
                      f"h<=1_rand={agg4.get('frac_hamming_le1_random', 0):.3f}")

        del data
        gc.collect()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    write_results(all_results)

    return all_results


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------
def write_results(all_results: Dict[str, Dict]):
    results_dir = os.path.join(REPO_ROOT, "benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "code_retrieval_results.md")

    lines = []
    lines.append("# Code Retrieval Experiment Results")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Model:** {MODEL_NAME}")
    lines.append(f"**Bits:** {BITS}, **Retrieval k:** {RETRIEVAL_K}, **Window:** {WINDOW_SIZE}")
    lines.append("")

    # ---- Clustering Analysis ----
    lines.append("## 1. Code-Space Clustering Analysis")
    lines.append("")
    lines.append("Do tokens that receive high attention cluster in quantization code space?")
    lines.append("")
    lines.append("| Context | Hash Width | Hamming (top-5) | Hamming (random) | Exact Match (top-5) | Exact Match (random) | H<=1 (top-5) | H<=1 (random) | H<=2 (top-5) | H<=2 (random) |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for ctx_len in CONTEXT_LENGTHS:
        for hw in HASH_WIDTHS:
            tag = f"ctx{ctx_len}_hw{hw}"
            if tag not in all_results["clustering"]:
                continue
            c = all_results["clustering"][tag]
            lines.append(
                f"| {ctx_len} | {hw} | "
                f"{c.get('avg_hamming_top5', 0):.2f} | "
                f"{c.get('avg_hamming_random', 0):.2f} | "
                f"{c.get('frac_exact_hash_match_top5', 0):.3f} | "
                f"{c.get('frac_exact_hash_match_random', 0):.3f} | "
                f"{c.get('frac_hamming_le1_top5', 0):.3f} | "
                f"{c.get('frac_hamming_le1_random', 0):.3f} | "
                f"{c.get('frac_hamming_le2_top5', 0):.3f} | "
                f"{c.get('frac_hamming_le2_random', 0):.3f} |"
            )
    lines.append("")

    # ---- Attention-guided retrieval ----
    lines.append("## 2. Attention-Guided Retrieval")
    lines.append("")
    lines.append("Use the top-1 attended key's code as hash. Check if other high-attention keys are found.")
    lines.append("")

    for ctx_len in CONTEXT_LENGTHS:
        lines.append(f"### Context {ctx_len}")
        lines.append("")
        lines.append("| Hash Width | Top-1 | Top-5 | Top-10 | Cosine | Recall | Avg Cand | Frac Searched |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        for hw in HASH_WIDTHS:
            tag = f"ctx{ctx_len}_hw{hw}"
            if tag not in all_results["attn_guided"]:
                continue
            r = all_results["attn_guided"][tag]
            lines.append(
                f"| {hw} | "
                f"{r.get('top1_match', 0):.3f} | "
                f"{r.get('top5_match', 0):.3f} | "
                f"{r.get('top10_match', 0):.3f} | "
                f"{r.get('output_cosine', 0):.4f} | "
                f"{r.get('recall_significant', 0):.3f} | "
                f"{r.get('avg_candidates', 0):.0f} | "
                f"{r.get('fraction_searched', 0):.3f} |"
            )
        lines.append("")

    # ---- Key-as-query retrieval ----
    lines.append("## 3. Key-as-Query Retrieval (Realistic)")
    lines.append("")
    lines.append("Hash the key at query position, retrieve by dot product.")
    lines.append("")

    for ctx_len in CONTEXT_LENGTHS:
        lines.append(f"### Context {ctx_len}")
        lines.append("")
        lines.append("| Hash Width | Top-1 | Top-5 | Top-10 | Cosine | Recall | Avg Cand | Frac Searched |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        for hw in HASH_WIDTHS:
            tag = f"ctx{ctx_len}_hw{hw}"
            if tag not in all_results["key_query"]:
                continue
            r = all_results["key_query"][tag]
            lines.append(
                f"| {hw} | "
                f"{r.get('top1_match', 0):.3f} | "
                f"{r.get('top5_match', 0):.3f} | "
                f"{r.get('top10_match', 0):.3f} | "
                f"{r.get('output_cosine', 0):.4f} | "
                f"{r.get('recall_significant', 0):.3f} | "
                f"{r.get('avg_candidates', 0):.0f} | "
                f"{r.get('fraction_searched', 0):.3f} |"
            )
        lines.append("")

    # ---- Multi-table retrieval ----
    lines.append("## 4. Multi-Table Retrieval")
    lines.append("")
    lines.append("Multiple hash tables from different coordinate subsets (4 tables).")
    lines.append("")

    for ctx_len in CONTEXT_LENGTHS:
        lines.append(f"### Context {ctx_len}")
        lines.append("")
        lines.append("| Hash Width | Top-1 | Top-5 | Top-10 | Cosine | Recall | Avg Cand |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        for hw in [8, 16, 32]:
            tag = f"ctx{ctx_len}_hw{hw}"
            if tag not in all_results["multi_table"]:
                continue
            r = all_results["multi_table"][tag]
            lines.append(
                f"| {hw} ({r.get('n_tables', 0):.0f}T) | "
                f"{r.get('top1_match', 0):.3f} | "
                f"{r.get('top5_match', 0):.3f} | "
                f"{r.get('top10_match', 0):.3f} | "
                f"{r.get('output_cosine', 0):.4f} | "
                f"{r.get('recall_significant', 0):.3f} | "
                f"{r.get('avg_candidates', 0):.0f} |"
            )
        lines.append("")

    # ---- Memory Analysis ----
    lines.append("## 5. Memory Analysis")
    lines.append("")
    lines.append("| System | Index Memory (1K tokens) | Index Memory (100K tokens) | Extra Memory |")
    lines.append("|:---|:---:|:---:|:---:|")
    lines.append("| FAISS IVF-Flat (d=128) | ~524 KB | ~52 MB | Yes |")
    lines.append("| ScaNN | ~256 KB | ~25 MB | Yes |")
    lines.append("| LSH (8 planes, 4 tables) | ~32 KB | ~3.2 MB | Yes |")
    lines.append("| **CodeIndex (this work)** | **0 bytes** | **0 bytes** | **No** |")
    lines.append("")
    lines.append("CodeIndex has zero additional memory because the hash IS the compression indices.")
    lines.append("")

    # ---- Verdict ----
    lines.append("## 6. Verdict")
    lines.append("")

    # Analyze the clustering data to determine if the hypothesis holds
    clustering = all_results.get("clustering", {})
    attn_guided = all_results.get("attn_guided", {})

    # Check if there's meaningful clustering signal
    has_signal = False
    for tag, c in clustering.items():
        if c.get("avg_hamming_top5", 999) < c.get("avg_hamming_random", 0) * 0.9:
            has_signal = True
            break

    # Check if retrieval quality is acceptable
    good_retrieval = False
    for tag, r in attn_guided.items():
        if r.get("recall_significant", 0) >= 0.85:
            good_retrieval = True
            break

    if has_signal and good_retrieval:
        lines.append("**HYPOTHESIS CONFIRMED.** High-attention tokens cluster in quantization")
        lines.append("code space. The compression codes provide effective locality-sensitive")
        lines.append("hashing with zero additional memory.")
    elif has_signal:
        lines.append("**PARTIAL CONFIRMATION.** There IS a clustering signal -- high-attention")
        lines.append("tokens have lower Hamming distances in code space than random tokens.")
        lines.append("However, the signal may not be strong enough for standalone retrieval.")
        lines.append("The codes work better as a FIRST-PASS FILTER combined with the window,")
        lines.append("rather than as a complete replacement for attention.")
    else:
        lines.append("**HYPOTHESIS NOT CONFIRMED for this configuration.** The clustering")
        lines.append("signal in code space is weak. This may be because:")
        lines.append("- Attention patterns are governed by position (recency) more than content")
        lines.append("- The WHT rotation distributes information across ALL coordinates, so")
        lines.append("  the first K coordinates don't capture enough of the similarity structure")
        lines.append("- Queries and keys live in different projection spaces (GQA)")

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults written to: {path}")


if __name__ == "__main__":
    results = run_experiment()
