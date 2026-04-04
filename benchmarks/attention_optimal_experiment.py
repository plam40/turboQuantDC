"""Attention-optimal quantization experiments on real Qwen2.5-3B KV caches.

Challenges the assumption that MSE-optimal quantization is best for attention.
Tests three approaches:
1. Mean-removed quantization (exploit softmax shift-invariance)
2. Importance-weighted quantization (allocate bits by attention mass)
3. Rank-preserving analysis (Spearman correlation of attention orderings)

Usage:
    python benchmarks/attention_optimal_experiment.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from turboquantdc.attention_optimal import (
    CombinedOptimalQuantizer,
    ImportanceWeightedQuantizer,
    MeanRemovedQuantizer,
    StandardQuantizer,
    attention_metrics,
    compute_attention_scores,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TARGET_LENGTHS = [2048, 4096]
BIT_WIDTHS = [2, 3, 4]
N_QUERY_SAMPLES = 8  # Number of random query positions to average over

NEEDLE = "The secret project code name is AURORA-7749."
NEEDLE_MARKER = "AURORA-7749"
QUESTION = "What is the secret project code name?"

FILLER = (
    "The quarterly financial review meeting covered several topics including "
    "budget allocations for the upcoming fiscal year, departmental spending reports, "
    "and projected revenue streams from various business units. The committee discussed "
    "infrastructure upgrades planned for the western regional offices and noted that "
    "maintenance schedules should be coordinated with the facilities management team. "
    "Several action items were assigned to team leads for follow-up before the next "
    "meeting cycle.\n\n"
)


# ---------------------------------------------------------------------------
# Prompt and model loading (reused from real_model.py)
# ---------------------------------------------------------------------------
def build_prompt(tokenizer, target_tokens: int = 2048, needle_pos: float = 0.25) -> str:
    filler_len = len(tokenizer.encode(FILLER, add_special_tokens=False))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)
    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Memo ---\n{NEEDLE}\n--- End ---\n\n")
        parts.append(FILLER)
    haystack = "".join(parts)
    return (
        f"<|im_start|>user\n{haystack}\n"
        f"Question: {QUESTION}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME} (4-bit NF4)...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    load_time = time.time() - t0
    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    head_dim = config.hidden_size // n_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Layers: {n_layers} | Heads: {n_heads} | KV heads: {n_kv_heads} | head_dim: {head_dim}")
    return model, tokenizer, n_layers, n_heads, n_kv_heads, head_dim


def extract_kv_cache(model, tokenizer, target_tokens: int):
    """Run forward pass and extract KV cache."""
    prompt = build_prompt(tokenizer, target_tokens=target_tokens)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=target_tokens + 256,
    ).to("cuda")
    seq_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)

    cache = outputs.past_key_values

    # Handle different cache formats across transformers versions
    if hasattr(cache, "key_cache"):
        # Older DynamicCache with .key_cache list
        get_keys = lambda i: cache.key_cache[i]
        n_layers = len(cache.key_cache)
    elif hasattr(cache, "layers") and len(cache.layers) > 0 and hasattr(cache.layers[0], "keys"):
        # Newer DynamicCache with .layers[i].keys / .layers[i].values
        get_keys = lambda i: cache.layers[i].keys
        n_layers = len(cache.layers)
    else:
        # Legacy tuple-of-tuples: ((keys, values), ...)
        get_keys = lambda i: cache[i][0]
        n_layers = len(cache)

    return get_keys, n_layers, seq_len


# ---------------------------------------------------------------------------
# Per-head experiment runner
# ---------------------------------------------------------------------------
def run_head_experiment(
    keys_fp: torch.Tensor,
    head_dim: int,
    bits: int,
    seed: int,
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """Run all quantization strategies on one head's keys.

    Args:
        keys_fp: (seq_len, head_dim) float32 keys for one head.
        head_dim: d.
        bits: Bit-width.
        seed: Unique seed for this head.
        device: Compute device.

    Returns:
        Dict mapping method name -> metrics dict.
    """
    seq_len = keys_fp.shape[0]
    keys = keys_fp.to(device).float()

    results = {}

    # Sample multiple query positions for robust metrics
    torch.manual_seed(seed + 9999)
    query_positions = torch.randint(0, seq_len, (N_QUERY_SAMPLES,))

    for method_name, QuantClass in [
        ("standard", StandardQuantizer),
        ("mean_removed", MeanRemovedQuantizer),
    ]:
        method_metrics = defaultdict(float)
        for qpos in query_positions:
            query = keys[qpos].unsqueeze(0)
            quant = QuantClass(d=head_dim, bits=bits, seed=seed, device=device)
            _, metrics = quant.quantize_and_score(query, keys)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    method_metrics[k] += v
        # Average over query samples
        for k in method_metrics:
            method_metrics[k] /= N_QUERY_SAMPLES
        results[method_name] = dict(method_metrics)

    # Importance-weighted: needs per-query pilot, average across queries
    iw_metrics = defaultdict(float)
    for qpos in query_positions:
        query = keys[qpos].unsqueeze(0)
        quant = ImportanceWeightedQuantizer(d=head_dim, bits=bits, seed=seed, device=device)
        _, metrics = quant.quantize_and_score(query, keys)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                iw_metrics[k] += v
    for k in iw_metrics:
        iw_metrics[k] /= N_QUERY_SAMPLES
    results["importance_weighted"] = dict(iw_metrics)

    # Combined: mean-removal + importance weighting
    comb_metrics = defaultdict(float)
    for qpos in query_positions:
        query = keys[qpos].unsqueeze(0)
        quant = CombinedOptimalQuantizer(d=head_dim, bits=bits, seed=seed, device=device)
        _, metrics = quant.quantize_and_score(query, keys)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                comb_metrics[k] += v
    for k in comb_metrics:
        comb_metrics[k] /= N_QUERY_SAMPLES
    results["combined"] = dict(comb_metrics)

    return results


# ---------------------------------------------------------------------------
# Variance analysis: why does mean-removal help?
# ---------------------------------------------------------------------------
def variance_analysis(keys_fp: torch.Tensor) -> Dict[str, float]:
    """Analyze how mean-removal affects key vector statistics.

    Args:
        keys_fp: (seq_len, head_dim) float32 keys.

    Returns:
        Dict with variance statistics.
    """
    mean_k = keys_fp.mean(dim=0, keepdim=True)
    keys_centered = keys_fp - mean_k

    return {
        "original_var": keys_fp.var().item(),
        "centered_var": keys_centered.var().item(),
        "variance_reduction": 1.0 - keys_centered.var().item() / (keys_fp.var().item() + 1e-10),
        "mean_norm": mean_k.norm().item(),
        "mean_norm_ratio": mean_k.norm().item() / keys_fp.norm(dim=-1).mean().item(),
        "original_coord_std": keys_fp.std(dim=0).mean().item(),
        "centered_coord_std": keys_centered.std(dim=0).mean().item(),
    }


# ---------------------------------------------------------------------------
# Attention concentration analysis
# ---------------------------------------------------------------------------
def attention_concentration_analysis(
    query: torch.Tensor,
    keys: torch.Tensor,
) -> Dict[str, float]:
    """Analyze how concentrated the attention distribution is.

    More concentrated = fewer tokens matter = importance weighting helps more.
    """
    attn = compute_attention_scores(query, keys)
    if attn.dim() == 1:
        attn = attn.unsqueeze(0)

    # Entropy (lower = more concentrated)
    entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1)
    # Max entropy for uniform: log(seq_len)
    max_entropy = math.log(attn.shape[-1])

    # Gini coefficient
    sorted_attn, _ = attn.sort(dim=-1)
    n = attn.shape[-1]
    idx = torch.arange(1, n + 1, device=attn.device, dtype=attn.dtype)
    gini = (2 * (idx * sorted_attn).sum(dim=-1) / (n * sorted_attn.sum(dim=-1)) - (n + 1) / n)

    # Top-k mass
    top1_mass = attn.max(dim=-1).values
    top10_mass = attn.topk(min(10, n), dim=-1).values.sum(dim=-1)
    top50_mass = attn.topk(min(50, n), dim=-1).values.sum(dim=-1)

    return {
        "entropy": entropy.mean().item(),
        "entropy_ratio": (entropy.mean().item() / max_entropy) if max_entropy > 0 else 0,
        "gini": gini.mean().item(),
        "top1_mass": top1_mass.mean().item(),
        "top10_mass": top10_mass.mean().item(),
        "top50_mass": top50_mass.mean().item(),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_full_experiment():
    """Run all attention-optimal quantization experiments."""
    print("=" * 72)
    print("ATTENTION-OPTIMAL QUANTIZATION EXPERIMENTS")
    print("Objective: minimize attention score error, not reconstruction error")
    print("=" * 72)
    print()

    model, tokenizer, n_layers, n_heads, n_kv_heads, head_dim = load_model()

    all_results = {}

    for target_len in TARGET_LENGTHS:
        print(f"\n{'=' * 72}")
        print(f"Context length: {target_len} tokens")
        print(f"{'=' * 72}")

        get_keys, actual_layers, actual_seq = extract_kv_cache(model, tokenizer, target_len)
        print(f"  Actual sequence: {actual_seq} tokens")

        sample_keys = get_keys(0)
        actual_kv_heads = sample_keys.shape[1]
        actual_head_dim = sample_keys.shape[3]
        print(f"  KV heads: {actual_kv_heads} | head_dim: {actual_head_dim}")

        # --- Variance analysis across all layers/heads ---
        print("\n  [Variance Analysis] ...")
        var_stats = defaultdict(list)
        conc_stats = defaultdict(list)

        for layer_idx in range(actual_layers):
            layer_keys = get_keys(layer_idx)  # (1, n_kv_heads, seq, head_dim)
            for h in range(actual_kv_heads):
                k = layer_keys[0, h].float()
                vstat = variance_analysis(k)
                for key, val in vstat.items():
                    var_stats[key].append(val)

                # Concentration analysis using last token as query
                query = k[-1:]
                cstat = attention_concentration_analysis(query, k)
                for key, val in cstat.items():
                    conc_stats[key].append(val)

        print(f"    Variance reduction (mean-removal): "
              f"{100 * sum(var_stats['variance_reduction']) / len(var_stats['variance_reduction']):.1f}%")
        print(f"    Mean norm ratio (||mean|| / avg ||k||): "
              f"{sum(var_stats['mean_norm_ratio']) / len(var_stats['mean_norm_ratio']):.4f}")
        print(f"    Attention entropy ratio: "
              f"{sum(conc_stats['entropy_ratio']) / len(conc_stats['entropy_ratio']):.3f}")
        print(f"    Attention Gini coefficient: "
              f"{sum(conc_stats['gini']) / len(conc_stats['gini']):.3f}")
        print(f"    Top-10 token mass: "
              f"{100 * sum(conc_stats['top10_mass']) / len(conc_stats['top10_mass']):.1f}%")

        all_results[f"variance_{target_len}"] = {
            k: sum(v) / len(v) for k, v in var_stats.items()
        }
        all_results[f"concentration_{target_len}"] = {
            k: sum(v) / len(v) for k, v in conc_stats.items()
        }

        # --- Per-bit-width experiments ---
        for bits in BIT_WIDTHS:
            print(f"\n  [{bits}-bit Quantization]")
            method_agg = defaultdict(lambda: defaultdict(list))

            # Sample layers evenly (don't need all 36 for the experiment)
            layer_step = max(1, actual_layers // 6)
            sample_layers = list(range(0, actual_layers, layer_step))
            sample_heads = list(range(min(2, actual_kv_heads)))  # first 2 heads per layer

            total_heads = len(sample_layers) * len(sample_heads)
            done = 0

            for layer_idx in sample_layers:
                layer_keys = get_keys(layer_idx)
                for h in sample_heads:
                    k = layer_keys[0, h].float().cuda()
                    seed = layer_idx * 10000 + h

                    head_results = run_head_experiment(
                        keys_fp=k,
                        head_dim=actual_head_dim,
                        bits=bits,
                        seed=seed,
                        device="cuda",
                    )

                    for method, metrics in head_results.items():
                        for metric_name, val in metrics.items():
                            if isinstance(val, (int, float)):
                                method_agg[method][metric_name].append(val)

                    done += 1
                    if done % 4 == 0 or done == total_heads:
                        print(f"    Progress: {done}/{total_heads} heads", end="\r")

            print()  # newline after progress

            # Aggregate and print results
            key_metrics = ["cosine_sim", "top1_match", "top5_match", "spearman_rho", "kl_div", "l1_error"]
            print(f"\n    {'Method':<22} {'CosSim':>8} {'Top1%':>8} {'Top5%':>8} {'Spearman':>8} {'KL-div':>10} {'L1':>8}")
            print(f"    {'-' * 78}")

            for method in ["standard", "mean_removed", "importance_weighted", "combined"]:
                if method not in method_agg:
                    continue
                agg = method_agg[method]
                row = {}
                for m in key_metrics:
                    if m in agg and len(agg[m]) > 0:
                        row[m] = sum(agg[m]) / len(agg[m])
                    else:
                        row[m] = float("nan")

                label = method.replace("_", " ").title()
                print(
                    f"    {label:<22} "
                    f"{row['cosine_sim']:>8.5f} "
                    f"{100 * row['top1_match']:>7.1f}% "
                    f"{100 * row['top5_match']:>7.1f}% "
                    f"{row['spearman_rho']:>8.5f} "
                    f"{row['kl_div']:>10.6f} "
                    f"{row['l1_error']:>8.5f}"
                )

                # Compute delta vs standard
                if method != "standard" and "standard" in method_agg:
                    std = method_agg["standard"]
                    for m in ["cosine_sim", "top1_match", "top5_match", "spearman_rho"]:
                        if m in agg and m in std and len(agg[m]) > 0 and len(std[m]) > 0:
                            delta = sum(agg[m]) / len(agg[m]) - sum(std[m]) / len(std[m])
                            row[f"delta_{m}"] = delta

                all_results[f"{method}_{bits}bit_{target_len}"] = row

            # Show deltas
            print(f"\n    Delta vs Standard:")
            for method in ["mean_removed", "importance_weighted", "combined"]:
                key = f"{method}_{bits}bit_{target_len}"
                if key not in all_results:
                    continue
                row = all_results[key]
                label = method.replace("_", " ").title()
                deltas = []
                for m in ["cosine_sim", "top1_match", "top5_match", "spearman_rho"]:
                    dk = f"delta_{m}"
                    if dk in row:
                        sign = "+" if row[dk] >= 0 else ""
                        short = m.replace("_match", "").replace("cosine_sim", "cos").replace("spearman_rho", "spear")
                        if "match" in m or m == "top1_match":
                            deltas.append(f"{short}: {sign}{100 * row[dk]:.1f}pp")
                        else:
                            deltas.append(f"{short}: {sign}{row[dk]:.5f}")
                if deltas:
                    print(f"      {label:<22} {', '.join(deltas)}")

    # --- Generate results report ---
    print("\n\n" + "=" * 72)
    print("GENERATING RESULTS REPORT")
    print("=" * 72)

    report = generate_report(all_results, TARGET_LENGTHS, BIT_WIDTHS)

    results_dir = os.path.join(REPO_ROOT, "benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "attention_optimal_results.md")
    with open(results_path, "w") as f:
        f.write(report)
    print(f"\nResults saved to: {results_path}")

    return all_results


def generate_report(
    all_results: Dict,
    target_lengths: List[int],
    bit_widths: List[int],
) -> str:
    """Generate markdown report from experiment results."""
    lines = []
    lines.append("# Attention-Optimal Quantization Results")
    lines.append("")
    lines.append("**Hypothesis:** Quantizing to minimize attention score error (not MSE)")
    lines.append("can improve attention preservation at the same bit budget.")
    lines.append("")
    lines.append("**Model:** Qwen2.5-3B-Instruct (BnB 4-bit)")
    lines.append(f"**Query samples per head:** {N_QUERY_SAMPLES}")
    lines.append("")

    # Variance analysis
    lines.append("## 1. Variance Analysis (Mean-Removal Opportunity)")
    lines.append("")
    lines.append("If softmax is shift-invariant, the per-head mean of K is wasted information.")
    lines.append("Removing it before quantization reduces variance and improves codebook utilization.")
    lines.append("")
    lines.append("| Context | Var Reduction | Mean Norm Ratio | Coord Std (orig) | Coord Std (centered) |")
    lines.append("|---------|--------------|-----------------|------------------|---------------------|")
    for tl in target_lengths:
        vk = f"variance_{tl}"
        if vk in all_results:
            v = all_results[vk]
            lines.append(
                f"| {tl} "
                f"| {100 * v.get('variance_reduction', 0):.1f}% "
                f"| {v.get('mean_norm_ratio', 0):.4f} "
                f"| {v.get('original_coord_std', 0):.4f} "
                f"| {v.get('centered_coord_std', 0):.4f} |"
            )
    lines.append("")

    # Concentration analysis
    lines.append("## 2. Attention Concentration (Importance-Weighting Opportunity)")
    lines.append("")
    lines.append("Higher concentration = fewer tokens carry most attention mass = importance")
    lines.append("weighting has more opportunity to help.")
    lines.append("")
    lines.append("| Context | Entropy Ratio | Gini | Top-1 Mass | Top-10 Mass | Top-50 Mass |")
    lines.append("|---------|--------------|------|-----------|------------|------------|")
    for tl in target_lengths:
        ck = f"concentration_{tl}"
        if ck in all_results:
            c = all_results[ck]
            lines.append(
                f"| {tl} "
                f"| {c.get('entropy_ratio', 0):.3f} "
                f"| {c.get('gini', 0):.3f} "
                f"| {100 * c.get('top1_mass', 0):.1f}% "
                f"| {100 * c.get('top10_mass', 0):.1f}% "
                f"| {100 * c.get('top50_mass', 0):.1f}% |"
            )
    lines.append("")

    # Per-bitwidth results
    lines.append("## 3. Quantization Comparison")
    lines.append("")
    methods = ["standard", "mean_removed", "importance_weighted", "combined"]
    method_labels = {
        "standard": "Standard (MSE-optimal)",
        "mean_removed": "Mean-Removed",
        "importance_weighted": "Importance-Weighted",
        "combined": "Combined (MR + IW)",
    }

    for bits in bit_widths:
        lines.append(f"### {bits}-bit")
        lines.append("")
        lines.append(f"| Method | Ctx | CosSim | Top-1% | Top-5% | Spearman | KL-div | L1 |")
        lines.append(f"|--------|-----|--------|--------|--------|----------|--------|----|")
        for tl in target_lengths:
            for method in methods:
                key = f"{method}_{bits}bit_{tl}"
                if key not in all_results:
                    continue
                r = all_results[key]
                label = method_labels.get(method, method)
                lines.append(
                    f"| {label} | {tl} "
                    f"| {r.get('cosine_sim', 0):.5f} "
                    f"| {100 * r.get('top1_match', 0):.1f}% "
                    f"| {100 * r.get('top5_match', 0):.1f}% "
                    f"| {r.get('spearman_rho', 0):.5f} "
                    f"| {r.get('kl_div', 0):.6f} "
                    f"| {r.get('l1_error', 0):.5f} |"
                )
        lines.append("")

        # Delta table
        lines.append(f"#### Deltas vs Standard ({bits}-bit)")
        lines.append("")
        lines.append(f"| Method | Ctx | dCosSim | dTop-1 | dTop-5 | dSpearman |")
        lines.append(f"|--------|-----|---------|--------|--------|-----------|")
        for tl in target_lengths:
            for method in ["mean_removed", "importance_weighted", "combined"]:
                key = f"{method}_{bits}bit_{tl}"
                if key not in all_results:
                    continue
                r = all_results[key]
                label = method_labels.get(method, method)
                d_cos = r.get("delta_cosine_sim", 0)
                d_t1 = r.get("delta_top1_match", 0)
                d_t5 = r.get("delta_top5_match", 0)
                d_sp = r.get("delta_spearman_rho", 0)
                sign = lambda x: "+" if x >= 0 else ""
                lines.append(
                    f"| {label} | {tl} "
                    f"| {sign(d_cos)}{d_cos:.5f} "
                    f"| {sign(d_t1)}{100 * d_t1:.1f}pp "
                    f"| {sign(d_t5)}{100 * d_t5:.1f}pp "
                    f"| {sign(d_sp)}{d_sp:.5f} |"
                )
        lines.append("")

    # Conclusions placeholder
    lines.append("## 4. Key Findings")
    lines.append("")
    lines.append("_(Auto-populated after experiments complete)_")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    results = run_full_experiment()
