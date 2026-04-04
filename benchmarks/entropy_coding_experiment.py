"""Entropy coding experiment — measuring free compression on real KV caches.

Loads Qwen2.5-3B-Instruct, captures the KV cache, quantizes with PolarQuant
at 2/3/4 bits, and measures the actual Shannon entropy of the indices.

If entropy < allocated bits, that gap is FREE lossless compression — same
quality, fewer bits.

Experiments:
    1. Global entropy vs allocated bits at 2/3/4 bits
    2. Per-layer entropy: is non-uniformity consistent across layers?
    3. Per-head entropy: variance across heads within a layer
    4. WHT vs QR rotation: does rotation type affect entropy?
    5. Sequential correlations: do adjacent indices predict each other?
    6. Run-length analysis: are there exploitable runs?
    7. Actual compressed size: ANS, zlib, lzma vs raw

Usage:
    python benchmarks/entropy_coding_experiment.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from turboquantdc.codebook import LloydMaxCodebook
from turboquantdc.entropy_analysis import (
    analyze_kv_cache_entropy,
    compare_rotation_entropy,
    measure_actual_compression,
    measure_per_coordinate_entropy,
    measure_real_entropy,
    measure_run_lengths,
    measure_sequential_correlation,
)
from turboquantdc.entropy_coding import (
    _symbol_probabilities,
    compression_opportunity,
    entropy_analysis_sweep,
    measure_index_entropy,
    theoretical_index_entropy,
)
from turboquantdc.polarquant import PolarQuant

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CONTEXT_LENGTH = 2048
BIT_WIDTHS = (2, 3, 4)

# Filler for building a long prompt
FILLER = (
    "The quarterly financial review meeting covered several topics including "
    "budget allocations for the upcoming fiscal year, departmental spending reports, "
    "and projected revenue streams from various business units. The committee discussed "
    "infrastructure upgrades planned for the western regional offices and noted that "
    "maintenance schedules should be coordinated with the facilities management team. "
    "Several action items were assigned to team leads for follow-up before the next "
    "meeting cycle.\n\n"
)

RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    """Load Qwen2.5-3B-Instruct in 4-bit with BitsAndBytes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"  Loading {MODEL_NAME} (4-bit NF4)...", flush=True)
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
    gpu_mb = torch.cuda.memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0

    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    head_dim = config.hidden_size // n_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)

    print(f"  Loaded in {load_time:.1f}s | GPU: {gpu_mb} MB")
    print(f"  Layers: {n_layers} | Heads: {n_heads} | KV heads: {n_kv_heads} | head_dim: {head_dim}")

    return model, tokenizer, n_layers, n_heads, n_kv_heads, head_dim


def capture_kv_cache(model, tokenizer, target_tokens: int = 2048):
    """Run a forward pass and capture the KV cache."""
    # Build a prompt of the target length
    filler_len = len(tokenizer.encode(FILLER, add_special_tokens=False))
    n_reps = max(1, target_tokens // filler_len)

    text = FILLER * n_reps
    prompt = f"<|im_start|>user\n{text}\nSummarize this.<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=target_tokens + 256,
    ).to("cuda")
    seq_len = inputs["input_ids"].shape[1]

    print(f"  Forward pass with {seq_len} tokens...", end="", flush=True)
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)
    fwd_time = time.time() - t0
    print(f" {fwd_time:.1f}s")

    cache = outputs.past_key_values

    # Extract key cache as list of tensors
    if hasattr(cache, "key_cache"):
        key_cache = list(cache.key_cache)
        n_layers = len(key_cache)
    elif hasattr(cache, "layers"):
        key_cache = [cache.layers[i].keys for i in range(len(cache.layers))]
        n_layers = len(key_cache)
    else:
        key_cache = [cache[i][0] for i in range(len(cache))]
        n_layers = len(key_cache)

    sample = key_cache[0]
    n_kv_heads = sample.shape[1]
    actual_seq = sample.shape[2]
    head_dim = sample.shape[3]

    print(f"  Cache: {n_layers} layers x {n_kv_heads} KV heads x {actual_seq} seq x {head_dim} head_dim")

    return key_cache, seq_len, n_layers, n_kv_heads, head_dim


# ---------------------------------------------------------------------------
# Experiment 1: Theoretical entropy sweep
# ---------------------------------------------------------------------------
def experiment_theoretical_sweep():
    """Compute theoretical entropy from codebook PDFs (no model needed)."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Theoretical Entropy Sweep (from PDF integration)")
    print("=" * 70)

    for d in (64, 128, 256):
        print(f"\n  d={d}:")
        print(f"  {'Bits':>4} | {'Levels':>6} | {'Entropy':>8} | {'Allocated':>9} | {'Savings':>8} | {'Distribution'}")
        print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*40}")

        results = entropy_analysis_sweep(d=d, bit_range=(2, 3, 4, 5, 6, 7, 8))
        for r in results:
            probs = r["symbol_probabilities"]
            # Show first few probabilities
            prob_str = ", ".join(f"{p:.3f}" for p in probs[:min(8, len(probs))])
            if len(probs) > 8:
                prob_str += ", ..."
            print(
                f"  {r['bits']:>4} | {r['n_levels']:>6} | "
                f"{r['theoretical_entropy']:>7.3f}b | "
                f"{r['allocated_bits']:>8.1f}b | "
                f"{r['savings_pct']:>7.1f}% | "
                f"[{prob_str}]"
            )

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Empirical entropy on real KV cache
# ---------------------------------------------------------------------------
def experiment_real_kv_entropy(key_cache, head_dim, n_kv_heads, n_layers):
    """Measure entropy on actual Qwen2.5-3B KV cache."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Empirical Entropy on Real KV Cache")
    print("=" * 70)

    results = {}
    for bits in BIT_WIDTHS:
        print(f"\n  --- {bits}-bit ---")
        all_entropies = []
        all_savings = []
        per_layer_avg = []

        for layer_idx in range(n_layers):
            keys = key_cache[layer_idx]  # (1, n_kv_heads, seq, head_dim)
            layer_entropies = []

            for h in range(n_kv_heads):
                k = keys[0, h].float()  # (seq, head_dim)
                seed = layer_idx * 10000 + h

                stats = measure_real_entropy(k, bits=bits, d=head_dim, seed=seed)
                layer_entropies.append(stats["empirical_entropy"])
                all_entropies.append(stats["empirical_entropy"])
                all_savings.append(stats["savings_pct"])

            avg_e = np.mean(layer_entropies)
            per_layer_avg.append(avg_e)

        all_e = np.array(all_entropies)
        all_s = np.array(all_savings)

        # Theoretical baseline
        cb = LloydMaxCodebook(d=head_dim, bits=bits)
        theory = theoretical_index_entropy(cb)

        print(f"  Theoretical entropy:     {theory:.4f} bits (from PDF)")
        print(f"  Empirical entropy (avg): {all_e.mean():.4f} bits")
        print(f"  Empirical entropy (std): {all_e.std():.4f} bits")
        print(f"  Empirical entropy (min): {all_e.min():.4f} bits")
        print(f"  Empirical entropy (max): {all_e.max():.4f} bits")
        print(f"  Average savings:         {all_s.mean():.2f}%")
        print(f"  Min savings:             {all_s.min():.2f}%")
        print(f"  Max savings:             {all_s.max():.2f}%")

        # Per-layer summary
        print(f"\n  Per-layer entropy ({bits}-bit):")
        print(f"  {'Layer':>5} | {'Avg Entropy':>11} | {'Savings':>8}")
        print(f"  {'-'*5}-+-{'-'*11}-+-{'-'*8}")
        for i, avg in enumerate(per_layer_avg):
            savings = (1.0 - avg / bits) * 100
            print(f"  {i:>5} | {avg:>10.4f}b | {savings:>7.2f}%")

        results[bits] = {
            "theory": theory,
            "empirical_mean": float(all_e.mean()),
            "empirical_std": float(all_e.std()),
            "empirical_min": float(all_e.min()),
            "empirical_max": float(all_e.max()),
            "savings_mean": float(all_s.mean()),
            "savings_min": float(all_s.min()),
            "savings_max": float(all_s.max()),
            "per_layer_avg": [float(x) for x in per_layer_avg],
        }

    return results


# ---------------------------------------------------------------------------
# Experiment 3: WHT vs QR rotation comparison
# ---------------------------------------------------------------------------
def experiment_wht_vs_qr(key_cache, head_dim, n_layers):
    """Compare entropy under WHT vs QR rotation."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: WHT vs QR Rotation Entropy")
    print("=" * 70)

    results = {}
    for bits in BIT_WIDTHS:
        print(f"\n  --- {bits}-bit ---")
        wht_entropies = []
        qr_entropies = []

        # Sample a few layers to keep runtime reasonable
        sample_layers = list(range(0, n_layers, max(1, n_layers // 4)))
        for layer_idx in sample_layers:
            keys = key_cache[layer_idx]
            k = keys[0, 0].float()  # First head

            comp = compare_rotation_entropy(k, bits=bits, d=head_dim, seed=layer_idx * 10000)
            wht_entropies.append(comp["wht"]["empirical_entropy"])
            qr_entropies.append(comp["qr"]["empirical_entropy"])

        wht_avg = np.mean(wht_entropies)
        qr_avg = np.mean(qr_entropies)
        diff = qr_avg - wht_avg

        print(f"  WHT avg entropy: {wht_avg:.4f} bits")
        print(f"  QR  avg entropy: {qr_avg:.4f} bits")
        print(f"  Difference:      {diff:+.4f} bits ({'QR higher' if diff > 0 else 'WHT higher'})")
        print(f"  WHT savings:     {(1.0 - wht_avg / bits) * 100:.2f}%")
        print(f"  QR  savings:     {(1.0 - qr_avg / bits) * 100:.2f}%")

        results[bits] = {
            "wht_avg": float(wht_avg),
            "qr_avg": float(qr_avg),
            "diff": float(diff),
        }

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Sequential correlation
# ---------------------------------------------------------------------------
def experiment_sequential_correlation(key_cache, head_dim, n_layers):
    """Check if adjacent coordinate indices are correlated."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: Sequential Correlation (Adjacent Indices)")
    print("=" * 70)

    results = {}
    for bits in BIT_WIDTHS:
        print(f"\n  --- {bits}-bit ---")
        gains = []

        # Sample a few layers
        sample_layers = list(range(0, n_layers, max(1, n_layers // 4)))
        for layer_idx in sample_layers:
            keys = key_cache[layer_idx]
            k = keys[0, 0].float()

            corr = measure_sequential_correlation(
                k, bits=bits, d=head_dim, seed=layer_idx * 10000, max_lag=4
            )
            gains.append(corr["correlation_gain_lag1"])

            if layer_idx == sample_layers[0]:
                # Print detailed for first layer
                print(f"  Layer {layer_idx} head 0:")
                print(f"    Zeroth-order entropy:     {corr['zeroth_order_entropy']:.4f} bits")
                print(f"    Conditional H(X|X_{'{-1}'}):    {corr['conditional_entropy_lag1']:.4f} bits")
                print(f"    Correlation gain (lag 1): {corr['correlation_gain_lag1']:.4f} bits")
                for lag, h_cond in enumerate(corr["conditional_entropies"], 1):
                    print(f"    Conditional H(X|X_{'{-' + str(lag) + '}'}):    {h_cond:.4f} bits")

        avg_gain = np.mean(gains)
        print(f"\n  Average correlation gain (lag 1): {avg_gain:.4f} bits ({avg_gain / bits * 100:.2f}%)")
        print(f"  Verdict: {'EXPLOITABLE' if avg_gain > 0.05 else 'NEGLIGIBLE'} sequential structure")

        results[bits] = {
            "avg_gain_lag1": float(avg_gain),
            "exploitable": avg_gain > 0.05,
        }

    return results


# ---------------------------------------------------------------------------
# Experiment 5: Run-length analysis
# ---------------------------------------------------------------------------
def experiment_run_lengths(key_cache, head_dim, n_layers):
    """Analyze run-length patterns."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 5: Run-Length Analysis")
    print("=" * 70)

    results = {}
    for bits in BIT_WIDTHS:
        print(f"\n  --- {bits}-bit ---")

        # Sample a few layers
        sample_layers = list(range(0, n_layers, max(1, n_layers // 4)))
        all_avg_runs = []
        all_expected_runs = []

        for layer_idx in sample_layers:
            keys = key_cache[layer_idx]
            k = keys[0, 0].float()

            rle = measure_run_lengths(k, bits=bits, d=head_dim, seed=layer_idx * 10000)
            all_avg_runs.append(rle["avg_run_length"])
            all_expected_runs.append(rle["expected_random_run_length"])

            if layer_idx == sample_layers[0]:
                print(f"  Layer {layer_idx} head 0:")
                print(f"    Avg run length:      {rle['avg_run_length']:.2f}")
                print(f"    Max run length:      {rle['max_run_length']}")
                print(f"    Total runs:          {rle['run_count']}")
                print(f"    Expected (random):   {rle['expected_random_run_length']:.2f}")
                print(f"    RLE compressible:    {rle['rle_compressible']}")

        avg_run = np.mean(all_avg_runs)
        expected_run = np.mean(all_expected_runs)
        print(f"\n  Average run length:       {avg_run:.2f}")
        print(f"  Expected (memoryless):    {expected_run:.2f}")
        print(f"  Verdict: {'RLE USEFUL' if avg_run > expected_run * 1.5 else 'RLE NOT USEFUL'}")

        results[bits] = {
            "avg_run_length": float(avg_run),
            "expected_random": float(expected_run),
            "rle_useful": avg_run > expected_run * 1.5,
        }

    return results


# ---------------------------------------------------------------------------
# Experiment 6: Actual compressed sizes
# ---------------------------------------------------------------------------
def experiment_actual_compression(key_cache, head_dim, n_layers, n_kv_heads):
    """Measure actual compressed sizes using ANS, zlib, lzma."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 6: Actual Compressed Sizes (ANS / zlib / lzma)")
    print("=" * 70)

    results = {}
    for bits in BIT_WIDTHS:
        print(f"\n  --- {bits}-bit ---")

        total_raw = 0
        total_ans = 0
        total_zlib = 0
        total_lzma = 0
        total_symbols = 0

        # Use all layers, first head for a representative cross-section
        for layer_idx in range(n_layers):
            keys = key_cache[layer_idx]
            k = keys[0, 0].float()
            seed = layer_idx * 10000

            comp = measure_actual_compression(k, bits=bits, d=head_dim, seed=seed)
            total_raw += comp["raw_bytes_byte_per_idx"]
            total_ans += comp["ans_bytes"]
            total_zlib += comp["zlib_bytes"]
            total_lzma += comp["lzma_bytes"]
            total_symbols += comp["n_symbols"]

        ans_bps = (total_ans * 8) / total_symbols if total_symbols > 0 else bits
        zlib_bps = (total_zlib * 8) / total_symbols if total_symbols > 0 else bits
        lzma_bps = (total_lzma * 8) / total_symbols if total_symbols > 0 else bits

        print(f"  Total symbols:       {total_symbols:,}")
        print(f"  Raw size (1B/idx):   {total_raw:,} bytes ({total_raw / 1024:.1f} KB)")
        print(f"  Raw size (packed):   {total_symbols * bits // 8:,} bytes ({total_symbols * bits / 8 / 1024:.1f} KB)")
        print(f"  ANS compressed:      {total_ans:,} bytes ({total_ans / 1024:.1f} KB)")
        print(f"  Zlib compressed:     {total_zlib:,} bytes ({total_zlib / 1024:.1f} KB)")
        print(f"  LZMA compressed:     {total_lzma:,} bytes ({total_lzma / 1024:.1f} KB)")
        print(f"  ANS ratio:           {total_raw / total_ans:.3f}x (vs 1B/idx)")
        print(f"  Zlib ratio:          {total_raw / total_zlib:.3f}x (vs 1B/idx)")
        print(f"  LZMA ratio:          {total_raw / total_lzma:.3f}x (vs 1B/idx)")
        print(f"  ANS bits/symbol:     {ans_bps:.3f} (allocated: {bits})")
        print(f"  Zlib bits/symbol:    {zlib_bps:.3f}")
        print(f"  LZMA bits/symbol:    {lzma_bps:.3f}")

        # Compute savings relative to packed bit representation
        packed_bits = total_symbols * bits
        packed_bytes = packed_bits / 8
        print(f"  ANS vs packed:       {packed_bytes / total_ans:.3f}x")
        print(f"  Zlib vs packed:      {packed_bytes / total_zlib:.3f}x")
        print(f"  LZMA vs packed:      {packed_bytes / total_lzma:.3f}x")

        results[bits] = {
            "total_symbols": total_symbols,
            "raw_bytes": total_raw,
            "packed_bytes": int(packed_bytes),
            "ans_bytes": total_ans,
            "zlib_bytes": total_zlib,
            "lzma_bytes": total_lzma,
            "ans_bps": float(ans_bps),
            "zlib_bps": float(zlib_bps),
            "lzma_bps": float(lzma_bps),
            "ans_vs_packed": float(packed_bytes / total_ans) if total_ans > 0 else 1.0,
            "zlib_vs_packed": float(packed_bytes / total_zlib) if total_zlib > 0 else 1.0,
            "lzma_vs_packed": float(packed_bytes / total_lzma) if total_lzma > 0 else 1.0,
        }

    return results


# ---------------------------------------------------------------------------
# Experiment 7: Per-coordinate entropy profile
# ---------------------------------------------------------------------------
def experiment_per_coordinate_entropy(key_cache, head_dim):
    """Check if entropy varies by coordinate position."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 7: Per-Coordinate Entropy Profile")
    print("=" * 70)

    results = {}
    for bits in BIT_WIDTHS:
        k = key_cache[0][0, 0].float()  # Layer 0, head 0
        per_coord = measure_per_coordinate_entropy(k, bits=bits, d=head_dim, seed=0)

        print(f"\n  --- {bits}-bit ---")
        print(f"  Mean per-coord entropy: {per_coord.mean():.4f} bits")
        print(f"  Std  per-coord entropy: {per_coord.std():.4f} bits")
        print(f"  Min  per-coord entropy: {per_coord.min():.4f} bits")
        print(f"  Max  per-coord entropy: {per_coord.max():.4f} bits")
        print(f"  Range:                  {per_coord.max() - per_coord.min():.4f} bits")

        results[bits] = {
            "mean": float(per_coord.mean()),
            "std": float(per_coord.std()),
            "min": float(per_coord.min()),
            "max": float(per_coord.max()),
        }

    return results


# ---------------------------------------------------------------------------
# Write results to markdown
# ---------------------------------------------------------------------------
def write_results_md(
    theoretical_results,
    empirical_results,
    wht_vs_qr_results,
    correlation_results,
    rle_results,
    compression_results,
    per_coord_results,
    seq_len: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
):
    """Write comprehensive results to markdown file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "entropy_coding_results.md")

    with open(path, "w") as f:
        f.write("# Entropy Coding Analysis Results\n\n")
        f.write(f"**Model:** {MODEL_NAME}\n")
        f.write(f"**Context length:** {seq_len} tokens\n")
        f.write(f"**Layers:** {n_layers} | **KV heads:** {n_kv_heads} | **head_dim:** {head_dim}\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")

        # ---- Summary table ----
        f.write("## Summary: Free Compression Available\n\n")
        f.write("| Bits | Allocated | Theory H | Empirical H | Savings | ANS bps | Zlib bps | LZMA bps |\n")
        f.write("|------|-----------|----------|-------------|---------|---------|----------|----------|\n")
        for bits in BIT_WIDTHS:
            emp = empirical_results.get(bits, {})
            comp = compression_results.get(bits, {})
            f.write(
                f"| {bits} | {bits}.0 b | "
                f"{emp.get('theory', 0):.3f} b | "
                f"{emp.get('empirical_mean', 0):.3f} b | "
                f"**{emp.get('savings_mean', 0):.1f}%** | "
                f"{comp.get('ans_bps', 0):.3f} | "
                f"{comp.get('zlib_bps', 0):.3f} | "
                f"{comp.get('lzma_bps', 0):.3f} |\n"
            )

        # ---- Interpretation ----
        f.write("\n## Interpretation\n\n")

        best_savings = max(
            empirical_results.get(b, {}).get("savings_mean", 0)
            for b in BIT_WIDTHS
        )
        f.write(f"- **Best entropy savings: {best_savings:.1f}%** (lossless, zero quality loss)\n")

        for bits in BIT_WIDTHS:
            emp = empirical_results.get(bits, {})
            comp = compression_results.get(bits, {})
            savings = emp.get("savings_mean", 0)
            f.write(
                f"- **{bits}-bit:** entropy = {emp.get('empirical_mean', 0):.3f} bits "
                f"(vs {bits}.0 allocated) = **{savings:.1f}% free compression**\n"
            )

        # ---- Effective compression ratios ----
        f.write("\n## Effective Compression Ratios (with entropy coding)\n\n")
        f.write("Baseline TurboQuant compression = 16 / (b + overhead). ")
        f.write("With entropy coding, the effective bits are lower.\n\n")
        f.write("| Bits | Without EC | With ANS | With LZMA | Improvement |\n")
        f.write("|------|-----------|----------|-----------|-------------|\n")
        for bits in BIT_WIDTHS:
            comp = compression_results.get(bits, {})
            # Baseline compression: 16 bits fp16 / (bits per coord for keys+values)
            # Simplified: ratio = 16 / bits
            baseline = 16.0 / bits
            ans_bps = comp.get("ans_bps", bits)
            lzma_bps = comp.get("lzma_bps", bits)
            with_ans = 16.0 / ans_bps if ans_bps > 0 else baseline
            with_lzma = 16.0 / lzma_bps if lzma_bps > 0 else baseline
            improvement = (with_ans / baseline - 1) * 100
            f.write(
                f"| {bits} | {baseline:.2f}x | {with_ans:.2f}x | {with_lzma:.2f}x | +{improvement:.1f}% |\n"
            )

        # ---- WHT vs QR ----
        f.write("\n## WHT vs QR Rotation\n\n")
        f.write("| Bits | WHT Entropy | QR Entropy | Difference |\n")
        f.write("|------|-------------|------------|------------|\n")
        for bits in BIT_WIDTHS:
            wq = wht_vs_qr_results.get(bits, {})
            f.write(
                f"| {bits} | {wq.get('wht_avg', 0):.4f} | "
                f"{wq.get('qr_avg', 0):.4f} | "
                f"{wq.get('diff', 0):+.4f} |\n"
            )

        # ---- Sequential correlations ----
        f.write("\n## Sequential Correlations\n\n")
        for bits in BIT_WIDTHS:
            cr = correlation_results.get(bits, {})
            f.write(f"- **{bits}-bit:** correlation gain = {cr.get('avg_gain_lag1', 0):.4f} bits ")
            f.write(f"({cr.get('avg_gain_lag1', 0) / bits * 100:.2f}% of allocated) ")
            f.write(f"-> {'EXPLOITABLE' if cr.get('exploitable', False) else 'negligible'}\n")

        # ---- Run lengths ----
        f.write("\n## Run-Length Analysis\n\n")
        for bits in BIT_WIDTHS:
            rl = rle_results.get(bits, {})
            f.write(
                f"- **{bits}-bit:** avg run = {rl.get('avg_run_length', 0):.2f} "
                f"(expected random: {rl.get('expected_random', 0):.2f}) "
                f"-> {'RLE useful' if rl.get('rle_useful', False) else 'RLE not useful'}\n"
            )

        # ---- Per-coordinate ----
        f.write("\n## Per-Coordinate Entropy\n\n")
        f.write("| Bits | Mean | Std | Min | Max | Range |\n")
        f.write("|------|------|-----|-----|-----|-------|\n")
        for bits in BIT_WIDTHS:
            pc = per_coord_results.get(bits, {})
            f.write(
                f"| {bits} | {pc.get('mean', 0):.4f} | {pc.get('std', 0):.4f} | "
                f"{pc.get('min', 0):.4f} | {pc.get('max', 0):.4f} | "
                f"{pc.get('max', 0) - pc.get('min', 0):.4f} |\n"
            )

        # ---- Per-layer detail ----
        f.write("\n## Per-Layer Entropy Detail\n\n")
        for bits in BIT_WIDTHS:
            emp = empirical_results.get(bits, {})
            per_layer = emp.get("per_layer_avg", [])
            if per_layer:
                f.write(f"### {bits}-bit\n\n")
                f.write("| Layer | Avg Entropy | Savings |\n")
                f.write("|-------|-------------|--------|\n")
                for i, avg in enumerate(per_layer):
                    savings = (1.0 - avg / bits) * 100
                    f.write(f"| {i} | {avg:.4f} b | {savings:.2f}% |\n")
                f.write("\n")

        # ---- Actual compression table ----
        f.write("\n## Actual Compressed Sizes (per layer, head 0)\n\n")
        f.write("| Bits | Raw (packed) | ANS | Zlib | LZMA | ANS/packed | Zlib/packed | LZMA/packed |\n")
        f.write("|------|-------------|-----|------|------|-----------|------------|------------|\n")
        for bits in BIT_WIDTHS:
            comp = compression_results.get(bits, {})
            f.write(
                f"| {bits} | {comp.get('packed_bytes', 0):,} B | "
                f"{comp.get('ans_bytes', 0):,} B | "
                f"{comp.get('zlib_bytes', 0):,} B | "
                f"{comp.get('lzma_bytes', 0):,} B | "
                f"{comp.get('ans_vs_packed', 1):.3f}x | "
                f"{comp.get('zlib_vs_packed', 1):.3f}x | "
                f"{comp.get('lzma_vs_packed', 1):.3f}x |\n"
            )

        # ---- Recommendation ----
        f.write("\n## Recommendation\n\n")

        total_savings_3bit = empirical_results.get(3, {}).get("savings_mean", 0)
        f.write(f"The 3-bit sweet spot shows **{total_savings_3bit:.1f}% free compression** ")
        f.write("from entropy coding alone. This is lossless — identical quality, fewer bits.\n\n")

        if total_savings_3bit > 10:
            f.write("**STRONG RECOMMEND:** >10% free compression. Implement rANS entropy coder ")
            f.write("in the production pipeline. Expected to improve 3-bit compression from ")
            f.write(f"5.0x to ~{16.0 / (3.0 * (1 - total_savings_3bit / 100)):.1f}x.\n")
        elif total_savings_3bit > 5:
            f.write("**RECOMMEND:** 5-10% free compression is meaningful at scale. ")
            f.write("Implement entropy coding for memory-constrained deployments. ")
            f.write("Use zlib as fast backend, ANS for maximum compression.\n")
        else:
            f.write("**MARGINAL:** <5% savings. The Lloyd-Max codebook is already well-adapted ")
            f.write("to the Gaussian distribution, leaving limited room for entropy coding.\n")

        corr_gain = correlation_results.get(3, {}).get("avg_gain_lag1", 0)
        if corr_gain > 0.05:
            f.write(f"\nSequential correlations provide an additional {corr_gain:.3f} bits/symbol ")
            f.write("opportunity. Consider context-adaptive coding (FSE with context) for a ")
            f.write("second-order gain.\n")
        else:
            f.write("\nSequential correlations are negligible. Simple zeroth-order entropy ")
            f.write("coding (ANS/Huffman) captures essentially all the available gain.\n")

    print(f"\n  Results written to {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print()
    print("=" * 70)
    print("  TurboQuantDC — Entropy Coding Experiment")
    print("  Measuring free lossless compression from index non-uniformity")
    print("=" * 70)

    # Experiment 1: Theoretical (no model needed)
    t0 = time.time()
    theoretical_results = experiment_theoretical_sweep()

    # Load model and capture KV cache
    print("\n" + "=" * 70)
    print("  Loading model and capturing KV cache...")
    print("=" * 70)
    model, tokenizer, n_layers, n_heads, n_kv_heads, head_dim = load_model()
    key_cache, seq_len, n_layers_actual, n_kv_heads_actual, head_dim_actual = \
        capture_kv_cache(model, tokenizer, target_tokens=CONTEXT_LENGTH)

    # Free model to save GPU memory
    del model
    torch.cuda.empty_cache()

    # Run experiments on real data
    empirical_results = experiment_real_kv_entropy(
        key_cache, head_dim_actual, n_kv_heads_actual, n_layers_actual
    )
    wht_vs_qr_results = experiment_wht_vs_qr(key_cache, head_dim_actual, n_layers_actual)
    correlation_results = experiment_sequential_correlation(key_cache, head_dim_actual, n_layers_actual)
    rle_results = experiment_run_lengths(key_cache, head_dim_actual, n_layers_actual)
    compression_results = experiment_actual_compression(
        key_cache, head_dim_actual, n_layers_actual, n_kv_heads_actual
    )
    per_coord_results = experiment_per_coordinate_entropy(key_cache, head_dim_actual)

    total_time = time.time() - t0
    print(f"\n  Total experiment time: {total_time:.1f}s")

    # Write results
    path = write_results_md(
        theoretical_results,
        empirical_results,
        wht_vs_qr_results,
        correlation_results,
        rle_results,
        compression_results,
        per_coord_results,
        seq_len=seq_len,
        n_layers=n_layers_actual,
        n_kv_heads=n_kv_heads_actual,
        head_dim=head_dim_actual,
    )

    # Print final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    for bits in BIT_WIDTHS:
        emp = empirical_results.get(bits, {})
        comp = compression_results.get(bits, {})
        print(
            f"  {bits}-bit: H={emp.get('empirical_mean', 0):.3f}b "
            f"(saves {emp.get('savings_mean', 0):.1f}%) | "
            f"ANS={comp.get('ans_bps', 0):.3f}b | "
            f"LZMA={comp.get('lzma_bps', 0):.3f}b"
        )
    print("=" * 70)

    return {
        "theoretical": theoretical_results,
        "empirical": empirical_results,
        "wht_vs_qr": wht_vs_qr_results,
        "correlation": correlation_results,
        "rle": rle_results,
        "compression": compression_results,
        "per_coordinate": per_coord_results,
    }


if __name__ == "__main__":
    main()
