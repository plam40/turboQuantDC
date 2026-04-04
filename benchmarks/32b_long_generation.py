#!/usr/bin/env python3
"""Stress test: 3-bit ResidualQuant vs FP16 at 200, 500, 1000 tokens on Qwen2.5-32B.

Previous validation showed 100% token match at 50 tokens. This pushes the
boundary to find the divergence threshold (if any) for long-form generation.

Setup:
  - Qwen2.5-32B-Instruct, BnB NF4 (4-bit weight quantization)
  - Greedy decoding (do_sample=False, temperature=0 equivalent)
  - 3-bit ResidualQuant KV cache with boundary anchoring + FP16 window

Usage:
    python benchmarks/32b_long_generation.py
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Allow running from repo root
REPO_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, REPO_ROOT)

# HF cache on Beast
HF_CACHE = "/media/dhawal/Beast/cache/hub"
os.environ["HF_HOME"] = "/media/dhawal/Beast/cache"
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

from turboquantdc.generation_cache import GenerationCache

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
DEVICE = "cuda"
SEED = 42

# 32B has 64 layers, 8 KV heads, head_dim=128
NUM_LAYERS = 64

PROMPT = (
    "Write a detailed essay about the history of artificial intelligence, "
    "covering the key milestones from the 1950s to 2025."
)

TOKEN_LENGTHS = [200, 500, 1000]

RESULTS_PATH = os.path.join(REPO_ROOT, "benchmarks", "results", "32b_long_generation.md")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME} (BnB NF4)...")
    t0 = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=HF_CACHE
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
        max_memory={0: "22GiB", "cpu": "40GiB"},
        cache_dir=HF_CACHE,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**2
        print(f"  GPU memory: {used:.0f}MB")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_tokens(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    cache=None,
    label: str = "",
) -> Tuple[List[int], str, float]:
    """Generate tokens and return (token_ids, text, elapsed_seconds)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"  [{label}] Generating {max_new_tokens} tokens (prompt={prompt_len})...")
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            past_key_values=cache,
            use_cache=True,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    gen_ids = outputs[0][prompt_len:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    print(f"  [{label}] Got {len(gen_ids)} tokens in {elapsed:.2f}s "
          f"({len(gen_ids)/elapsed:.1f} tok/s)")

    return gen_ids, gen_text, elapsed


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_sequences(
    fp16_ids: List[int],
    rq_ids: List[int],
    tokenizer,
) -> Dict[str, Any]:
    """Compare two token sequences and return detailed analysis."""
    min_len = min(len(fp16_ids), len(rq_ids))

    # Find all divergence points
    divergences = []
    for i in range(min_len):
        if fp16_ids[i] != rq_ids[i]:
            divergences.append(i)

    # Compute cumulative match rate
    matches = sum(1 for a, b in zip(fp16_ids[:min_len], rq_ids[:min_len]) if a == b)
    match_rate = matches / min_len if min_len > 0 else 0.0

    first_div = divergences[0] if divergences else min_len

    # Build context around first divergence
    div_context = None
    if divergences:
        pos = divergences[0]
        # Show 10 tokens before and after divergence
        start = max(0, pos - 10)
        end = min(min_len, pos + 11)

        fp16_context_ids = fp16_ids[start:end]
        rq_context_ids = rq_ids[start:end]

        fp16_context = tokenizer.decode(fp16_context_ids, skip_special_tokens=True)
        rq_context = tokenizer.decode(rq_context_ids, skip_special_tokens=True)

        fp16_token = tokenizer.decode([fp16_ids[pos]], skip_special_tokens=False)
        rq_token = tokenizer.decode([rq_ids[pos]], skip_special_tokens=False)

        div_context = {
            "position": pos,
            "fp16_token": fp16_token,
            "fp16_token_id": fp16_ids[pos],
            "rq_token": rq_token,
            "rq_token_id": rq_ids[pos],
            "fp16_context": fp16_context,
            "rq_context": rq_context,
        }

    return {
        "total_tokens": min_len,
        "matching_tokens": matches,
        "match_rate": match_rate,
        "first_divergence": first_div,
        "num_divergences": len(divergences),
        "divergence_positions": divergences[:20],  # First 20
        "div_context": div_context,
    }


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------

def run_stress_test():
    print("=" * 70)
    print("  32B Long Generation Stress Test")
    print("  3-bit ResidualQuant vs FP16 at 200, 500, 1000 tokens")
    print("=" * 70)
    print()

    model, tokenizer = load_model()

    all_results = []

    for target_len in TOKEN_LENGTHS:
        print(f"\n{'='*70}")
        print(f"  TARGET: {target_len} tokens")
        print(f"{'='*70}")

        # --- FP16 baseline ---
        fp16_ids, fp16_text, fp16_time = generate_tokens(
            model, tokenizer, PROMPT, target_len,
            cache=None, label="FP16"
        )

        # Force cleanup between runs
        gc.collect()
        torch.cuda.empty_cache()

        # --- 3-bit ResidualQuant ---
        rq_cache = GenerationCache(
            key_bits=3,
            val_bits=3,
            fp16_window=64,
            anchor_strategy="boundary",
            num_layers=NUM_LAYERS,
            use_residual_quant=True,
            seed=SEED,
        )

        try:
            rq_ids, rq_text, rq_time = generate_tokens(
                model, tokenizer, PROMPT, target_len,
                cache=rq_cache, label="RQ3"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  [RQ3] OOM at {target_len} tokens -- skipping")
            rq_ids, rq_text, rq_time = None, None, None

        if rq_ids is not None:
            # Compare
            comparison = compare_sequences(fp16_ids, rq_ids, tokenizer)
            comparison["target_tokens"] = target_len
            comparison["fp16_actual_tokens"] = len(fp16_ids)
            comparison["rq_actual_tokens"] = len(rq_ids)
            comparison["fp16_time"] = fp16_time
            comparison["rq_time"] = rq_time
            comparison["fp16_text"] = fp16_text
            comparison["rq_text"] = rq_text
            comparison["oom"] = False

            # Print summary
            print(f"\n  --- Results for {target_len} tokens ---")
            print(f"  Match rate: {comparison['match_rate']:.4f} "
                  f"({comparison['matching_tokens']}/{comparison['total_tokens']})")
            print(f"  First divergence: token {comparison['first_divergence']}")
            print(f"  Total divergences: {comparison['num_divergences']}")

            if comparison["div_context"]:
                dc = comparison["div_context"]
                print(f"\n  At divergence point (token {dc['position']}):")
                print(f"    FP16 token: {repr(dc['fp16_token'])} (id={dc['fp16_token_id']})")
                print(f"    RQ3  token: {repr(dc['rq_token'])} (id={dc['rq_token_id']})")
                print(f"    FP16 context: ...{dc['fp16_context']}...")
                print(f"    RQ3  context: ...{dc['rq_context']}...")
            else:
                print(f"\n  PERFECT MATCH: All {comparison['total_tokens']} tokens identical!")

            print(f"\n  FP16 speed: {len(fp16_ids)/fp16_time:.1f} tok/s")
            print(f"  RQ3  speed: {len(rq_ids)/rq_time:.1f} tok/s")
        else:
            comparison = {
                "target_tokens": target_len,
                "total_tokens": 0,
                "matching_tokens": 0,
                "match_rate": 0,
                "first_divergence": 0,
                "num_divergences": 0,
                "divergence_positions": [],
                "div_context": None,
                "fp16_actual_tokens": len(fp16_ids),
                "rq_actual_tokens": 0,
                "fp16_time": fp16_time,
                "rq_time": 0,
                "fp16_text": fp16_text,
                "rq_text": "[OOM]",
                "oom": True,
            }

        all_results.append(comparison)

        # Cleanup between length tests
        del rq_cache
        gc.collect()
        torch.cuda.empty_cache()

    # --- Save results ---
    md = format_results(all_results)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write(md)
    print(f"\nResults saved to {RESULTS_PATH}")
    print("\n" + md)

    return all_results


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def format_results(results: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("# 32B Long Generation Stress Test")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Model:** {MODEL_NAME} (BnB NF4, 4-bit weights)")
    lines.append(f"**Hardware:** RTX 4090 24GB")
    lines.append(f"**KV Cache:** 3-bit ResidualQuant (K3/V3, boundary anchors, FP16 window=64)")
    lines.append(f"**Decoding:** Greedy (do_sample=False)")
    lines.append(f"**Prompt:** \"{PROMPT}\"")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Target | Actual | Match Rate | First Diverge | Divergences | FP16 (s) | RQ3 (s) | FP16 tok/s | RQ3 tok/s |")
    lines.append("|--------|--------|-----------|---------------|-------------|----------|---------|------------|-----------|")

    for r in results:
        target = r["target_tokens"]
        actual = r["total_tokens"]
        match = r["match_rate"]
        first_div = r["first_divergence"]
        num_div = r["num_divergences"]
        fp16_t = r["fp16_time"]
        rq_t = r["rq_time"]
        fp16_tps = r["fp16_actual_tokens"] / fp16_t
        rq_tps = r["rq_actual_tokens"] / rq_t

        if r.get("oom"):
            div_str = "OOM"
            first_str = "OOM"
        elif num_div == 0:
            div_str = "NONE"
            first_str = "N/A (perfect)"
        else:
            div_str = str(num_div)
            first_str = str(first_div)

        lines.append(
            f"| {target} | {actual} | {match:.4f} | {first_str} | "
            f"{div_str} | {fp16_t:.1f} | {rq_t:.1f} | "
            f"{fp16_tps:.1f} | {rq_tps:.1f} |"
        )

    lines.append("")

    # Detailed results per length
    for r in results:
        target = r["target_tokens"]
        lines.append(f"## {target}-Token Generation")
        lines.append("")

        if r.get("oom"):
            lines.append(f"**OOM:** CUDA out of memory during RQ3 generation at {target} tokens.")
            lines.append(f"FP16 baseline completed successfully ({r['fp16_actual_tokens']} tokens).")
            lines.append("")
            continue

        if r["num_divergences"] == 0:
            lines.append(
                f"**PERFECT MATCH:** All {r['total_tokens']} tokens identical "
                f"between FP16 and 3-bit ResidualQuant."
            )
        else:
            lines.append(
                f"**DIVERGENCE at token {r['first_divergence']}** "
                f"({r['num_divergences']} total divergences out of {r['total_tokens']} tokens)"
            )
            lines.append(f"- Match rate: {r['match_rate']:.4f}")

            if r["div_context"]:
                dc = r["div_context"]
                lines.append("")
                lines.append("### First Divergence Point")
                lines.append("")
                lines.append(f"| | Token | Token ID |")
                lines.append(f"|---|---|---|")
                lines.append(f"| FP16 | `{dc['fp16_token']}` | {dc['fp16_token_id']} |")
                lines.append(f"| RQ3 | `{dc['rq_token']}` | {dc['rq_token_id']} |")
                lines.append("")
                lines.append("**FP16 context around divergence:**")
                lines.append(f"> ...{dc['fp16_context']}...")
                lines.append("")
                lines.append("**RQ3 context around divergence:**")
                lines.append(f"> ...{dc['rq_context']}...")

            if r["divergence_positions"]:
                lines.append("")
                lines.append(f"Divergence positions (first 20): {r['divergence_positions']}")

        lines.append("")

        # Show first 300 chars of each text for comparison
        lines.append("### Text Comparison (first 500 chars)")
        lines.append("")
        lines.append("**FP16:**")
        lines.append(f"```")
        lines.append(r["fp16_text"][:500])
        lines.append(f"```")
        lines.append("")
        lines.append("**RQ3:**")
        lines.append(f"```")
        lines.append(r["rq_text"][:500])
        lines.append(f"```")
        lines.append("")

    # Overall verdict
    lines.append("## Verdict")
    lines.append("")

    all_perfect = all(r["num_divergences"] == 0 for r in results)
    max_perfect = 0
    for r in results:
        if r["num_divergences"] == 0:
            max_perfect = max(max_perfect, r["total_tokens"])
        else:
            break

    if all_perfect:
        max_len = max(r["total_tokens"] for r in results)
        lines.append(
            f"3-bit ResidualQuant KV cache produces **IDENTICAL** generation "
            f"to FP16 across all tested lengths up to **{max_len} tokens** "
            f"on Qwen2.5-32B-Instruct."
        )
    else:
        first_fail = next(r for r in results if r["num_divergences"] > 0)
        lines.append(
            f"3-bit ResidualQuant matches FP16 up to **{max_perfect} tokens**, "
            f"with first divergence at token **{first_fail['first_divergence']}** "
            f"in the {first_fail['target_tokens']}-token test."
        )
        lines.append("")
        lines.append(
            f"Match rate at divergence point: {first_fail['match_rate']:.4f}"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_stress_test()
