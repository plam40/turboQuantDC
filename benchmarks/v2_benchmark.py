#!/usr/bin/env python3
"""TurboQuantDC v2 unified benchmark.

Tests TurboQuantV2Cache against FP16 baseline on Qwen2.5-3B:
  1. Calibrate PCA rotations (128 tokens)
  2. Generate 200 tokens in compress mode -- quality vs FP16
  3. Generate 200 tokens in full mode (FAISS retrieval) -- quality + speed
  4. Report: effective bits, compression ratio, token match rate, speed

Run:
    python benchmarks/v2_benchmark.py
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Allow running from repo root
REPO_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, REPO_ROOT)

from turboquantdc.v2_cache import TurboQuantV2Cache, V2Config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("V2_MODEL", "Qwen/Qwen2.5-3B-Instruct")
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MAX_NEW_TOKENS = 200

GENERATION_PROMPTS = [
    (
        "Explain the mathematical foundations of KV cache compression in "
        "transformer-based language models. Start from the attention mechanism "
        "and derive why quantization of key vectors requires careful treatment "
        "of inner product preservation:"
    ),
    "What is the capital of Australia? Answer in one sentence:",
    "Write a Python function to calculate the factorial of a number:",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    """Load model with 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME} (4-bit quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_with_cache(
    model,
    tokenizer,
    prompt: str,
    cache,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Tuple[str, List[int], float]:
    """Generate tokens with a given cache."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    torch.cuda.empty_cache()
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            past_key_values=cache,
            use_cache=True,
        )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return gen_text, gen_ids, elapsed


def token_match_rate(baseline_ids: List[int], test_ids: List[int]) -> float:
    min_len = min(len(baseline_ids), len(test_ids))
    if min_len == 0:
        return 0.0
    matches = sum(1 for a, b in zip(baseline_ids[:min_len], test_ids[:min_len]) if a == b)
    return matches / min_len


def first_divergence(baseline_ids: List[int], test_ids: List[int]) -> int:
    for i, (a, b) in enumerate(zip(baseline_ids, test_ids)):
        if a != b:
            return i
    return min(len(baseline_ids), len(test_ids))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    results = {
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Detect model config
    num_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    print(f"Model: {num_layers} layers, d={head_dim}, {num_kv_heads} KV heads")
    results["model_config"] = {
        "num_layers": num_layers,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
    }

    # ---------------------------------------------------------------
    # Step 1: Calibrate PCA rotations
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: Calibrate PCA rotations")
    print("=" * 70)

    cal_path = os.path.join(REPO_ROOT, "benchmarks", "results", "v2_pca_rotations.pt")
    t0 = time.perf_counter()
    pca_rotations = TurboQuantV2Cache.calibrate(
        model, tokenizer, n_tokens=128, save_path=cal_path, device=DEVICE,
    )
    cal_time = time.perf_counter() - t0
    print(f"Calibration time: {cal_time:.2f}s")
    results["calibration"] = {
        "n_tokens": 128,
        "time_sec": round(cal_time, 2),
        "n_layers_calibrated": len(pca_rotations),
    }

    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Step 2: FP16 baseline generation
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: FP16 baseline generation")
    print("=" * 70)

    baseline_results = []
    for i, prompt in enumerate(GENERATION_PROMPTS):
        print(f"\n  Prompt {i+1}: {prompt[:60]}...")
        text, ids, elapsed = generate_with_cache(model, tokenizer, prompt, cache=None)
        tps = len(ids) / elapsed if elapsed > 0 else 0
        print(f"  Generated {len(ids)} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")
        print(f"  Output: {text[:100]}...")
        baseline_results.append({
            "prompt_idx": i,
            "text": text,
            "ids": ids,
            "n_tokens": len(ids),
            "time_sec": round(elapsed, 2),
            "tokens_per_sec": round(tps, 1),
        })
        gc.collect()
        torch.cuda.empty_cache()

    results["fp16_baseline"] = baseline_results

    # ---------------------------------------------------------------
    # Step 3: Compress mode (no retrieval)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: V2 Compress mode (PCA + mean-removal + adaptive + delta)")
    print("=" * 70)

    compress_config = V2Config(
        key_bits=3,
        val_bits=3,
        window_size=64,
        boundary_layers=2,
        mode="compress",
        seed=SEED,
    )

    compress_results = []
    for i, prompt in enumerate(GENERATION_PROMPTS):
        print(f"\n  Prompt {i+1}: {prompt[:60]}...")

        # Use WHT rotation (more robust than PCA for generation)
        cache = TurboQuantV2Cache(
            config=compress_config,
            num_layers=num_layers,
            pca_rotations={},  # WHT fallback
        )

        text, ids, elapsed = generate_with_cache(model, tokenizer, prompt, cache=cache)
        tps = len(ids) / elapsed if elapsed > 0 else 0

        # Metrics
        eff_bits = cache.effective_bits()
        comp_ratio = cache.compression_ratio()
        match_rate = token_match_rate(baseline_results[i]["ids"], ids)
        first_div = first_divergence(baseline_results[i]["ids"], ids)
        tier_info = cache.tier_summary()

        print(f"  Generated {len(ids)} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")
        print(f"  Effective bits: {eff_bits:.2f}, Compression: {comp_ratio:.1f}x")
        print(f"  Token match: {match_rate:.1%}, First divergence: token {first_div}")
        print(f"  Tier summary: {tier_info}")
        print(f"  Output: {text[:100]}...")

        compress_results.append({
            "prompt_idx": i,
            "text": text,
            "ids": ids,
            "n_tokens": len(ids),
            "time_sec": round(elapsed, 2),
            "tokens_per_sec": round(tps, 1),
            "effective_bits": round(eff_bits, 2),
            "compression_ratio": round(comp_ratio, 1),
            "token_match_rate": round(match_rate, 4),
            "first_divergence": first_div,
            "tier_summary": tier_info,
        })

        del cache
        gc.collect()
        torch.cuda.empty_cache()

    results["compress_mode"] = compress_results

    # ---------------------------------------------------------------
    # Step 4: Full mode (with FAISS retrieval)
    # ---------------------------------------------------------------
    try:
        import faiss
        has_faiss = True
    except ImportError:
        has_faiss = False
        print("\n  FAISS not available, skipping full mode")

    if has_faiss:
        print("\n" + "=" * 70)
        print("STEP 4: V2 Full mode (compression + FAISS retrieval)")
        print("=" * 70)

        full_config = V2Config(
            key_bits=3,
            val_bits=3,
            window_size=64,
            boundary_layers=2,
            retrieval_k=128,
            faiss_nprobe=16,
            faiss_nlist=64,
            mode="full",
            seed=SEED,
        )

        full_results = []
        for i, prompt in enumerate(GENERATION_PROMPTS):
            print(f"\n  Prompt {i+1}: {prompt[:60]}...")

            # Use WHT rotation (more robust than PCA for generation)
            cache = TurboQuantV2Cache(
                config=full_config,
                num_layers=num_layers,
                pca_rotations={},  # WHT fallback
            )

            text, ids, elapsed = generate_with_cache(model, tokenizer, prompt, cache=cache)
            tps = len(ids) / elapsed if elapsed > 0 else 0

            eff_bits = cache.effective_bits()
            comp_ratio = cache.compression_ratio()
            match_rate = token_match_rate(baseline_results[i]["ids"], ids)
            first_div = first_divergence(baseline_results[i]["ids"], ids)
            tier_info = cache.tier_summary()

            print(f"  Generated {len(ids)} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")
            print(f"  Effective bits: {eff_bits:.2f}, Compression: {comp_ratio:.1f}x")
            print(f"  Token match: {match_rate:.1%}, First divergence: token {first_div}")
            print(f"  Output: {text[:100]}...")

            full_results.append({
                "prompt_idx": i,
                "text": text,
                "ids": ids,
                "n_tokens": len(ids),
                "time_sec": round(elapsed, 2),
                "tokens_per_sec": round(tps, 1),
                "effective_bits": round(eff_bits, 2),
                "compression_ratio": round(comp_ratio, 1),
                "token_match_rate": round(match_rate, 4),
                "first_divergence": first_div,
                "tier_summary": tier_info,
            })

            del cache
            gc.collect()
            torch.cuda.empty_cache()

        results["full_mode"] = full_results

    # ---------------------------------------------------------------
    # Step 5: Sweep configurations
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Configuration sweep")
    print("=" * 70)

    sweep_configs = [
        # WHT rotation (default, robust) -- prefix with wht_ to use WHT
        ("wht_w32_3bit", V2Config(key_bits=3, val_bits=3, window_size=32, mode="compress")),
        ("wht_w64_3bit", V2Config(key_bits=3, val_bits=3, window_size=64, mode="compress")),
        ("wht_w128_3bit", V2Config(key_bits=3, val_bits=3, window_size=128, mode="compress")),
        ("wht_w64_4bit", V2Config(key_bits=4, val_bits=3, window_size=64, tier_bits=[4, 3, 2], mode="compress")),
        ("wht_w32_2bit", V2Config(key_bits=2, val_bits=2, window_size=32, tier_bits=[3, 2, 1], mode="compress")),
        ("wht_w0_3bit", V2Config(key_bits=3, val_bits=3, window_size=0, boundary_layers=2, mode="compress")),
        # PCA rotation (calibrated) for comparison
        ("pca_w64_3bit", V2Config(key_bits=3, val_bits=3, window_size=64, mode="compress")),
        ("pca_w128_3bit", V2Config(key_bits=3, val_bits=3, window_size=128, mode="compress")),
    ]

    prompt = GENERATION_PROMPTS[0]
    sweep_results = []

    for name, cfg in sweep_configs:
        print(f"\n  Config: {name}")
        # PCA configs get pca_rotations, WHT configs get empty dict
        use_pca = name.startswith("pca_")
        cache = TurboQuantV2Cache(
            config=cfg,
            num_layers=num_layers,
            pca_rotations=pca_rotations if use_pca else {},
        )

        text, ids, elapsed = generate_with_cache(model, tokenizer, prompt, cache=cache)
        tps = len(ids) / elapsed if elapsed > 0 else 0
        eff_bits = cache.effective_bits()
        comp_ratio = cache.compression_ratio()
        match_rate = token_match_rate(baseline_results[0]["ids"], ids)
        first_div = first_divergence(baseline_results[0]["ids"], ids)

        print(f"    Eff bits: {eff_bits:.2f}, Compression: {comp_ratio:.1f}x, "
              f"Match: {match_rate:.1%}, Div@{first_div}, "
              f"{tps:.1f} tok/s")

        sweep_results.append({
            "name": name,
            "effective_bits": round(eff_bits, 2),
            "compression_ratio": round(comp_ratio, 1),
            "token_match_rate": round(match_rate, 4),
            "first_divergence": first_div,
            "tokens_per_sec": round(tps, 1),
            "time_sec": round(elapsed, 2),
        })

        del cache
        gc.collect()
        torch.cuda.empty_cache()

    results["sweep"] = sweep_results

    # ---------------------------------------------------------------
    # Generate report
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)

    report = generate_report(results)
    report_path = os.path.join(REPO_ROOT, "benchmarks", "results", "v2_results.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    return results


def generate_report(results: Dict) -> str:
    """Generate markdown report from benchmark results."""
    lines = []
    lines.append("# TurboQuantDC v2 Unified Cache Results")
    lines.append(f"\n**Date:** {results['timestamp']}")
    lines.append(f"**Model:** {results['model']}")
    lines.append(f"**Device:** {results['device']}")
    lines.append(f"**Max new tokens:** {results['max_new_tokens']}")

    mc = results.get("model_config", {})
    lines.append(f"\n**Model config:** {mc.get('num_layers', '?')} layers, "
                 f"d={mc.get('head_dim', '?')}, "
                 f"{mc.get('num_kv_heads', '?')} KV heads")

    # Calibration
    cal = results.get("calibration", {})
    lines.append(f"\n## Calibration")
    lines.append(f"- Tokens: {cal.get('n_tokens', '?')}")
    lines.append(f"- Time: {cal.get('time_sec', '?')}s")
    lines.append(f"- Layers calibrated: {cal.get('n_layers_calibrated', '?')}")

    # FP16 Baseline
    lines.append("\n## FP16 Baseline")
    lines.append("| Prompt | Tokens | Time (s) | Tok/s |")
    lines.append("|--------|--------|----------|-------|")
    for r in results.get("fp16_baseline", []):
        lines.append(f"| {r['prompt_idx']+1} | {r['n_tokens']} | {r['time_sec']} | {r['tokens_per_sec']} |")

    # Compress mode
    lines.append("\n## Compress Mode (PCA + Mean-Removal + Adaptive + DeltaQuant)")
    lines.append("| Prompt | Eff Bits | Compression | Token Match | First Div | Tok/s |")
    lines.append("|--------|----------|-------------|-------------|-----------|-------|")
    for r in results.get("compress_mode", []):
        lines.append(
            f"| {r['prompt_idx']+1} | {r['effective_bits']} | "
            f"{r['compression_ratio']}x | {r['token_match_rate']:.1%} | "
            f"{r['first_divergence']} | {r['tokens_per_sec']} |"
        )

    # Full mode
    if "full_mode" in results:
        lines.append("\n## Full Mode (Compression + FAISS Retrieval)")
        lines.append("| Prompt | Eff Bits | Compression | Token Match | First Div | Tok/s |")
        lines.append("|--------|----------|-------------|-------------|-----------|-------|")
        for r in results.get("full_mode", []):
            lines.append(
                f"| {r['prompt_idx']+1} | {r['effective_bits']} | "
                f"{r['compression_ratio']}x | {r['token_match_rate']:.1%} | "
                f"{r['first_divergence']} | {r['tokens_per_sec']} |"
            )

    # Configuration sweep
    lines.append("\n## Configuration Sweep")
    lines.append("| Config | Eff Bits | Compression | Token Match | First Div | Tok/s |")
    lines.append("|--------|----------|-------------|-------------|-----------|-------|")
    for r in results.get("sweep", []):
        lines.append(
            f"| {r['name']} | {r['effective_bits']} | "
            f"{r['compression_ratio']}x | {r['token_match_rate']:.1%} | "
            f"{r['first_divergence']} | {r['tokens_per_sec']} |"
        )

    # Summary
    lines.append("\n## Summary")

    # Average across compress mode prompts
    cm = results.get("compress_mode", [])
    if cm:
        avg_bits = sum(r["effective_bits"] for r in cm) / len(cm)
        avg_ratio = sum(r["compression_ratio"] for r in cm) / len(cm)
        avg_match = sum(r["token_match_rate"] for r in cm) / len(cm)
        avg_tps = sum(r["tokens_per_sec"] for r in cm) / len(cm)
        lines.append(f"\n**Compress mode averages:**")
        lines.append(f"- Effective bits: {avg_bits:.2f}")
        lines.append(f"- Compression ratio: {avg_ratio:.1f}x")
        lines.append(f"- Token match rate: {avg_match:.1%}")
        lines.append(f"- Speed: {avg_tps:.1f} tok/s")

    if "full_mode" in results:
        fm = results["full_mode"]
        avg_bits = sum(r["effective_bits"] for r in fm) / len(fm)
        avg_ratio = sum(r["compression_ratio"] for r in fm) / len(fm)
        avg_match = sum(r["token_match_rate"] for r in fm) / len(fm)
        avg_tps = sum(r["tokens_per_sec"] for r in fm) / len(fm)
        lines.append(f"\n**Full mode averages:**")
        lines.append(f"- Effective bits: {avg_bits:.2f}")
        lines.append(f"- Compression ratio: {avg_ratio:.1f}x")
        lines.append(f"- Token match rate: {avg_match:.1%}")
        lines.append(f"- Speed: {avg_tps:.1f} tok/s")

    # Sweep highlight
    sw = results.get("sweep", [])
    if sw:
        best_comp = max(sw, key=lambda x: x["compression_ratio"])
        best_qual = max(sw, key=lambda x: x["token_match_rate"])
        lines.append(f"\n**Best compression:** {best_comp['name']} "
                     f"({best_comp['compression_ratio']}x, {best_comp['token_match_rate']:.1%} match)")
        lines.append(f"**Best quality:** {best_qual['name']} "
                     f"({best_qual['compression_ratio']}x, {best_qual['token_match_rate']:.1%} match)")

    lines.append("\n## Architecture")
    lines.append("```")
    lines.append("Input KV -> Boundary check -> FP16 hot window -> PCA rotation")
    lines.append("  -> Mean removal -> Importance scoring -> Tier assignment")
    lines.append("  -> Tier 0 (top 5%):     4-bit ResidualQuant")
    lines.append("  -> Tier 1 (next 25%):   3-bit ResidualQuant")
    lines.append("  -> Tier 2 (bottom 70%): DeltaQuant (3-bit anchor + 1-bit delta)")
    lines.append("  -> FAISS index (full mode) -> Store compressed")
    lines.append("```")

    return "\n".join(lines)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(SEED)
    run_benchmark()
