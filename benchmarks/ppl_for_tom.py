#!/usr/bin/env python3
"""Perplexity benchmark for Tom: mean-removal vs production WHT.

Computes real perplexity on wikitext-2 test split using:
  a. FP16 KV baseline (no compression)
  b. WHT 3-bit (GenerationCache, center_before_quantize=False)
  c. WHT 3-bit + mean-removal (GenerationCache, center_before_quantize=True)
  d. WHT 4-bit (GenerationCache, center_before_quantize=False)
  e. WHT 4-bit + mean-removal (GenerationCache, center_before_quantize=True)

Models: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct (both BnB 4-bit loaded).
Dataset: wikitext-2 test split.
Method: sliding window PPL (context=512, stride=256).

Run:
    python benchmarks/ppl_for_tom.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Allow running from repo root
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from turboquantdc.generation_core import GenerationCache

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_CACHE_DIR = "/media/dhawal/Beast/cache/hub"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
CONTEXT_LEN = 512     # tokens per window
STRIDE = 256          # sliding window stride
MAX_EVAL_TOKENS = 4096  # max tokens from wikitext-2 test to evaluate

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

# Configurations to test: (name, key_bits, center_before_quantize, extra_kwargs)
CONFIGS = [
    ("FP16 baseline", None, None, {}),
    ("WHT 3-bit", 3, False, {}),
    ("WHT 3-bit + mean-removal", 3, True, {}),
    ("WHT 4-bit", 4, False, {}),
    ("WHT 4-bit + mean-removal", 4, True, {}),
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_wikitext2_test() -> str:
    """Load wikitext-2 test split and return as a single string."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                      cache_dir=HF_CACHE_DIR)
    # Concatenate all non-empty lines
    text = "\n".join(line for line in ds["text"] if line.strip())
    return text


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str):
    """Load model with BnB 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\nLoading {model_name} (BnB 4-bit)...")
    t0 = time.perf_counter()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=HF_CACHE_DIR,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    elapsed = time.perf_counter() - t0
    num_layers = model.config.num_hidden_layers
    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"  Loaded in {elapsed:.1f}s | {num_layers} layers, head_dim={head_dim}")
    return model, tokenizer, num_layers, head_dim


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------

def compute_ppl_fp16(
    model,
    tokenizer,
    text: str,
    context_len: int = CONTEXT_LEN,
    stride: int = STRIDE,
    max_tokens: int = MAX_EVAL_TOKENS,
) -> Tuple[float, int]:
    """Compute perplexity with no KV cache compression (FP16 baseline).

    Uses sliding window: for each window of context_len tokens, compute
    the loss on the last (context_len - stride) tokens.

    Returns (perplexity, num_tokens_evaluated).
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_tokens)
    input_ids = encodings["input_ids"].to(model.device)
    seq_len = input_ids.shape[1]

    nlls = []
    num_tokens = 0

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + context_len, seq_len)
        chunk = input_ids[:, begin:end]

        # Target: predict each token from the previous
        target = chunk.clone()
        # Mask out tokens we've already evaluated (overlap region)
        if begin > 0:
            target[:, :stride] = -100
        # The first token has no prediction target
        target[:, 0] = -100

        with torch.no_grad():
            outputs = model(chunk, labels=target)
        loss = outputs.loss

        # Count non-masked tokens
        n_tokens = (target != -100).sum().item()
        if n_tokens > 0:
            nlls.append(loss.item() * n_tokens)
            num_tokens += n_tokens

        if end >= seq_len:
            break

    if num_tokens == 0:
        return float("inf"), 0

    avg_nll = sum(nlls) / num_tokens
    ppl = math.exp(avg_nll)
    return ppl, num_tokens


def compute_ppl_compressed(
    model,
    tokenizer,
    text: str,
    key_bits: int,
    center_before_quantize: bool,
    num_layers: int,
    context_len: int = CONTEXT_LEN,
    stride: int = STRIDE,
    max_tokens: int = MAX_EVAL_TOKENS,
) -> Tuple[float, int]:
    """Compute perplexity with compressed KV cache.

    Strategy: use model.generate's KV cache integration. For each sliding
    window, do a forward pass with our GenerationCache as past_key_values.

    The model will call cache.update() for each layer, which compresses
    the KV states and returns dequantized versions for attention.

    Returns (perplexity, num_tokens_evaluated).
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_tokens)
    input_ids = encodings["input_ids"].to(model.device)
    seq_len = input_ids.shape[1]

    nlls = []
    num_tokens = 0

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + context_len, seq_len)
        chunk = input_ids[:, begin:end]

        target = chunk.clone()
        if begin > 0:
            target[:, :stride] = -100
        target[:, 0] = -100

        # Create a FRESH cache for each window (no cross-window leakage)
        cache = GenerationCache(
            key_bits=key_bits,
            val_bits=3,  # values always at 3-bit
            fp16_window=0,  # no FP16 window -- test pure compression
            anchor_interval=0,  # no anchor layers -- test pure compression
            num_layers=num_layers,
            anchor_strategy="fixed",
            seed=SEED,
            use_norm_correction=True,
            use_residual_quant=True,
            center_before_quantize=center_before_quantize,
        )

        with torch.no_grad():
            outputs = model(chunk, past_key_values=cache, use_cache=True,
                            labels=target)
        loss = outputs.loss

        n_tokens = (target != -100).sum().item()
        if n_tokens > 0:
            nlls.append(loss.item() * n_tokens)
            num_tokens += n_tokens

        # Free cache memory
        del cache
        torch.cuda.empty_cache()

        if end >= seq_len:
            break

    if num_tokens == 0:
        return float("inf"), 0

    avg_nll = sum(nlls) / num_tokens
    ppl = math.exp(avg_nll)
    return ppl, num_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ppl_benchmark():
    """Run the full perplexity benchmark."""
    print("=" * 70)
    print("PPL Benchmark for Tom: Mean-Removal vs Production WHT")
    print("=" * 70)

    # Load wikitext-2
    print("\nLoading wikitext-2 test split...")
    wikitext_text = load_wikitext2_test()
    print(f"  Text length: {len(wikitext_text)} chars")

    results = {}

    for model_name in MODELS:
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print(f"{'=' * 70}")

        try:
            model, tokenizer, num_layers, head_dim = load_model_and_tokenizer(model_name)
        except Exception as e:
            print(f"  FAILED to load model: {e}")
            print(f"  Skipping {model_name}")
            results[model_name] = {"error": str(e)}
            continue

        # Tokenize once to report token count
        test_enc = tokenizer(wikitext_text, truncation=True, max_length=MAX_EVAL_TOKENS)
        total_tokens = len(test_enc["input_ids"])
        print(f"  Eval tokens: {total_tokens} (max {MAX_EVAL_TOKENS})")
        print(f"  Window: {CONTEXT_LEN} tokens, stride: {STRIDE}")

        model_results = {}

        for config_name, key_bits, center, extra_kw in CONFIGS:
            print(f"\n  --- {config_name} ---")
            torch.cuda.empty_cache()
            gc.collect()

            t0 = time.perf_counter()
            try:
                if key_bits is None:
                    # FP16 baseline
                    ppl, n_tok = compute_ppl_fp16(
                        model, tokenizer, wikitext_text,
                    )
                else:
                    ppl, n_tok = compute_ppl_compressed(
                        model, tokenizer, wikitext_text,
                        key_bits=key_bits,
                        center_before_quantize=center,
                        num_layers=num_layers,
                    )
                elapsed = time.perf_counter() - t0
                print(f"  PPL = {ppl:.4f}  ({n_tok} tokens, {elapsed:.1f}s)")
                model_results[config_name] = {
                    "ppl": round(ppl, 4),
                    "tokens": n_tok,
                    "time_sec": round(elapsed, 1),
                }
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"  FAILED: {e} ({elapsed:.1f}s)")
                model_results[config_name] = {"error": str(e)}

        results[model_name] = model_results

        # Free model memory
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    # --- Report ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    report_lines = []
    report_lines.append("# PPL Benchmark: Mean-Removal vs Production WHT")
    report_lines.append("")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"Dataset: wikitext-2 test (max {MAX_EVAL_TOKENS} tokens)")
    report_lines.append(f"Window: {CONTEXT_LEN} tokens, stride {STRIDE}")
    report_lines.append(f"Compression: anchor=0, fp16_window=0, RQ=True, V3-bit")
    report_lines.append(f"Context: Tom's TQ 3-bit gets +62.95 PPL on Qwen2.5-3B (catastrophic)")
    report_lines.append("")

    for model_name, model_res in results.items():
        short_name = model_name.split("/")[-1]
        report_lines.append(f"## {short_name}")
        report_lines.append("")

        if "error" in model_res:
            report_lines.append(f"FAILED: {model_res['error']}")
            report_lines.append("")
            continue

        # Get baseline PPL for delta computation
        baseline_ppl = None
        for cfg_name in ["FP16 baseline"]:
            if cfg_name in model_res and "ppl" in model_res[cfg_name]:
                baseline_ppl = model_res[cfg_name]["ppl"]

        report_lines.append("| Config | PPL | Delta vs FP16 | Time |")
        report_lines.append("|--------|-----|---------------|------|")

        for config_name, _, _, _ in CONFIGS:
            if config_name not in model_res:
                continue
            r = model_res[config_name]
            if "error" in r:
                report_lines.append(f"| {config_name} | FAILED | - | - |")
                continue
            ppl = r["ppl"]
            t = r["time_sec"]
            if baseline_ppl and config_name != "FP16 baseline":
                delta = ppl - baseline_ppl
                delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
            else:
                delta_str = "baseline"
            report_lines.append(f"| {config_name} | {ppl:.4f} | {delta_str} | {t:.1f}s |")

        report_lines.append("")

    # Analysis
    report_lines.append("## Analysis")
    report_lines.append("")
    for model_name, model_res in results.items():
        if "error" in model_res:
            continue
        short_name = model_name.split("/")[-1]

        baseline = model_res.get("FP16 baseline", {}).get("ppl")
        wht3 = model_res.get("WHT 3-bit", {}).get("ppl")
        wht3m = model_res.get("WHT 3-bit + mean-removal", {}).get("ppl")
        wht4 = model_res.get("WHT 4-bit", {}).get("ppl")
        wht4m = model_res.get("WHT 4-bit + mean-removal", {}).get("ppl")

        if baseline and wht3 and wht3m:
            d3 = wht3 - baseline
            d3m = wht3m - baseline
            helps_3 = "YES" if d3m < d3 else "NO"
            report_lines.append(
                f"- {short_name} 3-bit: mean-removal helps? {helps_3} "
                f"(delta {d3:.2f} -> {d3m:.2f}, improvement {d3-d3m:.2f})"
            )
        if baseline and wht4 and wht4m:
            d4 = wht4 - baseline
            d4m = wht4m - baseline
            helps_4 = "YES" if d4m < d4 else "NO"
            report_lines.append(
                f"- {short_name} 4-bit: mean-removal helps? {helps_4} "
                f"(delta {d4:.2f} -> {d4m:.2f}, improvement {d4-d4m:.2f})"
            )

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save
    results_dir = REPO_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "ppl_for_tom.md", "w") as f:
        f.write(report_text)
    with open(results_dir / "ppl_for_tom.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_dir / 'ppl_for_tom.md'}")
    return results


if __name__ == "__main__":
    run_ppl_benchmark()
