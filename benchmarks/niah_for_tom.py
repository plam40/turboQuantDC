#!/usr/bin/env python3
"""Needle-in-a-Haystack benchmark for Tom: mean-removal vs WHT.

Tests KV cache compression at long context by inserting a unique fact
into filler text at different positions and checking recall.

Positions: 10%, 50%, 90% of context
Configs: FP16 baseline, WHT 3-bit, WHT 3-bit + mean-removal
Context: As long as fits on RTX 4090 with BnB 4-bit model.

Run:
    python benchmarks/niah_for_tom.py
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

# The needle: a unique fact to be recalled
NEEDLE = "The secret code is PINEAPPLE-77."
NEEDLE_ANSWER = "PINEAPPLE-77"

# Filler text: a repetitive but coherent paragraph that won't contain the needle
FILLER_PARAGRAPH = (
    "The history of mathematics spans thousands of years. Ancient civilizations "
    "like the Babylonians and Egyptians developed practical arithmetic for trade "
    "and construction. Greek mathematicians introduced rigorous proof and abstract "
    "reasoning. The development of algebra in the Islamic Golden Age transformed "
    "mathematical thought. European mathematicians during the Renaissance built on "
    "these foundations, leading to calculus, probability theory, and modern analysis. "
    "Today, mathematics underpins all of science and technology, from quantum physics "
    "to artificial intelligence. The beauty of mathematics lies in its universal "
    "applicability and its capacity to describe the fundamental patterns of nature. "
)

# Positions to test (fraction of context)
NEEDLE_POSITIONS = [0.10, 0.50, 0.90]

# Model to use
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Target context lengths to try (in tokens, descending -- use first that fits)
TARGET_CONTEXT_LENGTHS = [16384, 8192, 4096, 2048]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_haystack_chat(
    tokenizer,
    target_tokens: int,
    needle: str,
    needle_position: float,
) -> Tuple[str, int]:
    """Build a haystack as a chat-template prompt with needle at given position.

    Uses the model's chat template so the model actually follows instructions.

    Returns (formatted_prompt_text, approximate_needle_token_position).
    """
    filler_tokens = tokenizer(FILLER_PARAGRAPH, add_special_tokens=False)["input_ids"]
    filler_len = len(filler_tokens)

    # Reserve tokens for chat template overhead (~100 tokens) and the question
    question = "Based on the text above, what is the secret code? Answer with ONLY the code, nothing else."
    overhead = 200  # chat template tokens + question tokens

    available = target_tokens - overhead
    if available < 100:
        raise ValueError(f"Target {target_tokens} tokens too small for haystack")

    # Compute how many filler paragraphs we need
    num_paragraphs = max(1, math.ceil(available / filler_len))

    # Build pre-needle and post-needle filler
    needle_para_idx = int(num_paragraphs * needle_position)
    needle_para_idx = max(1, min(needle_para_idx, num_paragraphs - 1))

    pre_filler = FILLER_PARAGRAPH * needle_para_idx
    post_filler = FILLER_PARAGRAPH * (num_paragraphs - needle_para_idx)

    haystack_body = pre_filler + "\n" + needle + "\n" + post_filler

    # Build as a chat message
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Read the following document carefully and answer the question at the end."},
        {"role": "user", "content": haystack_body.strip() + "\n\n" + question},
    ]

    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    # Check token count and truncate filler if needed
    tokens = tokenizer(formatted, add_special_tokens=False)["input_ids"]
    while len(tokens) > target_tokens and num_paragraphs > 2:
        num_paragraphs -= 1
        needle_para_idx = int(num_paragraphs * needle_position)
        needle_para_idx = max(1, min(needle_para_idx, num_paragraphs - 1))
        pre_filler = FILLER_PARAGRAPH * needle_para_idx
        post_filler = FILLER_PARAGRAPH * (num_paragraphs - needle_para_idx)
        haystack_body = pre_filler + "\n" + needle + "\n" + post_filler
        messages[1]["content"] = haystack_body.strip() + "\n\n" + question
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        tokens = tokenizer(formatted, add_special_tokens=False)["input_ids"]

    needle_tok_pos = int(len(tokens) * needle_position)
    return formatted, needle_tok_pos


def load_model(model_name: str):
    """Load model with BnB 4-bit."""
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

    num_layers = model.config.num_hidden_layers
    elapsed = time.perf_counter() - t0
    print(f"  Loaded in {elapsed:.1f}s | {num_layers} layers")
    return model, tokenizer, num_layers


def generate_with_cache(
    model,
    tokenizer,
    input_text: str,
    cache,
    max_new_tokens: int = 30,
) -> str:
    """Generate tokens with optional compressed cache."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True,
                       max_length=32768)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            past_key_values=cache,
            use_cache=True,
        )

    gen_ids = outputs[0][input_ids.shape[1]:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen_text


def check_recall(generated: str, expected: str) -> bool:
    """Check if the generated text contains the expected answer."""
    return expected.lower() in generated.lower()


# ---------------------------------------------------------------------------
# NIAH Test
# ---------------------------------------------------------------------------

def run_niah_single(
    model,
    tokenizer,
    num_layers: int,
    context_tokens: int,
    needle_pos: float,
    config_name: str,
    key_bits: Optional[int],
    center: Optional[bool],
) -> Dict[str, Any]:
    """Run a single NIAH test."""
    torch.cuda.empty_cache()
    gc.collect()

    # Build haystack
    haystack_text, needle_tok_pos = build_haystack_chat(
        tokenizer, context_tokens, NEEDLE, needle_pos,
    )

    actual_tokens = len(tokenizer(haystack_text, add_special_tokens=False)["input_ids"])

    # Create cache
    if key_bits is None:
        cache = None
        cache_desc = "FP16"
    else:
        cache = GenerationCache(
            key_bits=key_bits,
            val_bits=3,
            fp16_window=0,
            anchor_interval=0,
            num_layers=num_layers,
            anchor_strategy="fixed",
            seed=SEED,
            use_norm_correction=True,
            use_residual_quant=True,
            center_before_quantize=center,
        )
        cache_desc = f"K{key_bits}{'_mean' if center else ''}"

    print(f"    [{config_name}] pos={needle_pos:.0%}, tokens={actual_tokens}, cache={cache_desc}...")

    t0 = time.perf_counter()
    try:
        generated = generate_with_cache(
            model, tokenizer, haystack_text, cache, max_new_tokens=30,
        )
        elapsed = time.perf_counter() - t0
        recalled = check_recall(generated, NEEDLE_ANSWER)
        print(f"      -> {'PASS' if recalled else 'FAIL'}: \"{generated.strip()[:80]}\" ({elapsed:.1f}s)")

        return {
            "config": config_name,
            "needle_position": needle_pos,
            "context_tokens": actual_tokens,
            "pass": recalled,
            "generated": generated.strip()[:200],
            "time_sec": round(elapsed, 1),
        }
    except torch.cuda.OutOfMemoryError as e:
        elapsed = time.perf_counter() - t0
        print(f"      -> OOM: {e} ({elapsed:.1f}s)")
        torch.cuda.empty_cache()
        return {
            "config": config_name,
            "needle_position": needle_pos,
            "context_tokens": actual_tokens,
            "pass": False,
            "error": "OOM",
            "time_sec": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"      -> ERROR: {e} ({elapsed:.1f}s)")
        return {
            "config": config_name,
            "needle_position": needle_pos,
            "context_tokens": actual_tokens,
            "pass": False,
            "error": str(e)[:200],
            "time_sec": round(elapsed, 1),
        }
    finally:
        if cache is not None:
            del cache
        torch.cuda.empty_cache()


def find_max_context(model, tokenizer, num_layers: int) -> int:
    """Find the largest context that fits without OOM."""
    for target in TARGET_CONTEXT_LENGTHS:
        print(f"  Trying context length {target}...")
        torch.cuda.empty_cache()
        gc.collect()

        try:
            haystack, _ = build_haystack_chat(tokenizer, target, NEEDLE, 0.5)
            # Quick test with compressed cache (most memory-intensive path)
            cache = GenerationCache(
                key_bits=3, val_bits=3, fp16_window=0,
                anchor_interval=0, num_layers=num_layers,
                anchor_strategy="fixed", seed=SEED,
                center_before_quantize=True,
            )
            inputs = tokenizer(haystack, return_tensors="pt", truncation=True,
                               max_length=target)
            input_ids = inputs["input_ids"].to(model.device)
            with torch.no_grad():
                _ = model(input_ids, past_key_values=cache, use_cache=True)
            del cache, inputs, input_ids
            torch.cuda.empty_cache()
            print(f"  -> {target} tokens fits!")
            return target
        except torch.cuda.OutOfMemoryError:
            print(f"  -> {target} tokens OOM, trying smaller...")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            print(f"  -> {target} tokens error: {e}, trying smaller...")
            torch.cuda.empty_cache()
            gc.collect()
            continue

    print("  WARNING: All context lengths OOM. Using minimum 1024.")
    return 1024


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_niah_benchmark():
    """Run the full NIAH benchmark."""
    print("=" * 70)
    print("NIAH Benchmark for Tom: Mean-Removal vs Production WHT")
    print("=" * 70)

    model, tokenizer, num_layers = load_model(MODEL_NAME)

    # Find max context
    print("\nFinding maximum context length...")
    max_ctx = find_max_context(model, tokenizer, num_layers)
    print(f"Using context length: {max_ctx} tokens")

    # NIAH configs
    niah_configs = [
        ("FP16 baseline", None, None),
        ("WHT 3-bit", 3, False),
        ("WHT 3-bit + mean-removal", 3, True),
    ]

    all_results = []

    for needle_pos in NEEDLE_POSITIONS:
        print(f"\n  === Needle at {needle_pos:.0%} ===")
        for config_name, key_bits, center in niah_configs:
            result = run_niah_single(
                model, tokenizer, num_layers, max_ctx,
                needle_pos, config_name, key_bits, center,
            )
            all_results.append(result)

    # Free model
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # --- Report ---
    print("\n" + "=" * 70)
    print("NIAH RESULTS SUMMARY")
    print("=" * 70)

    report_lines = []
    report_lines.append("# NIAH Benchmark: Mean-Removal vs Production WHT")
    report_lines.append("")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"Model: {MODEL_NAME}")
    report_lines.append(f"Context: {max_ctx} tokens")
    report_lines.append(f"Needle: \"{NEEDLE}\"")
    report_lines.append(f"Expected answer: {NEEDLE_ANSWER}")
    report_lines.append(f"Compression: anchor=0, fp16_window=0, RQ=True, V3-bit")
    report_lines.append("")
    report_lines.append("| Position | Config | Pass/Fail | Generated (truncated) | Time |")
    report_lines.append("|----------|--------|-----------|-----------------------|------|")

    for r in all_results:
        pos_str = f"{r['needle_position']:.0%}"
        status = "PASS" if r["pass"] else "FAIL"
        gen = r.get("generated", r.get("error", "N/A"))[:60]
        t = r.get("time_sec", "N/A")
        report_lines.append(f"| {pos_str} | {r['config']} | {status} | {gen} | {t}s |")

    report_lines.append("")

    # Summary per config
    report_lines.append("## Summary by Config")
    report_lines.append("")
    for config_name, _, _ in niah_configs:
        config_results = [r for r in all_results if r["config"] == config_name]
        passes = sum(1 for r in config_results if r["pass"])
        total = len(config_results)
        report_lines.append(f"- {config_name}: {passes}/{total} positions recalled")

    report_lines.append("")

    # Analysis
    report_lines.append("## Analysis")
    report_lines.append("")
    fp16_passes = sum(1 for r in all_results if r["config"] == "FP16 baseline" and r["pass"])
    wht3_passes = sum(1 for r in all_results if r["config"] == "WHT 3-bit" and r["pass"])
    wht3m_passes = sum(1 for r in all_results if r["config"] == "WHT 3-bit + mean-removal" and r["pass"])

    report_lines.append(f"- FP16 baseline: {fp16_passes}/3 (reference)")
    report_lines.append(f"- WHT 3-bit: {wht3_passes}/3")
    report_lines.append(f"- WHT 3-bit + mean-removal: {wht3m_passes}/3")

    if wht3m_passes > wht3_passes:
        report_lines.append("- CONCLUSION: Mean-removal HELPS needle recall at 3-bit")
    elif wht3m_passes == wht3_passes:
        report_lines.append("- CONCLUSION: Mean-removal has NO EFFECT on needle recall at 3-bit")
    else:
        report_lines.append("- CONCLUSION: Mean-removal HURTS needle recall at 3-bit")

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save
    results_dir = REPO_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "niah_for_tom.md", "w") as f:
        f.write(report_text)
    with open(results_dir / "niah_for_tom.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_dir / 'niah_for_tom.md'}")
    return all_results


if __name__ == "__main__":
    run_niah_benchmark()
