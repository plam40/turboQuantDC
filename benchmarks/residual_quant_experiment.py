"""ResidualQuant vs TurboQuant generation quality experiment.

Hypothesis: Directly quantizing the residual (sign of actual residual, no
random projection) produces better generation quality than QJL at the same
bit budget, because it trades unbiasedness for lower variance.

Compares 5 configurations on the same prompts:
1. FP16 baseline (no compression)
2. TQ-4 MSE-only (current best at 4-bit, no QJL)
3. TQ-3 MSE-only (garbled at 3-bit)
4. ResidualQuant 3-bit (MSE + direct residual signs)
5. ResidualQuant 4-bit (MSE + direct residual signs)
"""

import gc
import os
import sys
import time

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from turboquantdc.hf_integration import TurboQuantCache
from turboquantdc.residual_quant import ResidualQuantCache

# ---- Configuration ----
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 60
DO_SAMPLE = False

PROMPTS = [
    "What is the capital of Australia? Answer briefly:",
    "What is 15 + 27?",
    "Who wrote the novel 1984? Answer briefly:",
    "Explain what a neural network is in two sentences:",
    "Write a Python function that returns the factorial of n:",
]

EXPECTED_KEYWORDS = [
    ["canberra"],
    ["42"],
    ["george", "orwell"],
    ["layer", "neuron", "learn", "network", "input", "output", "weight"],
    ["def", "factorial", "return"],
]


def load_model():
    """Load model once for all experiments."""
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


def generate_with_cache(model, tokenizer, prompt, cache=None, max_new_tokens=MAX_NEW_TOKENS):
    """Generate text with optional KV cache."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
        )
        if cache is not None:
            kwargs["past_key_values"] = cache
        out = model.generate(**kwargs)

    response = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response


def check_coherence(response, keywords):
    """Check if response is coherent and contains expected keywords."""
    response_lower = response.lower()
    words = response.split()

    # Check 1: Not too short
    if len(words) < 2:
        return False, "too_short"

    # Check 2: Not repetitive (same 3-gram repeated > 3 times)
    if len(words) >= 6:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        for tg in set(trigrams):
            if trigrams.count(tg) > 3:
                return False, "repetitive"

    # Check 3: Contains at least one expected keyword
    has_keyword = any(kw in response_lower for kw in keywords)

    return has_keyword, "correct" if has_keyword else "wrong_content"


def run_experiment(model, tokenizer, name, cache_factory):
    """Run one experiment configuration across all prompts."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    correct = 0
    total = len(PROMPTS)
    results = []

    for i, (prompt, keywords) in enumerate(zip(PROMPTS, EXPECTED_KEYWORDS)):
        cache = cache_factory() if cache_factory is not None else None

        t0 = time.time()
        response = generate_with_cache(model, tokenizer, prompt, cache=cache)
        elapsed = time.time() - t0

        is_correct, status = check_coherence(response, keywords)
        if is_correct:
            correct += 1

        results.append({
            "prompt": prompt,
            "response": response[:200],
            "correct": is_correct,
            "status": status,
            "time": elapsed,
        })

        status_icon = "PASS" if is_correct else "FAIL"
        print(f"\n  [{status_icon}] Q: {prompt}")
        print(f"       A: {response[:200]}")
        print(f"       Status: {status}, Time: {elapsed:.2f}s")

        # Free cache memory
        del cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n  SCORE: {correct}/{total} correct")
    return correct, total, results


def main():
    model, tokenizer = load_model()

    print("\n" + "=" * 70)
    print("  RESIDUALQUANT EXPERIMENT: Direct residual vs QJL")
    print("  Hypothesis: sign(residual) beats sign(S @ residual)")
    print("=" * 70)

    all_results = {}

    # 1. FP16 baseline
    score, total, results = run_experiment(
        model, tokenizer,
        "FP16 BASELINE (no compression)",
        cache_factory=None,
    )
    all_results["fp16"] = (score, total)

    # 2. TQ-4 MSE-only (known good)
    score, total, results = run_experiment(
        model, tokenizer,
        "TQ-4 MSE-ONLY (4-bit, 8 centroids, no QJL)",
        cache_factory=lambda: TurboQuantCache(bits=4, seed=42, mse_only=True),
    )
    all_results["tq4_mse"] = (score, total)

    # 3. TQ-3 MSE-only (known garbled)
    score, total, results = run_experiment(
        model, tokenizer,
        "TQ-3 MSE-ONLY (3-bit, 8 centroids, no QJL) [expect garbled]",
        cache_factory=lambda: TurboQuantCache(bits=3, seed=42, mse_only=True),
    )
    all_results["tq3_mse"] = (score, total)

    # 4. ResidualQuant 3-bit (THE EXPERIMENT)
    score, total, results = run_experiment(
        model, tokenizer,
        "RESIDUALQUANT 3-BIT (2-bit MSE + 1-bit residual signs)",
        cache_factory=lambda: ResidualQuantCache(bits=3, seed=42),
    )
    all_results["rq3"] = (score, total)

    # 5. ResidualQuant 4-bit
    score, total, results = run_experiment(
        model, tokenizer,
        "RESIDUALQUANT 4-BIT (3-bit MSE + 1-bit residual signs)",
        cache_factory=lambda: ResidualQuantCache(bits=4, seed=42),
    )
    all_results["rq4"] = (score, total)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<45} {'Score':>10}")
    print(f"  {'-'*45} {'-'*10}")
    for name, (score, total) in all_results.items():
        labels = {
            "fp16": "FP16 Baseline",
            "tq4_mse": "TQ-4 MSE-only (4-bit)",
            "tq3_mse": "TQ-3 MSE-only (3-bit)",
            "rq3": "ResidualQuant 3-bit  <-- EXPERIMENT",
            "rq4": "ResidualQuant 4-bit",
        }
        print(f"  {labels[name]:<45} {score}/{total}")

    print()

    # ---- Verdict ----
    rq3_score = all_results["rq3"][0]
    tq3_score = all_results["tq3_mse"][0]
    tq4_score = all_results["tq4_mse"][0]
    fp16_score = all_results["fp16"][0]

    print("  VERDICT:")
    if rq3_score > tq3_score:
        print(f"  ResidualQuant 3-bit ({rq3_score}/5) beats TQ-3 MSE-only ({tq3_score}/5)")
        if rq3_score >= tq4_score:
            print(f"  ResidualQuant 3-bit MATCHES or BEATS TQ-4 MSE-only ({tq4_score}/5)!")
            print("  --> HYPOTHESIS CONFIRMED: Direct residual quantization at 3-bit")
            print("      achieves 4-bit quality. 5x compression with coherent generation.")
        else:
            print(f"  But still below TQ-4 MSE-only ({tq4_score}/5)")
            print("  --> PARTIAL SUCCESS: Better than 3-bit MSE but not 4-bit quality yet")
    elif rq3_score == tq3_score:
        print(f"  ResidualQuant 3-bit ({rq3_score}/5) ties with TQ-3 MSE-only ({tq3_score}/5)")
        print("  --> HYPOTHESIS NOT CONFIRMED at this sample size")
    else:
        print(f"  ResidualQuant 3-bit ({rq3_score}/5) loses to TQ-3 MSE-only ({tq3_score}/5)")
        print("  --> HYPOTHESIS REJECTED")


if __name__ == "__main__":
    main()
