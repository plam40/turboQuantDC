"""Sweep bit-widths to find the generation quality threshold (v2).

More rigorous quality assessment: tests factual accuracy, not just
non-repetition. Also tests at 3-bit and with longer generation.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from turboquantdc.hf_integration import TurboQuantCache

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Questions with verifiable answers
qa_pairs = [
    ("What is the capital of Australia? Answer in one word:", "Canberra"),
    ("What is 15 + 27? Answer with just the number:", "42"),
    ("Who wrote the novel 1984? Answer in two words:", "George Orwell"),
    ("What planet is closest to the Sun? Answer in one word:", "Mercury"),
    ("What is the chemical symbol for water? Answer in brief:", "H2O"),
]

def check_answer(response, expected):
    """Check if the expected answer appears in the response."""
    return expected.lower() in response.lower()

def run_baseline():
    """FP16 baseline (no compression)."""
    print("=" * 60)
    print("FP16 BASELINE (no compression)")
    print("=" * 60)
    correct = 0
    for prompt, expected in qa_pairs:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        ok = check_answer(response, expected)
        correct += ok
        print(f"  Q: {prompt}")
        print(f"  A: {response[:120]}")
        print(f"  Contains '{expected}': {ok}")
    print(f"\n  Factual accuracy: {correct}/{len(qa_pairs)}")
    print()
    return correct

def run_quantized(bits):
    """Run with MSE-only quantized KV cache at given bit-width."""
    print("=" * 60)
    print(f"MSE-ONLY {bits}-BIT ({2**bits} centroids)")
    print("=" * 60)
    correct = 0
    for prompt, expected in qa_pairs:
        cache = TurboQuantCache(bits=bits, seed=42, mse_only=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=50, past_key_values=cache, do_sample=False)
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        ok = check_answer(response, expected)
        correct += ok
        print(f"  Q: {prompt}")
        print(f"  A: {response[:120]}")
        print(f"  Contains '{expected}': {ok}")

    compression = 16.0 / bits
    print(f"\n  Factual accuracy: {correct}/{len(qa_pairs)}, ~{compression:.1f}x compression")
    print()
    return correct

def run_long_generation(bits):
    """Test longer generation (100 tokens) at a given bit-width."""
    prompt = "Explain step by step how photosynthesis works in plants:"
    print(f"--- Long generation test at {bits}-bit (100 tokens) ---")
    cache = TurboQuantCache(bits=bits, seed=42, mse_only=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=100, past_key_values=cache, do_sample=False)
    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # Quality metrics
    words = response.split()
    unique_words = set(w.lower() for w in words)
    unique_ratio = len(unique_words) / max(len(words), 1)

    # Check for degenerate repetition
    has_repetition = False
    for w in list(unique_words)[:20]:
        if words.count(w) > 8:
            has_repetition = True
            break

    print(f"  Prompt: {prompt}")
    print(f"  Response ({len(words)} words):")
    print(f"    {response[:300]}")
    print(f"  Unique word ratio: {unique_ratio:.2f}")
    print(f"  Degenerate repetition: {has_repetition}")
    print()

# Run everything
print("\n" + "#" * 60)
print("# BIT-WIDTH SWEEP: MSE-ONLY KV CACHE GENERATION QUALITY")
print("#" * 60 + "\n")

baseline = run_baseline()

results = {}
for bits in [3, 4, 5, 6, 8]:
    results[bits] = run_quantized(bits)

print("\n" + "#" * 60)
print("# LONG GENERATION TESTS")
print("#" * 60 + "\n")

for bits in [3, 4, 5, 6, 8]:
    run_long_generation(bits)

print("\n" + "#" * 60)
print("# SUMMARY")
print("#" * 60)
print(f"\n{'Bits':>6} | {'Centroids':>10} | {'Compression':>12} | {'Factual Acc':>12}")
print("-" * 50)
print(f"{'FP16':>6} | {'N/A':>10} | {'1.0x':>12} | {baseline}/{len(qa_pairs)}")
for bits in [3, 4, 5, 6, 8]:
    compression = 16.0 / bits
    print(f"{bits:>6} | {2**bits:>10} | {compression:>11.1f}x | {results[bits]}/{len(qa_pairs)}")
