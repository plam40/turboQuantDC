"""Sweep bit-widths to find the generation quality threshold.

Tests MSE-only KV cache at bit-widths 4 through 8 against an FP16 baseline
to determine the exact threshold where generation becomes coherent.
"""
import os
import sys

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from turboquantdc.hf_integration import TurboQuantCache

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

prompts = [
    "What is the capital of Australia? Answer briefly:",
    "What is 15 + 27?",
    "Who wrote the novel 1984? Answer briefly:",
    "Explain what a neural network is in two sentences:",
    "Write a Python function that returns the factorial of n:",
]

# Also test FP16 baseline (no compression)
print("FP16 BASELINE (no compression)")
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Q: {prompt}")
    print(f"  A: {response[:150]}")
print()

# Sweep bit-widths
for bits in [4, 5, 6, 8]:
    print(f"{'='*60}")
    print(f"MSE-ONLY {bits}-BIT ({2**bits} centroids)")
    print(f"{'='*60}")

    correct = 0
    total = len(prompts)

    for prompt in prompts:
        cache = TurboQuantCache(bits=bits, seed=42, mse_only=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=50, past_key_values=cache, do_sample=False)
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Simple coherence check
        is_coherent = len(response.split()) > 3 and not any(
            response.count(w) > 5 for w in response.split()[:3]
        )
        if is_coherent:
            correct += 1

        print(f"  Q: {prompt}")
        print(f"  A: {response[:150]}")
        print(f"  Coherent: {is_coherent}")

    compression = 16.0 / bits  # Approximate
    print(f"\n  Score: {correct}/{total} coherent, ~{compression:.1f}x compression")
    print()
