#!/usr/bin/env python3
"""Phase 4: 70B+ via llama-cpp-python with GGUF models.

Bypasses the broken HF transformers + accelerate + bitsandbytes stack.
Uses llama.cpp's native CUDA offload for GPU/CPU split.

Strategy:
  - Qwen2.5-72B-Instruct at IQ2_M quantization (~29GB)
  - Partial GPU offload: ~60 of 80 layers on GPU, rest on CPU
  - llama.cpp handles the split natively — no accelerate needed
"""

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path("/home/dhawal/turboQuantDC/overnight_results")

PROMPTS = [
    "What is the capital of France? Answer briefly.",
    "Write a Python function to compute fibonacci numbers.",
    "Explain quantum computing in 3 sentences.",
    "What is 15 * 37? Show your work.",
    "Translate to Spanish: The weather is beautiful today.",
]


def get_gpu_info():
    """Get GPU name and VRAM info."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True
        )
        parts = result.stdout.strip().split(", ")
        return {
            "name": parts[0],
            "total_mb": int(parts[1].replace(" MiB", "")),
            "free_mb": int(parts[2].replace(" MiB", "")),
            "used_mb": int(parts[3].replace(" MiB", "")),
        }
    except Exception:
        return {"name": "Unknown", "total_mb": 0, "free_mb": 0, "used_mb": 0}


def get_gpu_vram_mb():
    """Current VRAM usage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def find_gguf_model():
    """Find the downloaded GGUF model file."""
    search_paths = [
        "/media/dhawal/Beast/cache/huggingface",
        os.path.expanduser("~/.cache/huggingface"),
        "/home/dhawal/.cache/huggingface",
    ]
    for base in search_paths:
        for root, dirs, files in os.walk(base):
            for f in files:
                if "qwen2.5-72b" in f.lower() and f.endswith(".gguf"):
                    return os.path.join(root, f)
    return None


def run_llama_cpp(model_path, n_gpu_layers=-1, n_ctx=4096):
    """Run the model via llama-cpp-python."""
    from llama_cpp import Llama

    gpu_before = get_gpu_vram_mb()
    print(f"\n{'='*70}")
    print(f"Loading model: {os.path.basename(model_path)}")
    print(f"  File size: {os.path.getsize(model_path) / 1e9:.1f} GB")
    print(f"  GPU layers: {n_gpu_layers}")
    print(f"  Context: {n_ctx}")
    print(f"  VRAM before: {gpu_before} MB")
    print(f"{'='*70}")

    t_load_start = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_threads=8,
        verbose=True,
    )
    t_load = time.perf_counter() - t_load_start
    gpu_after = get_gpu_vram_mb()

    print(f"\nModel loaded in {t_load:.1f}s")
    print(f"VRAM after load: {gpu_after} MB (+{gpu_after - gpu_before} MB)")

    results = []
    total_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {i+1}/{len(PROMPTS)} ---")
        print(f"Q: {prompt}")

        t0 = time.perf_counter()
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Be concise."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        elapsed = time.perf_counter() - t0

        text = output["choices"][0]["message"]["content"]
        usage = output.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        tok_per_s = completion_tokens / elapsed if elapsed > 0 else 0

        total_tokens += completion_tokens
        total_time += elapsed

        print(f"A: {text[:300]}")
        print(f"   [{completion_tokens} tokens in {elapsed:.2f}s = {tok_per_s:.1f} tok/s]")

        results.append({
            "prompt": prompt,
            "output": text[:500],
            "elapsed_s": round(elapsed, 2),
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "tok_per_s": round(tok_per_s, 1),
        })

    gpu_peak = get_gpu_vram_mb()
    avg_tps = total_tokens / total_time if total_time > 0 else 0

    # Needle-in-haystack test
    print(f"\n{'='*70}")
    print("Needle-in-Haystack Test (~2K tokens)")
    print(f"{'='*70}")

    filler = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200
    parts = filler.split(". ")
    parts.insert(len(parts) // 3, "The secret code is BANANA42")
    long_text = ". ".join(parts)
    needle_prompt = f"Read the following text carefully and find the secret code:\n\n{long_text}\n\nWhat is the secret code? Answer with just the code."

    t0 = time.perf_counter()
    needle_output = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": needle_prompt},
        ],
        max_tokens=50,
        temperature=0.0,
    )
    needle_elapsed = time.perf_counter() - t0
    needle_text = needle_output["choices"][0]["message"]["content"]
    needle_found = "BANANA42" in needle_text.upper()
    needle_input_tokens = needle_output.get("usage", {}).get("prompt_tokens", 0)

    print(f"Input tokens: {needle_input_tokens}")
    print(f"Output: {needle_text}")
    print(f"Found BANANA42: {'YES' if needle_found else 'NO'}")
    print(f"Time: {needle_elapsed:.1f}s")

    del llm
    gc.collect()

    return {
        "model_path": model_path,
        "model_size_gb": round(os.path.getsize(model_path) / 1e9, 1),
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
        "load_time_s": round(t_load, 1),
        "vram_model_mb": gpu_after - gpu_before,
        "vram_peak_mb": gpu_peak,
        "avg_tok_per_s": round(avg_tps, 1),
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 1),
        "results": results,
        "needle": {
            "input_tokens": needle_input_tokens,
            "output": needle_text[:200],
            "found": needle_found,
            "elapsed_s": round(needle_elapsed, 1),
        },
    }


def run_ollama(model_name="qwen2.5:72b-instruct-q2_K"):
    """Run via Ollama as fallback."""
    import requests

    print(f"\n{'='*70}")
    print(f"Running via Ollama: {model_name}")
    print(f"{'='*70}")

    gpu_before = get_gpu_vram_mb()

    # Check if model is available
    try:
        resp = requests.get("http://localhost:11434/api/tags")
        models = [m["name"] for m in resp.json().get("models", [])]
        if model_name not in models:
            # Try without the tag suffix
            matching = [m for m in models if "72b" in m.lower()]
            if matching:
                model_name = matching[0]
                print(f"  Using available model: {model_name}")
            else:
                print(f"  Model {model_name} not found. Available: {models}")
                return None
    except Exception as e:
        print(f"  Ollama not running or not accessible: {e}")
        return None

    results = []
    total_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {i+1}/{len(PROMPTS)} ---")
        print(f"Q: {prompt}")

        t0 = time.perf_counter()
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Be concise."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 150},
            },
            timeout=300,
        )
        elapsed = time.perf_counter() - t0

        data = resp.json()
        text = data.get("message", {}).get("content", "ERROR")
        eval_count = data.get("eval_count", 0)
        tok_per_s = eval_count / elapsed if elapsed > 0 else 0

        total_tokens += eval_count
        total_time += elapsed

        print(f"A: {text[:300]}")
        print(f"   [{eval_count} tokens in {elapsed:.2f}s = {tok_per_s:.1f} tok/s]")

        results.append({
            "prompt": prompt,
            "output": text[:500],
            "elapsed_s": round(elapsed, 2),
            "completion_tokens": eval_count,
            "tok_per_s": round(tok_per_s, 1),
        })

    gpu_peak = get_gpu_vram_mb()
    avg_tps = total_tokens / total_time if total_time > 0 else 0

    # Needle test
    filler = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200
    parts = filler.split(". ")
    parts.insert(len(parts) // 3, "The secret code is BANANA42")
    long_text = ". ".join(parts)
    needle_prompt = f"Read the following text carefully and find the secret code:\n\n{long_text}\n\nWhat is the secret code? Answer with just the code."

    t0 = time.perf_counter()
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": needle_prompt}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 50},
        },
        timeout=300,
    )
    needle_elapsed = time.perf_counter() - t0
    needle_data = resp.json()
    needle_text = needle_data.get("message", {}).get("content", "ERROR")
    needle_found = "BANANA42" in needle_text.upper()

    print(f"\nNeedle test: {needle_text}")
    print(f"Found: {'YES' if needle_found else 'NO'}")

    return {
        "model_name": model_name,
        "method": "ollama",
        "vram_peak_mb": gpu_peak,
        "avg_tok_per_s": round(avg_tps, 1),
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 1),
        "results": results,
        "needle": {
            "output": needle_text[:200],
            "found": needle_found,
            "elapsed_s": round(needle_elapsed, 1),
        },
    }


def generate_report(data, method_label):
    """Generate markdown report."""
    gpu = get_gpu_info()
    import psutil
    ram = psutil.virtual_memory()

    lines = []
    lines.append(f"# Phase 4: 70B+ Model via {method_label}")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**GPU:** {gpu['name']}")
    lines.append(f"**VRAM:** {gpu['total_mb']} MB total")
    lines.append(f"**RAM:** {ram.total / 1024**3:.1f} GB total ({ram.available / 1024**3:.1f} GB available)")
    lines.append(f"**Method:** {method_label}")
    lines.append("")

    lines.append("## Configuration")
    if "model_path" in data:
        lines.append(f"- **Model:** {os.path.basename(data['model_path'])}")
        lines.append(f"- **Format:** GGUF (IQ2_M quantization)")
        lines.append(f"- **File size:** {data['model_size_gb']} GB")
        lines.append(f"- **GPU layers:** {data['n_gpu_layers']}")
        lines.append(f"- **Context window:** {data['n_ctx']}")
        lines.append(f"- **Load time:** {data['load_time_s']}s")
    else:
        lines.append(f"- **Model:** {data.get('model_name', 'unknown')}")
        lines.append(f"- **Format:** Ollama-managed GGUF")
    lines.append(f"- **Peak VRAM:** {data.get('vram_peak_mb', 'N/A')} MB")
    lines.append("")

    lines.append("## Generation Results")
    lines.append(f"- **Average speed:** {data['avg_tok_per_s']} tok/s")
    lines.append(f"- **Total tokens:** {data['total_tokens']}")
    lines.append(f"- **Total time:** {data['total_time_s']}s")
    lines.append("")

    lines.append("| # | Prompt | Output | tok/s |")
    lines.append("|---|--------|--------|-------|")
    for i, r in enumerate(data["results"]):
        prompt_short = r["prompt"][:45] + "..." if len(r["prompt"]) > 45 else r["prompt"]
        output_short = r["output"][:120].replace("\n", " ") + "..."
        lines.append(f"| {i+1} | {prompt_short} | {output_short} | {r['tok_per_s']} |")
    lines.append("")

    lines.append("## Needle-in-Haystack")
    needle = data["needle"]
    lines.append(f"- **Input tokens:** {needle.get('input_tokens', 'N/A')}")
    lines.append(f"- **Output:** {needle['output'][:150]}")
    lines.append(f"- **Found BANANA42:** {'YES' if needle['found'] else 'NO'}")
    lines.append(f"- **Time:** {needle['elapsed_s']}s")
    lines.append("")

    # Verdict
    quality_ok = all(len(r["output"]) > 10 for r in data["results"])
    verdict = "PASS" if quality_ok and needle["found"] else "PARTIAL" if quality_ok else "FAIL"

    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    if verdict == "PASS":
        lines.append("Qwen2.5-72B-Instruct is GENERATING TEXT on RTX 4090 via llama.cpp.")
        lines.append(f"Average generation speed: {data['avg_tok_per_s']} tok/s.")
        lines.append("Needle-in-haystack: PASSED.")
        lines.append("")
        lines.append("### What This Means for TurboQuantDC")
        lines.append("- 70B+ model running on single RTX 4090 -- the VRAM constraint is real")
        lines.append("- llama.cpp manages its own KV cache internally")
        lines.append("- TurboQuantDC integration path: intercept KV cache at the llama.cpp layer")
        lines.append("  or use HF loading with fixed CPU offload for full TurboQuantDC control")
        lines.append("- The algorithm is validated at 3B and 14B; 70B is an infrastructure proof")
    elif verdict == "PARTIAL":
        lines.append("Model generates text but needle-in-haystack failed.")
    else:
        lines.append("Generation quality issues detected.")
    lines.append("")

    return "\n".join(lines)


def main():
    print("Phase 4: 70B+ Model via llama-cpp-python")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M')}")

    gpu = get_gpu_info()
    print(f"GPU: {gpu['name']} ({gpu['total_mb']} MB total, {gpu['free_mb']} MB free)")

    data = None
    method = None

    # Strategy 1: llama-cpp-python with downloaded GGUF
    model_path = find_gguf_model()
    if model_path:
        print(f"\nFound GGUF model: {model_path}")
        try:
            # 80 layers total in Qwen2.5-72B
            # IQ2_M is ~29GB; GPU has ~22GB free
            # Try putting 55-60 layers on GPU (~20GB), rest on CPU
            data = run_llama_cpp(model_path, n_gpu_layers=60, n_ctx=4096)
            method = "llama-cpp-python (CUDA)"
        except Exception as e:
            print(f"llama-cpp-python failed: {e}")
            import traceback
            traceback.print_exc()

    # Strategy 2: Ollama fallback
    if data is None:
        print("\nTrying Ollama fallback...")
        try:
            data = run_ollama()
            method = "Ollama"
        except Exception as e:
            print(f"Ollama failed: {e}")
            import traceback
            traceback.print_exc()

    if data is None:
        print("\nBoth methods failed. Writing failure report.")
        report = f"""# Phase 4: 70B+ Model -- FAILED

**Date:** {time.strftime('%Y-%m-%d %H:%M')}

## Result: Could not load 70B+ model

Neither llama-cpp-python nor Ollama succeeded.
Check logs above for details.
"""
    else:
        report = generate_report(data, method)

    # Save report
    report_path = RESULTS_DIR / "phase4_70b_llama_cpp.md"
    report_path.write_text(report)
    print(f"\n{'='*70}")
    print(f"Report saved to: {report_path}")
    print(f"{'='*70}")
    print(report)


if __name__ == "__main__":
    main()
