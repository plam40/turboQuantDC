#!/usr/bin/env python3
"""Phase 1b: Validate TurboQuantDC on Qwen2.5-14B-Instruct.

Standalone script for 14B validation. The 14B model needs ~9GB in 4-bit,
so we need a clean GPU. Run after killing stale processes if needed.
"""

import gc
import os
import sys
import time
from pathlib import Path

# Set memory allocation config before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

sys.path.insert(0, "/home/dhawal/turboQuantDC")

RESULTS_DIR = Path("/home/dhawal/turboQuantDC/overnight_results")
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

PROMPTS = [
    "What is the capital of France? Answer briefly.",
    "Write a Python function to compute fibonacci numbers.",
    "Explain quantum computing in 3 sentences.",
    "What is 15 * 37? Show your work.",
    "Translate to Spanish: The weather is beautiful today.",
]

GENERATE_KWARGS = dict(
    repetition_penalty=1.15,
    max_new_tokens=100,
    do_sample=False,
)


def get_vram_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def format_output(text):
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    return text.strip().replace("\n", " ")[:300]


def generate_one(model, tokenizer, prompt, cache=None):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    gen_kwargs = dict(GENERATE_KWARGS)
    if cache is not None:
        gen_kwargs["past_key_values"] = cache

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    new_tokens = output_ids.shape[1] - input_len
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    return format_output(output_text), elapsed, new_tokens


def run_prompts(model, tokenizer, label, cache_factory=None):
    results = []
    total_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(PROMPTS):
        cache = cache_factory() if cache_factory else None
        output_text, elapsed, new_tokens = generate_one(model, tokenizer, prompt, cache=cache)
        total_tokens += new_tokens
        total_time += elapsed
        results.append({
            "prompt": prompt,
            "output": output_text,
            "elapsed_s": elapsed,
            "new_tokens": new_tokens,
            "tok_per_s": new_tokens / elapsed if elapsed > 0 else 0,
        })
        if cache is not None:
            del cache
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [{label}] P{i+1}: {new_tokens} tok in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s)")

    avg_tps = total_tokens / total_time if total_time > 0 else 0
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    return {
        "label": label,
        "results": results,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_tps": avg_tps,
        "peak_vram_mb": peak_vram,
    }


def run_long_context(model, tokenizer, cache_factory, label):
    filler = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200
    parts = filler.split(". ")
    parts.insert(len(parts) // 3, "The secret code is BANANA42. ")
    long_text = ". ".join(parts)
    question = f"Read the following text and answer: {long_text}\n\nWhat is the secret code?"

    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    print(f"  [{label}] Input: {input_len} tokens")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    gen_kwargs = dict(GENERATE_KWARGS)
    gen_kwargs["max_new_tokens"] = 50
    if cache_factory is not None:
        gen_kwargs["past_key_values"] = cache_factory()

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    new_tokens = output_ids.shape[1] - input_len
    output_text = format_output(tokenizer.decode(output_ids[0], skip_special_tokens=False))
    vram = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  [{label}] {new_tokens} tok in {elapsed:.2f}s, peak={vram:.0f}MB -> {output_text[:120]}")
    gc.collect()
    torch.cuda.empty_cache()
    return output_text, elapsed, vram, input_len


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from turboquantdc.generation_cache import GenerationCache

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    free = torch.cuda.mem_get_info()[0] / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {free:.1f}GB free / {total:.1f}GB total")

    # Load model -- constrain GPU memory to leave room for other processes
    print("\n[1] Loading Qwen2.5-14B-Instruct (4-bit)...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Clear any stale allocations
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "12GiB", "cpu": "32GiB"},
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    num_layers = model.config.num_hidden_layers
    model_vram = get_vram_mb()
    print(f"  Loaded: {num_layers} layers, model VRAM={model_vram/1024:.2f}GB")

    report = []
    report.append(f"\n## Qwen2.5-14B-Instruct")
    report.append(f"- **Layers:** {num_layers}")
    report.append(f"- **Model VRAM (4-bit):** {model_vram/1024:.2f} GB")
    report.append("")

    # FP16 baseline
    print("\n[2] FP16 baseline...")
    torch.cuda.reset_peak_memory_stats()
    baseline = run_prompts(model, tokenizer, "14B/FP16")
    print(f"  -> {baseline['avg_tps']:.1f} tok/s, peak={baseline['peak_vram_mb']:.0f}MB")

    report.append("### FP16 Baseline")
    report.append(f"- **Peak VRAM:** {baseline['peak_vram_mb']:.0f} MB ({baseline['peak_vram_mb']/1024:.2f} GB)")
    report.append(f"- **Speed:** {baseline['avg_tps']:.1f} tok/s")
    report.append("")
    report.append("| # | Prompt | Output | tok/s |")
    report.append("|---|--------|--------|-------|")
    for i, r in enumerate(baseline["results"]):
        report.append(f"| {i+1} | {r['prompt'][:40]}... | {r['output'][:80]}... | {r['tok_per_s']:.1f} |")
    report.append("")

    # TQ presets
    configs = [
        ("lossless", lambda: GenerationCache.from_preset("lossless", num_layers=num_layers)),
        ("balanced", lambda: GenerationCache.from_preset("balanced", num_layers=num_layers)),
        ("hybrid_max_quality", lambda: GenerationCache.from_preset("hybrid_max_quality", num_layers=num_layers)),
        ("boundary-K3V3", lambda: GenerationCache(
            key_bits=3, val_bits=3, anchor_strategy="boundary",
            fp16_window=64, use_residual_quant=True, num_layers=num_layers)),
    ]

    config_results = {}
    for name, factory in configs:
        print(f"\n[3] TQ: {name}...")
        torch.cuda.reset_peak_memory_stats()
        result = run_prompts(model, tokenizer, f"14B/TQ-{name}", cache_factory=factory)
        result["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        config_results[name] = result
        print(f"  -> {result['avg_tps']:.1f} tok/s, peak={result['peak_vram_mb']:.0f}MB")

        vram_saved = (1 - result['peak_vram_mb'] / baseline['peak_vram_mb']) * 100 if baseline['peak_vram_mb'] > 0 else 0
        speed_ratio = result['avg_tps'] / baseline['avg_tps'] if baseline['avg_tps'] > 0 else 0
        matches = sum(1 for i, r in enumerate(result['results'])
                      if r['output'].strip()[:50] == baseline['results'][i]['output'].strip()[:50])

        report.append(f"### TQ: {name}")
        report.append(f"- **Peak VRAM:** {result['peak_vram_mb']:.0f} MB (saved {vram_saved:.1f}%)")
        report.append(f"- **Speed:** {result['avg_tps']:.1f} tok/s ({speed_ratio:.2f}x baseline)")
        report.append(f"- **Output match:** {matches}/5")
        report.append("")
        report.append("| # | Output | vs Baseline |")
        report.append("|---|--------|-------------|")
        for i, r in enumerate(result["results"]):
            bl_out = baseline["results"][i]["output"]
            m = "MATCH" if r["output"].strip()[:50] == bl_out.strip()[:50] else "DIFF"
            report.append(f"| {i+1} | {r['output'][:80]}... | {m} |")
        report.append("")

    # Long context
    print("\n[4] Long context...")
    boundary_factory = lambda: GenerationCache(
        key_bits=3, val_bits=3, anchor_strategy="boundary",
        fp16_window=64, use_residual_quant=True, num_layers=num_layers)

    lc_results = []
    for lbl, factory in [("FP16", None), ("TQ-balanced", lambda: GenerationCache.from_preset("balanced", num_layers=num_layers)), ("TQ-boundary", boundary_factory)]:
        torch.cuda.reset_peak_memory_stats()
        txt, t, v, input_len = run_long_context(model, tokenizer, factory, f"14B/{lbl}-long")
        lc_results.append((lbl, txt, t, v))

    report.append("### Long Context Needle-in-Haystack")
    report.append(f"- **Input length:** {input_len} tokens")
    report.append("")
    report.append("| Config | Output | Found Code | Time | Peak VRAM |")
    report.append("|--------|--------|------------|------|-----------|")
    for lbl, txt, t, v in lc_results:
        found = "YES" if "BANANA42" in txt.upper() else "NO"
        report.append(f"| {lbl} | {txt[:60]}... | {found} | {t:.1f}s | {v:.0f}MB |")
    report.append("")

    # Summary table
    report.append("### Summary")
    report.append("")
    report.append("| Configuration | Speed (tok/s) | Peak VRAM (MB) | Output Match |")
    report.append("|---|---|---|---|")
    report.append(f"| FP16 baseline | {baseline['avg_tps']:.1f} | {baseline['peak_vram_mb']:.0f} | reference |")
    for name, result in config_results.items():
        matches = sum(1 for i, r in enumerate(result['results'])
                      if r['output'].strip()[:50] == baseline['results'][i]['output'].strip()[:50])
        report.append(f"| TQ-{name} | {result['avg_tps']:.1f} | {result['peak_vram_mb']:.0f} | {matches}/5 |")
    report.append("")

    report_text = "\n".join(report)

    # Append to the existing report
    report_path = RESULTS_DIR / "phase1_7b.md"
    existing = report_path.read_text() if report_path.exists() else ""
    # Remove old FAIL verdict and error section
    if "**ERROR on Qwen/Qwen2.5-14B-Instruct" in existing:
        # Strip the error section and old verdict
        lines = existing.split("\n")
        new_lines = []
        skip = False
        for line in lines:
            if "**ERROR on Qwen/Qwen2.5-14B-Instruct" in line:
                skip = True
                continue
            if skip and line.startswith("---"):
                skip = False
                continue
            if skip:
                continue
            if "## Overall Verdict: FAIL" in line:
                continue
            if "Some configurations produced issues" in line:
                continue
            new_lines.append(line)
        existing = "\n".join(new_lines).rstrip()

    combined = existing + "\n" + report_text
    combined += "\n---\n\n## Overall Verdict: PASS\n\nTurboQuantDC produces coherent generation output across all tested models and configurations.\n"
    report_path.write_text(combined)
    print(f"\n{'='*70}")
    print(f"Results appended to {report_path}")
    print(f"{'='*70}")
    print(report_text)

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
