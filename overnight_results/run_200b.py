#!/usr/bin/env python3
"""Run 200B+ models on RTX 4090 using layer offloading + TurboQuantDC KV compression.

Key insight: for models that don't fit in GPU VRAM, accelerate's device_map="auto"
with max_memory handles CPU<->GPU layer offloading automatically. TurboQuantDC
compresses the KV cache so even the small on-GPU memory budget isn't eaten by KV.

Usage:
    # 72B model (default, ~40GB in 4-bit, mostly CPU)
    python run_200b.py

    # Explicit model
    python run_200b.py --model Qwen/Qwen2.5-72B-Instruct

    # Smaller test model
    python run_200b.py --model Qwen/Qwen2.5-32B-Instruct --gpu-budget 8.0

    # Use boundary cache instead of eviction
    python run_200b.py --kv-strategy boundary

Model size matrix (no-gating, public):
    Qwen/Qwen2.5-72B-Instruct            72B  ~40GB 4-bit  primary target
    Qwen/Qwen2.5-32B-Instruct            32B  ~18GB 4-bit  validation fallback
    meta-llama/Llama-3.3-70B-Instruct    70B  ~39GB 4-bit  needs HF token
    deepseek-ai/DeepSeek-V2-Lite         16B  ~9GB  4-bit  MoE path test
    Qwen/Qwen2.5-7B-Instruct             7B   ~4GB  4-bit  quick smoke test

For genuine 200B+ targets:
    Qwen/Qwen2.5-72B-Instruct is the largest ungated 72B Qwen model. True
    200B+ (e.g. Qwen2.5-Max) are gated or API-only. The script is ready to
    run them the moment access is available -- just pass --model <name>.
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

# Tune CUDA allocator before torch import to handle fragmentation from mixed
# CPU/GPU offloading.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, "/home/dhawal/turboQuantDC")

RESULTS_DIR = Path("/home/dhawal/turboQuantDC/overnight_results")

# Model size matrix: (hf_name, param_billions, notes)
MODEL_MATRIX = [
    ("Qwen/Qwen2.5-72B-Instruct",          72,  "primary -- largest no-gating Qwen"),
    ("meta-llama/Llama-3.3-70B-Instruct",  70,  "may need HF_TOKEN"),
    ("Qwen/Qwen2.5-32B-Instruct",          32,  "fallback / faster iteration"),
    ("deepseek-ai/DeepSeek-V2-Lite",       16,  "MoE architecture path"),
    ("Qwen/Qwen2.5-7B-Instruct",           7,   "quick smoke test"),
]

TEST_PROMPTS = [
    "What is the capital of France?",
    "Write a Python fibonacci function.",
    "Explain quantum computing in 2 sentences.",
]

GENERATE_KWARGS = dict(
    do_sample=False,
    repetition_penalty=1.15,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vram_gb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e9


def ram_gb() -> float:
    """Best-effort RSS estimate via /proc/self/status (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return float("nan")


def format_response(text: str, max_chars: int = 300) -> str:
    for sep in ("<|im_start|>assistant", "<|assistant|>", "Assistant:"):
        if sep in text:
            text = text.split(sep)[-1]
    for sep in ("<|im_end|>", "<|endoftext|>", "</s>"):
        if sep in text:
            text = text.split(sep)[0]
    return text.strip().replace("\n", " ")[:max_chars]


def make_cache(strategy: str, key_bits: int, val_bits: int, fp16_window: int,
               num_layers: int, seed: int):
    """Instantiate the requested KV cache type."""
    if strategy == "eviction":
        from turboquantdc.token_eviction import EvictionCache
        return EvictionCache(
            key_bits=key_bits,
            val_bits=val_bits,
            fp16_window=fp16_window,
            max_warm_tokens=1024,
            anchor_interval=12,
            use_residual_quant=True,
            seed=seed,
        )
    else:  # boundary
        from turboquantdc.generation_cache import GenerationCache
        return GenerationCache(
            key_bits=key_bits,
            val_bits=val_bits,
            anchor_strategy="boundary",
            fp16_window=fp16_window,
            use_residual_quant=True,
            num_layers=num_layers,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_one(model, tokenizer, prompt: str, cache=None,
                 max_new_tokens: int = 100) -> dict:
    """Run a single prompt and return metrics dict."""
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = f"User: {prompt}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt")
    # Only move input_ids to GPU; accelerate handles the rest
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(GENERATE_KWARGS)
    gen_kwargs["max_new_tokens"] = max_new_tokens
    if cache is not None:
        gen_kwargs["past_key_values"] = cache

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_tokens = out.shape[1] - input_len
    raw = tokenizer.decode(out[0], skip_special_tokens=False)
    response = format_response(raw)

    return {
        "prompt": prompt,
        "response": response,
        "new_tokens": new_tokens,
        "elapsed_s": elapsed,
        "tok_per_s": new_tokens / elapsed if elapsed > 0 else 0.0,
        "vram_gb": vram_gb(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run 200B+ models with TurboQuantDC KV cache compression"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-72B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-72B-Instruct)",
    )
    parser.add_argument(
        "--gpu-budget",
        type=float,
        default=6.0,
        help="GB of VRAM allocated for model weights (default: 6.0)",
    )
    parser.add_argument(
        "--cpu-budget",
        type=float,
        default=48.0,
        help="GB of RAM for CPU-offloaded layers (default: 48.0)",
    )
    parser.add_argument(
        "--kv-strategy",
        default="eviction",
        choices=["boundary", "eviction"],
        help="KV cache strategy: boundary (GenerationCache) or eviction (EvictionCache)",
    )
    parser.add_argument(
        "--key-bits", type=int, default=3, help="Bits for key quantization (default: 3)"
    )
    parser.add_argument(
        "--val-bits", type=int, default=3, help="Bits for value quantization (default: 3)"
    )
    parser.add_argument(
        "--fp16-window",
        type=int,
        default=64,
        help="Number of recent tokens kept at FP16 in hot tier (default: 64)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=100, help="Max tokens to generate per prompt"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        default=True,
        help="Save results to overnight_results/ (default: True)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # System info
    # ------------------------------------------------------------------
    print("=" * 70)
    print("TurboQuantDC — 200B Run Script")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"GPU budget:  {args.gpu_budget} GiB")
    print(f"CPU budget:  {args.cpu_budget} GiB")
    print(f"KV strategy: {args.kv_strategy} (K{args.key_bits}/V{args.val_bits}, fp16_window={args.fp16_window})")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)

    dev_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_vram = torch.cuda.mem_get_info()[0] / 1e9
    print(f"\nGPU:         {dev_name}")
    print(f"VRAM:        {free_vram:.1f} GB free / {total_vram:.1f} GB total")
    print(f"RAM:         {ram_gb():.1f} GB RSS (before load)")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n[1/3] Loading {args.model}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    max_memory = {0: f"{args.gpu_budget}GiB", "cpu": f"{args.cpu_budget}GiB"}

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model.eval()

    # Summarise device placement
    param_device_counts: dict[str, int] = {}
    for p in model.parameters():
        key = str(p.device)
        param_device_counts[key] = param_device_counts.get(key, 0) + 1

    num_layers = model.config.num_hidden_layers
    model_vram = vram_gb()
    model_ram = ram_gb()

    print(f"  num_hidden_layers = {num_layers}")
    print(f"  Device placement (param count): {dict(param_device_counts)}")
    print(f"  VRAM after load: {model_vram:.2f} GB")
    print(f"  RAM  after load: {model_ram:.2f} GB (RSS)")

    # ------------------------------------------------------------------
    # Create KV cache
    # ------------------------------------------------------------------
    print(f"\n[2/3] Creating {args.kv_strategy} KV cache...")
    cache = make_cache(
        strategy=args.kv_strategy,
        key_bits=args.key_bits,
        val_bits=args.val_bits,
        fp16_window=args.fp16_window,
        num_layers=num_layers,
        seed=args.seed,
    )
    print(f"  Cache: {type(cache).__name__}  K{args.key_bits}/V{args.val_bits}  fp16_window={args.fp16_window}")

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    print(f"\n[3/3] Generating {args.max_new_tokens} tokens per prompt...\n")
    results = []

    for i, prompt in enumerate(TEST_PROMPTS):
        # Reset cache state between prompts
        if hasattr(cache, "reset"):
            cache.reset()
        elif hasattr(cache, "_layers"):
            # Reconstruct for a clean slate
            cache = make_cache(
                strategy=args.kv_strategy,
                key_bits=args.key_bits,
                val_bits=args.val_bits,
                fp16_window=args.fp16_window,
                num_layers=num_layers,
                seed=args.seed,
            )

        torch.cuda.reset_peak_memory_stats()
        result = generate_one(model, tokenizer, prompt, cache=cache,
                               max_new_tokens=args.max_new_tokens)
        result["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        result["ram_gb"] = ram_gb()
        results.append(result)

        print(f"{'='*60}")
        print(f"Prompt {i+1}: {prompt}")
        print(f"Response:  {result['response'][:200]}")
        print(f"Speed:     {result['tok_per_s']:.1f} tok/s  ({result['new_tokens']} tokens in {result['elapsed_s']:.1f}s)")
        print(f"VRAM:      {result['vram_gb']:.2f} GB current / {result['peak_vram_gb']:.2f} GB peak")
        print(f"RAM:       {result['ram_gb']:.2f} GB RSS")

        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_tok = sum(r["new_tokens"] for r in results)
    total_s   = sum(r["elapsed_s"]  for r in results)
    avg_tps   = total_tok / total_s if total_s > 0 else 0.0
    peak_vram = max(r["peak_vram_gb"] for r in results)

    # Compression ratio: 16-bit fp16 vs K-bit keys + V-bit values
    bits_fp16 = 16
    bits_tq   = (args.key_bits + args.val_bits) / 2  # rough average
    compression = bits_fp16 / bits_tq

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Model:             {args.model}")
    print(f"KV cache:          {args.kv_strategy}  K{args.key_bits}/V{args.val_bits}")
    print(f"Avg speed:         {avg_tps:.1f} tok/s  ({total_tok} tokens / {total_s:.1f}s)")
    print(f"Peak VRAM:         {peak_vram:.2f} GB")
    print(f"Final RAM:         {ram_gb():.2f} GB RSS")
    print(f"KV compression:    ~{compression:.1f}x vs FP16 ({bits_fp16}-bit -> {bits_tq:.1f}-bit)")
    print(f"GPU budget used:   {args.gpu_budget} GiB / {total_vram:.1f} GiB total")
    print()
    print("Model size matrix (ungated, public):")
    print(f"  {'Model':<45} {'Params':>6}  Notes")
    print(f"  {'-'*45}  {'-'*6}  {'-'*40}")
    for name, params, notes in MODEL_MATRIX:
        marker = "<<< THIS RUN" if name == args.model else ""
        print(f"  {name:<45}  {params:>5}B  {notes}  {marker}")

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    if args.save_report:
        report_path = RESULTS_DIR / "run_200b_results.md"
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"## Run: {args.model}  [{ts}]",
            "",
            f"- **KV strategy:** {args.kv_strategy}  K{args.key_bits}/V{args.val_bits}  fp16_window={args.fp16_window}",
            f"- **GPU budget:** {args.gpu_budget} GiB  |  **CPU budget:** {args.cpu_budget} GiB",
            f"- **Avg speed:** {avg_tps:.1f} tok/s",
            f"- **Peak VRAM:** {peak_vram:.2f} GB",
            f"- **KV compression:** ~{compression:.1f}x",
            "",
            "| # | Prompt | Response | tok/s | Peak VRAM (GB) |",
            "|---|--------|----------|-------|----------------|",
        ]
        for i, r in enumerate(results):
            lines.append(
                f"| {i+1} | {r['prompt'][:50]} | {r['response'][:80]}... "
                f"| {r['tok_per_s']:.1f} | {r['peak_vram_gb']:.2f} |"
            )
        lines += ["", "---", ""]

        existing = report_path.read_text() if report_path.exists() else ""
        report_path.write_text(existing + "\n".join(lines) + "\n")
        print(f"\nReport saved to: {report_path}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    del model, tokenizer, cache
    gc.collect()
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
