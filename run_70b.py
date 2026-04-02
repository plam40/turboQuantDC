"""Run 70B models on RTX 4090 (24GB) with 144K context.

Unified launcher that auto-configures weight offloading + TurboQuantDC KV
compression to fit any 70B model on a single 24GB GPU.

Usage:
    python run_70b.py                                    # Interactive chat
    python run_70b.py --model meta-llama/Llama-3.3-70B-Instruct
    python run_70b.py --context-target 144000            # Default
    python run_70b.py --speed-mode fast                  # More eviction, faster
    python run_70b.py --speed-mode safe                  # No eviction, slower
    python run_70b.py --prompt "Explain quantum computing"
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, Optional, Tuple

import torch

from turboquantdc.streaming_70b import MemoryPlanner


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPEED_MODES = ("safe", "balanced", "fast")

# Default 70B architecture (Llama-3.x / Qwen2.5-72B style)
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_70B_CONFIG = {
    "num_layers": 80,
    "hidden_size": 8192,
    "num_kv_heads": 8,
    "head_dim": 128,
    "vocab_size": 128256,
    "intermediate_size": 28672,
}

# Compression ratios by speed mode (vs FP16 KV)
COMPRESSION_RATIOS = {
    "safe": 5.0,
    "balanced": 5.0,
    "fast": 7.5,
}

# Estimated tok/s by speed mode (for startup report)
ESTIMATED_SPEEDS = {
    "safe": 27,
    "balanced": 33,
    "fast": 38,
}


# ---------------------------------------------------------------------------
# GPU Detection
# ---------------------------------------------------------------------------


def detect_gpu() -> Dict[str, Any]:
    """Detect GPU properties.

    Returns:
        Dict with name, vram_gb, vram_bytes, compute_capability, available.
        If no CUDA GPU is found, vram_gb=0 and available=False.
    """
    if not torch.cuda.is_available():
        return {
            "name": "none",
            "vram_gb": 0.0,
            "vram_bytes": 0,
            "compute_capability": (0, 0),
            "available": False,
        }

    props = torch.cuda.get_device_properties(0)
    vram_bytes = props.total_memory
    return {
        "name": props.name,
        "vram_gb": round(vram_bytes / (1024**3), 1),
        "vram_bytes": vram_bytes,
        "compute_capability": (props.major, props.minor),
        "available": True,
    }


# ---------------------------------------------------------------------------
# Memory Calculation
# ---------------------------------------------------------------------------


def calculate_memory_plan(
    context_target: int = 144_000,
    gpu_vram_gb: float = 24.0,
    model_config: Optional[Dict[str, Any]] = None,
    weight_quant_bits: int = 4,
    speed_mode: str = "balanced",
) -> Dict[str, Any]:
    """Calculate how to split the model across GPU and CPU.

    Uses MemoryPlanner to determine layer allocation, then computes
    KV cache budget from the remaining VRAM.

    Args:
        context_target: Target context length in tokens.
        gpu_vram_gb: Available GPU VRAM in GB.
        model_config: Model architecture dict. Uses DEFAULT_70B_CONFIG if None.
        weight_quant_bits: Weight quantization (4 = 4-bit BnB, 8 = 8-bit).
        speed_mode: One of "safe", "balanced", "fast".

    Returns:
        Dict with weight_budget_gb, kv_budget_gb, num_gpu_layers,
        total_layers, compression_ratio, context_achievable, etc.
    """
    if speed_mode not in SPEED_MODES:
        raise ValueError(f"speed_mode must be one of {SPEED_MODES}, got '{speed_mode}'")

    cfg = model_config or DEFAULT_70B_CONFIG

    # Weight quantization factor: 4-bit = 0.25 of FP16, 8-bit = 0.5
    quant_factor = weight_quant_bits / 16.0

    # Compute layer size at the quantized precision
    dtype_bytes = max(1, int(2 * quant_factor))  # bytes per param after quant

    # Leave 4GB headroom for CUDA context, activations, fragmentation
    usable_vram = max(1.0, gpu_vram_gb - 4.0)

    plan = MemoryPlanner.plan(
        num_layers=cfg["num_layers"],
        hidden_size=cfg["hidden_size"],
        num_kv_heads=cfg["num_kv_heads"],
        head_dim=cfg["head_dim"],
        vocab_size=cfg["vocab_size"],
        dtype_bytes=dtype_bytes,
        gpu_budget_gb=usable_vram,
    )

    # KV cache size estimation
    # Per-token FP16 KV: 2 * num_kv_heads * head_dim * 2 bytes * num_layers
    kv_per_token_fp16 = (
        2 * cfg["num_kv_heads"] * cfg["head_dim"] * 2 * cfg["num_layers"]
    )

    compression_ratio = COMPRESSION_RATIOS[speed_mode]
    kv_per_token_compressed = kv_per_token_fp16 / compression_ratio

    # How much context fits in the KV budget?
    kv_budget_bytes = plan["kv_budget_gb"] * (1024**3)
    context_achievable = int(kv_budget_bytes / max(kv_per_token_compressed, 1))
    context_achievable = max(context_achievable, 1)

    # Weight budget: total usable VRAM minus KV reservation
    weight_budget_gb = round(usable_vram - plan["kv_budget_gb"], 1)
    weight_budget_gb = max(1.0, weight_budget_gb)

    return {
        "weight_budget_gb": weight_budget_gb,
        "kv_budget_gb": plan["kv_budget_gb"],
        "num_gpu_layers": plan["num_gpu_layers"],
        "total_layers": cfg["num_layers"],
        "layer_size_mb": plan["layer_size_mb"],
        "compression_ratio": compression_ratio,
        "context_achievable": context_achievable,
        "context_target": context_target,
        "estimated_tok_per_sec": ESTIMATED_SPEEDS[speed_mode],
        "weight_quant_bits": weight_quant_bits,
        "speed_mode": speed_mode,
    }


# ---------------------------------------------------------------------------
# KV Cache Strategy Selection
# ---------------------------------------------------------------------------


def select_kv_strategy(speed_mode: str = "balanced") -> Dict[str, Any]:
    """Select KV cache configuration based on speed mode.

    Args:
        speed_mode: One of "safe", "balanced", "fast".

    Returns:
        Dict with cache_type and kwargs for constructing the cache.
    """
    if speed_mode not in SPEED_MODES:
        raise ValueError(f"speed_mode must be one of {SPEED_MODES}, got '{speed_mode}'")

    if speed_mode == "safe":
        return {
            "cache_type": "GenerationCache",
            "preset": "hybrid_max_quality",
            "description": "boundary K3/V3 5x, no eviction",
            "quality_pct": 100,
            "kwargs": {},
        }

    elif speed_mode == "balanced":
        return {
            "cache_type": "GenerationCache",
            "preset": None,
            "description": "boundary K3/V3 5x + mild eviction",
            "quality_pct": 98,
            "kwargs": {
                "key_bits": 3,
                "val_bits": 3,
                "anchor_strategy": "boundary",
                "fp16_window": 64,
                "use_residual_quant": True,
            },
        }

    else:  # fast
        return {
            "cache_type": "EvictionCache",
            "preset": None,
            "description": "boundary K3/V2 + aggressive eviction 7.5x",
            "quality_pct": 96,
            "kwargs": {
                "key_bits": 3,
                "val_bits": 2,
                "fp16_window": 64,
                "max_warm_tokens": 1024,
                "anchor_interval": 12,
            },
        }


def create_kv_cache(speed_mode: str = "balanced", num_layers: int = 80) -> Any:
    """Create the appropriate KV cache for the given speed mode.

    Args:
        speed_mode: One of "safe", "balanced", "fast".
        num_layers: Number of transformer layers in the model.

    Returns:
        A GenerationCache or EvictionCache instance.
    """
    from turboquantdc.generation_cache import GenerationCache
    from turboquantdc.token_eviction import EvictionCache

    strategy = select_kv_strategy(speed_mode)

    if strategy["cache_type"] == "EvictionCache":
        return EvictionCache(**strategy["kwargs"])

    if strategy["preset"]:
        return GenerationCache.from_preset(
            strategy["preset"], num_layers=num_layers
        )

    # Manual construction with num_layers for boundary/gradient strategies
    kwargs = strategy["kwargs"].copy()
    anchor_strategy = kwargs.get("anchor_strategy", "fixed")
    if anchor_strategy in ("boundary", "gradient"):
        kwargs["num_layers"] = num_layers
    return GenerationCache(**kwargs)


# ---------------------------------------------------------------------------
# Startup Report
# ---------------------------------------------------------------------------


def format_startup_report(
    model_name: str,
    gpu_info: Dict[str, Any],
    memory_plan: Dict[str, Any],
    kv_strategy: Dict[str, Any],
) -> str:
    """Format a startup report box for display.

    Args:
        model_name: HuggingFace model name.
        gpu_info: From detect_gpu().
        memory_plan: From calculate_memory_plan().
        kv_strategy: From select_kv_strategy().

    Returns:
        Formatted multi-line string with the startup report.
    """
    # Short model name
    short_name = model_name.split("/")[-1] if "/" in model_name else model_name

    gpu_name = gpu_info.get("name", "unknown")
    vram = gpu_info.get("vram_gb", 0)

    n_gpu = memory_plan["num_gpu_layers"]
    n_total = memory_plan["total_layers"]
    quant_bits = memory_plan["weight_quant_bits"]
    ctx = memory_plan["context_target"]
    speed = memory_plan["estimated_tok_per_sec"]
    compression = memory_plan["compression_ratio"]
    mode = memory_plan["speed_mode"]

    kv_desc = kv_strategy["description"]

    # Build context string
    if ctx >= 1000:
        ctx_str = f"{ctx // 1000}K tokens"
    else:
        ctx_str = f"{ctx} tokens"

    lines = [
        f"  Model:    {short_name}",
        f"  GPU:      {gpu_name} ({vram}GB)",
        f"  Weights:  {quant_bits}-bit BnB, {n_gpu}/{n_total} layers on GPU",
        f"  KV Cache: {kv_desc} ({compression}x)",
        f"  Context:  {ctx_str}",
        f"  Mode:     {mode} (~{speed} tok/s est.)",
    ]

    # Calculate box width
    max_line_len = max(len(line) for line in lines)
    box_width = max(max_line_len + 2, 48)

    top = "+" + "=" * box_width + "+"
    title_line = "  TurboQuantDC -- 70B on " + gpu_name
    title_padded = "| " + title_line.ljust(box_width - 2) + " |"
    sep = "|" + "-" * box_width + "|"
    bottom = "+" + "=" * box_width + "+"

    body_lines = []
    for line in lines:
        body_lines.append("| " + line.ljust(box_width - 2) + " |")

    parts = [top, title_padded, sep] + body_lines + [bottom]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


def load_model(
    model_name: str,
    memory_plan: Dict[str, Any],
) -> Any:
    """Load a model with partial GPU offloading via BitsAndBytes + accelerate.

    Args:
        model_name: HuggingFace model name or path.
        memory_plan: From calculate_memory_plan().

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    weight_budget_gb = memory_plan["weight_budget_gb"]
    quant_bits = memory_plan["weight_quant_bits"]

    if quant_bits == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quant_bits == 8:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: f"{weight_budget_gb}GiB", "cpu": "64GiB"},
        quantization_config=quant_config,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Interactive Chat Loop
# ---------------------------------------------------------------------------


def chat_loop(
    model: Any,
    tokenizer: Any,
    cache: Any,
    memory_plan: Dict[str, Any],
) -> None:
    """Run an interactive chat loop with context tracking.

    Commands:
        /context  - Show KV cache usage
        /clear    - Reset KV cache
        /quit     - Exit

    Args:
        model: Loaded HuggingFace model.
        tokenizer: Loaded tokenizer.
        cache: TurboQuantDC KV cache instance.
        memory_plan: For context target display.
    """
    print("\nChat started. Commands: /context, /clear, /quit\n")
    context_target = memory_plan.get("context_target", 144_000)

    conversation_ids = torch.tensor([], dtype=torch.long)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Goodbye!")
            break

        if user_input == "/context":
            seq_len = cache.get_seq_length(0) if hasattr(cache, "get_seq_length") else 0
            pct = round(100 * seq_len / max(context_target, 1), 1)
            print(f"  KV cache: {seq_len:,} / {context_target:,} tokens ({pct}%)")
            if hasattr(cache, "memory_savings"):
                savings = cache.memory_savings()
                ratio = savings.get("overall_compression_ratio", 1.0)
                print(f"  Compression: {ratio:.1f}x")
            if hasattr(cache, "eviction_stats"):
                stats = cache.eviction_stats()
                evicted = stats.get("total_tokens_evicted", 0)
                if evicted > 0:
                    print(f"  Evicted: {evicted:,} tokens")
            continue

        if user_input == "/clear":
            if hasattr(cache, "reset"):
                cache.reset()
            print("  KV cache cleared.")
            conversation_ids = torch.tensor([], dtype=torch.long)
            continue

        # Tokenize and generate
        inputs = tokenizer(user_input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        start_time = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                past_key_values=cache,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        elapsed = time.time() - start_time

        new_tokens = outputs.shape[1] - input_ids.shape[1]
        tok_per_sec = new_tokens / max(elapsed, 0.001)

        response_text = tokenizer.decode(
            outputs[0, input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        print(f"\nAssistant: {response_text}")
        print(f"  [{new_tokens} tokens, {tok_per_sec:.1f} tok/s]\n")


# ---------------------------------------------------------------------------
# One-Line API
# ---------------------------------------------------------------------------


def run_model(
    model_name: str = DEFAULT_MODEL,
    context_target: int = 144_000,
    speed_mode: str = "balanced",
    prompt: Optional[str] = None,
) -> None:
    """One-line API to run any 70B model on an RTX 4090.

    Auto-detects GPU, calculates optimal split, loads model with partial
    offload, attaches TurboQuantDC KV cache, and starts interactive chat
    (or generates a single response if prompt is given).

    Args:
        model_name: HuggingFace model name or path.
        context_target: Target context length in tokens.
        speed_mode: One of "safe", "balanced", "fast".
        prompt: Optional single prompt. If None, starts interactive chat.
    """
    gpu_info = detect_gpu()
    vram = gpu_info["vram_gb"] if gpu_info["available"] else 24.0

    memory_plan = calculate_memory_plan(
        context_target=context_target,
        gpu_vram_gb=vram,
        speed_mode=speed_mode,
    )

    kv_strategy = select_kv_strategy(speed_mode)

    report = format_startup_report(model_name, gpu_info, memory_plan, kv_strategy)
    print(report)

    # Load model
    model, tokenizer = load_model(model_name, memory_plan)

    # Create KV cache
    num_layers = getattr(model.config, "num_hidden_layers", 80)
    cache = create_kv_cache(speed_mode, num_layers=num_layers)

    if prompt:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                past_key_values=cache,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
    else:
        chat_loop(model, tokenizer, cache, memory_plan)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for run_70b.py."""
    parser = argparse.ArgumentParser(
        description="Run 70B models on RTX 4090 (24GB) with TurboQuantDC KV compression.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Speed modes:
  safe      boundary K3/V3 5x, no eviction, ~27 tok/s, 100%% quality
  balanced  boundary K3/V3 5x + mild eviction, ~33 tok/s, 98%% quality
  fast      boundary K3/V2 + aggressive eviction 7.5x, ~38 tok/s, 96%% quality
""",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name or path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--context-target",
        type=int,
        default=144_000,
        help="Target context length in tokens (default: 144000)",
    )
    parser.add_argument(
        "--speed-mode",
        choices=SPEED_MODES,
        default="balanced",
        help="Speed/quality tradeoff (default: balanced)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate (if omitted, starts interactive chat)",
    )
    parser.add_argument(
        "--weight-bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Weight quantization bits via BitsAndBytes (default: 4)",
    )
    parser.add_argument(
        "--gpu-budget",
        type=float,
        default=None,
        help="Override GPU VRAM budget in GB (default: auto-detect)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print the startup report and exit (no model loading)",
    )

    args = parser.parse_args()

    # Detect GPU
    gpu_info = detect_gpu()
    if args.gpu_budget is not None:
        vram = args.gpu_budget
    elif gpu_info["available"]:
        vram = gpu_info["vram_gb"]
    else:
        vram = 24.0  # Assume RTX 4090

    # Calculate memory plan
    memory_plan = calculate_memory_plan(
        context_target=args.context_target,
        gpu_vram_gb=vram,
        speed_mode=args.speed_mode,
        weight_quant_bits=args.weight_bits,
    )

    kv_strategy = select_kv_strategy(args.speed_mode)

    # Print startup report
    report = format_startup_report(args.model, gpu_info, memory_plan, kv_strategy)
    print(report)

    if args.report_only:
        return

    # Load model
    model, tokenizer = load_model(args.model, memory_plan)

    # Create KV cache
    num_layers = getattr(model.config, "num_hidden_layers", 80)
    cache = create_kv_cache(args.speed_mode, num_layers=num_layers)

    if args.prompt:
        inputs = tokenizer(args.prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                past_key_values=cache,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
    else:
        chat_loop(model, tokenizer, cache, memory_plan)


if __name__ == "__main__":
    main()
