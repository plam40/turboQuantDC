#!/usr/bin/env python3
"""TurboQuantDC V2 long-context validation benchmark.

Tests V2 cache at 2K, 4K, and 8K context lengths where adaptive tiering,
DeltaQuant grouping, and FAISS retrieval are designed to shine.

Compares:
  - FP16 baseline (model.generate with default cache)
  - V2 compress mode (PCA + mean-removal + adaptive bits + DeltaQuant)
  - V2 full mode (above + FAISS retrieval)
  - Production GenerationCache (Triton-optimized 3-bit)

Measures at each context length:
  - Effective bits per token
  - Compression ratio
  - Token match rate vs FP16 (first 100 generated tokens)
  - First divergence point
  - Generation speed (tok/s)
  - Peak GPU memory

Run:
    python benchmarks/v2_long_context_benchmark.py
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Allow running from repo root
REPO_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, REPO_ROOT)

from turboquantdc.v2_cache import TurboQuantV2Cache, V2Config
from turboquantdc.generation_cache import GenerationCache

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("V2_MODEL", "Qwen/Qwen2.5-3B-Instruct")
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MAX_NEW_TOKENS = 100
TARGET_CONTEXTS = [2048, 4096, 8192]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_gpu_mem_mb() -> float:
    """Current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def get_gpu_peak_mb() -> float:
    """Peak GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def reset_gpu_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


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
# Long text generation from wikitext
# ---------------------------------------------------------------------------

def build_long_prompts(tokenizer, targets: List[int]) -> Dict[int, Tuple[str, int]]:
    """Build prompts at target token counts from wikitext-2.

    Returns dict: target_len -> (prompt_text, actual_token_count)
    """
    from datasets import load_dataset

    print("Loading wikitext-2 for long prompts...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                      cache_dir=CACHE_DIR)
    paragraphs = [t for t in ds["text"] if t.strip() and len(t.strip()) > 50]
    long_text = " ".join(paragraphs)

    # Suffix to make the model generate meaningful continuation
    suffix = (
        "\n\nBased on the text above, provide a detailed summary of the key "
        "topics covered. Begin your summary:"
    )

    prompts = {}
    for target in sorted(targets):
        # Binary search for the right prefix length
        # We need to leave room for the suffix
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        target_prefix_tokens = target - len(suffix_ids)

        if target_prefix_tokens <= 0:
            target_prefix_tokens = target

        # Encode full text, truncate to target
        all_ids = tokenizer.encode(long_text, add_special_tokens=False)
        prefix_ids = all_ids[:target_prefix_tokens]
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)

        full_prompt = prefix_text + suffix
        actual_tokens = len(tokenizer.encode(full_prompt))

        prompts[target] = (full_prompt, actual_tokens)
        print(f"  Target {target}: built prompt with {actual_tokens} tokens")

    return prompts


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
) -> Tuple[str, List[int], float, float]:
    """Generate tokens with a given cache. Returns (text, ids, elapsed, peak_mb)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    torch.cuda.empty_cache()
    gc.collect()
    reset_gpu_peak()

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
    peak_mb = get_gpu_peak_mb()

    gen_ids = outputs[0][prompt_len:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return gen_text, gen_ids, elapsed, peak_mb


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    all_results = {
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "target_contexts": TARGET_CONTEXTS,
        "context_results": {},
    }

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    num_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_kv_heads = getattr(
        model.config, "num_key_value_heads", model.config.num_attention_heads
    )
    print(f"Model: {num_layers} layers, d={head_dim}, {num_kv_heads} KV heads")
    all_results["model_config"] = {
        "num_layers": num_layers,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
    }

    # Build long prompts
    prompts = build_long_prompts(tokenizer, TARGET_CONTEXTS)

    # Calibrate PCA rotations (use first prompt for calibration)
    print("\n" + "=" * 70)
    print("CALIBRATING PCA rotations")
    print("=" * 70)

    cal_path = os.path.join(
        REPO_ROOT, "benchmarks", "results", "v2_long_ctx_pca.pt"
    )
    t0 = time.perf_counter()
    pca_rotations = TurboQuantV2Cache.calibrate(
        model, tokenizer, n_tokens=128, save_path=cal_path, device=DEVICE,
    )
    cal_time = time.perf_counter() - t0
    print(f"Calibration time: {cal_time:.2f}s")
    all_results["calibration_time_sec"] = round(cal_time, 2)

    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Run at each context length
    # ---------------------------------------------------------------

    for target_ctx in TARGET_CONTEXTS:
        prompt_text, actual_tokens = prompts[target_ctx]
        ctx_label = f"{target_ctx // 1024}K"

        print("\n" + "=" * 70)
        print(f"CONTEXT LENGTH: {ctx_label} ({actual_tokens} tokens)")
        print("=" * 70)

        ctx_results = {
            "target_tokens": target_ctx,
            "actual_tokens": actual_tokens,
        }

        # ------ 1. FP16 baseline ------
        print(f"\n  [1/4] FP16 Baseline ({ctx_label})...")
        try:
            text, ids, elapsed, peak = generate_with_cache(
                model, tokenizer, prompt_text, cache=None,
            )
            tps = len(ids) / elapsed if elapsed > 0 else 0
            print(f"    Generated {len(ids)} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")
            print(f"    Peak GPU: {peak:.0f} MB")
            print(f"    Output: {text[:120]}...")
            ctx_results["fp16"] = {
                "tokens_generated": len(ids),
                "ids": ids,
                "time_sec": round(elapsed, 2),
                "tokens_per_sec": round(tps, 1),
                "peak_gpu_mb": round(peak, 0),
                "effective_bits": 16.0,
                "compression_ratio": 1.0,
                "text_preview": text[:200],
            }
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            ctx_results["fp16"] = {"error": str(e)}

        gc.collect()
        torch.cuda.empty_cache()

        # ------ 2. V2 compress mode ------
        print(f"\n  [2/4] V2 Compress mode ({ctx_label})...")
        try:
            # For 8K, use smaller window to avoid OOM
            window = 64 if target_ctx <= 4096 else 32
            compress_config = V2Config(
                key_bits=3,
                val_bits=3,
                window_size=window,
                boundary_layers=2,
                mode="compress",
                seed=SEED,
            )
            cache = TurboQuantV2Cache(
                config=compress_config,
                num_layers=num_layers,
                pca_rotations=pca_rotations,
            )
            text, ids, elapsed, peak = generate_with_cache(
                model, tokenizer, prompt_text, cache=cache,
            )
            tps = len(ids) / elapsed if elapsed > 0 else 0
            eff_bits = cache.effective_bits()
            comp_ratio = cache.compression_ratio()
            tier_info = cache.tier_summary()

            # Compare to FP16
            fp16_ids = ctx_results.get("fp16", {}).get("ids", [])
            match = token_match_rate(fp16_ids, ids) if fp16_ids else 0.0
            div_pt = first_divergence(fp16_ids, ids) if fp16_ids else -1

            print(f"    Generated {len(ids)} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")
            print(f"    Peak GPU: {peak:.0f} MB")
            print(f"    Effective bits: {eff_bits:.2f}, Compression: {comp_ratio:.2f}x")
            print(f"    Token match vs FP16: {match:.1%}, First diverge: token {div_pt}")
            print(f"    Tier summary: {tier_info}")
            print(f"    Output: {text[:120]}...")

            ctx_results["v2_compress"] = {
                "tokens_generated": len(ids),
                "ids": ids,
                "time_sec": round(elapsed, 2),
                "tokens_per_sec": round(tps, 1),
                "peak_gpu_mb": round(peak, 0),
                "effective_bits": round(eff_bits, 3),
                "compression_ratio": round(comp_ratio, 2),
                "token_match_rate": round(match, 4),
                "first_divergence": div_pt,
                "tier_summary": tier_info,
                "window_size": window,
                "text_preview": text[:200],
            }
            del cache
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            ctx_results["v2_compress"] = {"error": str(e)}

        gc.collect()
        torch.cuda.empty_cache()

        # ------ 3. V2 full mode (FAISS) ------
        print(f"\n  [3/4] V2 Full mode / FAISS ({ctx_label})...")
        try:
            window = 64 if target_ctx <= 4096 else 32
            full_config = V2Config(
                key_bits=3,
                val_bits=3,
                window_size=window,
                boundary_layers=2,
                retrieval_k=128,
                faiss_nprobe=16,
                faiss_nlist=64,
                mode="full",
                seed=SEED,
            )
            cache = TurboQuantV2Cache(
                config=full_config,
                num_layers=num_layers,
                pca_rotations=pca_rotations,
            )
            text, ids, elapsed, peak = generate_with_cache(
                model, tokenizer, prompt_text, cache=cache,
            )
            tps = len(ids) / elapsed if elapsed > 0 else 0
            eff_bits = cache.effective_bits()
            comp_ratio = cache.compression_ratio()
            tier_info = cache.tier_summary()

            fp16_ids = ctx_results.get("fp16", {}).get("ids", [])
            match = token_match_rate(fp16_ids, ids) if fp16_ids else 0.0
            div_pt = first_divergence(fp16_ids, ids) if fp16_ids else -1

            print(f"    Generated {len(ids)} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")
            print(f"    Peak GPU: {peak:.0f} MB")
            print(f"    Effective bits: {eff_bits:.2f}, Compression: {comp_ratio:.2f}x")
            print(f"    Token match vs FP16: {match:.1%}, First diverge: token {div_pt}")
            print(f"    Output: {text[:120]}...")

            ctx_results["v2_full"] = {
                "tokens_generated": len(ids),
                "ids": ids,
                "time_sec": round(elapsed, 2),
                "tokens_per_sec": round(tps, 1),
                "peak_gpu_mb": round(peak, 0),
                "effective_bits": round(eff_bits, 3),
                "compression_ratio": round(comp_ratio, 2),
                "token_match_rate": round(match, 4),
                "first_divergence": div_pt,
                "tier_summary": tier_info,
                "window_size": window,
                "text_preview": text[:200],
            }
            del cache
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            ctx_results["v2_full"] = {"error": str(e)}

        gc.collect()
        torch.cuda.empty_cache()

        # ------ 4. Production GenerationCache ------
        print(f"\n  [4/4] Production GenerationCache ({ctx_label})...")
        try:
            cache = GenerationCache(
                key_bits=3,
                val_bits=2,
                fp16_window=64,
                anchor_strategy="boundary",
                num_layers=num_layers,
                seed=SEED,
                use_residual_quant=True,
                use_norm_correction=True,
            )
            text, ids, elapsed, peak = generate_with_cache(
                model, tokenizer, prompt_text, cache=cache,
            )
            tps = len(ids) / elapsed if elapsed > 0 else 0

            # GenerationCache doesn't have effective_bits() at top level the same way
            # Compute manually
            eff_bits = 16.0
            total_bits = 0.0
            total_tokens = 0
            for layer in cache._layers:
                sl = layer.get_seq_length()
                if sl > 0:
                    total_tokens += sl
                    if hasattr(layer, "key_bits"):
                        # Compressed layer: key_bits + 1 (residual) + val_bits + overheads
                        kb = layer.key_bits + 1.0 + 32.0 / head_dim
                        vb = layer.val_bits + 16.0 / head_dim
                        # FP16 window portion
                        fp16_n = min(layer.fp16_window, sl)
                        comp_n = max(sl - fp16_n, 0)
                        avg = (comp_n * (kb + vb) / 2.0 + fp16_n * 16.0) / sl
                        total_bits += avg * sl
                    else:
                        total_bits += 16.0 * sl
            if total_tokens > 0:
                eff_bits = total_bits / total_tokens
            comp_ratio = 16.0 / max(eff_bits, 0.01)

            fp16_ids = ctx_results.get("fp16", {}).get("ids", [])
            match = token_match_rate(fp16_ids, ids) if fp16_ids else 0.0
            div_pt = first_divergence(fp16_ids, ids) if fp16_ids else -1

            print(f"    Generated {len(ids)} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")
            print(f"    Peak GPU: {peak:.0f} MB")
            print(f"    Effective bits: {eff_bits:.2f}, Compression: {comp_ratio:.2f}x")
            print(f"    Token match vs FP16: {match:.1%}, First diverge: token {div_pt}")
            print(f"    Output: {text[:120]}...")

            ctx_results["production"] = {
                "tokens_generated": len(ids),
                "ids": ids,
                "time_sec": round(elapsed, 2),
                "tokens_per_sec": round(tps, 1),
                "peak_gpu_mb": round(peak, 0),
                "effective_bits": round(eff_bits, 3),
                "compression_ratio": round(comp_ratio, 2),
                "token_match_rate": round(match, 4),
                "first_divergence": div_pt,
                "text_preview": text[:200],
            }
            del cache
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            ctx_results["production"] = {"error": str(e)}

        gc.collect()
        torch.cuda.empty_cache()

        all_results["context_results"][ctx_label] = ctx_results

    # ---------------------------------------------------------------
    # Write results markdown
    # ---------------------------------------------------------------
    write_results_markdown(all_results)
    return all_results


def write_results_markdown(results: Dict[str, Any]):
    """Write comprehensive markdown results file."""
    out_path = os.path.join(
        REPO_ROOT, "benchmarks", "results", "v2_long_context_results.md"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines = []
    lines.append("# TurboQuantDC V2 Long-Context Validation Results")
    lines.append("")
    lines.append(f"**Date:** {results['timestamp']}")
    lines.append(f"**Model:** {results['model']}")
    lines.append(f"**Device:** {results['device']}")
    lines.append(f"**Generation:** {results['max_new_tokens']} new tokens per run")
    mc = results.get("model_config", {})
    lines.append(
        f"**Config:** {mc.get('num_layers', '?')} layers, "
        f"d={mc.get('head_dim', '?')}, "
        f"{mc.get('num_kv_heads', '?')} KV heads"
    )
    lines.append("")

    # Summary table
    lines.append("## Summary Table")
    lines.append("")
    lines.append(
        "| Context | Method | Eff Bits | Compression | "
        "Token Match | 1st Diverge | Speed (tok/s) | Peak GPU (MB) |"
    )
    lines.append(
        "|---------|--------|----------|-------------|"
        "------------|-------------|---------------|---------------|"
    )

    for ctx_label, ctx_data in results.get("context_results", {}).items():
        for method_key, method_name in [
            ("fp16", "FP16 Baseline"),
            ("v2_compress", "V2 Compress"),
            ("v2_full", "V2 Full/FAISS"),
            ("production", "Production GC"),
        ]:
            m = ctx_data.get(method_key, {})
            if "error" in m:
                lines.append(
                    f"| {ctx_label} | {method_name} | "
                    f"ERROR | - | - | - | - | - |"
                )
                continue
            if not m:
                continue

            eff = m.get("effective_bits", "-")
            comp = m.get("compression_ratio", "-")
            match = m.get("token_match_rate", "-")
            div = m.get("first_divergence", "-")
            speed = m.get("tokens_per_sec", "-")
            peak = m.get("peak_gpu_mb", "-")

            eff_str = f"{eff:.2f}" if isinstance(eff, float) else str(eff)
            comp_str = f"{comp:.2f}x" if isinstance(comp, float) else str(comp)
            match_str = f"{match:.1%}" if isinstance(match, float) else str(match)
            div_str = str(div)
            speed_str = f"{speed:.1f}" if isinstance(speed, float) else str(speed)
            peak_str = f"{peak:.0f}" if isinstance(peak, (int, float)) else str(peak)

            lines.append(
                f"| {ctx_label} | {method_name} | "
                f"{eff_str} | {comp_str} | {match_str} | "
                f"{div_str} | {speed_str} | {peak_str} |"
            )

    # Detailed sections per context length
    for ctx_label, ctx_data in results.get("context_results", {}).items():
        lines.append("")
        lines.append(f"## {ctx_label} Context ({ctx_data.get('actual_tokens', '?')} tokens)")
        lines.append("")

        for method_key, method_name in [
            ("fp16", "FP16 Baseline"),
            ("v2_compress", "V2 Compress"),
            ("v2_full", "V2 Full/FAISS"),
            ("production", "Production GC"),
        ]:
            m = ctx_data.get(method_key, {})
            if not m:
                continue

            lines.append(f"### {method_name}")
            lines.append("")

            if "error" in m:
                lines.append(f"**Error:** `{m['error']}`")
                lines.append("")
                continue

            lines.append(f"- **Effective bits:** {m.get('effective_bits', '-')}")
            lines.append(f"- **Compression ratio:** {m.get('compression_ratio', '-')}x")
            lines.append(f"- **Token match vs FP16:** {m.get('token_match_rate', '-')}")
            lines.append(f"- **First divergence:** token {m.get('first_divergence', '-')}")
            lines.append(
                f"- **Speed:** {m.get('tokens_per_sec', '-')} tok/s "
                f"({m.get('time_sec', '-')}s total)"
            )
            lines.append(f"- **Peak GPU:** {m.get('peak_gpu_mb', '-')} MB")
            if "tier_summary" in m:
                lines.append(f"- **Tier summary:** {m['tier_summary']}")
            if "window_size" in m:
                lines.append(f"- **Window size:** {m['window_size']}")
            lines.append(f"- **Output preview:** {m.get('text_preview', '')[:150]}...")
            lines.append("")

    # Analysis section
    lines.append("## Analysis")
    lines.append("")

    # Check scaling trends
    ctx_keys = list(results.get("context_results", {}).keys())
    if len(ctx_keys) >= 2:
        lines.append("### Compression Scaling with Context Length")
        lines.append("")
        lines.append("Does V2 compression improve at longer context? (Asymptotic law)")
        lines.append("")

        for method_key, method_name in [
            ("v2_compress", "V2 Compress"),
            ("v2_full", "V2 Full"),
            ("production", "Production GC"),
        ]:
            bits_by_ctx = []
            for ck in ctx_keys:
                m = results["context_results"][ck].get(method_key, {})
                eb = m.get("effective_bits")
                if isinstance(eb, (int, float)):
                    bits_by_ctx.append((ck, eb))

            if len(bits_by_ctx) >= 2:
                first_bits = bits_by_ctx[0][1]
                last_bits = bits_by_ctx[-1][1]
                delta = first_bits - last_bits
                lines.append(
                    f"- **{method_name}:** {first_bits:.2f} bits ({bits_by_ctx[0][0]}) -> "
                    f"{last_bits:.2f} bits ({bits_by_ctx[-1][0]}), "
                    f"delta = {delta:+.2f} bits"
                )
        lines.append("")

        lines.append("### Quality Scaling with Context Length")
        lines.append("")
        for method_key, method_name in [
            ("v2_compress", "V2 Compress"),
            ("v2_full", "V2 Full"),
            ("production", "Production GC"),
        ]:
            match_by_ctx = []
            for ck in ctx_keys:
                m = results["context_results"][ck].get(method_key, {})
                mr = m.get("token_match_rate")
                if isinstance(mr, (int, float)):
                    match_by_ctx.append((ck, mr))

            if len(match_by_ctx) >= 2:
                first_m = match_by_ctx[0][1]
                last_m = match_by_ctx[-1][1]
                delta = last_m - first_m
                lines.append(
                    f"- **{method_name}:** {first_m:.1%} match ({match_by_ctx[0][0]}) -> "
                    f"{last_m:.1%} match ({match_by_ctx[-1][0]}), "
                    f"delta = {delta:+.1%}"
                )
        lines.append("")

        lines.append("### Key Questions")
        lines.append("")

        # Q1: Does V2 compress beat production at long context?
        lines.append("**Q1: Does V2 compress mode beat Production GenerationCache at long context?**")
        lines.append("")
        for ck in ctx_keys:
            v2c = results["context_results"][ck].get("v2_compress", {})
            prod = results["context_results"][ck].get("production", {})
            v2_bits = v2c.get("effective_bits")
            prod_bits = prod.get("effective_bits")
            v2_match = v2c.get("token_match_rate")
            prod_match = prod.get("token_match_rate")

            if isinstance(v2_bits, (int, float)) and isinstance(prod_bits, (int, float)):
                better_comp = "V2" if v2_bits < prod_bits else "Production"
                lines.append(
                    f"- {ck}: V2={v2_bits:.2f}b vs Prod={prod_bits:.2f}b "
                    f"(better compression: **{better_comp}**)"
                )
            if isinstance(v2_match, (int, float)) and isinstance(prod_match, (int, float)):
                better_qual = "V2" if v2_match > prod_match else "Production"
                lines.append(
                    f"  Quality: V2={v2_match:.1%} vs Prod={prod_match:.1%} "
                    f"(better quality: **{better_qual}**)"
                )
        lines.append("")

        # Q2: Does FAISS show speed gains at 4K+?
        lines.append("**Q2: Does FAISS retrieval mode show speed gains at 4K+ context?**")
        lines.append("")
        for ck in ctx_keys:
            v2c = results["context_results"][ck].get("v2_compress", {})
            v2f = results["context_results"][ck].get("v2_full", {})
            fp16 = results["context_results"][ck].get("fp16", {})

            c_speed = v2c.get("tokens_per_sec")
            f_speed = v2f.get("tokens_per_sec")
            fp_speed = fp16.get("tokens_per_sec")

            if isinstance(c_speed, (int, float)) and isinstance(f_speed, (int, float)):
                lines.append(
                    f"- {ck}: Compress={c_speed:.1f} tok/s, Full/FAISS={f_speed:.1f} tok/s"
                )
                if isinstance(fp_speed, (int, float)):
                    lines.append(f"  (FP16 baseline: {fp_speed:.1f} tok/s)")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_benchmark()
