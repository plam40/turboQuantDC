#!/usr/bin/env python3
"""Ultra-compression benchmark: 1-bit KV cache experiments.

Tests three approaches to 1-bit compression on real KV caches from
Qwen2.5-3B-Instruct, comparing against the 3-bit ResidualQuant baseline.

Measures:
  - Cosine similarity (vector reconstruction quality)
  - Top-1 and Top-5 attention match (attention score preservation)
  - Generation quality (50-token autoregressive, greedy)

Run:
    python benchmarks/ultra_compress_benchmark.py
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
REPO_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, REPO_ROOT)

from turboquantdc.ultra_compress import (
    AttentionGatedCache,
    MultiScaleResidualChain,
    OneBitResidualQuant,
    SignPredictionCompressor,
)
from turboquantdc.residual_quant import ResidualQuantEstimator
from turboquantdc.polarquant import PolarQuant

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
GENERATION_TOKENS = 50

PROMPT = (
    "The Walsh-Hadamard Transform is an important tool in signal processing "
    "and quantum computing. It operates on vectors whose length is a power of "
    "two and can be computed in O(n log n) time using a butterfly structure. "
    "In the context of KV cache compression for large language models, the "
    "WHT serves as a fast random orthogonal rotation that concentrates the "
    "coordinate distribution, enabling optimal scalar quantization. The key "
    "insight is that after rotation, each coordinate becomes nearly independent "
    "and follows a concentrated distribution close to Gaussian N(0, 1/d), "
    "which allows a precomputed Lloyd-Max codebook to achieve near-optimal "
    "mean squared error. This is the foundation of the TurboQuant algorithm."
)

GENERATION_PROMPTS = [
    "What is the capital of Australia? Answer briefly:",
    "Explain what a neural network is in two sentences:",
    "Write a Python function that returns the factorial of n:",
]

GENERATION_KEYWORDS = [
    ["canberra"],
    ["layer", "neuron", "learn", "network", "input", "output", "weight"],
    ["def", "factorial", "return"],
]


# ---------------------------------------------------------------------------
# Model loading + KV extraction
# ---------------------------------------------------------------------------

def load_model():
    """Load model with 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME} (4-bit quantized)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


def extract_kv_caches(model, tokenizer, prompt: str) -> Dict[int, Dict[str, torch.Tensor]]:
    """Run forward pass and extract raw KV caches as FP32."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    kv = {}
    for i, layer_kv in enumerate(outputs.past_key_values):
        k, v = layer_kv[0], layer_kv[1]
        kv[i] = {
            "keys": k.float(),    # (1, num_heads, seq_len, head_dim)
            "values": v.float(),
        }

    num_layers = len(kv)
    head_dim = kv[0]["keys"].shape[3]
    num_heads = kv[0]["keys"].shape[1]
    seq_len = kv[0]["keys"].shape[2]
    print(f"Extracted KV: {num_layers} layers, {num_heads} heads, "
          f"seq_len={seq_len}, head_dim={head_dim}")
    return kv


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_cosine_sim(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Mean cosine similarity across all vectors."""
    orig_flat = original.reshape(-1, original.shape[-1])
    recon_flat = reconstructed.reshape(-1, reconstructed.shape[-1])
    cos = F.cosine_similarity(orig_flat, recon_flat, dim=-1)
    return cos.mean().item()


def compute_attention_match(
    keys_orig: torch.Tensor,
    keys_recon: torch.Tensor,
    queries: torch.Tensor,
) -> Tuple[float, float]:
    """Compute top-1 and top-5 attention match rates.

    Uses the last token's query to compute attention scores against all keys,
    then checks if the top-K attended tokens match between original and
    reconstructed keys.
    """
    # queries: (1, num_heads, seq, head_dim) - use last token as query
    q = queries[:, :, -1:, :]  # (1, num_heads, 1, head_dim)

    scale = math.sqrt(keys_orig.shape[-1])

    # Original attention scores
    scores_orig = (q @ keys_orig.transpose(-2, -1)) / scale  # (1, H, 1, S)
    scores_orig = scores_orig.squeeze(2)  # (1, H, S)

    # Reconstructed attention scores
    scores_recon = (q @ keys_recon.transpose(-2, -1)) / scale
    scores_recon = scores_recon.squeeze(2)

    # Top-1 match
    top1_orig = scores_orig.argmax(dim=-1)  # (1, H)
    top1_recon = scores_recon.argmax(dim=-1)
    top1_match = (top1_orig == top1_recon).float().mean().item()

    # Top-5 match
    k = min(5, scores_orig.shape[-1])
    top5_orig = scores_orig.topk(k, dim=-1).indices  # (1, H, 5)
    top5_recon = scores_recon.topk(k, dim=-1).indices

    # Check if each top-5 from original appears in reconstructed top-5
    matches = 0
    total = top5_orig.numel()
    for i in range(top5_orig.shape[0]):
        for h in range(top5_orig.shape[1]):
            orig_set = set(top5_orig[i, h].tolist())
            recon_set = set(top5_recon[i, h].tolist())
            matches += len(orig_set & recon_set)

    top5_match = matches / total if total > 0 else 0.0

    return top1_match, top5_match


def compute_inner_product_mse(
    keys_orig: torch.Tensor,
    keys_recon: torch.Tensor,
    queries: torch.Tensor,
) -> float:
    """MSE between original and reconstructed inner products."""
    q = queries[:, :, -1:, :]
    ip_orig = (q @ keys_orig.transpose(-2, -1)).squeeze(2)
    ip_recon = (q @ keys_recon.transpose(-2, -1)).squeeze(2)
    return ((ip_orig - ip_recon) ** 2).mean().item()


# ---------------------------------------------------------------------------
# Per-approach compression + evaluation
# ---------------------------------------------------------------------------

def evaluate_approach(
    name: str,
    keys_flat: torch.Tensor,
    keys_orig: torch.Tensor,
    queries: torch.Tensor,
    compress_fn,
    decompress_fn,
    extra_info: str = "",
) -> Dict[str, Any]:
    """Compress, decompress, and evaluate one approach.

    Args:
        name: Method name.
        keys_flat: Flattened keys (N, d) for compression.
        keys_orig: Original keys (1, H, S, d) for attention match.
        queries: Query vectors for attention evaluation.
        compress_fn: Callable(keys_flat) -> compressed dict.
        decompress_fn: Callable(compressed) -> reconstructed flat keys.
        extra_info: Additional info string.

    Returns:
        Dict of metric results.
    """
    t0 = time.perf_counter()
    compressed = compress_fn(keys_flat)
    t_compress = time.perf_counter() - t0

    t0 = time.perf_counter()
    keys_recon_flat = decompress_fn(compressed)
    t_decompress = time.perf_counter() - t0

    # Reshape back to (1, H, S, d)
    keys_recon = keys_recon_flat.reshape(keys_orig.shape)

    cos_sim = compute_cosine_sim(keys_orig, keys_recon)
    top1, top5 = compute_attention_match(keys_orig, keys_recon, queries)
    ip_mse = compute_inner_product_mse(keys_orig, keys_recon, queries)

    result = {
        "name": name,
        "cosine_sim": cos_sim,
        "top1_match": top1,
        "top5_match": top5,
        "ip_mse": ip_mse,
        "compress_ms": t_compress * 1000,
        "decompress_ms": t_decompress * 1000,
        "extra": extra_info,
    }

    print(f"  {name:40s}  cos={cos_sim:.4f}  top1={top1:.1%}  "
          f"top5={top5:.1%}  ip_mse={ip_mse:.6f}")
    return result


def run_kv_evaluation(kv: Dict[int, Dict[str, torch.Tensor]]) -> List[Dict[str, Any]]:
    """Run all ultra-compression approaches on extracted KV caches.

    Tests on middle layers (most compressed in a real deployment).
    """
    results = []

    # Pick representative layers: early (5), middle (18), late (30)
    num_layers = len(kv)
    test_layers = [
        min(5, num_layers - 1),
        num_layers // 2,
        min(num_layers - 6, num_layers - 1),
    ]
    test_layers = sorted(set(test_layers))

    head_dim = kv[0]["keys"].shape[3]
    device = kv[0]["keys"].device

    for layer_idx in test_layers:
        print(f"\n{'='*75}")
        print(f"  Layer {layer_idx}")
        print(f"{'='*75}")

        keys_orig = kv[layer_idx]["keys"]  # (1, H, S, d)
        vals_orig = kv[layer_idx]["values"]
        batch, num_heads, seq_len, d = keys_orig.shape

        # Use keys as queries (self-attention simulation)
        queries = keys_orig

        # Flatten for compressor input
        keys_flat = keys_orig.reshape(-1, d)  # (H*S, d)

        # ---- Baseline: 3-bit ResidualQuant (current best) ----
        rq3 = ResidualQuantEstimator(d=d, bits=3, seed=SEED, device=str(device))
        r = evaluate_approach(
            "3-bit ResidualQuant (baseline)",
            keys_flat, keys_orig, queries,
            rq3.quantize, rq3.dequantize,
            "3 bits/coord",
        )
        r["layer"] = layer_idx
        r["bits"] = 3
        r["approach"] = "baseline_rq3"
        results.append(r)

        # ---- Baseline: 1-bit MSE only (no residual correction) ----
        pq1 = PolarQuant(d=d, bits=1, seed=SEED, device=str(device))

        def compress_1bit_mse(x):
            norms = x.norm(dim=-1, keepdim=True)
            x_n = x / (norms + 1e-8)
            idx = pq1.quantize(x_n)
            return {"indices": idx, "norms": norms}

        def decompress_1bit_mse(c):
            recon = pq1.dequantize(c["indices"])
            return recon * c["norms"]

        r = evaluate_approach(
            "1-bit MSE only (lower bound)",
            keys_flat, keys_orig, queries,
            compress_1bit_mse, decompress_1bit_mse,
            "1 bit/coord, no residual",
        )
        r["layer"] = layer_idx
        r["bits"] = 1
        r["approach"] = "baseline_1bit_mse"
        results.append(r)

        # ---- Baseline: 1-bit + residual sign (2 bits total) ----
        rq1 = OneBitResidualQuant(d=d, seed=SEED, device=str(device))
        r = evaluate_approach(
            "1-bit + residual sign (2 bits)",
            keys_flat, keys_orig, queries,
            rq1.quantize, rq1.dequantize,
            "2 bits/coord total",
        )
        r["layer"] = layer_idx
        r["bits"] = 2
        r["approach"] = "baseline_1bit_resid"
        results.append(r)

        # ---- Approach 1: Multi-scale residual chain ----
        for n_stages in [2, 3, 4]:
            chain = MultiScaleResidualChain(
                d=d, num_stages=n_stages, seed=SEED, device=str(device),
            )
            r = evaluate_approach(
                f"Multi-scale chain ({n_stages} stages = {n_stages} bits)",
                keys_flat, keys_orig, queries,
                chain.quantize, chain.dequantize,
                f"{n_stages} x 1-bit cascaded",
            )
            r["layer"] = layer_idx
            r["bits"] = n_stages
            r["approach"] = f"chain_{n_stages}stage"
            results.append(r)

        # ---- Approach 2: Sign prediction ----
        for window in [2, 4, 8]:
            sp = SignPredictionCompressor(
                d=d, window=window, use_residual_signs=True,
                seed=SEED, device=str(device),
            )
            # Measure prediction accuracy
            pred_acc = sp.get_prediction_accuracy(keys_flat)
            effective_bits = 1.0  # MSE
            if pred_acc > 0 and pred_acc < 1:
                h = -(pred_acc * math.log2(pred_acc) + (1 - pred_acc) * math.log2(1 - pred_acc))
                effective_bits += h
            else:
                effective_bits += 1.0

            r = evaluate_approach(
                f"Sign prediction (w={window}, acc={pred_acc:.1%})",
                keys_flat, keys_orig, queries,
                sp.quantize, sp.dequantize,
                f"pred_acc={pred_acc:.3f}, eff_bits={effective_bits:.2f}",
            )
            r["layer"] = layer_idx
            r["bits"] = effective_bits
            r["approach"] = f"sign_pred_w{window}"
            r["prediction_accuracy"] = pred_acc
            results.append(r)

        # ---- Approach 3: Attention-gated refinement ----
        agc = AttentionGatedCache(
            d=d, refine_bits=3, seed=SEED, device=str(device),
        )

        # Simulate attention to find which tokens to refine
        scale = math.sqrt(d)
        q_last = queries[:, :, -1:, :]  # (1, H, 1, d)
        attn_scores = (q_last @ keys_orig.transpose(-2, -1)) / scale  # (1,H,1,S)
        attn_weights = F.softmax(attn_scores.squeeze(2), dim=-1)  # (1, H, S)

        # Average across heads
        mean_attn = attn_weights.mean(dim=1).squeeze(0)  # (S,)

        for refine_frac in [0.0, 0.05, 0.1, 0.2, 0.5]:
            if refine_frac > 0:
                # Pick top refine_frac fraction of tokens by attention
                n_refine = max(1, int(seq_len * refine_frac))
                topk_indices = mean_attn.topk(n_refine).indices
                refine_mask_seq = torch.zeros(seq_len, dtype=torch.bool, device=device)
                refine_mask_seq[topk_indices] = True
            else:
                refine_mask_seq = torch.zeros(seq_len, dtype=torch.bool, device=device)

            # Expand mask to flat shape (H*S,)
            refine_mask_flat = refine_mask_seq.unsqueeze(0).expand(num_heads, -1).reshape(-1)

            eff_bits = AttentionGatedCache.compute_effective_bits(refine_mask_flat)

            def make_compress(agc_ref):
                def fn(x):
                    return agc_ref.quantize_both(x)
                return fn

            def make_decompress(agc_ref, mask):
                def fn(c):
                    return agc_ref.dequantize_selective(c, mask)
                return fn

            r = evaluate_approach(
                f"Attn-gated ({refine_frac:.0%} refined, {eff_bits:.2f} bits)",
                keys_flat, keys_orig, queries,
                make_compress(agc), make_decompress(agc, refine_mask_flat),
                f"refine_frac={refine_frac}, eff_bits={eff_bits:.2f}",
            )
            r["layer"] = layer_idx
            r["bits"] = eff_bits
            r["approach"] = f"attn_gated_{refine_frac:.0%}"
            r["refine_fraction"] = refine_frac
            results.append(r)

    return results


# ---------------------------------------------------------------------------
# Generation quality test
# ---------------------------------------------------------------------------

def test_generation_quality(model, tokenizer) -> List[Dict[str, Any]]:
    """Test autoregressive generation with ultra-compressed caches.

    Uses HF model.generate() with different cache configurations.
    """
    from turboquantdc.generation_cache import GenerationCache
    from turboquantdc.residual_quant import ResidualQuantCache

    gen_results = []

    # Configurations to test
    # Only use GenerationCache (has correct HF API) and FP16 baseline
    configs = [
        {
            "name": "FP16 Baseline",
            "factory": lambda: None,
        },
        {
            "name": "3-bit GenerationCache (prod)",
            "factory": lambda: GenerationCache(
                key_bits=3, val_bits=2, fp16_window=64,
                anchor_strategy="boundary", num_layers=36,
                use_residual_quant=True, seed=SEED,
            ),
        },
        {
            "name": "2-bit GenerationCache",
            "factory": lambda: GenerationCache(
                key_bits=2, val_bits=2, fp16_window=64,
                anchor_strategy="boundary", num_layers=36,
                use_residual_quant=True, seed=SEED,
            ),
        },
        {
            "name": "1-bit GenerationCache",
            "factory": lambda: GenerationCache(
                key_bits=1, val_bits=1, fp16_window=64,
                anchor_strategy="boundary", num_layers=36,
                use_residual_quant=True, seed=SEED,
            ),
        },
    ]

    for cfg in configs:
        print(f"\n--- Generation: {cfg['name']} ---")
        cfg_results = []

        for prompt_idx, (prompt, keywords) in enumerate(zip(GENERATION_PROMPTS, GENERATION_KEYWORDS)):
            cache = cfg["factory"]()

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            try:
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=GENERATION_TOKENS,
                        do_sample=False,
                        past_key_values=cache,
                        use_cache=True,
                    )

                gen_text = tokenizer.decode(
                    out[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
            except Exception as e:
                gen_text = f"[ERROR: {e}]"
                print(f"  [ERROR] Q{prompt_idx}: {e}")

            # Check coherence
            response_lower = gen_text.lower()
            has_keyword = any(kw in response_lower for kw in keywords)
            words = gen_text.split()
            is_repetitive = False
            if len(words) >= 6:
                trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
                for tg in set(trigrams):
                    if trigrams.count(tg) > 3:
                        is_repetitive = True
                        break

            status = "coherent" if (has_keyword and not is_repetitive) else "degraded"
            print(f"  [{status}] Q{prompt_idx}: {gen_text[:100]}...")

            cfg_results.append({
                "prompt_idx": prompt_idx,
                "status": status,
                "has_keyword": has_keyword,
                "is_repetitive": is_repetitive,
                "response_preview": gen_text[:200],
            })

            # Clean up cache
            del cache
            torch.cuda.empty_cache()

        coherent_count = sum(1 for r in cfg_results if r["status"] == "coherent")
        gen_results.append({
            "name": cfg["name"],
            "coherent": coherent_count,
            "total": len(cfg_results),
            "coherent_rate": coherent_count / len(cfg_results),
            "details": cfg_results,
        })

    return gen_results


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def format_results_markdown(
    kv_results: List[Dict[str, Any]],
    gen_results: List[Dict[str, Any]],
) -> str:
    """Format all results as a markdown report."""
    lines = [
        "# Ultra-Compression Results: 1-Bit KV Cache Experiments",
        "",
        f"**Model:** {MODEL_NAME}",
        f"**Device:** {DEVICE}",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Approach Summary",
        "",
        "| # | Approach | Bits/coord | Key Idea |",
        "|---|----------|-----------|----------|",
        "| 0 | 3-bit ResidualQuant | 3 | Current production baseline |",
        "| 1 | 1-bit MSE only | 1 | Lower bound (no residual) |",
        "| 2 | 1-bit + residual sign | 2 | Simple 1-bit base |",
        "| 3 | Multi-scale chain | N x 1-bit | Cascaded residual stages |",
        "| 4 | Sign prediction | ~1.7 | Predict signs from neighbors |",
        "| 5 | Attention-gated | 1-4 adaptive | Refine high-attention tokens |",
        "",
    ]

    # Group results by layer
    layers = sorted(set(r["layer"] for r in kv_results))
    for layer in layers:
        layer_results = [r for r in kv_results if r["layer"] == layer]
        lines.extend([
            f"## Layer {layer} Results",
            "",
            "| Approach | Bits | Cosine Sim | Top-1 | Top-5 | IP MSE |",
            "|----------|------|------------|-------|-------|--------|",
        ])
        for r in layer_results:
            bits_str = f"{r['bits']:.1f}" if isinstance(r['bits'], float) else str(r['bits'])
            lines.append(
                f"| {r['name']} | {bits_str} | {r['cosine_sim']:.4f} | "
                f"{r['top1_match']:.1%} | {r['top5_match']:.1%} | "
                f"{r['ip_mse']:.6f} |"
            )
        lines.append("")

    # Aggregated across layers
    lines.extend([
        "## Aggregated Results (Mean Across Layers)",
        "",
        "| Approach | Bits | Cosine Sim | Top-1 | Top-5 | IP MSE |",
        "|----------|------|------------|-------|-------|--------|",
    ])

    # Group by approach name
    approaches = {}
    for r in kv_results:
        key = r["approach"]
        if key not in approaches:
            approaches[key] = {"results": [], "name": r["name"], "bits": r["bits"]}
        approaches[key]["results"].append(r)

    for key, data in approaches.items():
        n = len(data["results"])
        avg_cos = sum(r["cosine_sim"] for r in data["results"]) / n
        avg_top1 = sum(r["top1_match"] for r in data["results"]) / n
        avg_top5 = sum(r["top5_match"] for r in data["results"]) / n
        avg_mse = sum(r["ip_mse"] for r in data["results"]) / n
        bits = data["bits"]
        bits_str = f"{bits:.1f}" if isinstance(bits, float) else str(bits)
        lines.append(
            f"| {data['name']} | {bits_str} | {avg_cos:.4f} | "
            f"{avg_top1:.1%} | {avg_top5:.1%} | {avg_mse:.6f} |"
        )
    lines.append("")

    # Breakthrough check
    lines.extend([
        "## Breakthrough Analysis",
        "",
        "**Threshold:** >90% top-5 attention match at effective 1-bit is a breakthrough.",
        "",
    ])

    # Find any approach with <2 effective bits and >90% top-5
    breakthroughs = []
    for key, data in approaches.items():
        avg_top5 = sum(r["top5_match"] for r in data["results"]) / len(data["results"])
        bits = data["bits"]
        eff_bits = bits if isinstance(bits, (int, float)) else 2.0
        if eff_bits < 2.5 and avg_top5 > 0.90:
            breakthroughs.append((data["name"], eff_bits, avg_top5))

    if breakthroughs:
        lines.append("**BREAKTHROUGH DETECTED:**")
        for name, bits, top5 in breakthroughs:
            lines.append(f"- {name}: {bits:.1f} bits, {top5:.1%} top-5 match")
    else:
        lines.append("No approaches achieved >90% top-5 at <2.5 effective bits.")

    lines.append("")

    # Generation results
    if gen_results:
        lines.extend([
            "## Generation Quality",
            "",
            "| Method | Coherent | Total | Rate |",
            "|--------|----------|-------|------|",
        ])
        for g in gen_results:
            lines.append(
                f"| {g['name']} | {g['coherent']} | {g['total']} | "
                f"{g['coherent_rate']:.0%} |"
            )
        lines.append("")

        # Show generation samples
        for g in gen_results:
            lines.append(f"### {g['name']}")
            for d in g["details"]:
                status_marker = "PASS" if d["status"] == "coherent" else "FAIL"
                lines.append(f"- [{status_marker}] Q{d['prompt_idx']}: "
                           f"`{d['response_preview'][:120]}...`")
            lines.append("")

    # Key findings
    lines.extend([
        "## Key Findings",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 75)
    print("  ULTRA-COMPRESSION BENCHMARK: 1-Bit KV Cache Experiments")
    print("=" * 75)

    # Phase 1: Load model and extract KV caches
    model, tokenizer = load_model()
    kv = extract_kv_caches(model, tokenizer, PROMPT)

    # Phase 2: Run KV-level evaluation (cosine sim, attention match)
    print("\n" + "=" * 75)
    print("  PHASE 1: KV Cache Quality Evaluation")
    print("=" * 75)
    kv_results = run_kv_evaluation(kv)

    # Free KV cache memory
    del kv
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 3: Generation quality
    print("\n" + "=" * 75)
    print("  PHASE 2: Autoregressive Generation Quality")
    print("=" * 75)
    gen_results = test_generation_quality(model, tokenizer)

    # Phase 4: Save results
    results_dir = Path(REPO_ROOT) / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    md_report = format_results_markdown(kv_results, gen_results)
    md_path = results_dir / "ultra_compress_results.md"
    md_path.write_text(md_report)
    print(f"\nResults saved to {md_path}")

    # Also save raw JSON
    json_data = {
        "kv_results": kv_results,
        "gen_results": gen_results,
        "config": {
            "model": MODEL_NAME,
            "device": DEVICE,
            "seed": SEED,
            "generation_tokens": GENERATION_TOKENS,
        },
    }
    json_path = results_dir / "ultra_compress_results.json"
    json_path.write_text(json.dumps(json_data, indent=2, default=str))
    print(f"Raw data saved to {json_path}")

    # Print summary
    print("\n" + "=" * 75)
    print("  SUMMARY")
    print("=" * 75)

    # Aggregate top-5 match across layers per approach
    approaches = {}
    for r in kv_results:
        key = r["approach"]
        if key not in approaches:
            approaches[key] = {"results": [], "name": r["name"], "bits": r["bits"]}
        approaches[key]["results"].append(r)

    print(f"\n{'Approach':<50} {'Bits':>5} {'Top-5':>7}")
    print("-" * 65)
    for key, data in sorted(approaches.items(), key=lambda x: -sum(r["top5_match"] for r in x[1]["results"])/len(x[1]["results"])):
        n = len(data["results"])
        avg_top5 = sum(r["top5_match"] for r in data["results"]) / n
        bits = data["bits"]
        bits_str = f"{bits:.1f}" if isinstance(bits, float) else str(bits)
        print(f"{data['name']:<50} {bits_str:>5} {avg_top5:>6.1%}")

    if gen_results:
        print(f"\n{'Generation Method':<40} {'Coherent':>10}")
        print("-" * 55)
        for g in gen_results:
            print(f"{g['name']:<40} {g['coherent']}/{g['total']:>8}")


if __name__ == "__main__":
    main()
