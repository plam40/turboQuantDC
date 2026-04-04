"""Spectral KV Cache Compression — Research Benchmark.

Extracts real KV caches from Qwen2.5-3B-Instruct, then analyzes:
1. DCT energy distribution of key and value vectors
2. SVD singular value spectrum per layer
3. DCT compression quality at various keep-K settings
4. SVD subspace projection quality at various k settings
5. Hybrid SVD+DCT compression
6. Head-to-head comparison vs 3-bit ResidualQuant baseline

Saves results to benchmarks/results/spectral_results.md

Usage:
    python benchmarks/spectral_analysis.py
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from turboquantdc.spectral_compress import (
    DCTCompressor,
    SVDCompressor,
    HybridCompressor,
    analyze_energy_spectrum,
    analyze_svd_spectrum,
    compute_quality_metrics,
    dct_scipy,
    idct_scipy,
)
from turboquantdc.residual_quant import ResidualQuantEstimator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Use 0.5B model in FP16 to fit in limited GPU memory; same Qwen2.5 architecture
# with d=64 head_dim (smaller but same structural properties as 3B's d=128)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CACHE_DIR = "/media/dhawal/Beast/cache/hub"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test prompt — enough context for meaningful KV statistics
PROMPT = (
    "The quarterly financial review meeting covered several topics including "
    "budget allocations for the upcoming fiscal year, departmental spending reports, "
    "and projected revenue streams from various business units. The committee discussed "
    "infrastructure upgrades planned for the western regional offices and noted that "
    "maintenance schedules should be coordinated with the facilities management team. "
    "Several action items were assigned to team leads for follow-up before the next "
    "meeting cycle. The technology division presented their analysis of cloud migration "
    "costs and anticipated savings over a five-year period. Human resources shared "
    "updated hiring projections and retention strategies for critical roles. The CFO "
    "emphasized the importance of maintaining cash reserves while investing in growth "
    "opportunities. Market analysis indicated strong demand in the Asia-Pacific region "
    "with particular growth in renewable energy sectors. The board approved a pilot "
    "program for automated quality assurance in manufacturing facilities. Research and "
    "development teams showcased prototypes for next-generation products expected to "
    "launch in Q3. Customer feedback surveys highlighted satisfaction improvements in "
    "technical support response times and product reliability metrics. Supply chain "
    "disruptions in the semiconductor sector continue to affect production timelines "
    "for several product lines. The legal department provided updates on pending "
    "intellectual property cases and recommended additional patent filings in emerging "
    "technology areas. Environmental sustainability initiatives exceeded their annual "
    "targets with a thirty percent reduction in carbon emissions across all facilities."
)


# ---------------------------------------------------------------------------
# KV Cache Extraction
# ---------------------------------------------------------------------------

def extract_kv_caches(prompt: str) -> Dict[str, torch.Tensor]:
    """Run model on prompt and extract all KV caches.

    Returns:
        Dict mapping 'keys_layer_{i}' and 'values_layer_{i}' to tensors
        of shape (num_kv_heads, seq_len, head_dim).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {MODEL_NAME} (CPU, float32 for analysis)...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]
    print(f"Prompt tokenized to {seq_len} tokens.")

    # Forward pass to capture KV cache
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_kv = outputs.past_key_values
    # Handle DynamicCache (transformers >= 4.45) and legacy tuple format
    if hasattr(past_kv, 'layers'):
        n_layers = len(past_kv.layers)
        result = {}
        for i, layer in enumerate(past_kv.layers):
            result[f"keys_layer_{i}"] = layer.keys[0].detach().float().cpu()
            result[f"values_layer_{i}"] = layer.values[0].detach().float().cpu()
    else:
        n_layers = len(past_kv)
        result = {}
        for i, (k, v) in enumerate(past_kv):
            result[f"keys_layer_{i}"] = k[0].detach().float().cpu()
            result[f"values_layer_{i}"] = v[0].detach().float().cpu()

    print(f"Extracted KV caches from {n_layers} layers.")
    k0 = result["keys_layer_0"]
    print(f"  Shape per layer: ({k0.shape[0]} heads, {k0.shape[1]} tokens, d={k0.shape[2]})")

    # Free model memory
    del model, outputs, past_kv
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Experiment 1: DCT Energy Spectrum Analysis
# ---------------------------------------------------------------------------

def experiment_dct_energy(
    kv_caches: Dict[str, torch.Tensor],
) -> Dict[str, any]:
    """Analyze DCT energy distribution across all layers.

    Key question: How concentrated is the frequency spectrum of real KV vectors?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: DCT Energy Spectrum Analysis")
    print("=" * 70)

    results = {"keys": {}, "values": {}}

    n_layers = sum(1 for k in kv_caches if k.startswith("keys_"))

    for kind in ["keys", "values"]:
        all_profiles = []
        for layer_idx in range(n_layers):
            tensor = kv_caches[f"{kind}_layer_{layer_idx}"]
            # Flatten heads: (heads, seq, d) -> (heads*seq, d)
            flat = tensor.reshape(-1, tensor.shape[-1])
            profile = analyze_energy_spectrum(flat, use_scipy=True)
            all_profiles.append(profile)

        # Aggregate across layers
        avg_k90 = np.mean([p.k_for_90 for p in all_profiles])
        avg_k95 = np.mean([p.k_for_95 for p in all_profiles])
        avg_k99 = np.mean([p.k_for_99 for p in all_profiles])
        avg_k999 = np.mean([p.k_for_999 for p in all_profiles])

        d = tensor.shape[-1]
        results[kind] = {
            "d": d,
            "avg_k_for_90": avg_k90,
            "avg_k_for_95": avg_k95,
            "avg_k_for_99": avg_k99,
            "avg_k_for_999": avg_k999,
            "pct_90": 100 * avg_k90 / d,
            "pct_95": 100 * avg_k95 / d,
            "pct_99": 100 * avg_k99 / d,
            "pct_999": 100 * avg_k999 / d,
            "per_layer_k90": [p.k_for_90 for p in all_profiles],
            "per_layer_k95": [p.k_for_95 for p in all_profiles],
            "per_layer_k99": [p.k_for_99 for p in all_profiles],
            "per_layer_energy": [p.energy_per_coeff for p in all_profiles],
        }

        print(f"\n--- {kind.upper()} ---")
        print(f"  Dimension: {d}")
        print(f"  Coefficients for 90% energy: {avg_k90:.1f}/{d} ({100*avg_k90/d:.1f}%)")
        print(f"  Coefficients for 95% energy: {avg_k95:.1f}/{d} ({100*avg_k95/d:.1f}%)")
        print(f"  Coefficients for 99% energy: {avg_k99:.1f}/{d} ({100*avg_k99/d:.1f}%)")
        print(f"  Coefficients for 99.9% energy: {avg_k999:.1f}/{d} ({100*avg_k999/d:.1f}%)")
        print(f"  Range across layers (90%):")
        print(f"    min={min(results[kind]['per_layer_k90'])}, "
              f"max={max(results[kind]['per_layer_k90'])}")

    return results


# ---------------------------------------------------------------------------
# Experiment 2: SVD Singular Value Spectrum
# ---------------------------------------------------------------------------

def experiment_svd_spectrum(
    kv_caches: Dict[str, torch.Tensor],
) -> Dict[str, any]:
    """Analyze singular value spectrum of KV caches per layer.

    Key question: How much variance is captured by a low-rank subspace?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: SVD Singular Value Spectrum")
    print("=" * 70)

    results = {"keys": {}, "values": {}}
    n_layers = sum(1 for k in kv_caches if k.startswith("keys_"))

    for kind in ["keys", "values"]:
        all_svd = []
        for layer_idx in range(n_layers):
            tensor = kv_caches[f"{kind}_layer_{layer_idx}"]
            flat = tensor.reshape(-1, tensor.shape[-1])
            svd_info = analyze_svd_spectrum(flat)
            all_svd.append(svd_info)

        d = tensor.shape[-1]
        avg_k90 = np.mean([s["k_for_90"] for s in all_svd])
        avg_k95 = np.mean([s["k_for_95"] for s in all_svd])
        avg_k99 = np.mean([s["k_for_99"] for s in all_svd])

        results[kind] = {
            "d": d,
            "avg_k_for_90": avg_k90,
            "avg_k_for_95": avg_k95,
            "avg_k_for_99": avg_k99,
            "pct_90": 100 * avg_k90 / d,
            "pct_95": 100 * avg_k95 / d,
            "pct_99": 100 * avg_k99 / d,
            "per_layer_k90": [s["k_for_90"] for s in all_svd],
            "per_layer_k95": [s["k_for_95"] for s in all_svd],
            "per_layer_k99": [s["k_for_99"] for s in all_svd],
            "per_layer_svd": all_svd,
        }

        print(f"\n--- {kind.upper()} ---")
        print(f"  Dimension: {d}")
        print(f"  SVD components for 90% variance: {avg_k90:.1f}/{d} ({100*avg_k90/d:.1f}%)")
        print(f"  SVD components for 95% variance: {avg_k95:.1f}/{d} ({100*avg_k95/d:.1f}%)")
        print(f"  SVD components for 99% variance: {avg_k99:.1f}/{d} ({100*avg_k99/d:.1f}%)")
        print(f"  Range across layers (90%):")
        print(f"    min={min(results[kind]['per_layer_k90'])}, "
              f"max={max(results[kind]['per_layer_k90'])}")

    return results


# ---------------------------------------------------------------------------
# Experiment 3: DCT Compression Quality
# ---------------------------------------------------------------------------

def experiment_dct_compression(
    kv_caches: Dict[str, torch.Tensor],
) -> Dict[str, any]:
    """Test DCT compression at various keep-K settings.

    Sweep K from 10% to 90% of d and measure reconstruction quality.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: DCT Compression Quality")
    print("=" * 70)

    # Get dimensions
    sample = kv_caches["keys_layer_0"]
    d = sample.shape[-1]
    n_layers = sum(1 for k in kv_caches if k.startswith("keys_"))

    keep_fractions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90]
    keep_ks = [max(1, int(f * d)) for f in keep_fractions]

    results = {}

    for kind in ["keys", "values"]:
        print(f"\n--- {kind.upper()} ---")
        sweep_results = []

        for keep_k in keep_ks:
            compressor = DCTCompressor(d, keep_k, value_bits=16, use_scipy=True)
            bits_per_dim = compressor.bits_per_dim()

            all_cos = []
            all_mse = []
            all_attn_cos = []
            all_top1 = []

            for layer_idx in range(n_layers):
                tensor = kv_caches[f"{kind}_layer_{layer_idx}"]
                flat = tensor.reshape(-1, d)

                # Compress and decompress
                compressed = compressor.compress(flat)
                reconstructed = compressor.decompress(compressed)

                # Generate random queries for attention quality
                queries = torch.randn(32, d) * flat.std()

                metrics = compute_quality_metrics(flat, reconstructed, queries)
                all_cos.append(metrics["cosine_sim_mean"])
                all_mse.append(metrics["mse"])
                if "attention_cosine_mean" in metrics:
                    all_attn_cos.append(metrics["attention_cosine_mean"])
                if "top1_match" in metrics:
                    all_top1.append(metrics["top1_match"])

            avg_cos = np.mean(all_cos)
            avg_mse = np.mean(all_mse)
            avg_attn = np.mean(all_attn_cos) if all_attn_cos else 0
            avg_top1 = np.mean(all_top1) if all_top1 else 0

            entry = {
                "keep_k": keep_k,
                "keep_frac": keep_k / d,
                "bits_per_dim": bits_per_dim,
                "compression_ratio": 16.0 / bits_per_dim,
                "cosine_sim": avg_cos,
                "mse": avg_mse,
                "attention_cosine": avg_attn,
                "top1_match": avg_top1,
            }
            sweep_results.append(entry)

            print(f"  K={keep_k:3d}/{d} ({100*keep_k/d:4.0f}%) | "
                  f"bits/dim={bits_per_dim:.2f} | "
                  f"cos={avg_cos:.6f} | "
                  f"attn_cos={avg_attn:.4f} | "
                  f"top1={avg_top1:.3f}")

        results[kind] = sweep_results

    return results


# ---------------------------------------------------------------------------
# Experiment 4: SVD Compression Quality
# ---------------------------------------------------------------------------

def experiment_svd_compression(
    kv_caches: Dict[str, torch.Tensor],
) -> Dict[str, any]:
    """Test SVD subspace compression at various k settings.

    For each layer, fit SVD on the KV vectors, then measure reconstruction
    quality at various subspace dimensions.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: SVD Subspace Compression Quality")
    print("=" * 70)

    sample = kv_caches["keys_layer_0"]
    d = sample.shape[-1]
    n_layers = sum(1 for k in kv_caches if k.startswith("keys_"))

    k_fractions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75]
    k_dims = [max(1, int(f * d)) for f in k_fractions]

    results = {}

    for kind in ["keys", "values"]:
        print(f"\n--- {kind.upper()} ---")
        sweep_results = []

        for k_dim in k_dims:
            all_cos = []
            all_mse = []
            all_attn_cos = []
            all_top1 = []
            all_explained = []

            for layer_idx in range(n_layers):
                tensor = kv_caches[f"{kind}_layer_{layer_idx}"]
                flat = tensor.reshape(-1, d)
                n_vecs = flat.shape[0]

                compressor = SVDCompressor(d, k_dim, value_bits=16)
                fit_stats = compressor.fit(flat)
                all_explained.append(fit_stats["explained_variance"])

                compressed = compressor.compress(flat)
                reconstructed = compressor.decompress(compressed)

                queries = torch.randn(32, d) * flat.std()
                metrics = compute_quality_metrics(flat, reconstructed, queries)

                all_cos.append(metrics["cosine_sim_mean"])
                all_mse.append(metrics["mse"])
                if "attention_cosine_mean" in metrics:
                    all_attn_cos.append(metrics["attention_cosine_mean"])
                if "top1_match" in metrics:
                    all_top1.append(metrics["top1_match"])

            bits_per_dim = SVDCompressor(d, k_dim).bits_per_dim(n_vecs)
            avg_cos = np.mean(all_cos)
            avg_mse = np.mean(all_mse)
            avg_attn = np.mean(all_attn_cos) if all_attn_cos else 0
            avg_top1 = np.mean(all_top1) if all_top1 else 0
            avg_expl = np.mean(all_explained)

            entry = {
                "k_dim": k_dim,
                "k_frac": k_dim / d,
                "bits_per_dim": bits_per_dim,
                "compression_ratio": 16.0 / bits_per_dim,
                "explained_variance": avg_expl,
                "cosine_sim": avg_cos,
                "mse": avg_mse,
                "attention_cosine": avg_attn,
                "top1_match": avg_top1,
            }
            sweep_results.append(entry)

            print(f"  k={k_dim:3d}/{d} ({100*k_dim/d:4.0f}%) | "
                  f"bits/dim={bits_per_dim:.2f} | "
                  f"expl={avg_expl:.4f} | "
                  f"cos={avg_cos:.6f} | "
                  f"attn_cos={avg_attn:.4f} | "
                  f"top1={avg_top1:.3f}")

        results[kind] = sweep_results

    return results


# ---------------------------------------------------------------------------
# Experiment 5: ResidualQuant Baseline
# ---------------------------------------------------------------------------

def experiment_residualquant_baseline(
    kv_caches: Dict[str, torch.Tensor],
) -> Dict[str, any]:
    """3-bit ResidualQuant baseline for comparison.

    This is the current state-of-the-art in this codebase.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: ResidualQuant Baseline (3-bit)")
    print("=" * 70)

    sample = kv_caches["keys_layer_0"]
    d = sample.shape[-1]
    n_layers = sum(1 for k in kv_caches if k.startswith("keys_"))

    results = {}

    for bits in [2, 3, 4]:
        print(f"\n--- {bits}-bit ResidualQuant ---")

        for kind in ["keys"]:
            all_cos = []
            all_mse = []
            all_attn_cos = []
            all_top1 = []

            rq = ResidualQuantEstimator(d=d, bits=bits, seed=42, device="cpu")

            for layer_idx in range(n_layers):
                tensor = kv_caches[f"{kind}_layer_{layer_idx}"]
                flat = tensor.reshape(-1, d)

                # Quantize and dequantize
                compressed = rq.quantize(flat)
                reconstructed = rq.dequantize(compressed)

                queries = torch.randn(32, d) * flat.std()
                metrics = compute_quality_metrics(flat, reconstructed, queries)

                all_cos.append(metrics["cosine_sim_mean"])
                all_mse.append(metrics["mse"])
                if "attention_cosine_mean" in metrics:
                    all_attn_cos.append(metrics["attention_cosine_mean"])
                if "top1_match" in metrics:
                    all_top1.append(metrics["top1_match"])

            avg_cos = np.mean(all_cos)
            avg_mse = np.mean(all_mse)
            avg_attn = np.mean(all_attn_cos) if all_attn_cos else 0
            avg_top1 = np.mean(all_top1) if all_top1 else 0

            # ResidualQuant storage: b*d + 32 bits per vector
            bits_per_dim = bits + 32.0 / d

            results[f"{bits}bit_{kind}"] = {
                "bits": bits,
                "bits_per_dim": bits_per_dim,
                "compression_ratio": 16.0 / bits_per_dim,
                "cosine_sim": avg_cos,
                "mse": avg_mse,
                "attention_cosine": avg_attn,
                "top1_match": avg_top1,
            }

            print(f"  {kind}: bits/dim={bits_per_dim:.2f} | "
                  f"cos={avg_cos:.6f} | "
                  f"attn_cos={avg_attn:.4f} | "
                  f"top1={avg_top1:.3f}")

    return results


# ---------------------------------------------------------------------------
# Experiment 6: Hybrid SVD + DCT
# ---------------------------------------------------------------------------

def experiment_hybrid(
    kv_caches: Dict[str, torch.Tensor],
) -> Dict[str, any]:
    """Test hybrid SVD + DCT residual compression."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Hybrid SVD + DCT Residual")
    print("=" * 70)

    sample = kv_caches["keys_layer_0"]
    d = sample.shape[-1]
    n_layers = sum(1 for k in kv_caches if k.startswith("keys_"))

    configs = [
        (16, 16),  # svd_k=16, dct_keep=16 residual coeffs
        (32, 16),
        (32, 32),
        (48, 16),
        (48, 32),
        (64, 16),
        (64, 32),
    ]

    results = []
    kind = "keys"

    for svd_k, dct_keep in configs:
        all_cos = []
        all_attn_cos = []
        all_top1 = []

        for layer_idx in range(n_layers):
            tensor = kv_caches[f"{kind}_layer_{layer_idx}"]
            flat = tensor.reshape(-1, d)

            hybrid = HybridCompressor(d, svd_k, dct_keep, value_bits=16)
            hybrid.fit(flat)

            compressed = hybrid.compress(flat)
            reconstructed = hybrid.decompress(compressed)

            queries = torch.randn(32, d) * flat.std()
            metrics = compute_quality_metrics(flat, reconstructed, queries)

            all_cos.append(metrics["cosine_sim_mean"])
            if "attention_cosine_mean" in metrics:
                all_attn_cos.append(metrics["attention_cosine_mean"])
            if "top1_match" in metrics:
                all_top1.append(metrics["top1_match"])

        # Approximate bits per dim for hybrid
        # SVD: k * 16 bits per vector + shared V_k overhead
        # DCT residual: dct_keep * (7 + 16) bits per vector
        svd_bits = svd_k * 16
        dct_bits = dct_keep * (math.ceil(math.log2(d)) + 16)
        total_bits = svd_bits + dct_bits
        bpd = total_bits / d

        avg_cos = np.mean(all_cos)
        avg_attn = np.mean(all_attn_cos) if all_attn_cos else 0
        avg_top1 = np.mean(all_top1) if all_top1 else 0

        entry = {
            "svd_k": svd_k,
            "dct_keep": dct_keep,
            "bits_per_dim": bpd,
            "compression_ratio": 16.0 / bpd,
            "cosine_sim": avg_cos,
            "attention_cosine": avg_attn,
            "top1_match": avg_top1,
        }
        results.append(entry)

        print(f"  SVD_k={svd_k:2d} + DCT_K={dct_keep:2d} | "
              f"bits/dim={bpd:.2f} | "
              f"cos={avg_cos:.6f} | "
              f"attn_cos={avg_attn:.4f} | "
              f"top1={avg_top1:.3f}")

    return {"keys": results}


# ---------------------------------------------------------------------------
# Experiment 7: Frequency-order vs magnitude-order DCT
# ---------------------------------------------------------------------------

def experiment_frequency_order(
    kv_caches: Dict[str, torch.Tensor],
) -> Dict[str, any]:
    """Compare keeping low-frequency vs top-magnitude DCT coefficients.

    Two strategies:
    1. Low-frequency: Keep first K coefficients (indices 0..K-1)
    2. Top-magnitude: Keep K largest coefficients by absolute value

    If KV vectors have smooth structure, low-frequency should work well.
    If they have distributed energy, top-magnitude is better.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Low-Frequency vs Top-Magnitude DCT")
    print("=" * 70)

    sample = kv_caches["keys_layer_0"]
    d = sample.shape[-1]
    n_layers = sum(1 for k in kv_caches if k.startswith("keys_"))

    keep_fractions = [0.15, 0.25, 0.40, 0.50, 0.75]
    results = {"low_freq": [], "top_mag": []}

    for frac in keep_fractions:
        keep_k = max(1, int(frac * d))

        for strategy in ["low_freq", "top_mag"]:
            all_cos = []

            for layer_idx in range(n_layers):
                tensor = kv_caches[f"keys_layer_{layer_idx}"]
                flat = tensor.reshape(-1, d)
                flat_np = flat.numpy()

                coeffs = dct_scipy(flat_np)

                if strategy == "low_freq":
                    # Zero out high frequencies
                    sparse = np.zeros_like(coeffs)
                    sparse[:, :keep_k] = coeffs[:, :keep_k]
                else:
                    # Keep top-K by magnitude
                    sparse = np.zeros_like(coeffs)
                    for i in range(coeffs.shape[0]):
                        top_idx = np.argsort(np.abs(coeffs[i]))[-keep_k:]
                        sparse[i, top_idx] = coeffs[i, top_idx]

                recon = idct_scipy(sparse)
                recon_t = torch.from_numpy(recon).float()

                cos = F.cosine_similarity(flat, recon_t, dim=-1).mean().item()
                all_cos.append(cos)

            avg_cos = np.mean(all_cos)
            results[strategy].append({
                "keep_k": keep_k,
                "keep_frac": frac,
                "cosine_sim": avg_cos,
            })

        lf_cos = results["low_freq"][-1]["cosine_sim"]
        tm_cos = results["top_mag"][-1]["cosine_sim"]
        print(f"  K={keep_k:3d}/{d} ({100*frac:4.0f}%) | "
              f"low_freq cos={lf_cos:.6f} | top_mag cos={tm_cos:.6f} | "
              f"winner={'top_mag' if tm_cos > lf_cos else 'low_freq'}")

    return results


# ---------------------------------------------------------------------------
# Experiment 8: Per-head vs per-layer analysis
# ---------------------------------------------------------------------------

def experiment_per_head(
    kv_caches: Dict[str, torch.Tensor],
) -> Dict[str, any]:
    """Check if energy concentration varies across attention heads.

    If some heads have more concentrated spectra, we could use adaptive
    compression — fewer coefficients for concentrated heads.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Per-Head Energy Analysis (select layers)")
    print("=" * 70)

    n_layers = sum(1 for k in kv_caches if k.startswith("keys_"))
    # Sample a few layers
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    sample_layers = sorted(set(min(l, n_layers - 1) for l in sample_layers))

    results = {}

    for layer_idx in sample_layers:
        tensor = kv_caches[f"keys_layer_{layer_idx}"]
        n_heads = tensor.shape[0]
        d = tensor.shape[-1]

        head_k90s = []
        head_k95s = []

        for head_idx in range(n_heads):
            head_vecs = tensor[head_idx]  # (seq, d)
            profile = analyze_energy_spectrum(head_vecs, use_scipy=True)
            head_k90s.append(profile.k_for_90)
            head_k95s.append(profile.k_for_95)

        results[f"layer_{layer_idx}"] = {
            "k90_per_head": head_k90s,
            "k95_per_head": head_k95s,
            "k90_mean": np.mean(head_k90s),
            "k90_std": np.std(head_k90s),
            "k95_mean": np.mean(head_k95s),
            "k95_std": np.std(head_k95s),
        }

        print(f"  Layer {layer_idx:2d}: "
              f"k90 mean={np.mean(head_k90s):.1f} std={np.std(head_k90s):.1f} "
              f"[{min(head_k90s)}-{max(head_k90s)}] | "
              f"k95 mean={np.mean(head_k95s):.1f} std={np.std(head_k95s):.1f} "
              f"[{min(head_k95s)}-{max(head_k95s)}]")

    return results


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(
    dct_energy: Dict,
    svd_spectrum: Dict,
    dct_compression: Dict,
    svd_compression: Dict,
    rq_baseline: Dict,
    hybrid: Dict,
    freq_order: Dict,
    per_head: Dict,
) -> str:
    """Generate comprehensive markdown report."""

    d = dct_energy["keys"]["d"]

    report = []
    report.append("# Spectral KV Cache Compression — Research Results")
    report.append(f"\n**Model:** {MODEL_NAME}")
    report.append(f"**Head dimension:** d={d}")
    report.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}")

    # --- Experiment 1: Energy Spectrum ---
    report.append("\n## 1. DCT Energy Spectrum Analysis")
    report.append("\nHow many DCT coefficients capture X% of vector energy?")
    report.append("")
    report.append("| Metric | Keys | Values |")
    report.append("|--------|------|--------|")
    for pct in [90, 95, 99]:
        kk = dct_energy["keys"][f"avg_k_for_{pct}"]
        vk = dct_energy["values"][f"avg_k_for_{pct}"]
        kp = dct_energy["keys"][f"pct_{pct}"]
        vp = dct_energy["values"][f"pct_{pct}"]
        report.append(f"| {pct}% energy | {kk:.0f}/{d} ({kp:.1f}%) | {vk:.0f}/{d} ({vp:.1f}%) |")

    # Per-layer variation
    report.append("\n### Per-Layer Variation (90% energy)")
    report.append("| Layer range | Keys (K for 90%) | Values (K for 90%) |")
    report.append("|-------------|------|--------|")
    k90_keys = dct_energy["keys"]["per_layer_k90"]
    k90_vals = dct_energy["values"]["per_layer_k90"]
    report.append(f"| Min | {min(k90_keys)} | {min(k90_vals)} |")
    report.append(f"| Max | {max(k90_keys)} | {max(k90_vals)} |")
    report.append(f"| Mean | {np.mean(k90_keys):.1f} | {np.mean(k90_vals):.1f} |")
    report.append(f"| Std | {np.std(k90_keys):.1f} | {np.std(k90_vals):.1f} |")

    # --- Experiment 2: SVD Spectrum ---
    report.append("\n## 2. SVD Singular Value Spectrum")
    report.append("\nHow many principal components capture X% of variance?")
    report.append("")
    report.append("| Metric | Keys | Values |")
    report.append("|--------|------|--------|")
    for pct in [90, 95, 99]:
        kk = svd_spectrum["keys"][f"avg_k_for_{pct}"]
        vk = svd_spectrum["values"][f"avg_k_for_{pct}"]
        kp = svd_spectrum["keys"][f"pct_{pct}"]
        vp = svd_spectrum["values"][f"pct_{pct}"]
        report.append(f"| {pct}% variance | {kk:.0f}/{d} ({kp:.1f}%) | {vk:.0f}/{d} ({vp:.1f}%) |")

    # --- Experiment 3: DCT Compression ---
    report.append("\n## 3. DCT Compression Quality")
    report.append("\n### Keys")
    report.append("| K kept | % of d | bits/dim | CR | cos sim | attn cos | top-1 match |")
    report.append("|--------|--------|----------|-----|---------|----------|-------------|")
    for e in dct_compression.get("keys", []):
        report.append(
            f"| {e['keep_k']} | {100*e['keep_frac']:.0f}% | "
            f"{e['bits_per_dim']:.2f} | {e['compression_ratio']:.1f}x | "
            f"{e['cosine_sim']:.6f} | {e['attention_cosine']:.4f} | "
            f"{e['top1_match']:.3f} |"
        )

    report.append("\n### Values")
    report.append("| K kept | % of d | bits/dim | CR | cos sim | attn cos | top-1 match |")
    report.append("|--------|--------|----------|-----|---------|----------|-------------|")
    for e in dct_compression.get("values", []):
        report.append(
            f"| {e['keep_k']} | {100*e['keep_frac']:.0f}% | "
            f"{e['bits_per_dim']:.2f} | {e['compression_ratio']:.1f}x | "
            f"{e['cosine_sim']:.6f} | {e['attention_cosine']:.4f} | "
            f"{e['top1_match']:.3f} |"
        )

    # --- Experiment 4: SVD Compression ---
    report.append("\n## 4. SVD Subspace Compression Quality")
    report.append("\n### Keys")
    report.append("| k dim | % of d | bits/dim | CR | var expl | cos sim | attn cos | top-1 |")
    report.append("|-------|--------|----------|-----|----------|---------|----------|-------|")
    for e in svd_compression.get("keys", []):
        report.append(
            f"| {e['k_dim']} | {100*e['k_frac']:.0f}% | "
            f"{e['bits_per_dim']:.2f} | {e['compression_ratio']:.1f}x | "
            f"{e['explained_variance']:.4f} | "
            f"{e['cosine_sim']:.6f} | {e['attention_cosine']:.4f} | "
            f"{e['top1_match']:.3f} |"
        )

    # --- Experiment 5: ResidualQuant Baseline ---
    report.append("\n## 5. ResidualQuant Baseline (for comparison)")
    report.append("| Config | bits/dim | CR | cos sim | attn cos | top-1 |")
    report.append("|--------|----------|-----|---------|----------|-------|")
    for key in sorted(rq_baseline.keys()):
        e = rq_baseline[key]
        report.append(
            f"| {key} | {e['bits_per_dim']:.2f} | {e['compression_ratio']:.1f}x | "
            f"{e['cosine_sim']:.6f} | {e['attention_cosine']:.4f} | "
            f"{e['top1_match']:.3f} |"
        )

    # --- Experiment 6: Hybrid ---
    report.append("\n## 6. Hybrid SVD + DCT Residual (Keys)")
    report.append("| SVD k | DCT K | bits/dim | CR | cos sim | attn cos | top-1 |")
    report.append("|-------|-------|----------|-----|---------|----------|-------|")
    for e in hybrid.get("keys", []):
        report.append(
            f"| {e['svd_k']} | {e['dct_keep']} | "
            f"{e['bits_per_dim']:.2f} | {e['compression_ratio']:.1f}x | "
            f"{e['cosine_sim']:.6f} | {e['attention_cosine']:.4f} | "
            f"{e['top1_match']:.3f} |"
        )

    # --- Experiment 7: Frequency order ---
    report.append("\n## 7. Low-Frequency vs Top-Magnitude DCT (Keys)")
    report.append("| K kept | Low-freq cos | Top-mag cos | Winner |")
    report.append("|--------|-------------|-------------|--------|")
    for lf, tm in zip(freq_order["low_freq"], freq_order["top_mag"]):
        winner = "top-mag" if tm["cosine_sim"] > lf["cosine_sim"] else "low-freq"
        report.append(
            f"| {lf['keep_k']}/{d} ({100*lf['keep_frac']:.0f}%) | "
            f"{lf['cosine_sim']:.6f} | {tm['cosine_sim']:.6f} | {winner} |"
        )

    # --- Experiment 8: Per-head ---
    report.append("\n## 8. Per-Head Energy Variation")
    report.append("| Layer | K90 mean | K90 std | K90 range | K95 mean | K95 range |")
    report.append("|-------|----------|---------|-----------|----------|-----------|")
    for layer_key in sorted(per_head.keys()):
        e = per_head[layer_key]
        k90 = e["k90_per_head"]
        k95 = e["k95_per_head"]
        report.append(
            f"| {layer_key} | {e['k90_mean']:.1f} | {e['k90_std']:.1f} | "
            f"{min(k90)}-{max(k90)} | {e['k95_mean']:.1f} | "
            f"{min(k95)}-{max(k95)} |"
        )

    # --- Summary ---
    report.append("\n## Summary & Conclusions")
    report.append("")

    # Find the best spectral config that matches or beats 3-bit RQ
    rq3 = rq_baseline.get("3bit_keys", {})
    rq3_cos = rq3.get("cosine_sim", 0)
    rq3_bpd = rq3.get("bits_per_dim", 3.25)

    report.append(f"### 3-bit ResidualQuant reference point:")
    report.append(f"- bits/dim: {rq3_bpd:.2f}")
    report.append(f"- cosine sim: {rq3_cos:.6f}")
    report.append("")

    # Check DCT
    better_dct = [e for e in dct_compression.get("keys", [])
                  if e["cosine_sim"] >= rq3_cos * 0.999]
    if better_dct:
        best = min(better_dct, key=lambda e: e["bits_per_dim"])
        report.append(f"### Best DCT matching 3-bit RQ quality:")
        report.append(f"- K={best['keep_k']}, bits/dim={best['bits_per_dim']:.2f}, "
                      f"cos={best['cosine_sim']:.6f}")
        if best["bits_per_dim"] < rq3_bpd:
            savings = (1 - best["bits_per_dim"] / rq3_bpd) * 100
            report.append(f"- **{savings:.1f}% fewer bits** than 3-bit ResidualQuant!")
        else:
            overhead = (best["bits_per_dim"] / rq3_bpd - 1) * 100
            report.append(f"- {overhead:.1f}% MORE bits than ResidualQuant "
                          "(DCT index overhead)")
    else:
        report.append("### DCT: No configuration matched 3-bit RQ quality.")

    report.append("")

    # Check SVD
    better_svd = [e for e in svd_compression.get("keys", [])
                  if e["cosine_sim"] >= rq3_cos * 0.999]
    if better_svd:
        best = min(better_svd, key=lambda e: e["bits_per_dim"])
        report.append(f"### Best SVD matching 3-bit RQ quality:")
        report.append(f"- k={best['k_dim']}, bits/dim={best['bits_per_dim']:.2f}, "
                      f"cos={best['cosine_sim']:.6f}")
        if best["bits_per_dim"] < rq3_bpd:
            savings = (1 - best["bits_per_dim"] / rq3_bpd) * 100
            report.append(f"- **{savings:.1f}% fewer bits** than 3-bit ResidualQuant!")
    else:
        report.append("### SVD: No configuration matched 3-bit RQ quality.")

    report.append("")

    # Key insight
    k90_keys = dct_energy["keys"]["avg_k_for_90"]
    k95_keys = dct_energy["keys"]["avg_k_for_95"]
    report.append("### Key Insight: Energy Concentration")
    report.append(f"- 90% DCT energy in {k90_keys:.0f}/{d} coefficients ({100*k90_keys/d:.1f}% of d)")
    report.append(f"- 95% DCT energy in {k95_keys:.0f}/{d} coefficients ({100*k95_keys/d:.1f}% of d)")

    energy_concentrated = k90_keys < 0.5 * d
    if energy_concentrated:
        report.append(f"- KV vectors DO show frequency-domain concentration.")
        report.append(f"  Spectral compression has theoretical advantage.")
    else:
        report.append(f"- KV vectors do NOT show strong frequency concentration.")
        report.append(f"  After orthogonal rotation, coordinates are nearly independent —")
        report.append(f"  this is expected from the TurboQuant theory (Lemma 1).")
        report.append(f"  The rotation already decorrelates coordinates, leaving little")
        report.append(f"  structure for DCT to exploit.")

    report.append("")
    report.append("### Verdict")

    # Determine verdict
    dct_wins = any(e["bits_per_dim"] < rq3_bpd and e["cosine_sim"] >= rq3_cos * 0.999
                   for e in dct_compression.get("keys", []))
    svd_wins = any(e["bits_per_dim"] < rq3_bpd and e["cosine_sim"] >= rq3_cos * 0.999
                   for e in svd_compression.get("keys", []))

    if dct_wins or svd_wins:
        report.append("**BREAKTHROUGH**: Spectral compression achieves 3-bit quality "
                      "at fewer bits per dimension!")
    else:
        report.append("Spectral compression does NOT beat spatial quantization for "
                      "KV cache compression.")
        report.append("The orthogonal rotation in TurboQuant/ResidualQuant already "
                      "decorrelates coordinates,")
        report.append("leaving minimal frequency-domain structure to exploit. "
                      "The per-coefficient index")
        report.append("overhead of DCT (ceil(log2(d)) bits per kept coefficient) "
                      "negates any sparsity gain.")
        report.append("")
        report.append("SVD subspace projection is more promising because it exploits "
                      "inter-vector correlations")
        report.append("(layer-specific patterns), but requires per-layer fitting "
                      "and shared overhead that")
        report.append("limits compression for short sequences.")

    return "\n".join(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SPECTRAL KV CACHE COMPRESSION — RESEARCH BENCHMARK")
    print("=" * 70)
    t0 = time.time()

    # Extract real KV caches
    kv_caches = extract_kv_caches(PROMPT)

    # Run all experiments
    dct_energy = experiment_dct_energy(kv_caches)
    svd_spectrum = experiment_svd_spectrum(kv_caches)
    dct_compression = experiment_dct_compression(kv_caches)
    svd_compression = experiment_svd_compression(kv_caches)
    rq_baseline = experiment_residualquant_baseline(kv_caches)
    hybrid = experiment_hybrid(kv_caches)
    freq_order = experiment_frequency_order(kv_caches)
    per_head = experiment_per_head(kv_caches)

    # Generate report
    report = generate_report(
        dct_energy, svd_spectrum, dct_compression, svd_compression,
        rq_baseline, hybrid, freq_order, per_head,
    )

    # Save
    results_dir = os.path.join(REPO_ROOT, "benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "spectral_results.md")
    with open(output_path, "w") as f:
        f.write(report)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_path}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
