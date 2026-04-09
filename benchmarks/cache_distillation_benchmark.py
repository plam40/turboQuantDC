#!/usr/bin/env python3
"""Cache distillation + TurboQuant compression benchmark.

Loads Qwen2.5-3B (BnB 4-bit), extracts real Q/K/V at 512+ tokens,
distills the cache at various ratios, then compresses with 3-bit TurboQuant.

Measures:
    1. Distillation quality: attention output cosine vs full cache
    2. Top-5 attention match at each distillation ratio
    3. Combined compression: distillation x TurboQuant
    4. Compressibility: entropy/variance of distilled vs real tokens after WHT

Run:
    python benchmarks/cache_distillation_benchmark.py

Saves results to benchmarks/results/cache_distillation_results.md
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, REPO_ROOT)

from turboquantdc.cache_distillation import CacheDistiller, _kmeans_init
from turboquantdc.residual_quant import ResidualQuantEstimator
from turboquantdc.polarquant import PolarQuant
from turboquantdc.rotation import apply_wht_rotation, generate_wht_rotation

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("DISTILL_MODEL", "Qwen/Qwen2.5-3B-Instruct")
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Distillation ratios to test: 512->128 (4x), 512->64 (8x), 512->32 (16x)
DISTILL_RATIOS = [4, 8, 16]

# Layers to benchmark (early, middle, late)
TEST_LAYERS = [0, 7, 15, 23, 35]

# TurboQuant bits
KEY_BITS = 3
VAL_BITS = 3

# Distillation optimization
DISTILL_STEPS = 50
DISTILL_LR = 0.01

# Long context prompt for 512+ tokens
CONTEXT_PROMPT = """You are an expert research assistant. Below is a collection of research notes about quantum computing. Read all notes carefully and answer the questions that follow.

Note 1: Quantum Error Correction
Quantum error correction (QEC) is essential for building fault-tolerant quantum computers. The surface code, proposed by Kitaev in 1997, is currently the most promising approach due to its high error threshold of approximately 1%. The surface code encodes a single logical qubit using a two-dimensional lattice of physical qubits, where the number of physical qubits scales as O(d^2) for a code distance d. Recent experimental demonstrations by Google's Sycamore processor have shown logical error rates below the threshold for distance-3 and distance-5 surface codes. IBM's Heron processor has demonstrated similar capabilities with their heavy-hexagonal lattice architecture. The key challenge remains scaling to larger code distances while maintaining low physical error rates.

Note 2: Quantum Advantage in Optimization
The quantum approximate optimization algorithm (QAOA), introduced by Farhi, Goldstone, and Gutmann in 2014, is designed to solve combinatorial optimization problems. Despite significant theoretical and experimental progress, definitive quantum advantage for optimization remains elusive. Classical algorithms, particularly simulated annealing and tensor network methods, continue to compete effectively on problems up to several hundred variables. The most promising applications appear to be in structured problems where the quantum speedup is polynomial rather than exponential, such as portfolio optimization and vehicle routing.

Note 3: Quantum Machine Learning
Quantum machine learning (QML) has seen explosive growth, with variational quantum eigensolvers (VQE) and quantum neural networks (QNN) being the most studied paradigms. However, the barren plateau phenomenon, identified by McClean et al. in 2018, poses a fundamental challenge: the gradients of randomly initialized quantum circuits vanish exponentially with system size, making training infeasible for large circuits. Recent work has proposed several mitigation strategies, including layer-wise training, identity-block initialization, and classical-quantum hybrid architectures. The most successful applications to date have been in quantum chemistry, where quantum computers can naturally represent electronic wave functions.

Note 4: Superconducting Qubits
Superconducting qubits, based on Josephson junctions, dominate the current quantum computing landscape. The transmon qubit, an improved charge qubit with reduced sensitivity to charge noise, achieves coherence times exceeding 100 microseconds in state-of-the-art devices. Google, IBM, and Rigetti all use transmon-based architectures. The key challenges include: improving gate fidelities beyond 99.9%, reducing crosstalk between adjacent qubits, scaling to thousands of qubits while maintaining connectivity, and operating at millikelvin temperatures, which requires expensive dilution refrigerators.

Note 5: Trapped Ion Quantum Computing
Trapped ion quantum computers, pioneered by groups at NIST, University of Innsbruck, and companies like IonQ and Quantinuum, offer several advantages over superconducting qubits. They typically achieve higher two-qubit gate fidelities (exceeding 99.9%), longer coherence times (seconds to minutes), and all-to-all connectivity between qubits in a single trap. However, they face challenges in scaling beyond several dozen qubits in a single trap, with proposed solutions including modular architectures using photonic interconnects and shuttling-based approaches. Quantinuum's H2 processor with 56 qubits represents the current state of the art.

Note 6: Quantum Networking and Communication
Quantum networking aims to connect quantum processors via quantum channels, enabling distributed quantum computing and quantum key distribution (QKD). The fundamental challenge is that quantum states cannot be copied (no-cloning theorem), requiring quantum repeaters for long-distance communication. Current QKD implementations achieve secure key rates of several kilobits per second over distances up to 400 km in optical fiber. Satellite-based QKD, demonstrated by the Chinese Micius satellite, has extended this to over 7,600 km. Quantum memory, essential for quantum repeaters, remains a significant bottleneck, with the best atomic ensemble memories achieving storage times of only a few seconds.

Note 7: Photonic Quantum Computing
Photonic quantum computing uses single photons as qubits, with encoding in polarization, time-bin, or path degrees of freedom. Xanadu's Borealis processor demonstrated quantum advantage in Gaussian boson sampling with 216 squeezed-state modes. Linear optical quantum computing faces the challenge that photon-photon interactions are extremely weak, requiring measurement-induced nonlinearity. PsiQuantum is pursuing a large-scale approach using silicon photonics, aiming for a million-qubit fault-tolerant machine.

Now answer these questions based on the notes above:

Question 1: What is the error threshold of the surface code, and which companies have demonstrated experimental results?
Question 2: What fundamental challenge does QAOA face in achieving quantum advantage?
Question 3: Explain the barren plateau phenomenon and list three proposed mitigation strategies.
Question 4: Compare coherence times of superconducting qubits versus trapped ion qubits.
Question 5: What is the current state of quantum key distribution in terms of distance and key rates?"""


# ---------------------------------------------------------------------------
# Model loading and KV extraction
# ---------------------------------------------------------------------------

def load_model():
    """Load Qwen2.5-3B with BnB 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME} (4-bit quantized)...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dt = time.time() - t0
    device = next(model.parameters()).device
    print(f"  Loaded in {dt:.1f}s on {device}")
    return model, tokenizer


def extract_qkv_data(model, tokenizer, prompt: str) -> Dict[str, Any]:
    """Extract Q, K, V from all layers via forward pass + hooks."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]
    print(f"  Prompt tokens: {seq_len}")

    # First pass: get KV cache
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=True)

    kv_cache = outputs.past_key_values
    keys_per_layer = []
    values_per_layer = []

    # Handle DynamicCache or tuple-based cache
    if hasattr(kv_cache, 'key_cache'):
        # transformers >= 4.36 DynamicCache
        for i in range(len(kv_cache.key_cache)):
            keys_per_layer.append(kv_cache.key_cache[i].cpu().float())
            values_per_layer.append(kv_cache.value_cache[i].cpu().float())
    elif hasattr(kv_cache, 'layers'):
        for layer in kv_cache.layers:
            keys_per_layer.append(layer.keys.cpu().float())
            values_per_layer.append(layer.values.cpu().float())
    else:
        # Tuple-based cache
        for layer_kv in kv_cache:
            keys_per_layer.append(layer_kv[0].cpu().float())
            values_per_layer.append(layer_kv[1].cpu().float())

    attn_per_layer = [a.cpu().float() for a in outputs.attentions]

    # Second pass: extract queries via hooks on q_proj
    query_outputs = []

    def q_proj_hook(module, input_args, output):
        query_outputs.append(output.detach().cpu().float())

    hooks = []
    for name, module in model.named_modules():
        if name.endswith(".q_proj"):
            hooks.append(module.register_forward_hook(q_proj_hook))

    with torch.no_grad():
        model(**inputs, output_attentions=False, use_cache=False)

    for h in hooks:
        h.remove()

    # Reshape queries to (batch, n_heads, seq, head_dim)
    config = model.config
    n_heads = getattr(config, "num_attention_heads", 32)
    head_dim = keys_per_layer[0].shape[-1]

    queries_per_layer = []
    for q_raw in query_outputs:
        batch, seq, total_dim = q_raw.shape
        per_head_dim = total_dim // n_heads
        q_reshaped = q_raw.view(batch, seq, n_heads, per_head_dim).transpose(1, 2)
        queries_per_layer.append(q_reshaped)

    n_kv_heads = keys_per_layer[0].shape[1]
    n_layers = len(keys_per_layer)

    # Clamp test layers to available layers
    global TEST_LAYERS
    TEST_LAYERS = [l for l in TEST_LAYERS if l < n_layers]
    if not TEST_LAYERS:
        TEST_LAYERS = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    return {
        "keys": keys_per_layer,
        "values": values_per_layer,
        "queries": queries_per_layer,
        "attention_weights": attn_per_layer,
        "seq_len": seq_len,
        "n_layers": n_layers,
        "head_dim": head_dim,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_attention_output_cosine(
    keys_full: torch.Tensor,
    values_full: torch.Tensor,
    keys_distilled: torch.Tensor,
    values_distilled: torch.Tensor,
    queries: torch.Tensor,
) -> float:
    """Cosine similarity between full and distilled attention outputs.

    Args:
        keys_full: (N, d)
        values_full: (N, d)
        keys_distilled: (M, d)
        values_distilled: (M, d)
        queries: (q, d)

    Returns:
        Mean cosine similarity across queries.
    """
    d = keys_full.shape[-1]
    scale = 1.0 / math.sqrt(d)

    # Full attention output
    attn_full = F.softmax(queries @ keys_full.T * scale, dim=-1)  # (q, N)
    out_full = attn_full @ values_full  # (q, d)

    # Distilled attention output
    attn_dist = F.softmax(queries @ keys_distilled.T * scale, dim=-1)  # (q, M)
    out_dist = attn_dist @ values_distilled  # (q, d)

    # Per-query cosine similarity
    cos = F.cosine_similarity(out_full, out_dist, dim=-1)  # (q,)
    return cos.mean().item()


def compute_top_k_match(
    keys_full: torch.Tensor,
    keys_distilled: torch.Tensor,
    values_full: torch.Tensor,
    values_distilled: torch.Tensor,
    queries: torch.Tensor,
    k: int = 5,
) -> float:
    """Top-K attention output match.

    Instead of matching token indices (different sizes), we check if the
    attention OUTPUT from the distilled cache recovers the same top-K
    query-output directions.

    Returns the fraction of queries where the top-K output dimensions
    match between full and distilled.
    """
    d = keys_full.shape[-1]
    scale = 1.0 / math.sqrt(d)

    attn_full = F.softmax(queries @ keys_full.T * scale, dim=-1)
    out_full = attn_full @ values_full  # (q, d)

    attn_dist = F.softmax(queries @ keys_distilled.T * scale, dim=-1)
    out_dist = attn_dist @ values_distilled  # (q, d)

    # Top-K dimensions by absolute value in the output
    _, top_full = out_full.abs().topk(k, dim=-1)  # (q, k)
    _, top_dist = out_dist.abs().topk(k, dim=-1)  # (q, k)

    # Compute overlap per query
    matches = 0
    total = queries.shape[0]
    for i in range(total):
        full_set = set(top_full[i].tolist())
        dist_set = set(top_dist[i].tolist())
        matches += len(full_set & dist_set) / k

    return matches / total


def compute_compression_ratio(
    n_original: int,
    n_distilled: int,
    d: int,
    quant_bits: int = 3,
) -> Dict[str, float]:
    """Compute compression ratios for distillation and quantization.

    FP16 = 16 bits per coordinate.
    TurboQuant at b bits = b bits per coordinate + overhead.
    Distillation: n_original / n_distilled reduction.

    Returns dict with distill_ratio, quant_ratio, total_ratio.
    """
    fp16_bits = 16
    # TurboQuant storage: b*d bits per vector + 32 bits overhead (norm + scale)
    quant_bits_per_vec = quant_bits * d + 32
    fp16_bits_per_vec = fp16_bits * d

    distill_ratio = n_original / n_distilled
    quant_ratio = fp16_bits_per_vec / quant_bits_per_vec
    total_ratio = distill_ratio * quant_ratio

    return {
        "distill_ratio": distill_ratio,
        "quant_ratio": quant_ratio,
        "total_ratio": total_ratio,
        "total_bits_per_coord": quant_bits * (n_distilled / n_original),
    }


# ---------------------------------------------------------------------------
# Compressibility analysis: distilled vs real tokens
# ---------------------------------------------------------------------------

def analyze_compressibility(
    real_keys: torch.Tensor,
    distilled_keys: torch.Tensor,
    d: int,
) -> Dict[str, Any]:
    """Compare compressibility of distilled vs real tokens after WHT rotation.

    Key hypothesis: distilled tokens live in a lower-dimensional subspace
    and should be MORE compressible (lower entropy, lower variance after WHT).

    Args:
        real_keys: (N, d) original key vectors.
        distilled_keys: (M, d) distilled key vectors.
        d: head dimension.

    Returns:
        Dict with variance, entropy, effective rank metrics for both.
    """
    results = {}

    for name, keys in [("real", real_keys), ("distilled", distilled_keys)]:
        # Normalize
        norms = keys.norm(dim=-1, keepdim=True)
        keys_norm = keys / (norms + 1e-8)

        # Apply WHT rotation
        if d > 0 and (d & (d - 1)) == 0:
            wht_params = generate_wht_rotation(d, seed=42, device=keys.device)
            keys_rotated = apply_wht_rotation(keys_norm, wht_params)
        else:
            keys_rotated = keys_norm

        # Per-coordinate variance after rotation
        coord_var = keys_rotated.var(dim=0)  # (d,)
        mean_var = coord_var.mean().item()
        max_var = coord_var.max().item()
        min_var = coord_var.min().item()
        var_ratio = max_var / (min_var + 1e-10)

        # Coordinate entropy: discretize to 256 bins and compute entropy
        keys_flat = keys_rotated.flatten()
        lo, hi = keys_flat.min().item(), keys_flat.max().item()
        if hi - lo < 1e-8:
            entropy = 0.0
        else:
            bins = 256
            hist = torch.histc(keys_flat, bins=bins, min=lo, max=hi)
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            entropy = -(probs * probs.log2()).sum().item()

        # Effective rank via singular values
        # Effective rank = exp(H(sigma/sum(sigma))) where sigma are SVs
        U, S, Vh = torch.linalg.svd(keys_norm, full_matrices=False)
        s_norm = S / S.sum()
        s_norm = s_norm[s_norm > 1e-10]
        effective_rank = torch.exp(-(s_norm * s_norm.log()).sum()).item()

        # Norm statistics
        norm_mean = norms.mean().item()
        norm_std = norms.std().item()

        results[name] = {
            "mean_var": mean_var,
            "max_var": max_var,
            "min_var": min_var,
            "var_ratio": var_ratio,
            "entropy_bits": entropy,
            "effective_rank": effective_rank,
            "n_vectors": keys.shape[0],
            "norm_mean": norm_mean,
            "norm_std": norm_std,
        }

    return results


# ---------------------------------------------------------------------------
# TurboQuant compression of distilled tokens
# ---------------------------------------------------------------------------

def compress_and_measure(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    keys_full: torch.Tensor,
    values_full: torch.Tensor,
    d: int,
    bits: int = 3,
) -> Dict[str, float]:
    """Compress keys/values with ResidualQuant and measure quality.

    Args:
        keys: (M, d) distilled (or real) keys to compress.
        values: (M, d) corresponding values.
        queries: (q, d) queries for evaluation.
        keys_full: (N, d) original full keys (ground truth).
        values_full: (N, d) original full values (ground truth).
        d: head dimension.
        bits: quantization bits.

    Returns:
        Dict with cosine_pre_quant, cosine_post_quant, quant_distortion.
    """
    device = keys.device

    # Pre-quantization quality (distilled but not quantized)
    cos_pre = compute_attention_output_cosine(
        keys_full, values_full, keys, values, queries,
    )

    # Quantize keys with ResidualQuant
    rq = ResidualQuantEstimator(d=d, bits=bits, seed=SEED, device=str(device))

    compressed = rq.quantize(keys)
    keys_deq = rq.dequantize(compressed)

    # Quantize values with PolarQuant (MSE only, no residual for values)
    pq = PolarQuant(d=d, bits=max(bits - 1, 1), seed=SEED + 100, device=str(device))
    val_indices = pq.quantize(values)
    values_deq = pq.dequantize(val_indices)

    # Post-quantization quality
    cos_post = compute_attention_output_cosine(
        keys_full, values_full, keys_deq, values_deq, queries,
    )

    # Quantization distortion on keys
    key_distortion = F.mse_loss(keys_deq, keys).item()
    val_distortion = F.mse_loss(values_deq, values).item()

    return {
        "cosine_pre_quant": cos_pre,
        "cosine_post_quant": cos_post,
        "key_distortion": key_distortion,
        "val_distortion": val_distortion,
        "quality_drop_from_quant": cos_pre - cos_post,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_distillation_benchmark(data: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full distillation benchmark across layers and ratios."""
    seq_len = data["seq_len"]
    head_dim = data["head_dim"]
    n_kv_heads = data["n_kv_heads"]
    n_heads = data["n_heads"]

    print(f"\n{'='*60}")
    print(f"Cache Distillation Benchmark")
    print(f"{'='*60}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dim: {head_dim}")
    print(f"  KV heads: {n_kv_heads}, Q heads: {n_heads}")
    print(f"  Distillation ratios: {DISTILL_RATIOS}")
    print(f"  Test layers: {TEST_LAYERS}")
    print(f"  TurboQuant bits: K{KEY_BITS}/V{VAL_BITS}")
    print(f"  Optimization: {DISTILL_STEPS} steps, lr={DISTILL_LR}")

    distiller = CacheDistiller(seed=SEED, device=DEVICE)

    all_results = []

    for layer_idx in TEST_LAYERS:
        print(f"\n--- Layer {layer_idx} ---")

        # Extract single head for benchmarking (head 0)
        # Keys/Values: (batch, n_kv_heads, seq, d) -> (seq, d) for head 0
        keys_full = data["keys"][layer_idx][0, 0, :, :].to(DEVICE)  # (N, d)
        values_full = data["values"][layer_idx][0, 0, :, :].to(DEVICE)  # (N, d)

        # Queries: may have more heads (GQA). Use head 0.
        if data["queries"] and layer_idx < len(data["queries"]):
            queries = data["queries"][layer_idx][0, 0, :, :].to(DEVICE)  # (seq, d)
        else:
            # Fallback: approximate from attention
            attn = data["attention_weights"][layer_idx][0, 0, :, :]
            queries = (attn.to(DEVICE) @ keys_full) * math.sqrt(head_dim)

        N = keys_full.shape[0]

        for ratio in DISTILL_RATIOS:
            target_size = max(1, N // ratio)
            print(f"  Distilling {N} -> {target_size} ({ratio}x)...")

            t0 = time.time()
            dk, dv = distiller.distill(
                keys_full, values_full, queries,
                target_size=target_size,
                steps=DISTILL_STEPS,
                lr=DISTILL_LR,
            )
            distill_time = time.time() - t0

            # Attention output cosine similarity
            cos_sim = compute_attention_output_cosine(
                keys_full, values_full, dk, dv, queries,
            )

            # Top-5 match
            top5 = compute_top_k_match(
                keys_full, dk, values_full, dv, queries, k=5,
            )

            # Compression ratios
            comp = compute_compression_ratio(N, target_size, head_dim, KEY_BITS)

            # Compressibility analysis
            compressibility = analyze_compressibility(keys_full, dk, head_dim)

            # Compress distilled tokens with TurboQuant
            quant_results = compress_and_measure(
                dk, dv, queries, keys_full, values_full, head_dim, KEY_BITS,
            )

            result = {
                "layer": layer_idx,
                "ratio": ratio,
                "n_original": N,
                "n_distilled": target_size,
                "distill_time_s": distill_time,
                "cosine_sim": cos_sim,
                "top5_match": top5,
                **comp,
                "real_entropy": compressibility["real"]["entropy_bits"],
                "distilled_entropy": compressibility["distilled"]["entropy_bits"],
                "real_effective_rank": compressibility["real"]["effective_rank"],
                "distilled_effective_rank": compressibility["distilled"]["effective_rank"],
                "real_var": compressibility["real"]["mean_var"],
                "distilled_var": compressibility["distilled"]["mean_var"],
                "real_var_ratio": compressibility["real"]["var_ratio"],
                "distilled_var_ratio": compressibility["distilled"]["var_ratio"],
                **quant_results,
            }
            all_results.append(result)

            print(f"    Time: {distill_time:.2f}s")
            print(f"    Attention output cosine: {cos_sim:.4f}")
            print(f"    Top-5 match: {top5:.3f}")
            print(f"    Compression: distill={comp['distill_ratio']:.1f}x, "
                  f"quant={comp['quant_ratio']:.1f}x, "
                  f"total={comp['total_ratio']:.1f}x")
            print(f"    Entropy: real={compressibility['real']['entropy_bits']:.2f}, "
                  f"distilled={compressibility['distilled']['entropy_bits']:.2f}")
            print(f"    Effective rank: real={compressibility['real']['effective_rank']:.1f}, "
                  f"distilled={compressibility['distilled']['effective_rank']:.1f}")
            print(f"    Post-quant cosine: {quant_results['cosine_post_quant']:.4f}")
            print(f"    Quant quality drop: {quant_results['quality_drop_from_quant']:.4f}")

    # Baseline: uniform 3-bit compression (no distillation)
    print(f"\n--- Baseline: 3-bit TurboQuant only (no distillation) ---")
    baseline_results = []
    for layer_idx in TEST_LAYERS:
        keys_full = data["keys"][layer_idx][0, 0, :, :].to(DEVICE)
        values_full = data["values"][layer_idx][0, 0, :, :].to(DEVICE)
        if data["queries"] and layer_idx < len(data["queries"]):
            queries = data["queries"][layer_idx][0, 0, :, :].to(DEVICE)
        else:
            attn = data["attention_weights"][layer_idx][0, 0, :, :]
            queries = (attn.to(DEVICE) @ keys_full) * math.sqrt(head_dim)

        quant_only = compress_and_measure(
            keys_full, values_full, queries, keys_full, values_full, head_dim, KEY_BITS,
        )
        baseline_results.append({
            "layer": layer_idx,
            "cosine_post_quant": quant_only["cosine_post_quant"],
            "key_distortion": quant_only["key_distortion"],
        })
        print(f"  Layer {layer_idx}: cosine={quant_only['cosine_post_quant']:.4f}, "
              f"distortion={quant_only['key_distortion']:.6f}")

    return {
        "distillation_results": all_results,
        "baseline_results": baseline_results,
        "config": {
            "model": MODEL_NAME,
            "seq_len": data["seq_len"],
            "head_dim": head_dim,
            "n_kv_heads": n_kv_heads,
            "n_heads": n_heads,
            "key_bits": KEY_BITS,
            "val_bits": VAL_BITS,
            "distill_steps": DISTILL_STEPS,
            "distill_lr": DISTILL_LR,
        },
    }


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def format_results(results: Dict[str, Any]) -> str:
    """Format results as markdown."""
    cfg = results["config"]
    lines = []
    lines.append("# Cache Distillation + TurboQuant Compression Results")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Model:** {cfg['model']}")
    lines.append(f"**Sequence length:** {cfg['seq_len']} tokens")
    lines.append(f"**Head dim:** {cfg['head_dim']}, KV heads: {cfg['n_kv_heads']}, Q heads: {cfg['n_heads']}")
    lines.append(f"**TurboQuant:** K{cfg['key_bits']}/V{cfg['val_bits']}")
    lines.append(f"**Distillation:** {DISTILL_STEPS} steps, lr={DISTILL_LR}")
    lines.append("")

    # Summary table
    lines.append("## Distillation Quality")
    lines.append("")
    lines.append("| Layer | Ratio | N->M | Attn Cosine | Top-5 | Distill x | Quant x | Total x | Post-Quant Cos |")
    lines.append("|-------|-------|------|-------------|-------|-----------|---------|---------|----------------|")

    for r in results["distillation_results"]:
        lines.append(
            f"| {r['layer']:>5} | {r['ratio']:>5}x | {r['n_original']}->{r['n_distilled']} "
            f"| {r['cosine_sim']:.4f} | {r['top5_match']:.3f} "
            f"| {r['distill_ratio']:.1f}x | {r['quant_ratio']:.1f}x "
            f"| {r['total_ratio']:.1f}x | {r['cosine_post_quant']:.4f} |"
        )

    lines.append("")

    # Baseline comparison
    lines.append("## Baseline: 3-bit TurboQuant Only (5.0x)")
    lines.append("")
    lines.append("| Layer | Cosine | Key Distortion |")
    lines.append("|-------|--------|----------------|")
    for r in results["baseline_results"]:
        lines.append(f"| {r['layer']:>5} | {r['cosine_post_quant']:.4f} | {r['key_distortion']:.6f} |")

    lines.append("")

    # Compressibility analysis
    lines.append("## Compressibility: Distilled vs Real Tokens")
    lines.append("")
    lines.append("Key hypothesis: distilled tokens live in a lower-dimensional subspace")
    lines.append("and should be MORE compressible (lower entropy, lower effective rank).")
    lines.append("")
    lines.append("| Layer | Ratio | Real Entropy | Distilled Entropy | Real EffRank | Distilled EffRank | Real VarRatio | Distilled VarRatio |")
    lines.append("|-------|-------|-------------|-------------------|-------------|-------------------|--------------|-------------------|")

    for r in results["distillation_results"]:
        lines.append(
            f"| {r['layer']:>5} | {r['ratio']:>5}x "
            f"| {r['real_entropy']:.2f} | {r['distilled_entropy']:.2f} "
            f"| {r['real_effective_rank']:.1f} | {r['distilled_effective_rank']:.1f} "
            f"| {r['real_var_ratio']:.1f} | {r['distilled_var_ratio']:.1f} |"
        )

    lines.append("")

    # Aggregated results
    lines.append("## Aggregated Results (Mean Across Layers)")
    lines.append("")

    for ratio in DISTILL_RATIOS:
        ratio_results = [r for r in results["distillation_results"] if r["ratio"] == ratio]
        if not ratio_results:
            continue

        avg_cos = sum(r["cosine_sim"] for r in ratio_results) / len(ratio_results)
        avg_top5 = sum(r["top5_match"] for r in ratio_results) / len(ratio_results)
        avg_post_cos = sum(r["cosine_post_quant"] for r in ratio_results) / len(ratio_results)
        avg_total = sum(r["total_ratio"] for r in ratio_results) / len(ratio_results)
        avg_time = sum(r["distill_time_s"] for r in ratio_results) / len(ratio_results)
        avg_entropy_real = sum(r["real_entropy"] for r in ratio_results) / len(ratio_results)
        avg_entropy_dist = sum(r["distilled_entropy"] for r in ratio_results) / len(ratio_results)
        avg_rank_real = sum(r["real_effective_rank"] for r in ratio_results) / len(ratio_results)
        avg_rank_dist = sum(r["distilled_effective_rank"] for r in ratio_results) / len(ratio_results)

        lines.append(f"### {ratio}x Distillation")
        lines.append(f"- **Attention cosine:** {avg_cos:.4f}")
        lines.append(f"- **Top-5 match:** {avg_top5:.3f}")
        lines.append(f"- **Post-quant cosine:** {avg_post_cos:.4f}")
        lines.append(f"- **Total compression:** {avg_total:.1f}x")
        lines.append(f"- **Distillation time:** {avg_time:.2f}s per head")
        lines.append(f"- **Entropy (real/distilled):** {avg_entropy_real:.2f} / {avg_entropy_dist:.2f}")
        lines.append(f"- **Effective rank (real/distilled):** {avg_rank_real:.1f} / {avg_rank_dist:.1f}")
        lines.append("")

    # Baseline average
    if results["baseline_results"]:
        avg_baseline = sum(r["cosine_post_quant"] for r in results["baseline_results"]) / len(results["baseline_results"])
        lines.append(f"### Baseline (3-bit TurboQuant only)")
        lines.append(f"- **Cosine:** {avg_baseline:.4f}")
        lines.append(f"- **Compression:** ~5.0x")
        lines.append("")

    # Conclusions
    lines.append("## Key Findings")
    lines.append("")

    # Determine if distilled tokens are more compressible
    ratio_4_results = [r for r in results["distillation_results"] if r["ratio"] == 4]
    if ratio_4_results:
        avg_real_rank = sum(r["real_effective_rank"] for r in ratio_4_results) / len(ratio_4_results)
        avg_dist_rank = sum(r["distilled_effective_rank"] for r in ratio_4_results) / len(ratio_4_results)
        more_compressible = avg_dist_rank < avg_real_rank

        lines.append(f"1. **Distilled tokens {'ARE' if more_compressible else 'are NOT'} more compressible** "
                      f"than real tokens (effective rank: {avg_dist_rank:.1f} vs {avg_real_rank:.1f})")

    avg_cos_4x = sum(r["cosine_sim"] for r in ratio_4_results) / len(ratio_4_results) if ratio_4_results else 0
    lines.append(f"2. **4x distillation quality:** {avg_cos_4x:.4f} attention cosine")

    ratio_16_results = [r for r in results["distillation_results"] if r["ratio"] == 16]
    if ratio_16_results:
        avg_total_16 = sum(r["total_ratio"] for r in ratio_16_results) / len(ratio_16_results)
        avg_cos_16 = sum(r["cosine_post_quant"] for r in ratio_16_results) / len(ratio_16_results)
        lines.append(f"3. **Maximum compression:** {avg_total_16:.0f}x total at {avg_cos_16:.4f} post-quant cosine")

    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Cache Distillation + TurboQuant Benchmark")
    print("=" * 60)

    model, tokenizer = load_model()

    print("\nExtracting Q/K/V data...")
    data = extract_qkv_data(model, tokenizer, CONTEXT_PROMPT)

    # Free model memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = run_distillation_benchmark(data)

    # Save results
    report = format_results(results)
    out_path = os.path.join(REPO_ROOT, "benchmarks", "results", "cache_distillation_results.md")
    with open(out_path, "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"{'='*60}")
    print(report)


if __name__ == "__main__":
    main()
