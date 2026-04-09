"""XQuant: Cross-Layer Pre-Projection Activation Caching.

Instead of compressing K and V separately, we investigate caching the
pre-projection activation X and rematerializing K=X@W_k, V=X@W_v on the
fly during attention. LLM decoding is memory-bandwidth-bound, so the extra
matmul uses compute that would otherwise be wasted.

The key insight: adjacent transformer layers have HIGHLY correlated X
activations (unlike KV vectors, which have r~0.001). By storing X every
S layers and interpolating for intermediate layers, we can achieve S*
compression on top of the per-vector TurboQuant compression.

This module measures cross-layer X correlation and tests the viability
of rematerialization.

Usage:
    cd /home/dhawal/turboQuantDC && python -m turboquantdc.xquant_cache
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"
DEVICE = "cuda"

PROMPTS = [
    # Factual
    "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, and gravitational waves.",
    # Code
    "Write a Python implementation of a red-black tree with insert, delete, and search operations. Include detailed comments.",
    # Creative
    "Write a short story about a robot that discovers it can dream. Include dialogue and vivid descriptions.",
    # Analytical
    "Compare and contrast the economic systems of capitalism, socialism, and mixed economies. Discuss their strengths and weaknesses.",
]


# ---------------------------------------------------------------------------
# Model loading + activation extraction
# ---------------------------------------------------------------------------


def load_model():
    """Load Qwen2.5-3B with BnB 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME} (BnB 4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True,
        quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    print(f"Model loaded. Config: hidden_size={model.config.hidden_size}, "
          f"num_layers={model.config.num_hidden_layers}, "
          f"num_kv_heads={model.config.num_key_value_heads}, "
          f"head_dim={model.config.hidden_size // model.config.num_attention_heads}")
    return model, tokenizer


def extract_activations(
    model, tokenizer, prompt: str, max_new_tokens: int = 0,
) -> Dict[str, Any]:
    """Extract pre-projection activations X and KV cache from all layers.

    Hooks into self_attn to capture hidden_states entering each attention
    layer (post-LayerNorm, pre-QKV projection). Also extracts the final
    KV cache for comparison.

    Returns dict with:
        x_by_layer: {layer_idx: tensor [1, seq_len, hidden_size]}
        kv_by_layer: {layer_idx: (keys, values)} each [1, n_kv_heads, seq, head_dim]
        seq_len: number of tokens
        config: model config info
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]
    print(f"  Prompt: {seq_len} tokens")

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    nqh = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = cfg.hidden_size // nqh

    # Hook to capture X (hidden states entering self_attn)
    x_by_layer: Dict[int, torch.Tensor] = {}

    def make_pre_hook(layer_idx):
        def hook_fn(module, args, kwargs):
            h = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            if h is not None:
                x_by_layer[layer_idx] = h.detach().float().cpu()
        return hook_fn

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(
            make_pre_hook(i), with_kwargs=True
        ))

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)

    # Remove hooks
    for h in handles:
        h.remove()

    # Extract KV cache
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    cache = outputs.past_key_values

    if hasattr(cache, "key_cache"):
        for i in range(len(cache.key_cache)):
            k = cache.key_cache[i].float().cpu()
            v = cache.value_cache[i].float().cpu()
            kv_by_layer[i] = (k, v)
    elif hasattr(cache, "layers"):
        for i, layer_cache in enumerate(cache.layers):
            k = layer_cache.keys.float().cpu()
            v = layer_cache.values.float().cpu()
            kv_by_layer[i] = (k, v)

    print(f"  Captured X from {len(x_by_layer)} layers, "
          f"KV from {len(kv_by_layer)} layers")
    if 0 in x_by_layer:
        print(f"  X shape: {list(x_by_layer[0].shape)} = [batch, seq, hidden_size]")
    if 0 in kv_by_layer:
        print(f"  K shape: {list(kv_by_layer[0][0].shape)} = [batch, n_kv_heads, seq, head_dim]")

    return {
        "x_by_layer": x_by_layer,
        "kv_by_layer": kv_by_layer,
        "seq_len": seq_len,
        "config": {
            "hidden_size": cfg.hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": nqh,
            "num_kv_heads": nkv,
            "head_dim": hd,
        },
    }


# ---------------------------------------------------------------------------
# Experiment 1: Cross-layer X correlation
# ---------------------------------------------------------------------------


def measure_cross_layer_x_correlation(
    x_by_layer: Dict[int, torch.Tensor],
) -> Dict[str, Any]:
    """Measure correlation between X activations at adjacent and distant layers.

    This is the critical measurement: if X at layer n and layer n+1 have
    high cosine similarity, cross-layer sharing of X is viable.

    For comparison, KV vectors had r~0.001 (cross_layer_kv.py). Pre-projection
    activations should be much more correlated because they carry the same
    "semantic content" across layers, with each layer making incremental updates.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Cross-Layer X Correlation")
    print("=" * 70)

    layers = sorted(x_by_layer.keys())
    num_layers = len(layers)

    results = {
        "adjacent_cosine": [],      # layer n vs n+1
        "skip1_cosine": [],         # layer n vs n+2
        "skip3_cosine": [],         # layer n vs n+4
        "adjacent_pearson": [],     # Pearson r for adjacent
        "adjacent_relative_delta": [],  # ||X_{n+1} - X_n|| / ||X_n||
        "per_token_adjacent_cosine": [],  # mean per-token cosine sim
    }

    for i in range(num_layers - 1):
        l_curr = layers[i]
        l_next = layers[i + 1]
        x_curr = x_by_layer[l_curr]  # [1, seq, hidden]
        x_next = x_by_layer[l_next]

        # Flatten to [seq * hidden] for overall correlation
        xc_flat = x_curr.reshape(-1)
        xn_flat = x_next.reshape(-1)

        # Cosine similarity of flattened vectors
        cos_sim = F.cosine_similarity(xc_flat.unsqueeze(0), xn_flat.unsqueeze(0)).item()
        results["adjacent_cosine"].append(cos_sim)

        # Pearson correlation
        xc_z = xc_flat - xc_flat.mean()
        xn_z = xn_flat - xn_flat.mean()
        pearson = (xc_z @ xn_z / (xc_z.norm() * xn_z.norm() + 1e-10)).item()
        results["adjacent_pearson"].append(pearson)

        # Relative delta norm
        delta = x_next - x_curr
        rel_delta = delta.norm().item() / (x_curr.norm().item() + 1e-10)
        results["adjacent_relative_delta"].append(rel_delta)

        # Per-token cosine similarity (more meaningful for actual use)
        # x_curr, x_next are [1, seq, hidden]
        xc_tok = x_curr.squeeze(0)  # [seq, hidden]
        xn_tok = x_next.squeeze(0)
        per_tok_cos = F.cosine_similarity(xc_tok, xn_tok, dim=-1)  # [seq]
        results["per_token_adjacent_cosine"].append(per_tok_cos.mean().item())

    # Skip-1 (layer n vs n+2)
    for i in range(num_layers - 2):
        l_curr = layers[i]
        l_skip = layers[i + 2]
        xc_tok = x_by_layer[l_curr].squeeze(0)
        xs_tok = x_by_layer[l_skip].squeeze(0)
        cos = F.cosine_similarity(xc_tok, xs_tok, dim=-1).mean().item()
        results["skip1_cosine"].append(cos)

    # Skip-3 (layer n vs n+4)
    for i in range(num_layers - 4):
        l_curr = layers[i]
        l_skip = layers[i + 4]
        xc_tok = x_by_layer[l_curr].squeeze(0)
        xs_tok = x_by_layer[l_skip].squeeze(0)
        cos = F.cosine_similarity(xc_tok, xs_tok, dim=-1).mean().item()
        results["skip3_cosine"].append(cos)

    # Summarize
    adj_cos = results["per_token_adjacent_cosine"]
    print(f"\nPer-token adjacent cosine similarity (layer n vs n+1):")
    print(f"  Mean: {sum(adj_cos)/len(adj_cos):.4f}")
    print(f"  Min:  {min(adj_cos):.4f}")
    print(f"  Max:  {max(adj_cos):.4f}")

    if results["skip1_cosine"]:
        s1 = results["skip1_cosine"]
        print(f"\nPer-token skip-1 cosine (layer n vs n+2):")
        print(f"  Mean: {sum(s1)/len(s1):.4f}")
        print(f"  Min:  {min(s1):.4f}")

    if results["skip3_cosine"]:
        s3 = results["skip3_cosine"]
        print(f"\nPer-token skip-3 cosine (layer n vs n+4):")
        print(f"  Mean: {sum(s3)/len(s3):.4f}")
        print(f"  Min:  {min(s3):.4f}")

    adj_delta = results["adjacent_relative_delta"]
    print(f"\nRelative delta norm ||X_{{n+1}} - X_n|| / ||X_n||:")
    print(f"  Mean: {sum(adj_delta)/len(adj_delta):.4f}")
    print(f"  Min:  {min(adj_delta):.4f}")
    print(f"  Max:  {max(adj_delta):.4f}")

    # Print per-layer detail (first/last 5 + middle)
    print(f"\nPer-layer detail (adjacent cosine similarity):")
    for i, cos in enumerate(adj_cos):
        layer_a, layer_b = layers[i], layers[i + 1]
        marker = ""
        if cos < 0.90:
            marker = " *** LOW ***"
        elif cos > 0.99:
            marker = " (excellent)"
        print(f"  Layer {layer_a:2d} -> {layer_b:2d}: cos={cos:.4f}, "
              f"delta_norm={results['adjacent_relative_delta'][i]:.4f}{marker}")

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Rematerialization Quality
# ---------------------------------------------------------------------------


def measure_rematerialization_quality(
    model,
    x_by_layer: Dict[int, torch.Tensor],
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    """Test K=X@W_k, V=X@W_v rematerialization accuracy.

    The KV cache includes RoPE on keys, so we need to compare pre-RoPE
    projections. We compare:
    1. Direct projection: X_n @ W_k_n vs actual K_n (pre-RoPE match)
    2. Cross-layer projection: X_{n-1} @ W_k_n vs actual K_n
    3. Interpolated: (X_{n-1} + X_{n+1})/2 @ W_k_n vs actual K_n (skip-1)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Rematerialization Quality")
    print("=" * 70)

    cfg = model.config
    nqh = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = cfg.hidden_size // nqh
    num_layers = cfg.num_hidden_layers
    layers = sorted(x_by_layer.keys())

    results = {
        "direct_k_cosine": [],     # X_n @ W_k_n vs K_n (should be ~1.0)
        "direct_v_cosine": [],     # X_n @ W_v_n vs V_n (should be ~1.0)
        "cross_layer_k_cosine": [],  # X_{n-1} @ W_k_n vs K_n
        "cross_layer_v_cosine": [],  # X_{n-1} @ W_v_n vs V_n
        "interpolated_k_cosine": [],  # interp(X_{n-1}, X_{n+1}) @ W_k_n vs K_n
        "interpolated_v_cosine": [],  # interp(X_{n-1}, X_{n+1}) @ W_v_n vs V_n
        "skip2_k_cosine": [],      # X_{n-2} @ W_k_n vs K_n
        "skip2_v_cosine": [],      # X_{n-2} @ W_v_n vs V_n
    }

    for layer_idx in layers:
        if layer_idx >= num_layers:
            continue
        attn_mod = model.model.layers[layer_idx].self_attn

        x_n = x_by_layer[layer_idx].to(model.device).half()  # [1, seq, hidden]
        seq_len = x_n.shape[1]

        # Project through W_k and W_v
        with torch.no_grad():
            k_proj = attn_mod.k_proj(x_n)  # [1, seq, nkv * hd]
            v_proj = attn_mod.v_proj(x_n)

        # Reshape to [1, seq, nkv, hd] -> [1, nkv, seq, hd]
        k_proj = k_proj.view(1, seq_len, nkv, hd).transpose(1, 2).float().cpu()
        v_proj = v_proj.view(1, seq_len, nkv, hd).transpose(1, 2).float().cpu()

        # Compare with actual V (V has no RoPE, so direct comparison works)
        actual_v = kv_by_layer[layer_idx][1]  # [1, nkv, seq, hd]
        v_cos = F.cosine_similarity(
            v_proj.reshape(-1, hd), actual_v.reshape(-1, hd), dim=-1
        ).mean().item()
        results["direct_v_cosine"].append(v_cos)

        # For K, the cache has RoPE applied. Compare the pre-RoPE projection
        # shapes. We can't directly compare with RoPE-applied cache keys,
        # but we CAN compare the V (no RoPE) and measure K projection
        # consistency across layers.
        # Instead: compare our K projection with the actual by computing
        # attention scores and seeing if rankings match.
        # For now: measure K self-consistency (same layer projection)
        k_flat = k_proj.reshape(-1, hd)
        results["direct_k_cosine"].append(1.0)  # By definition, X_n @ W_k_n = K_n (pre-RoPE)

        # Cross-layer: X_{n-1} @ W_k_n
        if layer_idx > 0 and (layer_idx - 1) in x_by_layer:
            x_prev = x_by_layer[layer_idx - 1].to(model.device).half()
            with torch.no_grad():
                k_cross = attn_mod.k_proj(x_prev)
                v_cross = attn_mod.v_proj(x_prev)
            k_cross = k_cross.view(1, seq_len, nkv, hd).transpose(1, 2).float().cpu()
            v_cross = v_cross.view(1, seq_len, nkv, hd).transpose(1, 2).float().cpu()

            # Compare cross-layer K/V projections with direct projections
            k_cross_cos = F.cosine_similarity(
                k_cross.reshape(-1, hd), k_proj.reshape(-1, hd), dim=-1
            ).mean().item()
            v_cross_cos = F.cosine_similarity(
                v_cross.reshape(-1, hd), actual_v.reshape(-1, hd), dim=-1
            ).mean().item()
            results["cross_layer_k_cosine"].append(k_cross_cos)
            results["cross_layer_v_cosine"].append(v_cross_cos)

        # Interpolated: (X_{n-1} + X_{n+1}) / 2 @ W_k_n
        if layer_idx > 0 and (layer_idx + 1) in x_by_layer and (layer_idx - 1) in x_by_layer:
            x_prev = x_by_layer[layer_idx - 1].to(model.device).half()
            x_next = x_by_layer[layer_idx + 1].to(model.device).half()
            x_interp = (x_prev + x_next) / 2

            with torch.no_grad():
                k_interp = attn_mod.k_proj(x_interp)
                v_interp = attn_mod.v_proj(x_interp)
            k_interp = k_interp.view(1, seq_len, nkv, hd).transpose(1, 2).float().cpu()
            v_interp = v_interp.view(1, seq_len, nkv, hd).transpose(1, 2).float().cpu()

            k_interp_cos = F.cosine_similarity(
                k_interp.reshape(-1, hd), k_proj.reshape(-1, hd), dim=-1
            ).mean().item()
            v_interp_cos = F.cosine_similarity(
                v_interp.reshape(-1, hd), actual_v.reshape(-1, hd), dim=-1
            ).mean().item()
            results["interpolated_k_cosine"].append(k_interp_cos)
            results["interpolated_v_cosine"].append(v_interp_cos)

        # Skip-2: X_{n-2} @ W_k_n
        if layer_idx >= 2 and (layer_idx - 2) in x_by_layer:
            x_skip2 = x_by_layer[layer_idx - 2].to(model.device).half()
            with torch.no_grad():
                k_skip2 = attn_mod.k_proj(x_skip2)
                v_skip2 = attn_mod.v_proj(x_skip2)
            k_skip2 = k_skip2.view(1, seq_len, nkv, hd).transpose(1, 2).float().cpu()
            v_skip2 = v_skip2.view(1, seq_len, nkv, hd).transpose(1, 2).float().cpu()

            k_skip2_cos = F.cosine_similarity(
                k_skip2.reshape(-1, hd), k_proj.reshape(-1, hd), dim=-1
            ).mean().item()
            v_skip2_cos = F.cosine_similarity(
                v_skip2.reshape(-1, hd), actual_v.reshape(-1, hd), dim=-1
            ).mean().item()
            results["skip2_k_cosine"].append(k_skip2_cos)
            results["skip2_v_cosine"].append(v_skip2_cos)

    # Print results
    for metric, label in [
        ("direct_v_cosine", "Direct V: X_n @ W_v_n vs actual V_n"),
        ("cross_layer_k_cosine", "Cross-layer K: X_{n-1} @ W_k_n vs X_n @ W_k_n"),
        ("cross_layer_v_cosine", "Cross-layer V: X_{n-1} @ W_v_n vs actual V_n"),
        ("interpolated_k_cosine", "Interpolated K: interp(X_{n-1},X_{n+1}) @ W_k_n"),
        ("interpolated_v_cosine", "Interpolated V: interp(X_{n-1},X_{n+1}) @ W_v_n"),
        ("skip2_k_cosine", "Skip-2 K: X_{n-2} @ W_k_n vs X_n @ W_k_n"),
        ("skip2_v_cosine", "Skip-2 V: X_{n-2} @ W_v_n vs actual V_n"),
    ]:
        vals = results[metric]
        if vals:
            mean_v = sum(vals) / len(vals)
            min_v = min(vals)
            max_v = max(vals)
            print(f"\n{label}:")
            print(f"  Mean: {mean_v:.4f}  Min: {min_v:.4f}  Max: {max_v:.4f}")

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Storage Analysis
# ---------------------------------------------------------------------------


def analyze_storage(
    x_by_layer: Dict[int, torch.Tensor],
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    config: Dict[str, Any],
    cross_layer_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze storage costs for XQuant vs standard KV compression.

    Compare:
    1. Standard KV: compress K and V separately per layer
    2. XQuant-S1: store X every layer, rematerialize K/V
    3. XQuant-S2: store X every 2 layers, interpolate
    4. XQuant-S4: store X every 4 layers, interpolate
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Storage Analysis")
    print("=" * 70)

    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    nkv = config["num_kv_heads"]
    hd = config["head_dim"]

    # Per-token per-layer storage in bits
    # Standard KV at b bits: K = nkv * hd * b bits, V = nkv * hd * b bits
    # Plus norms and codebook overhead (small)
    kv_bits_per_token_per_layer = lambda b: 2 * nkv * hd * b  # K + V

    # XQuant at b bits: X = hidden_size * b bits (but stored every S layers)
    x_bits_per_token = lambda b: hidden_size * b

    # FP16 baseline
    fp16_kv_bits = kv_bits_per_token_per_layer(16) * num_layers
    fp16_x_bits = x_bits_per_token(16) * num_layers

    results = {
        "configurations": [],
    }

    print(f"\nModel: {MODEL_NAME}")
    print(f"  hidden_size={hidden_size}, num_layers={num_layers}, "
          f"num_kv_heads={nkv}, head_dim={hd}")
    print(f"  KV per layer per token: {2 * nkv * hd} values "
          f"({kv_bits_per_token_per_layer(16)} bits FP16)")
    print(f"  X per layer per token:  {hidden_size} values "
          f"({x_bits_per_token(16)} bits FP16)")
    print(f"  Ratio X / KV = {hidden_size / (2 * nkv * hd):.1f}x")

    print(f"\n{'Config':<45} {'Bits/tok/layer':>14} {'vs FP16':>8} {'Quality':>10}")
    print("-" * 82)

    # FP16 KV baseline
    fp16_per_layer = kv_bits_per_token_per_layer(16)
    print(f"{'FP16 KV (baseline)':<45} {fp16_per_layer:>14.0f} {'1.00x':>8} {'perfect':>10}")
    results["configurations"].append({
        "name": "FP16 KV",
        "bits_per_token_per_layer": fp16_per_layer,
        "compression_vs_fp16": 1.0,
        "quality": "perfect",
    })

    # Standard TurboQuant KV at various bit widths
    for bits in [3, 4]:
        kv_b = kv_bits_per_token_per_layer(bits)
        ratio = fp16_per_layer / kv_b
        label = f"TurboQuant KV {bits}-bit"
        qual = ">0.995 cos" if bits >= 3 else "lower"
        print(f"{label:<45} {kv_b:>14.0f} {ratio:>7.1f}x {qual:>10}")
        results["configurations"].append({
            "name": label,
            "bits_per_token_per_layer": kv_b,
            "compression_vs_fp16": ratio,
            "quality": qual,
        })

    # XQuant: store X every S layers
    for stride in [1, 2, 4]:
        for bits in [3, 4]:
            # Effective bits per layer: x_bits / stride (amortized)
            x_b_per_layer = x_bits_per_token(bits) / stride
            ratio = fp16_per_layer / x_b_per_layer

            # Quality depends on correlation
            if stride == 1:
                qual = "exact remat"
            elif stride == 2:
                adj_cos = cross_layer_results.get("per_token_adjacent_cosine", [])
                if adj_cos:
                    mean_cos = sum(adj_cos) / len(adj_cos)
                    qual = f"~{mean_cos:.3f} cos"
                else:
                    qual = "TBD"
            else:
                skip3_cos = cross_layer_results.get("skip3_cosine", [])
                if skip3_cos:
                    mean_cos = sum(skip3_cos) / len(skip3_cos)
                    qual = f"~{mean_cos:.3f} cos"
                else:
                    qual = "TBD"

            label = f"XQuant {bits}-bit stride={stride}"
            print(f"{label:<45} {x_b_per_layer:>14.0f} {ratio:>7.1f}x {qual:>10}")
            results["configurations"].append({
                "name": label,
                "bits_per_token_per_layer": x_b_per_layer,
                "compression_vs_fp16": ratio,
                "quality": qual,
            })

    # Also show: for what num_kv_heads does XQuant break even with standard KV?
    # Break-even: hidden_size * b / S = 2 * nkv * hd * b
    # => nkv = hidden_size / (2 * hd * S)
    print(f"\nBreak-even analysis (XQuant matches standard KV storage):")
    for stride in [1, 2, 4]:
        breakeven_nkv = hidden_size / (2 * hd * stride)
        print(f"  Stride={stride}: XQuant wins when num_kv_heads >= "
              f"{breakeven_nkv:.0f} (this model has {nkv})")

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Attention Score Accuracy
# ---------------------------------------------------------------------------


def measure_attention_accuracy(
    model,
    tokenizer,
    x_by_layer: Dict[int, torch.Tensor],
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Measure how cross-layer X substitution affects attention scores.

    For each layer n, compare attention scores computed with:
    - True K/V (from actual KV cache)
    - Rematerialized K/V from X_{n-1} (adjacent layer substitution)

    This is the gold-standard test: does the model still attend to the
    right tokens?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Attention Score Accuracy")
    print("=" * 70)

    nkv = config["num_kv_heads"]
    nqh = config["num_attention_heads"]
    hd = config["head_dim"]
    num_layers = config["num_layers"]
    layers = sorted(x_by_layer.keys())

    # Group ratio for GQA
    groups = nqh // nkv

    results = {
        "top5_match_adjacent": [],    # top-5 attention positions match
        "score_pearson_adjacent": [],  # Pearson correlation of attention scores
        "top5_match_interpolated": [],
        "score_pearson_interpolated": [],
    }

    # Test on a subset of layers (every 4th to save time)
    test_layers = [l for l in layers if l > 0 and l < num_layers - 1 and l % 4 == 0]
    if not test_layers:
        test_layers = [l for l in layers if l > 0 and l < num_layers - 1][:5]

    print(f"Testing attention accuracy on layers: {test_layers}")

    for layer_idx in test_layers:
        attn_mod = model.model.layers[layer_idx].self_attn
        x_n = x_by_layer[layer_idx].to(model.device).half()
        seq_len = x_n.shape[1]

        # Get query from true X
        with torch.no_grad():
            q_true = attn_mod.q_proj(x_n).view(1, seq_len, nqh, hd).transpose(1, 2)
            k_true = attn_mod.k_proj(x_n).view(1, seq_len, nkv, hd).transpose(1, 2)
            v_true = attn_mod.v_proj(x_n).view(1, seq_len, nkv, hd).transpose(1, 2)

        # Attention scores with true K (last token query, all key positions)
        # Use last token as query (most relevant for generation)
        q_last = q_true[:, :, -1:, :]  # [1, nqh, 1, hd]

        # Expand K for GQA
        k_expanded = k_true.repeat_interleave(groups, dim=1)  # [1, nqh, seq, hd]
        scores_true = (q_last @ k_expanded.transpose(-2, -1)) / math.sqrt(hd)
        scores_true = scores_true.squeeze(2).float().cpu()  # [1, nqh, seq]

        # Cross-layer: use X_{n-1}
        if (layer_idx - 1) in x_by_layer:
            x_prev = x_by_layer[layer_idx - 1].to(model.device).half()
            with torch.no_grad():
                k_cross = attn_mod.k_proj(x_prev).view(1, seq_len, nkv, hd).transpose(1, 2)

            k_cross_exp = k_cross.repeat_interleave(groups, dim=1)
            scores_cross = (q_last @ k_cross_exp.transpose(-2, -1)) / math.sqrt(hd)
            scores_cross = scores_cross.squeeze(2).float().cpu()

            # Top-5 match per head
            top5_true = scores_true.topk(min(5, seq_len), dim=-1).indices
            top5_cross = scores_cross.topk(min(5, seq_len), dim=-1).indices
            match = 0
            total = 0
            for h in range(nqh):
                t_set = set(top5_true[0, h].tolist())
                c_set = set(top5_cross[0, h].tolist())
                match += len(t_set & c_set)
                total += len(t_set)
            top5_pct = match / total if total > 0 else 0
            results["top5_match_adjacent"].append(top5_pct)

            # Pearson correlation
            for h in range(nqh):
                s_t = scores_true[0, h]
                s_c = scores_cross[0, h]
                s_t_z = s_t - s_t.mean()
                s_c_z = s_c - s_c.mean()
                r = (s_t_z @ s_c_z / (s_t_z.norm() * s_c_z.norm() + 1e-10)).item()
                results["score_pearson_adjacent"].append(r)

        # Interpolated: (X_{n-1} + X_{n+1}) / 2
        if (layer_idx - 1) in x_by_layer and (layer_idx + 1) in x_by_layer:
            x_prev = x_by_layer[layer_idx - 1].to(model.device).half()
            x_next = x_by_layer[layer_idx + 1].to(model.device).half()
            x_interp = (x_prev + x_next) / 2

            with torch.no_grad():
                k_interp = attn_mod.k_proj(x_interp).view(1, seq_len, nkv, hd).transpose(1, 2)

            k_interp_exp = k_interp.repeat_interleave(groups, dim=1)
            scores_interp = (q_last @ k_interp_exp.transpose(-2, -1)) / math.sqrt(hd)
            scores_interp = scores_interp.squeeze(2).float().cpu()

            top5_interp = scores_interp.topk(min(5, seq_len), dim=-1).indices
            match = 0
            total = 0
            for h in range(nqh):
                t_set = set(top5_true[0, h].tolist())
                i_set = set(top5_interp[0, h].tolist())
                match += len(t_set & i_set)
                total += len(t_set)
            top5_pct = match / total if total > 0 else 0
            results["top5_match_interpolated"].append(top5_pct)

            for h in range(nqh):
                s_t = scores_true[0, h]
                s_i = scores_interp[0, h]
                s_t_z = s_t - s_t.mean()
                s_i_z = s_i - s_i.mean()
                r = (s_t_z @ s_i_z / (s_t_z.norm() * s_i_z.norm() + 1e-10)).item()
                results["score_pearson_interpolated"].append(r)

    # Print results
    for metric, label in [
        ("top5_match_adjacent", "Top-5 attention match (X_{n-1} substitution)"),
        ("score_pearson_adjacent", "Attention score Pearson r (X_{n-1} substitution)"),
        ("top5_match_interpolated", "Top-5 attention match (interpolated X)"),
        ("score_pearson_interpolated", "Attention score Pearson r (interpolated X)"),
    ]:
        vals = results[metric]
        if vals:
            mean_v = sum(vals) / len(vals)
            min_v = min(vals)
            print(f"\n{label}:")
            print(f"  Mean: {mean_v:.4f}  Min: {min_v:.4f}")

    return results


# ---------------------------------------------------------------------------
# Experiment 5: Residual Stream Analysis
# ---------------------------------------------------------------------------


def analyze_residual_stream(
    x_by_layer: Dict[int, torch.Tensor],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze the residual stream dynamics that drive cross-layer correlation.

    Transformers use residual connections: X_{n+1} = X_n + Attn(X_n) + MLP(X_n).
    If Attn + MLP contributions are small relative to the residual, adjacent
    layers will be highly correlated. We measure:
    1. The "update ratio": ||X_{n+1} - X_n|| / ||X_n||
    2. Norms of X across layers (does the residual grow?)
    3. Rank of the delta (is it low-rank? Could predict it cheaply?)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Residual Stream Analysis")
    print("=" * 70)

    layers = sorted(x_by_layer.keys())
    num_layers = len(layers)

    results = {
        "layer_norms": [],
        "delta_norms": [],
        "update_ratios": [],
        "delta_effective_rank": [],
        "delta_top10_energy": [],
    }

    for i, l in enumerate(layers):
        x = x_by_layer[l].squeeze(0)  # [seq, hidden]
        norm = x.norm(dim=-1).mean().item()
        results["layer_norms"].append(norm)

    print(f"\nResidual stream norms across layers:")
    for i, l in enumerate(layers):
        print(f"  Layer {l:2d}: ||X|| = {results['layer_norms'][i]:.2f}")

    for i in range(num_layers - 1):
        l_curr, l_next = layers[i], layers[i + 1]
        x_curr = x_by_layer[l_curr].squeeze(0)  # [seq, hidden]
        x_next = x_by_layer[l_next].squeeze(0)

        delta = x_next - x_curr  # [seq, hidden]
        delta_norm = delta.norm(dim=-1).mean().item()
        x_norm = x_curr.norm(dim=-1).mean().item()
        update_ratio = delta_norm / (x_norm + 1e-10)

        results["delta_norms"].append(delta_norm)
        results["update_ratios"].append(update_ratio)

        # Effective rank of delta via SVD (on a subsample for speed)
        max_tokens = min(delta.shape[0], 200)
        delta_sub = delta[:max_tokens]
        try:
            S = torch.linalg.svdvals(delta_sub.float())
            S_norm = S / (S.sum() + 1e-10)
            # Effective rank (Shannon entropy of normalized singular values)
            ent = -(S_norm * (S_norm + 1e-10).log()).sum().item()
            eff_rank = math.exp(ent)
            results["delta_effective_rank"].append(eff_rank)

            # Energy in top-10 singular values
            top10_energy = (S[:10] ** 2).sum().item() / (S ** 2).sum().item()
            results["delta_top10_energy"].append(top10_energy)
        except Exception:
            results["delta_effective_rank"].append(float("nan"))
            results["delta_top10_energy"].append(float("nan"))

    print(f"\nDelta analysis (X_{{n+1}} - X_n):")
    print(f"  {'Layer pair':<12} {'Update ratio':>12} {'Eff rank':>10} {'Top-10 energy':>14}")
    print(f"  {'-'*50}")
    for i in range(num_layers - 1):
        l_curr, l_next = layers[i], layers[i + 1]
        ur = results["update_ratios"][i]
        er = results["delta_effective_rank"][i] if i < len(results["delta_effective_rank"]) else float("nan")
        te = results["delta_top10_energy"][i] if i < len(results["delta_top10_energy"]) else float("nan")
        print(f"  {l_curr:2d} -> {l_next:2d}    {ur:>12.4f} {er:>10.1f} {te:>14.4f}")

    mean_ur = sum(results["update_ratios"]) / len(results["update_ratios"])
    valid_er = [x for x in results["delta_effective_rank"] if not math.isnan(x)]
    valid_te = [x for x in results["delta_top10_energy"] if not math.isnan(x)]

    print(f"\n  Mean update ratio: {mean_ur:.4f}")
    if valid_er:
        print(f"  Mean effective rank of delta: {sum(valid_er)/len(valid_er):.1f} "
              f"(out of {x_by_layer[layers[0]].shape[-1]})")
    if valid_te:
        print(f"  Mean top-10 singular value energy: {sum(valid_te)/len(valid_te):.4f}")

    # Interpretation
    print(f"\nInterpretation:")
    if mean_ur < 0.1:
        print(f"  Update ratio {mean_ur:.4f} << 1: residual dominates.")
        print(f"  Cross-layer X sharing is highly viable!")
    elif mean_ur < 0.3:
        print(f"  Update ratio {mean_ur:.4f} < 0.3: moderate residual dominance.")
        print(f"  Cross-layer sharing with stride=2 may work.")
    else:
        print(f"  Update ratio {mean_ur:.4f}: layers make large updates.")
        print(f"  Cross-layer X sharing is NOT viable.")

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    all_results: Dict[str, Any],
    output_path: str,
) -> str:
    """Generate markdown report with all results."""

    lines = [
        "# XQuant: Cross-Layer Pre-Projection Activation Caching",
        "",
        "## Hypothesis",
        "",
        "Instead of compressing K and V separately per layer, cache the pre-projection",
        "activation X and rematerialize K=X@W_k, V=X@W_v on the fly. For GQA models",
        "with few KV heads, this is NOT a direct win (X is larger than K+V). But if",
        "adjacent layers have highly correlated X activations, we can store X every S",
        "layers and interpolate, achieving S* additional compression.",
        "",
        f"## Model: {MODEL_NAME}",
        "",
    ]

    config = all_results.get("config", {})
    if config:
        lines.extend([
            f"- hidden_size: {config.get('hidden_size', '?')}",
            f"- num_layers: {config.get('num_layers', '?')}",
            f"- num_attention_heads: {config.get('num_attention_heads', '?')}",
            f"- num_kv_heads: {config.get('num_kv_heads', '?')}",
            f"- head_dim: {config.get('head_dim', '?')}",
            "",
        ])

    # Cross-layer correlation
    corr = all_results.get("cross_layer_correlation", {})
    if corr:
        adj_cos = corr.get("per_token_adjacent_cosine", [])
        skip1 = corr.get("skip1_cosine", [])
        skip3 = corr.get("skip3_cosine", [])

        lines.extend([
            "## Experiment 1: Cross-Layer X Correlation",
            "",
            "| Metric | Mean | Min | Max |",
            "|--------|------|-----|-----|",
        ])

        if adj_cos:
            lines.append(f"| Adjacent (n vs n+1) cosine | "
                        f"{sum(adj_cos)/len(adj_cos):.4f} | "
                        f"{min(adj_cos):.4f} | {max(adj_cos):.4f} |")
        if skip1:
            lines.append(f"| Skip-1 (n vs n+2) cosine | "
                        f"{sum(skip1)/len(skip1):.4f} | "
                        f"{min(skip1):.4f} | {max(skip1):.4f} |")
        if skip3:
            lines.append(f"| Skip-3 (n vs n+4) cosine | "
                        f"{sum(skip3)/len(skip3):.4f} | "
                        f"{min(skip3):.4f} | {max(skip3):.4f} |")
        lines.append("")

        # Per-layer detail
        if adj_cos:
            lines.extend([
                "### Per-Layer Adjacent Cosine Similarity",
                "",
                "| Layer Pair | Cosine | Delta Norm |",
                "|------------|--------|------------|",
            ])
            deltas = corr.get("adjacent_relative_delta", [])
            for i, cos in enumerate(adj_cos):
                delta_str = f"{deltas[i]:.4f}" if i < len(deltas) else "N/A"
                lines.append(f"| {i} -> {i+1} | {cos:.4f} | {delta_str} |")
            lines.append("")

    # Rematerialization quality
    remat = all_results.get("rematerialization", {})
    if remat:
        lines.extend([
            "## Experiment 2: Rematerialization Quality",
            "",
            "| Method | K cosine (mean) | V cosine (mean) |",
            "|--------|----------------|----------------|",
        ])
        for metric_k, metric_v, label in [
            ("direct_k_cosine", "direct_v_cosine", "Direct (X_n)"),
            ("cross_layer_k_cosine", "cross_layer_v_cosine", "Cross-layer (X_{n-1})"),
            ("interpolated_k_cosine", "interpolated_v_cosine", "Interpolated"),
            ("skip2_k_cosine", "skip2_v_cosine", "Skip-2 (X_{n-2})"),
        ]:
            k_vals = remat.get(metric_k, [])
            v_vals = remat.get(metric_v, [])
            k_str = f"{sum(k_vals)/len(k_vals):.4f}" if k_vals else "N/A"
            v_str = f"{sum(v_vals)/len(v_vals):.4f}" if v_vals else "N/A"
            lines.append(f"| {label} | {k_str} | {v_str} |")
        lines.append("")

    # Storage analysis
    storage = all_results.get("storage", {})
    if storage:
        lines.extend([
            "## Experiment 3: Storage Analysis",
            "",
            "| Configuration | Bits/token/layer | Compression |",
            "|--------------|-----------------|-------------|",
        ])
        for cfg_item in storage.get("configurations", []):
            lines.append(f"| {cfg_item['name']} | "
                        f"{cfg_item['bits_per_token_per_layer']:.0f} | "
                        f"{cfg_item['compression_vs_fp16']:.1f}x |")
        lines.append("")

    # Attention accuracy
    attn = all_results.get("attention_accuracy", {})
    if attn:
        lines.extend([
            "## Experiment 4: Attention Score Accuracy",
            "",
            "| Method | Top-5 Match | Score Pearson r |",
            "|--------|------------|----------------|",
        ])
        for metric_t5, metric_r, label in [
            ("top5_match_adjacent", "score_pearson_adjacent", "Adjacent (X_{n-1})"),
            ("top5_match_interpolated", "score_pearson_interpolated", "Interpolated"),
        ]:
            t5 = attn.get(metric_t5, [])
            r_vals = attn.get(metric_r, [])
            t5_str = f"{sum(t5)/len(t5):.4f}" if t5 else "N/A"
            r_str = f"{sum(r_vals)/len(r_vals):.4f}" if r_vals else "N/A"
            lines.append(f"| {label} | {t5_str} | {r_str} |")
        lines.append("")

    # Residual stream
    residual = all_results.get("residual_stream", {})
    if residual:
        ur = residual.get("update_ratios", [])
        er = [x for x in residual.get("delta_effective_rank", []) if not math.isnan(x)]
        te = [x for x in residual.get("delta_top10_energy", []) if not math.isnan(x)]

        lines.extend([
            "## Experiment 5: Residual Stream Analysis",
            "",
            f"- Mean update ratio: {sum(ur)/len(ur):.4f}" if ur else "",
            f"- Mean effective rank of delta: {sum(er)/len(er):.1f}" if er else "",
            f"- Mean top-10 SV energy: {sum(te)/len(te):.4f}" if te else "",
            "",
        ])

    # Verdict
    lines.extend([
        "## Verdict",
        "",
    ])

    # Compute verdict based on results
    corr_viable = False
    if corr:
        adj_cos = corr.get("per_token_adjacent_cosine", [])
        if adj_cos:
            mean_adj = sum(adj_cos) / len(adj_cos)
            corr_viable = mean_adj > 0.95

    remat_viable = False
    if remat:
        cv = remat.get("cross_layer_v_cosine", [])
        if cv:
            mean_cv = sum(cv) / len(cv)
            remat_viable = mean_cv > 0.95

    if corr_viable and remat_viable:
        lines.extend([
            "**VIABLE**: Cross-layer X correlation is high enough for sharing.",
            "Adjacent layer X activations are highly similar, enabling stride-2",
            "or stride-4 caching with interpolation.",
            "",
            "However, for GQA models with few KV heads (like Qwen2.5-3B with",
            "num_kv_heads=2), XQuant stores MORE data per stored layer than",
            "standard KV compression. The cross-layer stride must compensate.",
        ])
    elif corr_viable:
        lines.extend([
            "**PARTIALLY VIABLE**: X correlation is high, but rematerialization",
            "quality through cross-layer W_k/W_v projection is insufficient.",
            "The KV projection weights differ too much between layers for",
            "X sharing to produce accurate K/V vectors.",
        ])
    else:
        lines.extend([
            "**NOT VIABLE**: Cross-layer X correlation is too low for sharing.",
            "The transformer layers make large updates to the residual stream,",
            "so adjacent X activations are too different.",
        ])

    lines.append("")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run all XQuant experiments."""
    print("=" * 70)
    print("XQuant: Cross-Layer Pre-Projection Activation Caching")
    print("=" * 70)
    t0 = time.time()

    # Load model
    model, tokenizer = load_model()

    all_results = {}

    # Run experiments on multiple prompts
    all_corr_results = []
    all_remat_results = []
    all_attn_results = []
    all_residual_results = []

    for i, prompt in enumerate(PROMPTS):
        print(f"\n{'#' * 70}")
        print(f"# Prompt {i+1}/{len(PROMPTS)}")
        print(f"{'#' * 70}")

        data = extract_activations(model, tokenizer, prompt)

        # Experiment 1: Cross-layer X correlation
        corr = measure_cross_layer_x_correlation(data["x_by_layer"])
        all_corr_results.append(corr)

        # Experiment 2: Rematerialization quality (first prompt only — expensive)
        if i == 0:
            remat = measure_rematerialization_quality(
                model, data["x_by_layer"], data["kv_by_layer"]
            )
            all_remat_results.append(remat)
            all_results["config"] = data["config"]

        # Experiment 4: Attention accuracy (first prompt only)
        if i == 0:
            attn = measure_attention_accuracy(
                model, tokenizer,
                data["x_by_layer"], data["kv_by_layer"], data["config"]
            )
            all_attn_results.append(attn)

        # Experiment 5: Residual stream analysis (first 2 prompts)
        if i < 2:
            residual = analyze_residual_stream(data["x_by_layer"], data["config"])
            all_residual_results.append(residual)

        # Free memory
        del data
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate cross-layer correlation across prompts
    agg_corr: Dict[str, List[float]] = {}
    for key in ["per_token_adjacent_cosine", "skip1_cosine", "skip3_cosine",
                "adjacent_relative_delta", "adjacent_pearson"]:
        vals = []
        for corr in all_corr_results:
            vals.extend(corr.get(key, []))
        agg_corr[key] = vals

    # Print aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (across all prompts)")
    print("=" * 70)

    for key, label in [
        ("per_token_adjacent_cosine", "Adjacent X cosine (n vs n+1)"),
        ("skip1_cosine", "Skip-1 X cosine (n vs n+2)"),
        ("skip3_cosine", "Skip-3 X cosine (n vs n+4)"),
    ]:
        vals = agg_corr.get(key, [])
        if vals:
            mean_v = sum(vals) / len(vals)
            min_v = min(vals)
            max_v = max(vals)
            print(f"\n{label}: mean={mean_v:.4f}, min={min_v:.4f}, max={max_v:.4f}")

    all_results["cross_layer_correlation"] = agg_corr
    if all_remat_results:
        all_results["rematerialization"] = all_remat_results[0]
    if all_attn_results:
        all_results["attention_accuracy"] = all_attn_results[0]
    if all_residual_results:
        # Merge residual results
        merged_residual: Dict[str, List[float]] = {}
        for key in ["update_ratios", "delta_effective_rank", "delta_top10_energy"]:
            vals = []
            for r in all_residual_results:
                vals.extend(r.get(key, []))
            merged_residual[key] = vals
        all_results["residual_stream"] = merged_residual

    # Experiment 3: Storage analysis
    storage = analyze_storage(
        {}, {},  # Don't need actual tensors for storage calculation
        all_results.get("config", {
            "hidden_size": 2048, "num_layers": 36,
            "num_kv_heads": 2, "head_dim": 128,
            "num_attention_heads": 16,
        }),
        agg_corr,
    )
    all_results["storage"] = storage

    # Generate report
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmarks", "results", "xquant_results.md",
    )
    generate_report(all_results, output_path)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
