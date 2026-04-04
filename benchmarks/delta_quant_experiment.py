"""DeltaQuant experiment: cross-token delta coding in WHT-rotated space.

Tests anchor-based delta coding against uniform 3-bit TurboQuant:
  - Cluster tokens by key similarity (greedy + k-means)
  - Encode anchors at 3-bit, deltas at 1-bit and 2-bit
  - Measure quality (cosine sim, attention score match) vs baseline
  - Analyze delta distribution (entropy, tightness)
  - Compute effective bits per token

Model: Qwen2.5-3B-Instruct (BnB 4-bit)
Target: 2 bits/token at 3-bit quality => 8x compression

Usage:
    cd /home/dhawal/turboQuantDC
    python benchmarks/delta_quant_experiment.py
"""

import gc
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from turboquantdc.codebook import LloydMaxCodebook
from turboquantdc.rotation import apply_wht_rotation, generate_wht_rotation
from turboquantdc.polarquant import PolarQuant
from turboquantdc.delta_quant import (
    DeltaQuantEncoder,
    greedy_group_by_similarity,
    kmeans_grouping,
    analyze_delta_entropy,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"

PROMPT = """The following is a detailed technical analysis of machine learning optimization.

Gradient descent is the foundational optimization algorithm in deep learning. The basic idea
is simple: compute the gradient of the loss function with respect to each parameter, then
update each parameter in the direction that reduces the loss. The learning rate controls
the step size. Too large a learning rate causes divergence; too small causes slow convergence.

Stochastic gradient descent (SGD) approximates the full gradient using mini-batches. This
introduces noise but enables training on large datasets. The noise can actually help escape
local minima. Mini-batch sizes typically range from 32 to 4096, with larger batches providing
more stable gradients but less regularization.

Momentum adds a velocity term that accumulates past gradients. This helps traverse flat regions
and dampens oscillations in ravines. The momentum coefficient, typically 0.9, controls how much
history influences the current update. Nesterov momentum evaluates the gradient at the predicted
next position, providing better convergence.

Adam combines momentum with adaptive learning rates. It maintains running averages of both the
first moment (mean) and second moment (variance) of gradients. The bias correction terms ensure
stable early training. Adam's default hyperparameters (lr=0.001, beta1=0.9, beta2=0.999) work
well across many problems, making it the default optimizer for most practitioners.

Learning rate scheduling is critical for good performance. Common strategies include step decay,
cosine annealing, warmup followed by decay, and cyclical learning rates. The one-cycle policy
uses a warmup phase followed by cosine decay to the minimum learning rate. This has been shown
to enable super-convergence, reaching good accuracy much faster than constant learning rates.

Weight decay (L2 regularization) adds a penalty proportional to the squared magnitude of weights.
In Adam, decoupled weight decay (AdamW) is preferred because it separates the regularization
from the adaptive learning rate mechanism. This is important because L2 regularization in Adam
effectively scales the regularization by the inverse of the second moment estimate, which can
lead to under-regularization of parameters with large gradients.

Batch normalization normalizes activations within each mini-batch, stabilizing training and
allowing higher learning rates. Layer normalization, used in transformers, normalizes across
features instead of the batch dimension. RMSNorm, a simpler variant, normalizes by the root
mean square without centering, and has become popular in modern language models.

The transformer architecture relies on self-attention, which computes queries, keys, and values
from the input. The attention scores are computed as softmax(QK^T / sqrt(d)), where d is the
head dimension. Multi-head attention splits the representation into multiple heads, each
attending to different aspects of the input. The outputs are concatenated and projected.

Key-value caching is essential for efficient autoregressive generation. During the prefill
phase, all tokens are processed in parallel and their key-value pairs are stored. During
generation, only the new token's query is computed, and attention is computed against all
cached keys and values. This avoids redundant computation of previous tokens' representations.

Quantization reduces the precision of model weights and activations to save memory and compute.
Post-training quantization (PTQ) quantizes a pre-trained model without retraining. GPTQ and
AWQ are popular PTQ methods for large language models. Quantization-aware training (QAT)
fine-tunes the model with simulated quantization during forward passes.

The key-value cache grows linearly with sequence length and is often the memory bottleneck
for long-context inference. At 128K context with a 70B model, the KV cache alone can consume
over 40GB of memory. Compression techniques like quantization, eviction, and delta coding
can dramatically reduce this memory footprint while maintaining generation quality."""


# ---------------------------------------------------------------------------
# Model loading and KV extraction (same as temporal_delta_experiment)
# ---------------------------------------------------------------------------

def load_model():
    """Load Qwen2.5-3B in BnB 4-bit."""
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
    return model, tokenizer


def extract_kv_caches(model, tokenizer, prompt: str) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Run prefill and extract per-layer KV caches."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]
    print(f"Prompt tokenized: {seq_len} tokens")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)

    cache = outputs.past_key_values
    kv_by_layer = {}

    if hasattr(cache, "key_cache"):
        for i in range(len(cache.key_cache)):
            k = cache.key_cache[i].float().cpu()
            v = cache.value_cache[i].float().cpu()
            kv_by_layer[i] = (k, v)
    elif hasattr(cache, "layers"):
        for i, layer in enumerate(cache.layers):
            k = layer.keys.float().cpu()
            v = layer.values.float().cpu()
            kv_by_layer[i] = (k, v)
    else:
        for i, entry in enumerate(cache):
            k = entry[0].float().cpu()
            v = entry[1].float().cpu()
            kv_by_layer[i] = (k, v)

    print(f"Extracted KV from {len(kv_by_layer)} layers")
    if 0 in kv_by_layer:
        k0 = kv_by_layer[0][0]
        print(f"  Shape: {list(k0.shape)} = [batch, n_kv_heads, seq, head_dim]")
    return kv_by_layer


# ---------------------------------------------------------------------------
# Experiment 1: Intra-group similarity analysis
# ---------------------------------------------------------------------------

def experiment_group_similarity(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    group_sizes: List[int] = [4, 8, 16],
) -> Dict[str, Any]:
    """Measure how similar tokens are within each group.

    For delta coding to work, intra-group cosine must be high (>0.8).
    Higher similarity means tighter deltas, fewer bits needed.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Intra-Group Similarity Analysis")
    print("=" * 70)

    results = {}
    # Test on a representative subset of layers
    test_layers = [0, 6, 12, 18, 24, 30, 35]
    test_layers = [l for l in test_layers if l in kv_by_layer]

    for gs in group_sizes:
        print(f"\n  --- Group size G={gs} ---")
        layer_sims = []
        layer_delta_var_ratios = []

        for layer_idx in test_layers:
            keys, _ = kv_by_layer[layer_idx]
            keys = keys.squeeze(0)  # [n_heads, seq, d]
            n_heads, seq_len, d = keys.shape

            # Average over heads; reshape to (n_heads * seq, d)
            all_keys = keys.reshape(-1, d)
            # Just use first head for speed
            head_keys = keys[0]  # (seq, d)

            # Group by similarity
            group_ids, medoid_mask = greedy_group_by_similarity(head_keys, gs)
            n_groups = group_ids.max().item() + 1

            # Compute intra-group cosine similarity
            norms = head_keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            head_keys_norm = head_keys / norms

            intra_sims = []
            for g in range(n_groups):
                mask = group_ids == g
                group_vecs = head_keys_norm[mask]
                if len(group_vecs) > 1:
                    sim_mat = group_vecs @ group_vecs.T
                    # Upper triangle (exclude diagonal)
                    n = len(group_vecs)
                    upper_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
                    intra_sims.append(sim_mat[upper_mask].mean().item())

            avg_intra_sim = np.mean(intra_sims) if intra_sims else 0.0
            layer_sims.append(avg_intra_sim)

            # Compute delta variance ratio in WHT space
            wht_params = generate_wht_rotation(d, seed=42, device="cpu")
            rotated = apply_wht_rotation(head_keys_norm, wht_params)

            # For each group, compute variance of deltas vs absolute
            abs_var = rotated.var().item()
            delta_vars = []
            for g in range(n_groups):
                mask = group_ids == g
                medoid_in_group = (mask & medoid_mask).nonzero(as_tuple=True)[0]
                if len(medoid_in_group) == 0:
                    continue
                medoid_vec = rotated[medoid_in_group[0]]
                group_vecs = rotated[mask]
                deltas = group_vecs - medoid_vec.unsqueeze(0)
                delta_vars.append(deltas.var().item())

            avg_delta_var = np.mean(delta_vars) if delta_vars else abs_var
            var_ratio = avg_delta_var / (abs_var + 1e-10)
            layer_delta_var_ratios.append(var_ratio)

            if layer_idx % 12 == 0 or layer_idx == test_layers[-1]:
                print(f"    Layer {layer_idx:2d}: intra_cos={avg_intra_sim:.4f} "
                      f"delta_var_ratio={var_ratio:.4f} n_groups={n_groups}")

        avg_sim = np.mean(layer_sims)
        avg_var_ratio = np.mean(layer_delta_var_ratios)
        print(f"    AVERAGE: intra_cos={avg_sim:.4f}  delta_var_ratio={avg_var_ratio:.4f}")

        results[f"G{gs}"] = {
            "per_layer_intra_sim": layer_sims,
            "per_layer_delta_var_ratio": layer_delta_var_ratios,
            "avg_intra_sim": avg_sim,
            "avg_delta_var_ratio": avg_var_ratio,
        }

    return results


# ---------------------------------------------------------------------------
# Experiment 2: DeltaQuant quality vs 3-bit baseline
# ---------------------------------------------------------------------------

def experiment_quality_comparison(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    group_sizes: List[int] = [4, 8],
    delta_bits_list: List[int] = [1, 2],
) -> Dict[str, Any]:
    """Compare DeltaQuant quality against uniform 3-bit TurboQuant.

    Metrics:
        - Cosine similarity of reconstructed keys
        - Attention score correlation (using extracted queries)
        - Top-5 attention index match
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Quality Comparison vs 3-bit Baseline")
    print("=" * 70)

    results = {}
    test_layers = [0, 6, 12, 18, 24, 30, 35]
    test_layers = [l for l in test_layers if l in kv_by_layer]

    # Detect head_dim from data
    sample_keys = kv_by_layer[test_layers[0]][0].squeeze(0)
    d = sample_keys.shape[-1]
    print(f"  Head dimension: d={d}")

    # 3-bit baseline
    baseline_pq = PolarQuant(d, bits=3, seed=42, device="cpu")
    print(f"  Baseline: 3-bit PolarQuant (uniform)")

    for gs in group_sizes:
        for db in delta_bits_list:
            config_name = f"G{gs}_delta{db}bit"
            print(f"\n  --- Config: {config_name} ---")

            layer_cosines_baseline = []
            layer_cosines_delta = []
            layer_attn_corrs_baseline = []
            layer_attn_corrs_delta = []
            layer_top5_baseline = []
            layer_top5_delta = []
            layer_effective_bits = []

            encoder = DeltaQuantEncoder(
                d=d, anchor_bits=3, delta_bits=db,
                group_size=gs, seed=42, device="cpu",
            )

            for layer_idx in test_layers:
                keys, values = kv_by_layer[layer_idx]
                keys = keys.squeeze(0)  # [n_heads, seq, d]
                n_heads, seq_len, head_dim = keys.shape

                # Use first head
                head_keys = keys[0]  # (seq, d)
                norms = head_keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)

                # --- Baseline: 3-bit PolarQuant ---
                head_keys_unit = head_keys / norms
                baseline_recon, _ = baseline_pq.forward(head_keys_unit)
                baseline_recon = baseline_recon * norms

                cos_baseline = F.cosine_similarity(
                    head_keys, baseline_recon, dim=-1
                ).mean().item()

                # --- DeltaQuant ---
                encoded = encoder.encode(head_keys, clustering="greedy")
                delta_recon = encoder.decode(encoded)

                cos_delta = F.cosine_similarity(
                    head_keys, delta_recon, dim=-1
                ).mean().item()

                # --- Attention score comparison ---
                # Use random query vectors (or use last few keys as queries)
                n_queries = min(32, seq_len)
                queries = head_keys[-n_queries:]  # last tokens as queries

                # True attention scores
                true_scores = (queries @ head_keys.T) / math.sqrt(d)
                true_attn = F.softmax(true_scores, dim=-1)

                # Baseline attention
                baseline_scores = (queries @ baseline_recon.T) / math.sqrt(d)
                baseline_attn = F.softmax(baseline_scores, dim=-1)

                # Delta attention
                delta_scores = (queries @ delta_recon.T) / math.sqrt(d)
                delta_attn = F.softmax(delta_scores, dim=-1)

                # Correlation of attention weights
                def attn_corr(a, b):
                    a_flat = a.flatten()
                    b_flat = b.flatten()
                    a_c = a_flat - a_flat.mean()
                    b_c = b_flat - b_flat.mean()
                    return (a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-10)

                corr_baseline = attn_corr(true_attn, baseline_attn).item()
                corr_delta = attn_corr(true_attn, delta_attn).item()

                # Top-5 attention match
                _, true_top5 = true_attn.topk(5, dim=-1)
                _, base_top5 = baseline_attn.topk(5, dim=-1)
                _, delta_top5 = delta_attn.topk(5, dim=-1)

                def top5_match(pred_top5, true_top5):
                    matches = 0
                    total = true_top5.numel()
                    for q in range(true_top5.shape[0]):
                        t_set = set(true_top5[q].tolist())
                        p_set = set(pred_top5[q].tolist())
                        matches += len(t_set & p_set)
                    return matches / total

                top5_base = top5_match(base_top5, true_top5)
                top5_delta = top5_match(delta_top5, true_top5)

                # Effective bits
                eff = encoder.compute_effective_bits(encoded)

                layer_cosines_baseline.append(cos_baseline)
                layer_cosines_delta.append(cos_delta)
                layer_attn_corrs_baseline.append(corr_baseline)
                layer_attn_corrs_delta.append(corr_delta)
                layer_top5_baseline.append(top5_base)
                layer_top5_delta.append(top5_delta)
                layer_effective_bits.append(eff["total_bits_per_dim"])

                if layer_idx % 12 == 0 or layer_idx == test_layers[-1]:
                    print(f"    Layer {layer_idx:2d}: cos_3bit={cos_baseline:.4f} cos_delta={cos_delta:.4f} "
                          f"eff_bits={eff['total_bits_per_dim']:.2f} "
                          f"attn_corr_3bit={corr_baseline:.4f} attn_corr_delta={corr_delta:.4f} "
                          f"top5_3bit={top5_base:.3f} top5_delta={top5_delta:.3f}")

            avg_cos_base = np.mean(layer_cosines_baseline)
            avg_cos_delta = np.mean(layer_cosines_delta)
            avg_corr_base = np.mean(layer_attn_corrs_baseline)
            avg_corr_delta = np.mean(layer_attn_corrs_delta)
            avg_top5_base = np.mean(layer_top5_baseline)
            avg_top5_delta = np.mean(layer_top5_delta)
            avg_eff_bits = np.mean(layer_effective_bits)

            print(f"\n    AVERAGE {config_name}:")
            print(f"      Cosine sim:    3-bit={avg_cos_base:.4f}  delta={avg_cos_delta:.4f}")
            print(f"      Attn corr:     3-bit={avg_corr_base:.4f}  delta={avg_corr_delta:.4f}")
            print(f"      Top-5 match:   3-bit={avg_top5_base:.3f}  delta={avg_top5_delta:.3f}")
            print(f"      Effective bits: {avg_eff_bits:.2f} bits/dim")
            print(f"      Compression:   {16.0 / avg_eff_bits:.1f}x (vs 3-bit: {16.0 / 3.0:.1f}x)")

            results[config_name] = {
                "per_layer_cos_baseline": layer_cosines_baseline,
                "per_layer_cos_delta": layer_cosines_delta,
                "per_layer_attn_corr_baseline": layer_attn_corrs_baseline,
                "per_layer_attn_corr_delta": layer_attn_corrs_delta,
                "per_layer_top5_baseline": layer_top5_baseline,
                "per_layer_top5_delta": layer_top5_delta,
                "per_layer_effective_bits": layer_effective_bits,
                "avg_cos_baseline": avg_cos_base,
                "avg_cos_delta": avg_cos_delta,
                "avg_attn_corr_baseline": avg_corr_base,
                "avg_attn_corr_delta": avg_corr_delta,
                "avg_top5_baseline": avg_top5_base,
                "avg_top5_delta": avg_top5_delta,
                "avg_effective_bits": avg_eff_bits,
                "compression_ratio": 16.0 / avg_eff_bits,
            }

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Delta distribution and entropy analysis
# ---------------------------------------------------------------------------

def experiment_delta_entropy(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    group_sizes: List[int] = [4, 8],
    delta_bits_list: List[int] = [1, 2],
) -> Dict[str, Any]:
    """Analyze the entropy of delta indices.

    If deltas are tightly distributed around zero, entropy coding can
    compress them below the nominal bit rate.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Delta Index Entropy Analysis")
    print("=" * 70)

    results = {}
    test_layers = [0, 12, 24, 35]
    test_layers = [l for l in test_layers if l in kv_by_layer]

    sample_keys = kv_by_layer[test_layers[0]][0].squeeze(0)
    d = sample_keys.shape[-1]

    for gs in group_sizes:
        for db in delta_bits_list:
            config_name = f"G{gs}_delta{db}bit"
            print(f"\n  --- Config: {config_name} ---")

            all_delta_indices = []
            all_anchor_indices = []

            encoder = DeltaQuantEncoder(
                d=d, anchor_bits=3, delta_bits=db,
                group_size=gs, seed=42, device="cpu",
            )

            for layer_idx in test_layers:
                keys, _ = kv_by_layer[layer_idx]
                keys = keys.squeeze(0)
                head_keys = keys[0]  # first head

                encoded = encoder.encode(head_keys, clustering="greedy")
                if encoded["delta_indices"].numel() > 0:
                    all_delta_indices.append(encoded["delta_indices"])
                all_anchor_indices.append(encoded["anchor_indices"])

            # Analyze delta index distribution
            if all_delta_indices:
                combined_deltas = torch.cat(all_delta_indices, dim=0)
                entropy_stats = analyze_delta_entropy(combined_deltas, db)

                print(f"    Delta entropy: {entropy_stats['entropy']:.3f} bits "
                      f"(max={entropy_stats['max_entropy']:.1f}, ratio={entropy_stats['entropy_ratio']:.3f})")
                print(f"    Mode index: {entropy_stats['mode_index']} "
                      f"(probability={entropy_stats['mode_probability']:.3f})")
                print(f"    Effective bits with entropy coding: {entropy_stats['effective_bits_with_entropy']:.3f}")

                # Per-index histogram
                n_levels = 1 << db
                flat = combined_deltas.flatten().long()
                counts = torch.bincount(flat, minlength=n_levels).float()
                probs = counts / counts.sum()
                print(f"    Index distribution:")
                for i in range(n_levels):
                    bar = "#" * int(probs[i].item() * 60)
                    print(f"      [{i}] {probs[i].item():.4f} {bar}")
            else:
                entropy_stats = {"entropy": 0, "entropy_ratio": 0}

            # Analyze anchor index distribution
            combined_anchors = torch.cat(all_anchor_indices, dim=0)
            anchor_entropy = analyze_delta_entropy(combined_anchors, 3)
            print(f"    Anchor entropy: {anchor_entropy['entropy']:.3f} bits "
                  f"(max=3.0, ratio={anchor_entropy['entropy_ratio']:.3f})")

            results[config_name] = {
                "delta_entropy": entropy_stats,
                "anchor_entropy": anchor_entropy,
            }

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Delta tightness in WHT space
# ---------------------------------------------------------------------------

def experiment_delta_tightness(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    group_sizes: List[int] = [4, 8, 16],
) -> Dict[str, Any]:
    """Measure how tight the deltas are in WHT space.

    Theory predicts: for tokens with cosine similarity c,
    delta variance = (1 - c^2) * (1/d) per coordinate.

    We verify this empirically and check if the actual delta
    distribution matches the predicted tightness.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Delta Tightness in WHT Space")
    print("=" * 70)

    results = {}
    test_layers = [0, 12, 24, 35]
    test_layers = [l for l in test_layers if l in kv_by_layer]

    sample_keys = kv_by_layer[test_layers[0]][0].squeeze(0)
    d = sample_keys.shape[-1]
    wht_params = generate_wht_rotation(d, seed=42, device="cpu")

    for gs in group_sizes:
        print(f"\n  --- Group size G={gs} ---")
        all_actual_vars = []
        all_predicted_vars = []
        all_cosines = []

        for layer_idx in test_layers:
            keys, _ = kv_by_layer[layer_idx]
            keys = keys.squeeze(0)
            head_keys = keys[0]  # (seq, d)
            norms = head_keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            head_keys_unit = head_keys / norms

            # WHT rotate
            rotated = apply_wht_rotation(head_keys_unit, wht_params)

            # Group
            group_ids, medoid_mask = greedy_group_by_similarity(head_keys_unit, gs)
            n_groups = group_ids.max().item() + 1

            for g in range(n_groups):
                mask = group_ids == g
                group_vecs = head_keys_unit[mask]
                if group_vecs.shape[0] < 2:
                    continue

                # Find medoid
                medoid_candidates = (mask & medoid_mask).nonzero(as_tuple=True)[0]
                if len(medoid_candidates) == 0:
                    continue
                medoid_idx = medoid_candidates[0]
                medoid_vec = head_keys_unit[medoid_idx]

                # Cosine similarities within group
                cos_sims = F.cosine_similarity(
                    group_vecs, medoid_vec.unsqueeze(0), dim=-1
                )
                non_self = cos_sims[cos_sims < 0.999]
                if len(non_self) == 0:
                    continue

                mean_cos = non_self.mean().item()
                all_cosines.append(mean_cos)

                # Predicted delta variance per coordinate
                predicted_var = (1 - mean_cos ** 2) / d
                all_predicted_vars.append(predicted_var)

                # Actual delta variance in WHT space
                medoid_rotated = rotated[medoid_idx]
                group_rotated = rotated[mask]
                deltas = group_rotated - medoid_rotated.unsqueeze(0)
                # Exclude medoid's own delta (zero)
                non_medoid = (mask.nonzero(as_tuple=True)[0] != medoid_idx)
                if non_medoid.any():
                    actual_deltas = deltas[non_medoid]
                    actual_var = actual_deltas.var().item()
                    all_actual_vars.append(actual_var)

        if all_cosines:
            avg_cos = np.mean(all_cosines)
            avg_pred_var = np.mean(all_predicted_vars)
            avg_actual_var = np.mean(all_actual_vars)
            var_prediction_ratio = avg_actual_var / (avg_pred_var + 1e-10)

            print(f"    Avg intra-group cosine: {avg_cos:.4f}")
            print(f"    Predicted delta var/coord: {avg_pred_var:.6f}")
            print(f"    Actual delta var/coord:    {avg_actual_var:.6f}")
            print(f"    Prediction ratio:          {var_prediction_ratio:.3f} (1.0 = perfect)")
            print(f"    Delta std vs anchor std:   {math.sqrt(avg_actual_var) / (1.0 / math.sqrt(d)):.3f}")

            # Cosine distribution
            cos_arr = np.array(all_cosines)
            print(f"    Cosine distribution: min={cos_arr.min():.3f} "
                  f"p25={np.percentile(cos_arr, 25):.3f} "
                  f"p50={np.percentile(cos_arr, 50):.3f} "
                  f"p75={np.percentile(cos_arr, 75):.3f} "
                  f"max={cos_arr.max():.3f}")

            results[f"G{gs}"] = {
                "avg_intra_cos": avg_cos,
                "avg_predicted_var": avg_pred_var,
                "avg_actual_var": avg_actual_var,
                "var_prediction_ratio": var_prediction_ratio,
                "cosine_percentiles": {
                    "min": float(cos_arr.min()),
                    "p25": float(np.percentile(cos_arr, 25)),
                    "p50": float(np.percentile(cos_arr, 50)),
                    "p75": float(np.percentile(cos_arr, 75)),
                    "max": float(cos_arr.max()),
                },
            }

    return results


# ---------------------------------------------------------------------------
# Experiment 5: Compression-quality Pareto frontier
# ---------------------------------------------------------------------------

def experiment_pareto_frontier(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    """Sweep configurations to find the Pareto frontier of bits vs quality.

    Configurations tested:
        - Baseline: 2-bit, 3-bit, 4-bit uniform PolarQuant
        - DeltaQuant: G={4,8,16} x delta_bits={1,2} x anchor_bits={2,3}
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Compression-Quality Pareto Frontier")
    print("=" * 70)

    test_layers = [0, 12, 24, 35]
    test_layers = [l for l in test_layers if l in kv_by_layer]
    sample_keys = kv_by_layer[test_layers[0]][0].squeeze(0)
    d = sample_keys.shape[-1]

    configs = []

    # Baselines: uniform PolarQuant at various bit rates
    for bits in [2, 3, 4]:
        configs.append({
            "name": f"PolarQuant-{bits}bit",
            "type": "baseline",
            "bits": bits,
        })

    # DeltaQuant configurations
    for gs in [4, 8, 16]:
        for ab in [2, 3]:
            for db in [1, 2]:
                configs.append({
                    "name": f"Delta-G{gs}-a{ab}b-d{db}b",
                    "type": "delta",
                    "group_size": gs,
                    "anchor_bits": ab,
                    "delta_bits": db,
                })

    results = []

    for cfg in configs:
        cos_sims = []
        eff_bits = []

        if cfg["type"] == "baseline":
            pq = PolarQuant(d, bits=cfg["bits"], seed=42, device="cpu")
            for layer_idx in test_layers:
                keys, _ = kv_by_layer[layer_idx]
                keys = keys.squeeze(0)
                head_keys = keys[0]
                norms = head_keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                head_keys_unit = head_keys / norms
                recon, _ = pq.forward(head_keys_unit)
                recon = recon * norms
                cos = F.cosine_similarity(head_keys, recon, dim=-1).mean().item()
                cos_sims.append(cos)
                eff_bits.append(float(cfg["bits"]))

        else:
            encoder = DeltaQuantEncoder(
                d=d,
                anchor_bits=cfg["anchor_bits"],
                delta_bits=cfg["delta_bits"],
                group_size=cfg["group_size"],
                seed=42, device="cpu",
            )
            for layer_idx in test_layers:
                keys, _ = kv_by_layer[layer_idx]
                keys = keys.squeeze(0)
                head_keys = keys[0]
                encoded = encoder.encode(head_keys, clustering="greedy")
                recon = encoder.decode(encoded)
                cos = F.cosine_similarity(head_keys, recon, dim=-1).mean().item()
                eff = encoder.compute_effective_bits(encoded)
                cos_sims.append(cos)
                eff_bits.append(eff["total_bits_per_dim"])

        avg_cos = np.mean(cos_sims)
        avg_bits = np.mean(eff_bits)
        compression = 16.0 / avg_bits

        results.append({
            "name": cfg["name"],
            "avg_cosine": avg_cos,
            "avg_bits": avg_bits,
            "compression": compression,
        })

        print(f"  {cfg['name']:30s}: cos={avg_cos:.4f}  bits={avg_bits:.2f}  "
              f"compression={compression:.1f}x")

    # Sort by bits for Pareto analysis
    results.sort(key=lambda x: x["avg_bits"])
    print("\n  Pareto frontier (sorted by bits):")
    print(f"  {'Config':<30s} {'Bits':>6s} {'Cosine':>8s} {'Compress':>8s} {'Pareto?':>8s}")
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    best_cos = -1.0
    pareto_points = []
    for r in results:
        is_pareto = r["avg_cosine"] > best_cos
        if is_pareto:
            best_cos = r["avg_cosine"]
            pareto_points.append(r)
        print(f"  {r['name']:<30s} {r['avg_bits']:>6.2f} {r['avg_cosine']:>8.4f} "
              f"{r['compression']:>7.1f}x {'  YES' if is_pareto else '   no':>8s}")

    return {"all_configs": results, "pareto_points": pareto_points}


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------

def write_results(
    exp1: Dict, exp2: Dict, exp3: Dict, exp4: Dict, exp5: Dict,
    seq_len: int, n_layers: int, runtime: float,
):
    """Write results to benchmarks/results/delta_quant_results.md."""
    out_path = os.path.join(REPO_ROOT, "benchmarks", "results", "delta_quant_results.md")

    with open(out_path, "w") as f:
        f.write("# DeltaQuant Experiment Results\n\n")
        f.write(f"**Model:** {MODEL_NAME}\n")
        f.write(f"**Prompt tokens:** {seq_len}\n")
        f.write(f"**Layers analyzed:** {n_layers}\n")
        f.write(f"**Runtime:** {runtime:.1f}s\n\n")

        # Experiment 1
        f.write("## Experiment 1: Intra-Group Similarity\n\n")
        f.write("How similar are tokens within each group? Higher = tighter deltas.\n\n")
        f.write("| Group Size | Avg Intra-Group Cosine | Avg Delta Var Ratio | Viable? |\n")
        f.write("|-----------|------------------------|--------------------|---------|\n")
        for gs_key, data in sorted(exp1.items()):
            viable = "YES" if data["avg_delta_var_ratio"] < 0.5 else "NO"
            f.write(f"| {gs_key} | {data['avg_intra_sim']:.4f} | "
                    f"{data['avg_delta_var_ratio']:.4f} | {viable} |\n")

        # Experiment 2
        f.write("\n## Experiment 2: Quality vs 3-bit Baseline\n\n")
        f.write("| Config | Cosine (3-bit) | Cosine (delta) | Attn Corr (3-bit) | "
                "Attn Corr (delta) | Top-5 (3-bit) | Top-5 (delta) | Eff Bits | Compression |\n")
        f.write("|--------|----------------|----------------|--------------------"
                "|-------------------|---------------|---------------|----------|-------------|\n")
        for cfg_name, data in sorted(exp2.items()):
            f.write(f"| {cfg_name} | {data['avg_cos_baseline']:.4f} | "
                    f"{data['avg_cos_delta']:.4f} | "
                    f"{data['avg_attn_corr_baseline']:.4f} | "
                    f"{data['avg_attn_corr_delta']:.4f} | "
                    f"{data['avg_top5_baseline']:.3f} | "
                    f"{data['avg_top5_delta']:.3f} | "
                    f"{data['avg_effective_bits']:.2f} | "
                    f"{data['compression_ratio']:.1f}x |\n")

        # Experiment 3
        f.write("\n## Experiment 3: Delta Entropy Analysis\n\n")
        f.write("Lower entropy = more compressible with entropy coding.\n\n")
        f.write("| Config | Delta Entropy | Max Entropy | Ratio | Mode Prob | "
                "Eff Bits (entropy) |\n")
        f.write("|--------|---------------|-------------|-------|-----------|---"
                "-------------------|\n")
        for cfg_name, data in sorted(exp3.items()):
            de = data["delta_entropy"]
            if isinstance(de, dict) and "entropy" in de:
                f.write(f"| {cfg_name} | {de.get('entropy', 0):.3f} | "
                        f"{de.get('max_entropy', 0):.1f} | "
                        f"{de.get('entropy_ratio', 0):.3f} | "
                        f"{de.get('mode_probability', 0):.3f} | "
                        f"{de.get('effective_bits_with_entropy', 0):.3f} |\n")

        # Experiment 4
        f.write("\n## Experiment 4: Delta Tightness in WHT Space\n\n")
        f.write("Theory: delta var = (1 - cos^2) / d. Prediction ratio near 1.0 = theory holds.\n\n")
        f.write("| Group Size | Avg Intra-Cos | Predicted Var | Actual Var | "
                "Prediction Ratio |\n")
        f.write("|-----------|---------------|---------------|------------|---"
                "----------------|\n")
        for gs_key, data in sorted(exp4.items()):
            f.write(f"| {gs_key} | {data['avg_intra_cos']:.4f} | "
                    f"{data['avg_predicted_var']:.6f} | "
                    f"{data['avg_actual_var']:.6f} | "
                    f"{data['var_prediction_ratio']:.3f} |\n")

        # Experiment 5
        f.write("\n## Experiment 5: Pareto Frontier\n\n")
        f.write("| Config | Eff Bits | Cosine | Compression | Pareto? |\n")
        f.write("|--------|----------|--------|-------------|--------|\n")
        pareto_names = {p["name"] for p in exp5.get("pareto_points", [])}
        for r in exp5.get("all_configs", []):
            is_p = "YES" if r["name"] in pareto_names else "no"
            f.write(f"| {r['name']} | {r['avg_bits']:.2f} | "
                    f"{r['avg_cosine']:.4f} | {r['compression']:.1f}x | {is_p} |\n")

        # Summary verdict
        f.write("\n## Summary Verdict\n\n")

        # Find best DeltaQuant config that matches 3-bit quality
        baseline_3bit_cos = None
        for r in exp5.get("all_configs", []):
            if r["name"] == "PolarQuant-3bit":
                baseline_3bit_cos = r["avg_cosine"]
                break

        best_delta = None
        if baseline_3bit_cos is not None:
            matching = [r for r in exp5.get("all_configs", [])
                       if r["name"].startswith("Delta-") and r["avg_cosine"] >= baseline_3bit_cos * 0.99]
            if matching:
                matching.sort(key=lambda x: x["avg_bits"])
                best_delta = matching[0]

        if best_delta:
            f.write(f"**Best DeltaQuant matching 3-bit quality:** {best_delta['name']}\n")
            f.write(f"- Effective bits: {best_delta['avg_bits']:.2f}\n")
            f.write(f"- Cosine similarity: {best_delta['avg_cosine']:.4f} "
                    f"(vs 3-bit: {baseline_3bit_cos:.4f})\n")
            f.write(f"- Compression ratio: {best_delta['compression']:.1f}x "
                    f"(vs 3-bit: {16.0/3.0:.1f}x)\n")
        else:
            f.write("No DeltaQuant config matched 3-bit quality within 1%.\n")

            # Find best anyway
            all_delta = [r for r in exp5.get("all_configs", [])
                        if r["name"].startswith("Delta-")]
            if all_delta:
                best_quality = max(all_delta, key=lambda x: x["avg_cosine"])
                best_compress = min(all_delta, key=lambda x: x["avg_bits"])
                f.write(f"\nBest quality DeltaQuant: {best_quality['name']} "
                        f"(cos={best_quality['avg_cosine']:.4f}, "
                        f"bits={best_quality['avg_bits']:.2f})\n")
                f.write(f"Best compression DeltaQuant: {best_compress['name']} "
                        f"(cos={best_compress['avg_cosine']:.4f}, "
                        f"bits={best_compress['avg_bits']:.2f})\n")

        if baseline_3bit_cos is not None:
            f.write(f"\n3-bit PolarQuant baseline cosine: {baseline_3bit_cos:.4f}\n")

    print(f"\n  Results written to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("=" * 70)
    print("DELTAQUANT: Cross-Token Delta Coding in WHT-Rotated Space")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Key idea: Cluster tokens by similarity, encode medoids at 3-bit,")
    print(f"          encode deltas at 1-2 bits. NO cumulative error.")
    print()

    # Load model and extract KV caches
    model, tokenizer = load_model()
    kv_by_layer = extract_kv_caches(model, tokenizer, PROMPT)

    # Get metadata
    sample_k = kv_by_layer[0][0]
    seq_len = sample_k.shape[2]
    n_layers = len(kv_by_layer)

    # Free model memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Run experiments
    exp1 = experiment_group_similarity(kv_by_layer)
    exp2 = experiment_quality_comparison(kv_by_layer)
    exp3 = experiment_delta_entropy(kv_by_layer)
    exp4 = experiment_delta_tightness(kv_by_layer)
    exp5 = experiment_pareto_frontier(kv_by_layer)

    runtime = time.time() - t0

    # Write results
    write_results(exp1, exp2, exp3, exp4, exp5, seq_len, n_layers, runtime)

    print(f"\n{'=' * 70}")
    print(f"Total runtime: {runtime:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
