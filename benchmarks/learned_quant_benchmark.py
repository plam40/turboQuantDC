"""
Learned Quantization Benchmark: Attention-Optimal KV Cache Compression
======================================================================

Benchmarks the differentiable learned quantizer against all fixed approaches
on real Qwen2.5-3B-Instruct KV caches.

Configurations tested at 3-bit:
  1. Random Givens rotation (baseline)
  2. Random Givens + mean-removal (current best fixed approach)
  3. Learned rotation only (50 calibration steps)
  4. Learned rotation + mean-removal (full stack)
  5. Learned rotation + learned centroids + mean-removal

Sweeps:
  - Calibration steps: 10, 25, 50, 100
  - Calibration tokens: 32, 64, 128, 256
  - Transfer: calibrate on prompt A, test on prompt B

Metrics: attention cosine, top-1, top-5, KL divergence from FP16
"""

import sys
import os
import time
import math
import json
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/dhawal/turboQuantDC")

from turboquantdc.learned_quant import LearnedQuantizer
from turboquantdc.residual_quant import ResidualQuantEstimator
from turboquantdc.attention_optimal import compute_attention_scores, attention_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"
SEED = 42
HEAD_DIM = 128
BITS = 3

RESULTS_DIR = Path("/home/dhawal/turboQuantDC/benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Step 1: Extract real KV caches
# ============================================================================

def load_model_and_extract_kv(prompt_text: str, n_tokens: int = 256):
    """Load Qwen2.5-3B and extract real Q/K per layer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
    )
    model.eval()

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=n_tokens,
        truncation=True,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]
    gen_target = max(n_tokens, prompt_len + 32)
    gen_tokens = gen_target - prompt_len

    print(f"  Prompt tokens: {prompt_len}, generating {gen_tokens} more...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )

    past_kv = outputs.past_key_values
    total_tokens = outputs.sequences.shape[1]
    print(f"  Total tokens in cache: {total_tokens}")

    # Extract per-layer K and derive Q (use K as Q proxy for self-attention)
    layer_data = {}

    # Handle different cache formats
    if hasattr(past_kv, "layers"):
        # New DynamicCache: past_kv.layers[i].keys / .values
        n_layers = len(past_kv.layers)
        for layer_idx in range(n_layers):
            layer = past_kv.layers[layer_idx]
            if not hasattr(layer, "keys") or layer.keys.numel() == 0:
                continue
            K_all = layer.keys  # (batch, n_heads, seq, head_dim)
            K = K_all[0, 0].float().to(DEVICE)
            Q = K.clone()
            layer_data[layer_idx] = {"Q": Q, "K": K}
    elif hasattr(past_kv, "key_cache"):
        # Older DynamicCache with key_cache list
        n_layers = len(past_kv.key_cache)
        for layer_idx in range(n_layers):
            K_all = past_kv.key_cache[layer_idx]
            K = K_all[0, 0].float().to(DEVICE)
            Q = K.clone()
            layer_data[layer_idx] = {"Q": Q, "K": K}
    elif isinstance(past_kv, (list, tuple)):
        n_layers = len(past_kv)
        for layer_idx in range(n_layers):
            K_all = past_kv[layer_idx][0]
            K = K_all[0, 0].float().to(DEVICE)
            Q = K.clone()
            layer_data[layer_idx] = {"Q": Q, "K": K}
    else:
        raise RuntimeError(f"Unsupported cache type: {type(past_kv)}")

    return layer_data, model, tokenizer


# ============================================================================
# Step 2: Quantizer configurations
# ============================================================================

def evaluate_config(Q, K, config_name, quantizer_fn):
    """Evaluate a quantizer configuration on real Q/K."""
    attn_true = compute_attention_scores(Q, K)

    t0 = time.perf_counter()
    K_quant = quantizer_fn(K)
    quant_time = time.perf_counter() - t0

    attn_quant = compute_attention_scores(Q, K_quant)
    metrics = attention_metrics(attn_true, attn_quant)
    metrics["quant_time_ms"] = quant_time * 1000
    return metrics


def run_baseline_configs(Q, K, bits=BITS):
    """Run all fixed (non-learned) baseline configurations."""
    results = {}

    # 1. Random Givens (no learning, no mean-removal)
    rq_givens = ResidualQuantEstimator(
        d=HEAD_DIM, bits=bits, seed=SEED, device=DEVICE,
        center_before_quantize=False, rotation_type="givens",
    )
    def givens_quant(K):
        comp = rq_givens.quantize(K)
        return rq_givens.dequantize(comp)
    results["random_givens"] = evaluate_config(Q, K, "random_givens", givens_quant)

    # 2. Random Givens + mean-removal
    rq_givens_mean = ResidualQuantEstimator(
        d=HEAD_DIM, bits=bits, seed=SEED, device=DEVICE,
        center_before_quantize=True, rotation_type="givens",
    )
    def givens_mean_quant(K):
        comp = rq_givens_mean.quantize(K)
        return rq_givens_mean.dequantize(comp)
    results["givens_mean_removal"] = evaluate_config(Q, K, "givens+mean", givens_mean_quant)

    # 3. WHT baseline + mean-removal
    rq_wht = ResidualQuantEstimator(
        d=HEAD_DIM, bits=bits, seed=SEED, device=DEVICE,
        center_before_quantize=True,
    )
    def wht_mean_quant(K):
        comp = rq_wht.quantize(K)
        return rq_wht.dequantize(comp)
    results["wht_mean_removal"] = evaluate_config(Q, K, "wht+mean", wht_mean_quant)

    return results


def run_learned_configs(Q, K, bits=BITS, steps=50, lr=0.01):
    """Run learned quantizer configurations."""
    results = {}

    # 3. Learned rotation only
    lq_rot = LearnedQuantizer(d=HEAD_DIM, bits=bits, center=False, seed=SEED, device=DEVICE)
    t0 = time.perf_counter()
    losses_rot = lq_rot.calibrate(Q, K, lr=lr, steps=steps)
    cal_time_rot = time.perf_counter() - t0

    def learned_rot_quant(K):
        return lq_rot.forward(K).detach()
    metrics = evaluate_config(Q, K, "learned_rotation", learned_rot_quant)
    metrics["calibration_time_ms"] = cal_time_rot * 1000
    metrics["initial_kl"] = losses_rot[0]
    metrics["best_kl"] = min(losses_rot)
    results["learned_rotation"] = metrics

    # 4. Learned rotation + mean-removal
    lq_mean = LearnedQuantizer(d=HEAD_DIM, bits=bits, center=True, seed=SEED, device=DEVICE)
    t0 = time.perf_counter()
    losses_mean = lq_mean.calibrate(Q, K, lr=lr, steps=steps)
    cal_time_mean = time.perf_counter() - t0

    def learned_mean_quant(K):
        return lq_mean.forward(K).detach()
    metrics = evaluate_config(Q, K, "learned_rotation+mean", learned_mean_quant)
    metrics["calibration_time_ms"] = cal_time_mean * 1000
    metrics["initial_kl"] = losses_mean[0]
    metrics["best_kl"] = min(losses_mean)
    results["learned_rotation_mean"] = metrics

    # 5. Learned rotation + learned centroids + mean-removal
    lq_full = LearnedQuantizer(
        d=HEAD_DIM, bits=bits, center=True, learn_centroids=True,
        seed=SEED, device=DEVICE,
    )
    t0 = time.perf_counter()
    losses_full = lq_full.calibrate(Q, K, lr=lr, steps=steps)
    cal_time_full = time.perf_counter() - t0

    def learned_full_quant(K):
        return lq_full.forward(K).detach()
    metrics = evaluate_config(Q, K, "learned_full", learned_full_quant)
    metrics["calibration_time_ms"] = cal_time_full * 1000
    metrics["initial_kl"] = losses_full[0]
    metrics["best_kl"] = min(losses_full)
    results["learned_full"] = metrics

    return results


# ============================================================================
# Step 3: Calibration sweeps
# ============================================================================

def sweep_calibration_steps(Q, K, step_counts=[10, 25, 50, 100]):
    """How many calibration steps are needed?"""
    results = {}
    for steps in step_counts:
        lq = LearnedQuantizer(d=HEAD_DIM, bits=BITS, center=True, seed=SEED, device=DEVICE)
        t0 = time.perf_counter()
        losses = lq.calibrate(Q, K, lr=0.01, steps=steps)
        cal_time = time.perf_counter() - t0

        attn_true = compute_attention_scores(Q, K)
        K_q = lq.forward(K).detach()
        attn_q = compute_attention_scores(Q, K_q)
        metrics = attention_metrics(attn_true, attn_q)
        metrics["calibration_time_ms"] = cal_time * 1000
        metrics["best_kl_during_training"] = min(losses)
        results[f"{steps}_steps"] = metrics

    return results


def sweep_calibration_tokens(layer_data, model, tokenizer, token_counts=[32, 64, 128, 256]):
    """How many calibration tokens are needed?"""
    results = {}

    # Use layer 8 (middle layer) for this sweep
    test_layer = min(8, max(layer_data.keys()))
    Q_test = layer_data[test_layer]["Q"]
    K_test = layer_data[test_layer]["K"]

    for n_tok in token_counts:
        # Use first n_tok tokens for calibration, full set for eval
        n_cal = min(n_tok, K_test.shape[0])
        Q_cal = Q_test[:n_cal]
        K_cal = K_test[:n_cal]

        lq = LearnedQuantizer(d=HEAD_DIM, bits=BITS, center=True, seed=SEED, device=DEVICE)
        t0 = time.perf_counter()
        losses = lq.calibrate(Q_cal, K_cal, lr=0.01, steps=50)
        cal_time = time.perf_counter() - t0

        # Evaluate on FULL context
        lq._update_running_mean(K_test)
        attn_true = compute_attention_scores(Q_test, K_test)
        K_q = lq.forward(K_test).detach()
        attn_q = compute_attention_scores(Q_test, K_q)
        metrics = attention_metrics(attn_true, attn_q)
        metrics["calibration_time_ms"] = cal_time * 1000
        metrics["n_cal_tokens"] = n_cal
        metrics["n_eval_tokens"] = K_test.shape[0]
        results[f"{n_tok}_tokens"] = metrics

    return results


# ============================================================================
# Step 4: Transfer test
# ============================================================================

def test_transfer(model, tokenizer):
    """Calibrate on prompt A, test on prompt B."""
    prompt_a = (
        "Explain the theory of general relativity in detail, covering "
        "spacetime curvature, the equivalence principle, gravitational "
        "waves, and the experimental confirmations from Mercury's orbit "
        "to LIGO. Include the mathematical framework of tensors."
    )
    prompt_b = (
        "Write a comprehensive guide to machine learning, covering "
        "supervised vs unsupervised learning, neural network architectures, "
        "gradient descent, backpropagation, regularization techniques, "
        "and the transformer architecture revolution."
    )

    print("\n  Extracting KV from prompt A (calibration)...")
    layer_data_a, _, _ = load_model_and_extract_kv(prompt_a, n_tokens=128)

    print("  Extracting KV from prompt B (evaluation)...")
    layer_data_b, _, _ = load_model_and_extract_kv(prompt_b, n_tokens=128)

    # Test on a few layers
    test_layers = [0, 8, 16, max(layer_data_a.keys())]
    test_layers = sorted(set(l for l in test_layers if l in layer_data_a and l in layer_data_b))

    results = {}
    for layer_idx in test_layers:
        Q_a, K_a = layer_data_a[layer_idx]["Q"], layer_data_a[layer_idx]["K"]
        Q_b, K_b = layer_data_b[layer_idx]["Q"], layer_data_b[layer_idx]["K"]

        # Calibrate on A
        lq = LearnedQuantizer(d=HEAD_DIM, bits=BITS, center=True, seed=SEED, device=DEVICE)
        lq.calibrate(Q_a, K_a, lr=0.01, steps=50)

        # Evaluate on B (reset running mean)
        lq.running_mean.zero_()
        lq.running_count.zero_()
        lq._update_running_mean(K_b)

        attn_true_b = compute_attention_scores(Q_b, K_b)
        K_b_quant = lq.forward(K_b).detach()
        attn_quant_b = compute_attention_scores(Q_b, K_b_quant)
        metrics_transfer = attention_metrics(attn_true_b, attn_quant_b)

        # Baseline: no calibration on B
        lq_baseline = LearnedQuantizer(d=HEAD_DIM, bits=BITS, center=True, seed=SEED, device=DEVICE)
        lq_baseline._update_running_mean(K_b)
        K_b_base = lq_baseline.forward(K_b).detach()
        attn_base_b = compute_attention_scores(Q_b, K_b_base)
        metrics_baseline = attention_metrics(attn_true_b, attn_base_b)

        # Also: calibrate directly on B (upper bound)
        lq_direct = LearnedQuantizer(d=HEAD_DIM, bits=BITS, center=True, seed=SEED, device=DEVICE)
        lq_direct.calibrate(Q_b, K_b, lr=0.01, steps=50)
        K_b_direct = lq_direct.forward(K_b).detach()
        attn_direct_b = compute_attention_scores(Q_b, K_b_direct)
        metrics_direct = attention_metrics(attn_true_b, attn_direct_b)

        results[f"layer_{layer_idx}"] = {
            "no_calibration": metrics_baseline,
            "transfer_from_A": metrics_transfer,
            "calibrated_on_B": metrics_direct,
        }

    return results


# ============================================================================
# Step 5: Main
# ============================================================================

def format_metrics(metrics, prefix=""):
    """Format metrics for printing."""
    parts = []
    for k in ["cosine_sim", "top1_match", "top5_match", "kl_div", "spearman_rho"]:
        if k in metrics:
            if "kl" in k:
                parts.append(f"{k}={metrics[k]:.6f}")
            elif "time" in k:
                parts.append(f"{k}={metrics[k]:.1f}ms")
            else:
                parts.append(f"{k}={metrics[k]:.4f}")
    return f"{prefix}{', '.join(parts)}"


def main():
    print("=" * 70)
    print("LEARNED QUANTIZATION BENCHMARK")
    print("Differentiable attention-optimal KV cache compression")
    print("=" * 70)

    prompt = (
        "You are a world-class computer scientist giving a comprehensive lecture on "
        "the history of computing, starting from Charles Babbage's Analytical Engine "
        "through Alan Turing's theoretical foundations, the development of ENIAC, "
        "the transistor revolution, the birth of the internet at ARPANET, the rise "
        "of personal computing with Apple and IBM, the open source movement with "
        "Linux, the mobile revolution with iPhone, cloud computing with AWS, and "
        "finally the current AI revolution with large language models. Cover the "
        "key innovations, the people behind them, and the societal impact."
    )

    print("\nStep 1: Loading model and extracting KV caches...")
    layer_data, model, tokenizer = load_model_and_extract_kv(prompt, n_tokens=256)
    n_layers = len(layer_data)
    print(f"  Extracted {n_layers} layers, head_dim={HEAD_DIM}")

    # Test on representative layers
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    test_layers = sorted(set(test_layers))

    all_results = {"config": {}, "step_sweep": {}, "token_sweep": {}, "transfer": {}}

    # ---- Main comparison ----
    print("\n" + "=" * 70)
    print("Step 2: Main comparison (3-bit, per layer)")
    print("=" * 70)

    for layer_idx in test_layers:
        Q = layer_data[layer_idx]["Q"]
        K = layer_data[layer_idx]["K"]
        seq_len = K.shape[0]

        print(f"\n--- Layer {layer_idx} (seq_len={seq_len}) ---")

        # Baselines
        baseline_results = run_baseline_configs(Q, K)
        for name, metrics in baseline_results.items():
            print(f"  {name:30s}: {format_metrics(metrics)}")

        # Learned
        learned_results = run_learned_configs(Q, K, steps=50)
        for name, metrics in learned_results.items():
            cal_ms = metrics.get("calibration_time_ms", 0)
            print(f"  {name:30s}: {format_metrics(metrics)}, cal={cal_ms:.0f}ms")

        all_results["config"][f"layer_{layer_idx}"] = {
            **baseline_results,
            **learned_results,
        }

    # ---- Step sweep ----
    print("\n" + "=" * 70)
    print("Step 3: Calibration step sweep (layer 0)")
    print("=" * 70)

    Q0 = layer_data[0]["Q"]
    K0 = layer_data[0]["K"]
    step_results = sweep_calibration_steps(Q0, K0, step_counts=[10, 25, 50, 100])
    for name, metrics in step_results.items():
        cal_ms = metrics.get("calibration_time_ms", 0)
        print(f"  {name:15s}: {format_metrics(metrics)}, cal={cal_ms:.0f}ms")
    all_results["step_sweep"] = step_results

    # ---- Token sweep ----
    print("\n" + "=" * 70)
    print("Step 4: Calibration token sweep")
    print("=" * 70)

    token_results = sweep_calibration_tokens(layer_data, model, tokenizer)
    for name, metrics in token_results.items():
        n_cal = metrics.get("n_cal_tokens", "?")
        n_eval = metrics.get("n_eval_tokens", "?")
        print(f"  {name:15s} (cal={n_cal}, eval={n_eval}): {format_metrics(metrics)}")
    all_results["token_sweep"] = token_results

    # ---- Transfer test ----
    print("\n" + "=" * 70)
    print("Step 5: Transfer test (calibrate A, evaluate B)")
    print("=" * 70)

    transfer_results = test_transfer(model, tokenizer)
    for layer_name, configs in transfer_results.items():
        print(f"\n  {layer_name}:")
        for config_name, metrics in configs.items():
            print(f"    {config_name:25s}: {format_metrics(metrics)}")
    all_results["transfer"] = transfer_results

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Aggregate across layers
    config_names = ["random_givens", "givens_mean_removal", "wht_mean_removal",
                    "learned_rotation", "learned_rotation_mean", "learned_full"]
    agg = {name: {"cosine_sim": [], "kl_div": [], "top5_match": []} for name in config_names}

    for layer_name, layer_results in all_results["config"].items():
        for config_name in config_names:
            if config_name in layer_results:
                for metric in ["cosine_sim", "kl_div", "top5_match"]:
                    agg[config_name][metric].append(layer_results[config_name][metric])

    print(f"\n{'Config':35s} | {'Cosine':>8s} | {'KL Div':>10s} | {'Top-5':>8s}")
    print("-" * 70)
    for name in config_names:
        if agg[name]["cosine_sim"]:
            cos = sum(agg[name]["cosine_sim"]) / len(agg[name]["cosine_sim"])
            kl = sum(agg[name]["kl_div"]) / len(agg[name]["kl_div"])
            t5 = sum(agg[name]["top5_match"]) / len(agg[name]["top5_match"])
            print(f"  {name:33s} | {cos:>8.4f} | {kl:>10.6f} | {t5:>8.4f}")

    # Save results
    results_path = RESULTS_DIR / "learned_quant_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Generate markdown report
    generate_report(all_results)


def generate_report(all_results):
    """Generate markdown report."""
    lines = []
    lines.append("# Learned Quantization Results")
    lines.append("")
    lines.append("## Differentiable Attention-Optimal KV Cache Compression")
    lines.append("")
    lines.append("**Model:** Qwen2.5-3B-Instruct (BnB 4-bit)")
    lines.append(f"**Bits:** {BITS}")
    lines.append(f"**Head dim:** {HEAD_DIM}")
    lines.append("")

    # Main comparison table
    lines.append("## Main Comparison (3-bit)")
    lines.append("")
    lines.append("| Config | Cosine | KL Div | Top-1 | Top-5 | Spearman |")
    lines.append("|--------|--------|--------|-------|-------|----------|")

    config_names = ["random_givens", "givens_mean_removal", "wht_mean_removal",
                    "learned_rotation", "learned_rotation_mean", "learned_full"]

    agg = {name: {"cosine_sim": [], "kl_div": [], "top1_match": [], "top5_match": [], "spearman_rho": []}
           for name in config_names}

    for layer_name, layer_results in all_results["config"].items():
        for config_name in config_names:
            if config_name in layer_results:
                for metric in agg[config_name]:
                    if metric in layer_results[config_name]:
                        agg[config_name][metric].append(layer_results[config_name][metric])

    for name in config_names:
        if agg[name]["cosine_sim"]:
            avg = lambda lst: sum(lst) / len(lst) if lst else 0
            cos = avg(agg[name]["cosine_sim"])
            kl = avg(agg[name]["kl_div"])
            t1 = avg(agg[name]["top1_match"])
            t5 = avg(agg[name]["top5_match"])
            sp = avg(agg[name]["spearman_rho"])
            lines.append(f"| {name} | {cos:.4f} | {kl:.6f} | {t1:.4f} | {t5:.4f} | {sp:.4f} |")

    lines.append("")

    # Step sweep
    if all_results.get("step_sweep"):
        lines.append("## Calibration Steps Sweep")
        lines.append("")
        lines.append("| Steps | Cosine | KL Div | Top-5 | Cal Time (ms) |")
        lines.append("|-------|--------|--------|-------|---------------|")
        for name, metrics in all_results["step_sweep"].items():
            cos = metrics.get("cosine_sim", 0)
            kl = metrics.get("kl_div", 0)
            t5 = metrics.get("top5_match", 0)
            cal = metrics.get("calibration_time_ms", 0)
            lines.append(f"| {name} | {cos:.4f} | {kl:.6f} | {t5:.4f} | {cal:.0f} |")
        lines.append("")

    # Token sweep
    if all_results.get("token_sweep"):
        lines.append("## Calibration Tokens Sweep")
        lines.append("")
        lines.append("| Tokens | Cosine | KL Div | Top-5 | Cal Time (ms) |")
        lines.append("|--------|--------|--------|-------|---------------|")
        for name, metrics in all_results["token_sweep"].items():
            cos = metrics.get("cosine_sim", 0)
            kl = metrics.get("kl_div", 0)
            t5 = metrics.get("top5_match", 0)
            cal = metrics.get("calibration_time_ms", 0)
            lines.append(f"| {name} | {cos:.4f} | {kl:.6f} | {t5:.4f} | {cal:.0f} |")
        lines.append("")

    # Transfer
    if all_results.get("transfer"):
        lines.append("## Transfer Test")
        lines.append("")
        lines.append("Calibrate on prompt A, evaluate on prompt B.")
        lines.append("")
        lines.append("| Layer | No Cal Cosine | Transfer Cosine | Direct Cosine |")
        lines.append("|-------|---------------|-----------------|---------------|")
        for layer_name, configs in all_results["transfer"].items():
            no_cal = configs.get("no_calibration", {}).get("cosine_sim", 0)
            transfer = configs.get("transfer_from_A", {}).get("cosine_sim", 0)
            direct = configs.get("calibrated_on_B", {}).get("cosine_sim", 0)
            lines.append(f"| {layer_name} | {no_cal:.4f} | {transfer:.4f} | {direct:.4f} |")
        lines.append("")

    report = "\n".join(lines)
    report_path = RESULTS_DIR / "learned_quant_results.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
