#!/usr/bin/env python3
"""Head-to-head comparison: ResidualQuant vs QJL vs MSE-only.

Loads Qwen2.5-3B-Instruct, extracts real KV caches from a forward pass,
then compresses the SAME data with all 3 methods and compares:
  - Cosine similarity (reconstructed vs original)
  - Top-1 and Top-5 attention match
  - Inner product MSE
  - Per-head breakdown
  - Quantize + dequantize latency

Tests at 2-bit, 3-bit, and 4-bit.  Outputs JSON + markdown table.

Run:
    python benchmarks/head_to_head.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from turboquantdc.polarquant import PolarQuant
from turboquantdc.estimator import TurboQuantEstimator
from turboquantdc.residual_quant import ResidualQuantEstimator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BIT_WIDTHS = [2, 3, 4]

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

TIMING_WARMUP = 3
TIMING_ITERS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_extract_kv(
    model_name: str,
    prompt: str,
    device: str,
) -> Tuple[Dict[str, torch.Tensor], int, int]:
    """Load model, run forward pass, return raw KV caches.

    Returns:
        Tuple of (kv_dict, num_layers, head_dim) where kv_dict maps
        layer index to {"keys": Tensor, "values": Tensor} in FP32.
        Shapes are [batch=1, num_heads, seq_len, head_dim].
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {model_name} (4-bit quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)

    past = outputs.past_key_values

    # Handle DynamicCache (.layers[i].keys/.values), legacy key_cache/value_cache,
    # and old-style tuple format
    if hasattr(past, "layers") and len(past.layers) > 0:
        # Modern DynamicCache: .layers is a list of layer objects
        num_layers = len(past.layers)
        first_k = past.layers[0].keys
        head_dim = first_k.shape[-1]
        num_heads = first_k.shape[1]
    elif hasattr(past, "key_cache"):
        num_layers = len(past.key_cache)
        head_dim = past.key_cache[0].shape[-1]
        num_heads = past.key_cache[0].shape[1]
    else:
        num_layers = len(past)
        head_dim = past[0][0].shape[-1]
        num_heads = past[0][0].shape[1]

    print(f"Extracted KV cache: {num_layers} layers, {num_heads} heads, d={head_dim}")

    kv_dict = {}
    for layer_idx in range(num_layers):
        if hasattr(past, "layers") and len(past.layers) > 0:
            k = past.layers[layer_idx].keys.float()
            v = past.layers[layer_idx].values.float()
        elif hasattr(past, "key_cache"):
            k = past.key_cache[layer_idx].float()
            v = past.value_cache[layer_idx].float()
        else:
            k = past[layer_idx][0].float()
            v = past[layer_idx][1].float()
        kv_dict[layer_idx] = {
            "keys": k,      # [1, heads, seq, d]
            "values": v,     # [1, heads, seq, d]
        }

    del model, outputs, past
    torch.cuda.empty_cache()

    return kv_dict, num_layers, num_heads, head_dim


def compute_attention_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """Compute softmax attention scores.

    Args:
        queries: [num_heads, seq, d]
        keys: [num_heads, seq, d]

    Returns:
        Attention weights [num_heads, seq, seq] after softmax.
    """
    d = queries.shape[-1]
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / (d ** 0.5)
    return F.softmax(scores, dim=-1)


def top_k_match(
    attn_orig: torch.Tensor,
    attn_approx: torch.Tensor,
    k: int,
) -> float:
    """Fraction of top-k attention positions that match.

    Args:
        attn_orig: [num_heads, seq, seq]
        attn_approx: [num_heads, seq, seq]
        k: Number of top positions to compare.

    Returns:
        Match rate in [0, 1].
    """
    _, topk_orig = attn_orig.topk(k, dim=-1)
    _, topk_approx = attn_approx.topk(k, dim=-1)

    # For each query position, check what fraction of top-k keys match
    matches = 0
    total = 0
    for head in range(attn_orig.shape[0]):
        for q in range(attn_orig.shape[1]):
            orig_set = set(topk_orig[head, q].tolist())
            approx_set = set(topk_approx[head, q].tolist())
            matches += len(orig_set & approx_set)
            total += k
    return matches / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Method wrappers
# ---------------------------------------------------------------------------

class MSEOnlyMethod:
    """PolarQuant (Stage 1 only) — no residual correction."""

    name = "MSE-only"

    def __init__(self, d: int, bits: int, device: str, seed: int = SEED):
        self.pq = PolarQuant(d, bits, seed=seed, device=device)

    def compress_and_reconstruct(
        self, keys: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compress keys and return reconstruction.

        Args:
            keys: [num_heads, seq, d] in float32.

        Returns:
            (reconstructed_keys, compressed_data)
        """
        heads, seq, d = keys.shape
        all_recon = []
        all_compressed = []
        for h in range(heads):
            norms = keys[h].norm(dim=-1, keepdim=True)
            normalized = keys[h] / (norms + 1e-8)
            indices = self.pq.quantize(normalized)
            recon = self.pq.dequantize(indices) * norms
            all_recon.append(recon)
            all_compressed.append({"indices": indices, "norms": norms})
        return torch.stack(all_recon), all_compressed

    def inner_product(
        self, queries: torch.Tensor, compressed: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Compute inner products via MSE reconstruction."""
        heads, seq_q, d = queries.shape
        ips = []
        for h in range(heads):
            recon = self.pq.dequantize(compressed[h]["indices"]) * compressed[h]["norms"]
            ip = queries[h] @ recon.T  # [seq_q, seq_k]
            ips.append(ip)
        return torch.stack(ips)


class TurboQuantMethod:
    """MSE + QJL (paper's Stage 1 + Stage 2)."""

    name = "TurboQuant (QJL)"

    def __init__(self, d: int, bits: int, device: str, seed: int = SEED):
        self.est = TurboQuantEstimator(d, bits, seed=seed, device=device)

    def compress_and_reconstruct(
        self, keys: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        heads, seq, d = keys.shape
        # Compress and reconstruct per head to keep shapes aligned
        all_recon = []
        all_compressed = []
        for h in range(heads):
            comp = self.est.quantize(keys[h])  # [seq, d]
            recon_mse = self.est.dequantize_mse(comp)
            all_recon.append(recon_mse)
            all_compressed.append(comp)
        recon = torch.stack(all_recon)  # [heads, seq, d]
        return recon, all_compressed

    def inner_product(
        self, queries: torch.Tensor, compressed: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Compute inner products using the full two-stage estimator, per head."""
        heads, seq_q, d = queries.shape
        ips = []
        for h in range(heads):
            ip = self.est.inner_product(queries[h], compressed[h])  # [seq_q, seq_k]
            ips.append(ip)
        return torch.stack(ips)  # [heads, seq_q, seq_k]


class ResidualQuantMethod:
    """MSE + direct residual sign correction (our improvement)."""

    name = "ResidualQuant"

    def __init__(self, d: int, bits: int, device: str, seed: int = SEED):
        self.est = ResidualQuantEstimator(d, bits, seed=seed, device=device)

    def compress_and_reconstruct(
        self, keys: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        heads, seq, d = keys.shape
        all_recon = []
        all_compressed = []
        for h in range(heads):
            comp = self.est.quantize(keys[h])
            recon = self.est.dequantize(comp)
            all_recon.append(recon)
            all_compressed.append(comp)
        return torch.stack(all_recon), all_compressed

    def inner_product(
        self, queries: torch.Tensor, compressed: List[Dict[str, Any]]
    ) -> torch.Tensor:
        heads, seq_q, d = queries.shape
        ips = []
        for h in range(heads):
            ip = self.est.inner_product(queries[h], compressed[h])
            ips.append(ip)
        return torch.stack(ips)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_method(
    method,
    keys_orig: torch.Tensor,
    queries: torch.Tensor,
    attn_orig: torch.Tensor,
) -> Dict[str, Any]:
    """Run full evaluation of a compression method on one layer.

    Args:
        method: One of the method wrappers above.
        keys_orig: [num_heads, seq, d] original keys.
        queries: [num_heads, seq, d] original queries (= keys for self-attention proxy).
        attn_orig: [num_heads, seq, seq] ground-truth attention.

    Returns:
        Dict of metrics.
    """
    heads, seq, d = keys_orig.shape

    # Compress and reconstruct
    recon, compressed = method.compress_and_reconstruct(keys_orig)

    # 1. Cosine similarity (per-head, then averaged)
    flat_orig = keys_orig.reshape(-1, d)
    flat_recon = recon.reshape(-1, d)
    cos_sim = F.cosine_similarity(flat_orig, flat_recon, dim=-1)
    per_head_cos = []
    for h in range(heads):
        start, end = h * seq, (h + 1) * seq
        per_head_cos.append(cos_sim[start:end].mean().item())

    # 2. Inner product MSE
    ip_orig = torch.matmul(queries, keys_orig.transpose(-2, -1))
    ip_approx = method.inner_product(queries, compressed)
    ip_mse = ((ip_orig - ip_approx) ** 2).mean().item()
    ip_rel_error = ((ip_orig - ip_approx).abs() / (ip_orig.abs() + 1e-8)).mean().item()

    # 3. Attention match (top-1 and top-5)
    attn_approx = compute_attention_scores(queries, recon)
    top1 = top_k_match(attn_orig, attn_approx, 1)
    top5 = top_k_match(attn_orig, attn_approx, min(5, seq))

    # 4. Reconstruction MSE
    recon_mse = ((keys_orig - recon) ** 2).mean().item()
    recon_rel = (
        (keys_orig - recon).norm(dim=-1) / (keys_orig.norm(dim=-1) + 1e-8)
    ).mean().item()

    # 5. Timing (quantize + dequantize)
    flat = keys_orig.reshape(-1, d)
    # Warmup
    for _ in range(TIMING_WARMUP):
        method.compress_and_reconstruct(keys_orig)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(TIMING_ITERS):
        method.compress_and_reconstruct(keys_orig)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / TIMING_ITERS
    vectors_per_sec = (heads * seq) / elapsed

    return {
        "method": method.name,
        "cosine_sim_mean": float(sum(per_head_cos) / len(per_head_cos)),
        "cosine_sim_min": float(min(per_head_cos)),
        "cosine_sim_max": float(max(per_head_cos)),
        "cosine_sim_per_head": per_head_cos,
        "top1_match": top1,
        "top5_match": top5,
        "ip_mse": ip_mse,
        "ip_relative_error": ip_rel_error,
        "recon_mse": recon_mse,
        "recon_relative_error": recon_rel,
        "latency_ms": elapsed * 1000,
        "vectors_per_sec": vectors_per_sec,
    }


def evaluate_all_methods_on_layer(
    keys: torch.Tensor,
    values: torch.Tensor,
    bits: int,
    d: int,
) -> List[Dict[str, Any]]:
    """Evaluate all 3 methods on a single layer's KV data."""
    heads, seq, _ = keys.shape

    # Use keys as queries for attention score comparison (self-attention proxy)
    queries = keys.clone()
    attn_orig = compute_attention_scores(queries, keys)

    methods = [
        MSEOnlyMethod(d, bits, DEVICE),
        TurboQuantMethod(d, bits, DEVICE),
        ResidualQuantMethod(d, bits, DEVICE),
    ]

    results = []
    for method in methods:
        r = evaluate_method(method, keys, queries, attn_orig)
        r["bits"] = bits
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def format_markdown_table(all_results: Dict[str, Any]) -> str:
    """Format results as a publishable markdown table."""
    lines = []
    lines.append("# Head-to-Head Benchmark: ResidualQuant vs QJL vs MSE-only\n")
    lines.append(f"Model: {MODEL_NAME}")
    lines.append(f"Device: {DEVICE}")
    lines.append(f"Prompt tokens: {all_results.get('prompt_tokens', 'N/A')}")
    lines.append(f"Layers evaluated: {all_results.get('num_layers', 'N/A')}")
    lines.append(f"Head dimension: {all_results.get('head_dim', 'N/A')}")
    lines.append(f"Num heads: {all_results.get('num_heads', 'N/A')}")
    lines.append("")

    for bits in BIT_WIDTHS:
        key = f"{bits}bit"
        if key not in all_results:
            continue

        data = all_results[key]
        lines.append(f"## {bits}-bit Comparison\n")
        lines.append(
            "| Metric | MSE-only | TurboQuant (QJL) | ResidualQuant | Winner |"
        )
        lines.append("|--------|----------|------------------|---------------|--------|")

        methods = ["MSE-only", "TurboQuant (QJL)", "ResidualQuant"]
        metrics = [
            ("Cosine Similarity", "cosine_sim_mean", "{:.6f}", True),
            ("Cosine Sim (worst head)", "cosine_sim_min", "{:.6f}", True),
            ("Top-1 Attention Match", "top1_match", "{:.4f}", True),
            ("Top-5 Attention Match", "top5_match", "{:.4f}", True),
            ("Inner Product MSE", "ip_mse", "{:.6e}", False),
            ("IP Relative Error", "ip_relative_error", "{:.4f}", False),
            ("Reconstruction MSE", "recon_mse", "{:.6e}", False),
            ("Recon Relative Error", "recon_relative_error", "{:.4f}", False),
            ("Latency (ms)", "latency_ms", "{:.2f}", False),
            ("Throughput (vec/s)", "vectors_per_sec", "{:.0f}", True),
        ]

        for metric_name, metric_key, fmt, higher_is_better in metrics:
            vals = []
            for m in methods:
                v = data.get(m, {}).get(metric_key, None)
                vals.append(v)

            if all(v is not None for v in vals):
                strs = [fmt.format(v) for v in vals]
                if higher_is_better:
                    best_idx = vals.index(max(vals))
                else:
                    best_idx = vals.index(min(vals))
                winner = methods[best_idx]
                # Bold the winner
                strs[best_idx] = f"**{strs[best_idx]}**"
                lines.append(
                    f"| {metric_name} | {strs[0]} | {strs[1]} | {strs[2]} | {winner} |"
                )

        lines.append("")

    # Summary
    lines.append("## Summary\n")
    for bits in BIT_WIDTHS:
        key = f"{bits}bit"
        if key not in all_results:
            continue
        data = all_results[key]
        rq_cos = data.get("ResidualQuant", {}).get("cosine_sim_mean", 0)
        qjl_cos = data.get("TurboQuant (QJL)", {}).get("cosine_sim_mean", 0)
        mse_cos = data.get("MSE-only", {}).get("cosine_sim_mean", 0)

        rq_top5 = data.get("ResidualQuant", {}).get("top5_match", 0)
        qjl_top5 = data.get("TurboQuant (QJL)", {}).get("top5_match", 0)
        mse_top5 = data.get("MSE-only", {}).get("top5_match", 0)

        lines.append(f"**{bits}-bit:** ResidualQuant cos={rq_cos:.6f}, "
                      f"QJL cos={qjl_cos:.6f}, MSE cos={mse_cos:.6f}")
        lines.append(f"  Top-5 match: RQ={rq_top5:.4f}, QJL={qjl_top5:.4f}, MSE={mse_top5:.4f}")
        lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  Head-to-Head Benchmark: ResidualQuant vs QJL vs MSE-only")
    print("=" * 70)
    print()

    # Load model and extract KV caches
    kv_dict, num_layers, num_heads, head_dim = load_model_and_extract_kv(
        MODEL_NAME, PROMPT, DEVICE
    )
    seq_len = kv_dict[0]["keys"].shape[2]

    all_results: Dict[str, Any] = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "prompt_tokens": seq_len,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "seed": SEED,
    }

    # Evaluate across layers (sample 6 evenly spaced layers for speed)
    sample_layers = sorted(set([
        0,
        num_layers // 4,
        num_layers // 2,
        3 * num_layers // 4,
        num_layers - 2,
        num_layers - 1,
    ]))
    sample_layers = [l for l in sample_layers if 0 <= l < num_layers]

    for bits in BIT_WIDTHS:
        print(f"\n{'='*60}")
        print(f"  {bits}-bit evaluation")
        print(f"{'='*60}")

        aggregated: Dict[str, Dict[str, List[float]]] = {}

        for layer_idx in sample_layers:
            print(f"  Layer {layer_idx}/{num_layers-1}...", end=" ", flush=True)

            keys = kv_dict[layer_idx]["keys"][0]  # [heads, seq, d]
            values = kv_dict[layer_idx]["values"][0]

            results = evaluate_all_methods_on_layer(keys, values, bits, head_dim)

            for r in results:
                name = r["method"]
                if name not in aggregated:
                    aggregated[name] = {}
                for k, v in r.items():
                    if k in ("method", "bits", "cosine_sim_per_head"):
                        continue
                    if isinstance(v, (int, float)):
                        aggregated[name].setdefault(k, []).append(v)

            print("done")

        # Average across layers
        bit_key = f"{bits}bit"
        all_results[bit_key] = {}
        for method_name, metrics in aggregated.items():
            avg = {}
            for k, vals in metrics.items():
                avg[k] = sum(vals) / len(vals)
            all_results[bit_key][method_name] = avg

        # Print summary for this bit-width
        print(f"\n  {bits}-bit Summary:")
        print(f"  {'Method':<22} {'Cos Sim':>10} {'Top-1':>8} {'Top-5':>8} {'IP MSE':>12} {'ms':>8}")
        print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*8} {'-'*12} {'-'*8}")
        for name in ["MSE-only", "TurboQuant (QJL)", "ResidualQuant"]:
            if name in all_results[bit_key]:
                m = all_results[bit_key][name]
                print(
                    f"  {name:<22} {m['cosine_sim_mean']:>10.6f} "
                    f"{m['top1_match']:>8.4f} {m['top5_match']:>8.4f} "
                    f"{m['ip_mse']:>12.6e} {m['latency_ms']:>8.2f}"
                )

    # Per-head analysis at 3-bit
    print(f"\n{'='*60}")
    print("  Per-Head Analysis (3-bit, middle layer)")
    print(f"{'='*60}")

    mid_layer = num_layers // 2
    keys_mid = kv_dict[mid_layer]["keys"][0]  # [heads, seq, d]
    queries_mid = keys_mid.clone()

    for MethodClass, label in [
        (MSEOnlyMethod, "MSE-only"),
        (TurboQuantMethod, "TurboQuant (QJL)"),
        (ResidualQuantMethod, "ResidualQuant"),
    ]:
        method = MethodClass(head_dim, 3, DEVICE)
        attn_orig = compute_attention_scores(queries_mid, keys_mid)
        result = evaluate_method(method, keys_mid, queries_mid, attn_orig)
        per_head = result["cosine_sim_per_head"]

        worst_head = per_head.index(min(per_head))
        best_head = per_head.index(max(per_head))
        print(f"\n  {label}:")
        print(f"    Mean cos sim: {sum(per_head)/len(per_head):.6f}")
        print(f"    Best head  {best_head}: {max(per_head):.6f}")
        print(f"    Worst head {worst_head}: {min(per_head):.6f}")
        print(f"    Std dev: {torch.tensor(per_head).std().item():.6f}")

        all_results.setdefault("per_head_3bit", {})[label] = {
            "per_head_cosine": per_head,
            "mean": sum(per_head) / len(per_head),
            "worst_head": worst_head,
            "best_head": best_head,
            "worst_cos": min(per_head),
            "best_cos": max(per_head),
        }

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    json_path = results_dir / "head_to_head.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON results saved to: {json_path}")

    md_text = format_markdown_table(all_results)
    md_path = results_dir / "head_to_head.md"
    with open(md_path, "w") as f:
        f.write(md_text)
    print(f"Markdown table saved to: {md_path}")

    print("\n" + md_text)
    return all_results


if __name__ == "__main__":
    main()
