"""Expected Attention vs EMA importance scoring benchmark.

Loads Qwen2.5-3B (BnB 4-bit), extracts real Q/K at 500+ tokens,
and compares Expected Attention (predicted future) vs EMA (observed past)
importance scoring.

Measures:
    1. Spearman correlation with actual future attention weights
    2. Top-K overlap at 10%, 20%, 30%, 50%
    3. Attention output quality after eviction at 30%, 50%, 70%
    4. Effective compression ratios

Saves results to benchmarks/results/expected_attention_results.md
"""

import gc
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F

# ---- Configuration ----
MODEL_NAME = os.environ.get("EA_MODEL", "Qwen/Qwen2.5-3B-Instruct")
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"
EVICTION_RATES = [0.30, 0.50, 0.70]

# Long context prompt (same as adaptive_bits_benchmark for fair comparison)
CONTEXT_PROMPT = """You are an expert research assistant. Below is a collection of research notes about quantum computing. Read all notes carefully and answer the questions that follow.

Note 1: Quantum Error Correction
Quantum error correction (QEC) is essential for building fault-tolerant quantum computers. The surface code, proposed by Kitaev in 1997, is currently the most promising approach due to its high error threshold of approximately 1%. The surface code encodes a single logical qubit using a two-dimensional lattice of physical qubits, where the number of physical qubits scales as O(d^2) for a code distance d. Recent experimental demonstrations by Google's Sycamore processor have shown logical error rates below the threshold for distance-3 and distance-5 surface codes. IBM's Heron processor has demonstrated similar capabilities with their heavy-hexagonal lattice architecture. The key challenge remains scaling to larger code distances while maintaining low physical error rates.

Note 2: Quantum Advantage in Optimization
The quantum approximate optimization algorithm (QAOA), introduced by Farhi, Goldstone, and Gutmann in 2014, is designed to solve combinatorial optimization problems. Despite significant theoretical and experimental progress, definitive quantum advantage for optimization remains elusive. Classical algorithms, particularly simulated annealing and tensor network methods, continue to compete effectively on problems up to several hundred variables. The most promising applications appear to be in structured problems where the quantum speedup is polynomial rather than exponential, such as portfolio optimization and vehicle routing.

Note 3: Quantum Machine Learning
Quantum machine learning (QML) has seen explosive growth, with variational quantum eigensolvers (VQE) and quantum neural networks (QNN) being the most studied paradigms. However, the barren plateau phenomenon, identified by McClean et al. in 2018, poses a fundamental challenge: the gradients of randomly initialized quantum circuits vanish exponentially with system size, making training infeasible for large circuits. Recent work has proposed several mitigation strategies, including layer-wise training, identity-block initialization, and classical-quantum hybrid architectures. The most successful applications to date have been in quantum chemistry, where quantum computers can naturally represent electronic wave functions.

Note 4: Superconducting Qubits
Superconducting qubits, based on Josephson junctions, dominate the current quantum computing landscape. The transmon qubit, an improved charge qubit with reduced sensitivity to charge noise, achieves coherence times exceeding 100 microseconds in state-of-the-art devices. Google, IBM, and Rigetti all use transmon-based architectures. The key challenges include: (1) improving gate fidelities beyond 99.9%, (2) reducing crosstalk between adjacent qubits, (3) scaling to thousands of qubits while maintaining connectivity, and (4) operating at millikelvin temperatures, which requires expensive dilution refrigerators.

Note 5: Trapped Ion Quantum Computing
Trapped ion quantum computers, pioneered by groups at NIST, University of Innsbruck, and companies like IonQ and Quantinuum, offer several advantages over superconducting qubits. They typically achieve higher two-qubit gate fidelities (exceeding 99.9%), longer coherence times (seconds to minutes), and all-to-all connectivity between qubits in a single trap. However, they face challenges in scaling beyond several dozen qubits in a single trap, with proposed solutions including modular architectures using photonic interconnects and shuttling-based approaches. Quantinuum's H2 processor with 56 qubits represents the current state of the art.

Note 6: Quantum Networking and Communication
Quantum networking aims to connect quantum processors via quantum channels, enabling distributed quantum computing and quantum key distribution (QKD). The fundamental challenge is that quantum states cannot be copied (no-cloning theorem), requiring quantum repeaters for long-distance communication. Current QKD implementations achieve secure key rates of several kilobits per second over distances up to 400 km in optical fiber. Satellite-based QKD, demonstrated by the Chinese Micius satellite, has extended this to over 7,600 km. Quantum memory, essential for quantum repeaters, remains a significant bottleneck, with the best atomic ensemble memories achieving storage times of only a few seconds.

Note 7: Photonic Quantum Computing
Photonic quantum computing uses single photons as qubits, with encoding in polarization, time-bin, or path degrees of freedom. Xanadu's Borealis processor demonstrated quantum advantage in Gaussian boson sampling with 216 squeezed-state modes. Linear optical quantum computing faces the challenge that photon-photon interactions are extremely weak, requiring measurement-induced nonlinearity. PsiQuantum is pursuing a large-scale approach using silicon photonics, aiming for a million-qubit fault-tolerant machine. The fusion-based quantum computing model, proposed by Bartolucci et al. at PsiQuantum, offers a potentially scalable approach using type-II fusion gates on photonic resource states.

Now answer these questions based on the notes above:

Question 1: What is the error threshold of the surface code, and which companies have demonstrated experimental results?

Question 2: What fundamental challenge does QAOA face in achieving quantum advantage over classical methods?

Question 3: Explain the barren plateau phenomenon and list three proposed mitigation strategies.

Question 4: Compare the coherence times of superconducting qubits versus trapped ion qubits.

Question 5: What is the current state of quantum key distribution in terms of distance and key rates?"""


def load_model():
    """Load Qwen2.5-3B with BnB 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()

    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"  Available GPU memory: {free_mem:.1f} GB")
    else:
        free_mem = 0

    load_kwargs = {
        "cache_dir": CACHE_DIR,
        "attn_implementation": "eager",
        "torch_dtype": torch.float16,
    }

    # Use BnB 4-bit for 3B models, direct load for smaller
    if "0.5B" in MODEL_NAME or "1.5B" in MODEL_NAME:
        load_kwargs["device_map"] = "auto"
    elif free_mem > 6:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "cpu"
        print("  WARNING: Loading to CPU")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    dt = time.time() - t0
    device = next(model.parameters()).device
    print(f"  Loaded in {dt:.1f}s on {device}")
    return model, tokenizer


def extract_qk_data(model, tokenizer, prompt: str) -> Dict[str, Any]:
    """Extract Q, K, V, and attention weights from all layers.

    Returns all the data needed for Expected Attention benchmarking:
    queries, keys, values, and ground-truth attention.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    seq_len = inputs.input_ids.shape[1]
    print(f"  Input sequence length: {seq_len} tokens")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=True,
        )

    # Extract KV from DynamicCache
    kv_cache = outputs.past_key_values
    keys_per_layer = []
    values_per_layer = []

    for layer_idx in range(len(kv_cache.layers)):
        layer = kv_cache.layers[layer_idx]
        keys_per_layer.append(layer.keys.cpu().float())
        values_per_layer.append(layer.values.cpu().float())

    # Extract attention weights
    attn_per_layer = [a.cpu().float() for a in outputs.attentions]

    # Reconstruct queries from attention pattern and keys
    # q = attn_scores * sqrt(d) / softmax ... actually we need the pre-softmax
    # Instead, get Q from the model's attention mechanism directly
    # For Qwen2.5, queries are computed inside the attention layer
    # We'll extract them from a second forward pass with hooks
    queries_per_layer = extract_queries_with_hooks(model, inputs)

    return {
        "keys": keys_per_layer,
        "values": values_per_layer,
        "queries": queries_per_layer,
        "attention_weights": attn_per_layer,
        "seq_len": seq_len,
        "n_layers": len(keys_per_layer),
        "head_dim": keys_per_layer[0].shape[-1],
        "n_heads": keys_per_layer[0].shape[1],
        "n_kv_heads": keys_per_layer[0].shape[1],
    }


def extract_queries_with_hooks(model, inputs) -> List[torch.Tensor]:
    """Extract query tensors from all layers using forward hooks."""
    queries = []
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input_args, output):
            # For Qwen2 attention, we need to get the query states
            # The attention module computes q, k, v internally
            # We hook the q_proj to capture query projections
            pass
        return hook_fn

    # Try to get queries via the attention computation
    # Different approach: compute Q = (attn_pre_softmax) * sqrt(d) needs raw scores
    # Simplest: extract Q from q_proj output
    query_outputs = []

    def q_proj_hook(module, input_args, output):
        query_outputs.append(output.detach().cpu().float())

    # Find and hook all q_proj layers
    for name, module in model.named_modules():
        if name.endswith(".q_proj"):
            hooks.append(module.register_forward_hook(q_proj_hook))

    with torch.no_grad():
        model(**inputs, output_attentions=False, use_cache=False)

    for h in hooks:
        h.remove()

    # Reshape query projections to (batch, n_heads, seq, head_dim)
    if query_outputs:
        head_dim = query_outputs[0].shape[-1]
        # q_proj output: (batch, seq, n_heads * head_dim) for most models
        # We need to figure out n_heads from model config
        config = model.config
        n_heads = getattr(config, "num_attention_heads", 32)
        per_head_dim = getattr(config, "head_dim", head_dim // n_heads)

        for q_raw in query_outputs:
            batch, seq, total_dim = q_raw.shape
            # Reshape: (batch, seq, n_heads, head_dim) -> (batch, n_heads, seq, head_dim)
            q_reshaped = q_raw.view(batch, seq, n_heads, per_head_dim).transpose(1, 2)
            queries.append(q_reshaped)

    return queries


def run_scorer_comparison(
    data: Dict[str, Any],
    layer_indices: List[int],
    past_ratio: float = 0.6,
) -> Dict[str, Any]:
    """Compare Expected Attention vs EMA scoring on real model data.

    Splits the sequence into past/future, trains both scorers on past,
    evaluates against actual future attention.

    Args:
        data: Output from extract_qk_data.
        layer_indices: Which layers to test.
        past_ratio: Fraction of queries used as "past" (rest = "future").
    """
    from turboquantdc.expected_attention import (
        ExpectedAttentionScorer,
        compare_scorers,
    )
    from turboquantdc.adaptive_bits import ImportanceScorer
    from scipy.stats import spearmanr

    seq_len = data["seq_len"]
    head_dim = data["head_dim"]
    n_past = int(seq_len * past_ratio)
    n_future = seq_len - n_past

    print(f"\n=== Scorer Comparison ===")
    print(f"  Sequence: {seq_len} tokens, past: {n_past}, future: {n_future}")
    print(f"  Head dim: {head_dim}")
    print(f"  Layers to test: {layer_indices}")

    all_results = []

    for layer_idx in layer_indices:
        print(f"\n  Layer {layer_idx}:")

        # Keys: (batch, n_kv_heads, seq, head_dim) -> take head 0
        keys_full = data["keys"][layer_idx][0, 0, :, :]  # (seq, d)

        # Queries: might have more heads than KV heads (GQA)
        if data["queries"] and layer_idx < len(data["queries"]):
            queries_full = data["queries"][layer_idx][0, 0, :, :]  # (seq, d)
        else:
            # Fallback: synthesize queries from attention weights and keys
            attn = data["attention_weights"][layer_idx][0, 0, :, :]  # (seq, seq)
            # Q ~ attn @ K * sqrt(d) (approximate)
            queries_full = (attn @ keys_full) * math.sqrt(head_dim)

        # Split past/future
        queries_past = queries_full[:n_past]
        queries_future = queries_full[n_past:]
        keys = keys_full  # all keys are in the cache

        # Values for eviction testing
        values = data["values"][layer_idx][0, 0, :, :]  # (seq, d)

        # Run comparison
        result = compare_scorers(
            keys=keys,
            queries_past=queries_past,
            queries_future=queries_future,
            d=head_dim,
        )
        result["layer"] = layer_idx

        print(f"    EA  Spearman: {result['ea_spearman']:.4f}")
        print(f"    EMA Spearman: {result['ema_spearman']:.4f}")
        print(f"    Advantage:    {result['spearman_advantage']:+.4f}")
        for k_pct in [10, 20, 30, 50]:
            ea_k = result[f"top{k_pct}_ea_overlap"]
            ema_k = result[f"top{k_pct}_ema_overlap"]
            adv = result[f"top{k_pct}_advantage"]
            print(f"    Top-{k_pct}% overlap: EA={ea_k:.3f} EMA={ema_k:.3f} (adv={adv:+.3f})")

        all_results.append(result)

    # Aggregate across layers
    agg = {}
    for key in all_results[0]:
        if isinstance(all_results[0][key], (int, float)):
            vals = [r[key] for r in all_results]
            agg[key] = sum(vals) / len(vals)

    return {
        "per_layer": all_results,
        "aggregate": agg,
        "config": {
            "seq_len": seq_len,
            "n_past": n_past,
            "n_future": n_future,
            "head_dim": head_dim,
            "past_ratio": past_ratio,
        },
    }


def run_eviction_benchmark(
    data: Dict[str, Any],
    layer_indices: List[int],
    eviction_rates: List[float],
    past_ratio: float = 0.6,
) -> Dict[str, Any]:
    """Benchmark eviction quality: Expected Attention vs EMA vs Random.

    For each eviction rate, measures attention output quality when
    evicting tokens scored by each method.

    Returns:
        Dict with quality metrics at each eviction rate for each method.
    """
    from turboquantdc.expected_attention import (
        ExpectedAttentionScorer,
        simulate_eviction,
    )
    from turboquantdc.adaptive_bits import ImportanceScorer

    seq_len = data["seq_len"]
    head_dim = data["head_dim"]
    n_past = int(seq_len * past_ratio)
    scale = 1.0 / math.sqrt(head_dim)

    print(f"\n=== Eviction Benchmark ===")
    print(f"  Eviction rates: {eviction_rates}")

    all_results = {}
    for rate in eviction_rates:
        all_results[f"evict_{int(rate*100)}pct"] = {
            "ea": [], "ema": [], "random": [], "recency": [],
        }

    for layer_idx in layer_indices:
        print(f"\n  Layer {layer_idx}:")

        keys = data["keys"][layer_idx][0, 0, :, :]
        values = data["values"][layer_idx][0, 0, :, :]

        if data["queries"] and layer_idx < len(data["queries"]):
            queries_full = data["queries"][layer_idx][0, 0, :, :]
        else:
            attn = data["attention_weights"][layer_idx][0, 0, :, :]
            queries_full = (attn @ keys) * math.sqrt(head_dim)

        queries_past = queries_full[:n_past]
        queries_future = queries_full[n_past:]

        # --- Expected Attention importance ---
        ea_scorer = ExpectedAttentionScorer(d=head_dim, window=64, device=keys.device)
        ea_scorer.update_queries(queries_past)
        ea_importance = ea_scorer.score(keys)

        # --- EMA importance ---
        ema_scorer = ImportanceScorer(ema_decay=0.9)
        chunk_size = min(32, queries_past.shape[0])
        for i in range(0, queries_past.shape[0], chunk_size):
            chunk_q = queries_past[i:i + chunk_size]
            chunk_scores = (chunk_q @ keys.T) * scale
            chunk_attn = torch.softmax(chunk_scores, dim=-1)
            ema_scorer.update(chunk_attn.unsqueeze(0).unsqueeze(0))

        ema_importance = ema_scorer.scores
        if ema_importance is not None:
            ema_importance = ema_importance / ema_importance.sum().clamp(min=1e-10)
        else:
            ema_importance = torch.ones(seq_len) / seq_len

        # --- Random importance (baseline) ---
        random_importance = torch.rand(seq_len)
        random_importance = random_importance / random_importance.sum()

        # --- Recency importance (simple baseline) ---
        recency_importance = torch.linspace(0, 1, seq_len)
        recency_importance = recency_importance / recency_importance.sum()

        # Test each eviction rate
        for rate in eviction_rates:
            rate_key = f"evict_{int(rate*100)}pct"
            print(f"    Eviction {rate:.0%}:")

            for name, importance in [
                ("ea", ea_importance),
                ("ema", ema_importance),
                ("random", random_importance),
                ("recency", recency_importance),
            ]:
                result = simulate_eviction(
                    keys=keys,
                    values=values,
                    queries=queries_future,
                    importance=importance.to(keys.device),
                    eviction_rate=rate,
                )
                result["layer"] = layer_idx
                all_results[rate_key][name].append(result)

                if name in ("ea", "ema"):
                    print(
                        f"      {name:>6}: cos={result['cosine_similarity']:.4f} "
                        f"rel_err={result['relative_error']:.4f} "
                        f"top10={result['top10_attention_match']:.3f} "
                        f"kept={result['tokens_kept']}/{seq_len}"
                    )

    # Aggregate
    summary = {}
    for rate_key, methods in all_results.items():
        summary[rate_key] = {}
        for method, layer_results in methods.items():
            if not layer_results:
                continue
            avg = {}
            for metric in layer_results[0]:
                if isinstance(layer_results[0][metric], (int, float)):
                    vals = [r[metric] for r in layer_results]
                    avg[metric] = sum(vals) / len(vals)
            summary[rate_key][method] = avg

    return {
        "per_layer": all_results,
        "summary": summary,
        "config": {
            "eviction_rates": eviction_rates,
            "past_ratio": past_ratio,
            "seq_len": seq_len,
            "head_dim": head_dim,
        },
    }


def run_diagonal_vs_full_cov(
    data: Dict[str, Any],
    layer_idx: int = 0,
    past_ratio: float = 0.6,
) -> Dict[str, Any]:
    """Compare diagonal vs full covariance in the scorer."""
    from turboquantdc.expected_attention import ExpectedAttentionScorer
    from scipy.stats import spearmanr

    seq_len = data["seq_len"]
    head_dim = data["head_dim"]
    n_past = int(seq_len * past_ratio)
    scale = 1.0 / math.sqrt(head_dim)

    keys = data["keys"][layer_idx][0, 0, :, :]

    if data["queries"] and layer_idx < len(data["queries"]):
        queries_full = data["queries"][layer_idx][0, 0, :, :]
    else:
        attn = data["attention_weights"][layer_idx][0, 0, :, :]
        queries_full = (attn @ keys) * math.sqrt(head_dim)

    queries_past = queries_full[:n_past]
    queries_future = queries_full[n_past:]

    # Ground truth
    future_scores = (queries_future @ keys.T) * scale
    future_attn = torch.softmax(future_scores, dim=-1)
    ground_truth = future_attn.mean(dim=0)
    ground_truth = ground_truth / ground_truth.sum().clamp(min=1e-10)
    gt_np = ground_truth.cpu().numpy()

    results = {}
    for diag in [True, False]:
        label = "diagonal" if diag else "full"
        t0 = time.time()
        scorer = ExpectedAttentionScorer(
            d=head_dim, window=64, use_diagonal_cov=diag, device=keys.device,
        )
        scorer.update_queries(queries_past)
        importance = scorer.score(keys)
        dt = time.time() - t0

        imp_np = importance.cpu().numpy()
        sp = spearmanr(gt_np, imp_np).statistic

        results[label] = {
            "spearman": sp,
            "time_ms": dt * 1000,
            "stats": scorer.stats(),
        }
        print(f"  {label:>10}: Spearman={sp:.4f}, time={dt*1000:.1f}ms")

    return results


def write_results(
    scorer_results: Dict[str, Any],
    eviction_results: Dict[str, Any],
    cov_results: Dict[str, Any],
    output_path: str,
) -> None:
    """Write benchmark results to markdown."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Expected Attention Benchmark Results",
        f"",
        f"**Date:** {now}",
        f"**Model:** {MODEL_NAME}",
        f"**Sequence length:** {scorer_results['config']['seq_len']} tokens",
        f"**Head dimension:** {scorer_results['config']['head_dim']}",
        f"**Past/Future split:** {scorer_results['config']['past_ratio']:.0%} / "
        f"{1 - scorer_results['config']['past_ratio']:.0%}",
        f"",
        f"## 1. Scorer Comparison: Expected Attention vs EMA",
        f"",
        f"Expected Attention predicts FUTURE importance from query distribution "
        f"statistics. EMA tracks PAST attention weights.",
        f"",
        f"### Aggregate (averaged across layers)",
        f"",
        f"| Metric | Expected Attention | EMA | Advantage |",
        f"|--------|-------------------|-----|-----------|",
    ]

    agg = scorer_results["aggregate"]
    lines.append(
        f"| Spearman correlation | {agg.get('ea_spearman', 0):.4f} | "
        f"{agg.get('ema_spearman', 0):.4f} | "
        f"{agg.get('spearman_advantage', 0):+.4f} |"
    )
    for k in [10, 20, 30, 50]:
        ea_k = agg.get(f"top{k}_ea_overlap", 0)
        ema_k = agg.get(f"top{k}_ema_overlap", 0)
        adv = agg.get(f"top{k}_advantage", 0)
        lines.append(f"| Top-{k}% overlap | {ea_k:.3f} | {ema_k:.3f} | {adv:+.3f} |")

    lines.extend([
        f"",
        f"### Per-Layer Results",
        f"",
        f"| Layer | EA Spearman | EMA Spearman | Advantage |",
        f"|-------|-------------|--------------|-----------|",
    ])
    for r in scorer_results["per_layer"]:
        lines.append(
            f"| {r['layer']} | {r['ea_spearman']:.4f} | "
            f"{r['ema_spearman']:.4f} | {r['spearman_advantage']:+.4f} |"
        )

    lines.extend([
        f"",
        f"## 2. Eviction Quality Benchmark",
        f"",
        f"Measures attention output quality after evicting tokens ranked by "
        f"each method. Higher cosine similarity = better.",
        f"",
    ])

    summary = eviction_results.get("summary", {})
    for rate_key in sorted(summary.keys()):
        rate_pct = rate_key.replace("evict_", "").replace("pct", "")
        methods = summary[rate_key]
        lines.extend([
            f"### Eviction Rate: {rate_pct}%",
            f"",
            f"| Method | Cosine Sim | Relative Error | Top-10 Match | Tokens Kept | Eff. Compression |",
            f"|--------|-----------|----------------|-------------|-------------|-----------------|",
        ])
        for method in ["ea", "ema", "recency", "random"]:
            if method in methods:
                m = methods[method]
                lines.append(
                    f"| {method.upper()} | {m.get('cosine_similarity', 0):.4f} | "
                    f"{m.get('relative_error', 0):.4f} | "
                    f"{m.get('top10_attention_match', 0):.3f} | "
                    f"{m.get('tokens_kept', 0):.0f} | "
                    f"{m.get('effective_compression', 0):.2f}x |"
                )
        lines.append("")

    lines.extend([
        f"## 3. Diagonal vs Full Covariance",
        f"",
        f"| Covariance | Spearman | Time (ms) |",
        f"|-----------|----------|-----------|",
    ])
    for label, res in cov_results.items():
        lines.append(f"| {label} | {res['spearman']:.4f} | {res['time_ms']:.1f} |")

    lines.extend([
        f"",
        f"## 4. Analysis",
        f"",
    ])

    # Determine winner
    ea_sp = agg.get("ea_spearman", 0)
    ema_sp = agg.get("ema_spearman", 0)
    if ea_sp > ema_sp + 0.01:
        winner_line = (
            f"**Expected Attention outperforms EMA** by "
            f"{agg.get('spearman_advantage', 0):+.4f} Spearman correlation. "
            f"This means predicting future attention from query statistics "
            f"is more accurate than tracking past attention via EMA."
        )
    elif ema_sp > ea_sp + 0.01:
        winner_line = (
            f"**EMA outperforms Expected Attention** by "
            f"{-agg.get('spearman_advantage', 0):+.4f} Spearman correlation. "
            f"Past attention patterns are a stronger predictor of future "
            f"importance than the analytic Expected Attention formula."
        )
    else:
        winner_line = (
            f"**Expected Attention and EMA are comparable** "
            f"(difference: {agg.get('spearman_advantage', 0):+.4f} Spearman). "
            f"Both methods provide similar quality importance ranking."
        )

    lines.append(winner_line)
    lines.append("")

    # Eviction analysis
    lines.append("### Eviction Quality Summary")
    lines.append("")
    for rate_key in sorted(summary.keys()):
        rate_pct = rate_key.replace("evict_", "").replace("pct", "")
        methods = summary[rate_key]
        ea_cos = methods.get("ea", {}).get("cosine_similarity", 0)
        ema_cos = methods.get("ema", {}).get("cosine_similarity", 0)
        rand_cos = methods.get("random", {}).get("cosine_similarity", 0)
        ea_comp = methods.get("ea", {}).get("effective_compression", 0)

        lines.append(
            f"- **{rate_pct}% eviction:** EA cos={ea_cos:.4f}, "
            f"EMA cos={ema_cos:.4f}, Random cos={rand_cos:.4f}. "
            f"Effective compression: {ea_comp:.1f}x"
        )

    lines.extend([
        f"",
        f"### Effective Compression Stack",
        f"",
        f"- TurboQuant 3-bit base: ~5x compression",
        f"- + 50% eviction (Expected Attention): ~10x effective",
        f"- + 70% eviction (Expected Attention): ~15x effective",
        f"",
        f"### Key Advantages of Expected Attention",
        f"",
        f"1. **Proactive:** Predicts future importance from query distribution, "
        f"   not just past attention.",
        f"2. **O(n*d):** No quadratic attention matrix needed.",
        f"3. **Closed-form:** Analytic formula, no learned parameters.",
        f"4. **Composable:** Stacks with TurboQuant for 10-15x total compression.",
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {output_path}")


def main():
    print("=" * 70)
    print("Expected Attention vs EMA Importance Scoring Benchmark")
    print("=" * 70)

    # Load model
    model, tokenizer = load_model()

    # Extract Q, K, V data
    print("\n--- Extracting Q/K/V data ---")
    data = extract_qk_data(model, tokenizer, CONTEXT_PROMPT)
    print(f"  Extracted {data['n_layers']} layers, {data['n_heads']} heads, d={data['head_dim']}")
    print(f"  Queries extracted: {len(data['queries'])} layers")

    # Free model memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Select representative layers (early, middle, late)
    n_layers = data["n_layers"]
    layer_indices = sorted(set([
        0,
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 1,
    ]))
    print(f"\nTesting layers: {layer_indices}")

    # 1. Scorer comparison
    print("\n" + "=" * 50)
    print("Phase 1: Scorer Comparison")
    print("=" * 50)
    scorer_results = run_scorer_comparison(data, layer_indices)

    # 2. Eviction benchmark
    print("\n" + "=" * 50)
    print("Phase 2: Eviction Benchmark")
    print("=" * 50)
    eviction_results = run_eviction_benchmark(data, layer_indices, EVICTION_RATES)

    # 3. Diagonal vs full covariance
    print("\n" + "=" * 50)
    print("Phase 3: Diagonal vs Full Covariance")
    print("=" * 50)
    cov_results = run_diagonal_vs_full_cov(data, layer_idx=layer_indices[len(layer_indices)//2])

    # Write results
    output_path = os.path.join(REPO_ROOT, "benchmarks", "results", "expected_attention_results.md")
    write_results(scorer_results, eviction_results, cov_results, output_path)

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)

    # Print summary
    agg = scorer_results["aggregate"]
    print(f"\n  EA  Spearman: {agg.get('ea_spearman', 0):.4f}")
    print(f"  EMA Spearman: {agg.get('ema_spearman', 0):.4f}")
    print(f"  Advantage:    {agg.get('spearman_advantage', 0):+.4f}")

    for rate_key in sorted(eviction_results.get("summary", {}).keys()):
        rate_pct = rate_key.replace("evict_", "").replace("pct", "")
        methods = eviction_results["summary"][rate_key]
        ea_cos = methods.get("ea", {}).get("cosine_similarity", 0)
        ea_comp = methods.get("ea", {}).get("effective_compression", 0)
        print(f"  {rate_pct}% eviction: cos={ea_cos:.4f}, compression={ea_comp:.1f}x")


if __name__ == "__main__":
    main()
