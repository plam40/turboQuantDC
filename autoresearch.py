"""Autoresearch -- autonomous overnight configuration sweep for TurboQuantDC.

Inspired by Karpathy's autoresearch concept: one script, loads model ONCE,
sweeps hundreds of KV cache compression configurations, auto-scores each,
saves results after every round, and reports the Pareto frontier of
compression vs generation quality.

The search space covers:
    Uniform quantizer:
        key_bits: [3, 4, 5, 6, 8]
        val_bits: [2, 3, 4]
        anchor_interval: [0, 6, 12, 18, 36]
        fp16_window: [0, 64, 128, 256, 512]
        use_residual_quant: [True, False]
        mse_only: [True]  (QJL is dead; always MSE-only)
    Channel-adaptive quantizer (KITTY-style mixed precision):
        high_bits / low_bits replace key_bits (e.g. 4+2 = 2.5-bit avg)
        boost_fraction: 0.25 (top 25% channels at high_bits)

Scoring (v2 -- perplexity + generation quality):
    Primary:   Wikitext-2 perplexity vs FP16 baseline (0.6 weight)
    Secondary: 12-prompt generation quality via self-judge comparison (0.4 weight)
    Combined:  0.6 * ppl_score + 0.4 * gen_score
    Legacy keyword-matching score preserved as ``legacy_score`` for comparison.

Usage:
    cd /home/dhawal/turboQuantDC
    python autoresearch.py                          # full sweep
    python autoresearch.py --max-rounds 200         # first 200 (most promising)
    python autoresearch.py --resume                 # skip already-tested configs
    python autoresearch.py --skip-needle            # skip needle-in-haystack (faster)
    nohup python autoresearch.py > autoresearch.log 2>&1 &
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import torch

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NUM_LAYERS = 36
MAX_NEW_TOKENS = 80
DO_SAMPLE = False

SEARCH_SPACE = {
    "key_bits": [3, 4, 5, 6, 8],
    "val_bits": [2, 3, 4],
    "anchor_interval": [0, 6, 12, 18, 36],
    "fp16_window": [0, 64, 128, 256, 512],
    "use_residual_quant": [True, False],
    "mse_only": [True],
}

# Channel-adaptive configs: (high_bits, low_bits, val_bits) with boost_fraction=0.25
ADAPTIVE_CONFIGS = [
    {"high_bits": 4, "low_bits": 2, "val_bits": 2},   # 2.5-bit avg keys, 2-bit vals
    {"high_bits": 4, "low_bits": 3, "val_bits": 2},   # 3.25-bit avg keys, 2-bit vals
    {"high_bits": 4, "low_bits": 3, "val_bits": 3},   # 3.25-bit avg keys, 3-bit vals
]
ADAPTIVE_BOOST_FRACTION = 0.25

from benchmark import BenchmarkRunner, GENERATION_PROMPTS, score_response_legacy


def compute_compression_ratio(config: Dict) -> float:
    """Compute theoretical compression ratio for a configuration.

    Handles both uniform and channel-adaptive quantizer types.
    Uses the UltimateCache theoretical formula without storing any data.
    """
    val_bits = config["val_bits"]
    fp16_window = config["fp16_window"]

    # Determine effective key bits
    if config.get("quantizer_type") == "adaptive":
        high_bits = config["high_bits"]
        low_bits = config["low_bits"]
        boost_fraction = config.get("boost_fraction", ADAPTIVE_BOOST_FRACTION)
        eff_key_bits = boost_fraction * high_bits + (1 - boost_fraction) * low_bits
        anchor_interval = 0  # adaptive configs don't use anchors
        use_residual_quant = False
    else:
        eff_key_bits = config["key_bits"]
        anchor_interval = config["anchor_interval"]
        use_residual_quant = config["use_residual_quant"]

    if anchor_interval > 0 and anchor_interval < NUM_LAYERS:
        n_fp16 = len([i for i in range(NUM_LAYERS) if i % anchor_interval == 0])
    else:
        n_fp16 = 0
    n_comp = NUM_LAYERS - n_fp16

    # FP16 layers: 32 bits per coordinate (K+V each at 16-bit)
    fp16_cost = n_fp16 * 32.0

    # Compressed layers
    if use_residual_quant:
        comp_key = eff_key_bits + 32 / 128.0  # MSE + signs + scale + norm overhead
    else:
        comp_key = eff_key_bits + 16 / 128.0  # MSE + norm overhead

    comp_val = val_bits + 16 / 128.0
    comp_cost = n_comp * (comp_key + comp_val)

    # FP16 window adjustment (approximate for ~512 tokens)
    if fp16_window > 0 and n_comp > 0:
        window_frac = min(fp16_window / 512.0, 1.0)
        comp_cost_windowed = n_comp * (
            (1 - window_frac) * (comp_key + comp_val)
            + window_frac * 32
        )
        comp_cost = comp_cost_windowed

    total_cost = fp16_cost + comp_cost
    baseline = NUM_LAYERS * 32.0

    return round(baseline / total_cost, 2) if total_cost > 0 else 1.0


# ---------------------------------------------------------------------------
# Cache building
# ---------------------------------------------------------------------------

def build_cache(config: Dict):
    """Build the appropriate cache for a configuration.

    Dispatches based on quantizer_type:
    - "adaptive": ChannelAdaptiveCache (KITTY-style mixed precision)
    - "eviction": EvictionCache (token eviction for higher compression)
    - default: GenerationCache (uniform quantization)

    For uniform configs, supports anchor_strategy parameter:
    - "fixed" (default): every Nth layer FP16
    - "boundary": first 2 + last 2 layers FP16
    - "gradient": boundary FP16 + gradient bit allocation
    """
    if config.get("quantizer_type") == "adaptive":
        return build_adaptive_cache(config)
    if config.get("quantizer_type") == "eviction":
        return build_eviction_cache(config)
    if config.get("quantizer_type") == "hybrid":
        return build_hybrid_cache(config)

    from turboquantdc.generation_cache import GenerationCache

    key_bits = config["key_bits"]
    val_bits = config["val_bits"]
    fp16_window = config["fp16_window"]
    anchor_interval = config["anchor_interval"]
    use_residual_quant = config.get("use_residual_quant", True)
    anchor_strategy = config.get("anchor_strategy", "fixed")

    cache = GenerationCache(
        key_bits=key_bits,
        val_bits=val_bits,
        fp16_window=fp16_window,
        anchor_interval=anchor_interval,
        seed=42,
        use_residual_quant=use_residual_quant,
        anchor_strategy=anchor_strategy,
        num_layers=NUM_LAYERS,
    )
    return cache


def build_hybrid_cache(config: Dict):
    """Build a HybridCache combining all winning strategies."""
    from turboquantdc.generation_cache import HybridCache

    cache = HybridCache(
        num_layers=NUM_LAYERS,
        base_key_bits=config.get("key_bits", 3),
        base_val_bits=config.get("val_bits", 3),
        fp16_window=config.get("fp16_window", 64),
        warmup_tokens=config.get("warmup_tokens", 20),
        high_entropy_pct=config.get("high_entropy_pct", 75),
        low_entropy_pct=config.get("low_entropy_pct", 25),
        seed=42,
    )
    return cache


def build_eviction_cache(config: Dict):
    """Build an EvictionCache for token eviction + quantization."""
    from turboquantdc.token_eviction import EvictionCache

    cache = EvictionCache(
        key_bits=config.get("key_bits", 3),
        val_bits=config.get("val_bits", 3),
        fp16_window=config.get("fp16_window", 64),
        max_warm_tokens=config.get("max_warm_tokens", 512),
        eviction_threshold=config.get("eviction_threshold", 0.01),
        anchor_interval=config.get("anchor_interval", 12),
        use_residual_quant=config.get("use_residual_quant", True),
        seed=42,
    )
    return cache


def build_adaptive_cache(config: Dict):
    """Build a ChannelAdaptiveCache for mixed-precision key quantization.

    Uses KITTY-style channel sensitivity analysis to allocate more bits
    to sensitive channels and fewer bits to the rest.
    """
    from turboquantdc.channel_adaptive import ChannelAdaptiveCache

    cache = ChannelAdaptiveCache(
        high_bits=config["high_bits"],
        low_bits=config["low_bits"],
        val_bits=config["val_bits"],
        boost_fraction=config.get("boost_fraction", ADAPTIVE_BOOST_FRACTION),
        fp16_window=config["fp16_window"],
        seed=42,
    )
    return cache


# ---------------------------------------------------------------------------
# Score a full configuration (via benchmark module)
# ---------------------------------------------------------------------------

def score_config_v2(
    benchmark_runner: BenchmarkRunner,
    config: Dict,
    baseline_ppl: float,
    baseline_responses: Optional[Dict[str, str]] = None,
    run_needle: bool = True,
) -> Tuple[Dict[str, Any], float]:
    """Score a config using the benchmark module.

    Returns:
        (result_dict, compression_ratio) where result_dict is serializable
        and contains total_score, ppl_score, gen_score, legacy_score, per_prompt, etc.
    """
    result = benchmark_runner.evaluate_config(
        config=config,
        build_cache_fn=build_cache,
        baseline_ppl=baseline_ppl,
        baseline_responses=baseline_responses,
        run_needle=run_needle,
    )

    compression = compute_compression_ratio(config)
    return result.to_dict(), compression


# ---------------------------------------------------------------------------
# Config generation -- priority-ordered
# ---------------------------------------------------------------------------

def config_key(config: Dict) -> str:
    """Stable string key for a configuration (for dedup/resume)."""
    if config.get("quantizer_type") == "adaptive":
        return (
            f"adaptive_h{config['high_bits']}_l{config['low_bits']}"
            f"_v{config['val_bits']}"
            f"_w{config['fp16_window']}"
        )
    if config.get("quantizer_type") == "hybrid":
        return (
            f"hybrid_k{config['key_bits']}_v{config['val_bits']}"
            f"_w{config['fp16_window']}"
        )
    if config.get("quantizer_type") == "eviction":
        return (
            f"evict_k{config['key_bits']}_v{config['val_bits']}"
            f"_w{config['fp16_window']}"
            f"_warm{config.get('max_warm_tokens', 512)}"
            f"_rq{int(config['use_residual_quant'])}"
        )
    strategy = config.get("anchor_strategy", "fixed")
    prefix = {"fixed": "", "gradient": "grad_", "boundary": "bnd_"}.get(strategy, "")
    return (
        f"{prefix}k{config['key_bits']}_v{config['val_bits']}"
        f"_a{config['anchor_interval']}"
        f"_w{config['fp16_window']}"
        f"_rq{int(config['use_residual_quant'])}"
    )


def generate_config_list() -> List[Dict]:
    """Generate all configurations, priority-ordered.

    Strategy: most promising first, then systematic sweep.
    Priority 1: known breakthroughs + K8 near-lossless + adaptive
    Priority 2: K8 combos + combined stacks at moderate compression + adaptive variants
    Priority 3: full systematic sweep of remaining uniform space
    """
    seen = set()
    configs = []

    def add(cfg: Dict):
        key = config_key(cfg)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    def uniform(kb, vb, ai, fw, rq):
        """Shorthand for uniform config dict."""
        return {"key_bits": kb, "val_bits": vb, "anchor_interval": ai, "fp16_window": fw, "use_residual_quant": rq, "mse_only": True}

    def adaptive(hb, lb, vb, fw):
        """Shorthand for adaptive config dict."""
        return {"quantizer_type": "adaptive", "high_bits": hb, "low_bits": lb, "val_bits": vb, "boost_fraction": ADAPTIVE_BOOST_FRACTION, "fp16_window": fw, "anchor_interval": 0, "use_residual_quant": False, "mse_only": True}

    def gradient(kb, vb, fw, rq):
        """Shorthand for gradient anchor strategy config."""
        return {"key_bits": kb, "val_bits": vb, "anchor_interval": 0, "fp16_window": fw, "use_residual_quant": rq, "mse_only": True, "anchor_strategy": "gradient"}

    def boundary(kb, vb, fw, rq):
        """Shorthand for boundary anchor strategy config."""
        return {"key_bits": kb, "val_bits": vb, "anchor_interval": 0, "fp16_window": fw, "use_residual_quant": rq, "mse_only": True, "anchor_strategy": "boundary"}

    def eviction(kb, vb, fw, rq, max_warm=1024):
        """Shorthand for eviction cache config."""
        return {"quantizer_type": "eviction", "key_bits": kb, "val_bits": vb, "fp16_window": fw, "anchor_interval": 12, "use_residual_quant": rq, "mse_only": True, "max_warm_tokens": max_warm, "eviction_threshold": 0.01}

    def hybrid(kb, vb, fw):
        """Shorthand for hybrid cache config (boundary+gradient+per-head)."""
        return {"quantizer_type": "hybrid", "key_bits": kb, "val_bits": vb, "fp16_window": fw, "anchor_interval": 0, "use_residual_quant": True, "mse_only": True, "warmup_tokens": 20, "high_entropy_pct": 75, "low_entropy_pct": 25}

    # --- Priority 0: HybridCache (the main hypothesis) ---
    # HybridCache combines boundary+gradient anchoring + per-head bit allocation
    add(hybrid(3, 3, 0))      # K3/V3 hybrid, no window — the headline test
    add(hybrid(3, 3, 64))     # K3/V3 hybrid + small window
    add(hybrid(3, 2, 0))      # K3/V2 hybrid aggressive
    add(hybrid(3, 2, 64))     # K3/V2 hybrid + window
    add(hybrid(4, 3, 0))      # K4/V3 hybrid
    add(hybrid(4, 3, 64))     # K4/V3 hybrid + window
    add(hybrid(4, 2, 0))      # K4/V2 hybrid aggressive
    add(hybrid(4, 2, 64))     # K4/V2 hybrid + window
    add(hybrid(3, 3, 128))    # K3/V3 hybrid + medium window
    add(hybrid(5, 3, 0))      # K5/V3 hybrid
    # Fixed eviction (improved with prompt protection + key norm weighting)
    add(eviction(3, 3, 64, True, 1024))   # K3 eviction, 1024 warm (new default)
    add(eviction(3, 3, 64, True, 512))    # K3 eviction, 512 warm
    add(eviction(4, 3, 64, True, 1024))   # K4 eviction
    add(eviction(3, 2, 64, True, 1024))   # K3 aggressive eviction

    # --- Priority 1: known good configs + K8 near-lossless + adaptive ---
    # Anchor-12 + 4-bit keys + 4-bit values (known clean)
    add(uniform(4, 4, 12, 0, False))
    # Anchor-12 + 4-bit keys + 2-bit values (known clean, higher compression)
    add(uniform(4, 2, 12, 0, False))
    # Anchor-12 + ResQ-4 keys + 2-bit values
    add(uniform(4, 2, 12, 0, True))
    # No anchors + ResQ-4 keys + 2-bit values (highest compression that may work)
    add(uniform(4, 2, 0, 0, True))
    # Anchor-12 + 5-bit keys + 2-bit values
    add(uniform(5, 2, 12, 0, False))
    # Windowed variants
    add(uniform(4, 2, 12, 128, True))
    add(uniform(4, 2, 12, 64, False))
    # 3-bit keys (aggressive)
    add(uniform(3, 2, 12, 0, False))
    add(uniform(3, 2, 6, 0, False))
    add(uniform(3, 3, 6, 0, False))
    add(uniform(3, 2, 12, 128, True))
    # K8 near-lossless: 8-bit keys should be near-FP16 quality for keys,
    # paired with aggressive value compression for high overall compression
    add(uniform(8, 2, 0, 0, False))
    add(uniform(8, 2, 12, 0, False))
    add(uniform(8, 3, 0, 0, False))
    add(uniform(8, 3, 12, 0, False))
    add(uniform(8, 2, 0, 128, False))
    add(uniform(8, 2, 0, 256, False))
    # fp16_window=512 variants of best configs
    add(uniform(4, 2, 12, 512, True))
    add(uniform(4, 2, 12, 512, False))
    add(uniform(3, 2, 12, 512, True))
    add(uniform(3, 2, 6, 512, False))
    # Channel-adaptive: mixed-precision keys (KITTY-style)
    add(adaptive(4, 2, 2, 0))      # 2.5-bit avg keys, 2-bit vals
    add(adaptive(4, 2, 2, 128))    # 2.5-bit avg keys, 2-bit vals, windowed
    add(adaptive(4, 3, 2, 0))      # 3.25-bit avg keys, 2-bit vals
    add(adaptive(4, 3, 2, 128))    # 3.25-bit avg keys, 2-bit vals, windowed
    add(adaptive(4, 3, 3, 0))      # 3.25-bit avg keys, 3-bit vals
    add(adaptive(4, 3, 3, 128))    # 3.25-bit avg keys, 3-bit vals, windowed
    # --- NEW: Gradient anchor strategy (boundary FP16 + gradient bits) ---
    # These use smart per-layer bit allocation: 8-bit at edges, base_bits in middle
    add(gradient(3, 3, 0, True))    # K3 gradient, RQ — the big hypothesis
    add(gradient(3, 3, 64, True))   # K3 gradient + small window
    add(gradient(3, 2, 0, True))    # K3 aggressive gradient
    add(gradient(3, 2, 64, True))   # K3 aggressive gradient + window
    add(gradient(4, 3, 0, True))    # K4 gradient
    add(gradient(4, 2, 0, True))    # K4 gradient aggressive
    add(gradient(3, 3, 128, True))  # K3 gradient + medium window
    add(gradient(4, 3, 64, False))  # K4 gradient no RQ
    # Boundary strategy (first 2 + last 2 layers FP16)
    add(boundary(3, 3, 0, True))    # K3 boundary
    add(boundary(3, 3, 64, True))   # K3 boundary + window
    add(boundary(4, 3, 0, True))    # K4 boundary
    add(boundary(4, 2, 0, True))    # K4 boundary aggressive
    # --- NEW: Token eviction (quantize + evict for higher compression) ---
    add(eviction(3, 3, 64, True, 512))    # K3 eviction, 512 warm
    add(eviction(3, 3, 64, True, 256))    # K3 eviction, 256 warm (aggressive)
    add(eviction(3, 2, 64, True, 512))    # K3 aggressive eviction
    add(eviction(4, 3, 64, True, 512))    # K4 eviction
    add(eviction(3, 3, 128, True, 512))   # K3 eviction, larger window
    add(eviction(4, 2, 64, True, 256))    # K4 aggressive eviction

    # --- Priority 2: K8 combos + combined stacks + adaptive with fp16_window ---
    # K8 with all value bit-widths and anchor intervals
    for vb in [2, 3, 4]:
        for ai in [0, 6, 12, 18]:
            for rq in [True, False]:
                add(uniform(8, vb, ai, 0, rq))
    # Standard moderate-compression combos
    for kb in [4, 5, 6]:
        for vb in [2, 3]:
            for ai in [6, 12, 18]:
                for rq in [True, False]:
                    add(uniform(kb, vb, ai, 0, rq))
    # Adaptive with all fp16_window sizes
    for ac in ADAPTIVE_CONFIGS:
        for fw in SEARCH_SPACE["fp16_window"]:
            add(adaptive(ac["high_bits"], ac["low_bits"], ac["val_bits"], fw))
    # Gradient strategy: systematic sweep of key/val bits with windows
    for kb in [3, 4, 5]:
        for vb in [2, 3]:
            for fw in [0, 64, 128, 256]:
                for rq in [True, False]:
                    add(gradient(kb, vb, fw, rq))
    # Boundary strategy: systematic sweep
    for kb in [3, 4, 5]:
        for vb in [2, 3]:
            for fw in [0, 64, 128]:
                for rq in [True, False]:
                    add(boundary(kb, vb, fw, rq))
    # Eviction: systematic sweep of warm token limits
    for kb in [3, 4]:
        for vb in [2, 3]:
            for fw in [64, 128]:
                for max_warm in [256, 512]:
                    add(eviction(kb, vb, fw, True, max_warm))

    # --- Priority 3: full systematic sweep of uniform space ---
    for kb, vb, ai, fw, rq, mo in product(
        SEARCH_SPACE["key_bits"],
        SEARCH_SPACE["val_bits"],
        SEARCH_SPACE["anchor_interval"],
        SEARCH_SPACE["fp16_window"],
        SEARCH_SPACE["use_residual_quant"],
        SEARCH_SPACE["mse_only"],
    ):
        add(uniform(kb, vb, ai, fw, rq))

    return configs


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed_keys(results_file: str) -> set:
    """Load config keys already tested from results file."""
    completed = set()
    if not os.path.exists(results_file):
        return completed
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cfg = entry.get("config", {})
                if cfg:
                    completed.add(config_key(cfg))
            except json.JSONDecodeError:
                continue
    return completed


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def update_pareto(
    frontier: List[Tuple[float, float]],
    compression: float,
    score: float,
) -> List[Tuple[float, float]]:
    """Update the Pareto frontier (compression, score) and return pruned list.

    A point (c, s) is Pareto-dominated if there exists another point
    (c2, s2) with c2 >= c AND s2 >= s AND (c2, s2) != (c, s).
    """
    candidate = (compression, score)
    # Check if new point is dominated
    dominated = any(
        c2 >= compression and s2 >= score and (c2, s2) != candidate
        for c2, s2 in frontier
    )
    if dominated:
        return frontier

    # Add new point and remove points it dominates
    frontier.append(candidate)
    frontier = [
        (c, s) for c, s in frontier
        if not any(
            c2 >= c and s2 >= s and (c2, s2) != (c, s)
            for c2, s2 in frontier
        )
    ]
    # Sort by compression descending
    frontier.sort(key=lambda x: x[0], reverse=True)
    return frontier


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def print_final_summary(results_file: str):
    """Read results file and print a ranked summary."""
    if not os.path.exists(results_file):
        print("No results file found.")
        return

    results = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not results:
        print("No results to summarize.")
        return

    # Separate baseline
    baseline = [r for r in results if r.get("config", {}).get("baseline")]
    configs = [r for r in results if not r.get("config", {}).get("baseline")]

    # Sort by score descending, then compression descending
    configs.sort(key=lambda r: (r["total_score"], r["compression"]), reverse=True)

    print("\n" + "=" * 80)
    print("AUTORESEARCH FINAL SUMMARY")
    print("=" * 80)

    if baseline:
        b = baseline[0]
        ppl_info = ""
        if "ppl_score" in b:
            ppl_info = f"  (ppl={b.get('ppl_score', 0):.4f}, gen={b.get('gen_score', 0):.4f})"
        print(f"\nFP16 Baseline score: {b['total_score']:.4f}{ppl_info}")
        if "baseline_ppl" in b:
            print(f"FP16 Baseline perplexity: {b['baseline_ppl']:.2f}")

    print(f"\nTotal configurations tested: {len(configs)}")

    # Top 20 by score
    print(f"\n{'='*80}")
    print("TOP 20 BY QUALITY SCORE")
    print(f"{'='*80}")
    print(f"{'#':>3}  {'Score':>6}  {'PPL':>5}  {'Gen':>5}  {'PPL+%':>6}  {'Comp':>6}  {'K':>2}b  {'V':>2}b  {'Anchor':>6}  {'Window':>6}  {'ResQ':>4}")
    print("-" * 85)
    for i, r in enumerate(configs[:20]):
        c = r["config"]
        ppl_s = f"{r.get('ppl_score', 0):>5.3f}" if "ppl_score" in r else "  n/a"
        gen_s = f"{r.get('gen_score', 0):>5.3f}" if "gen_score" in r else "  n/a"
        ppl_pct = f"{r.get('ppl_increase_pct', 0):>5.1f}%" if "ppl_increase_pct" in r else "   n/a"
        print(
            f"{i+1:>3}  {r['total_score']:>6.4f}  {ppl_s}  {gen_s}  {ppl_pct}  {r['compression']:>5.2f}x  "
            f"{c['key_bits']:>2}   {c['val_bits']:>2}   {c['anchor_interval']:>6}  "
            f"{c['fp16_window']:>6}  {str(c['use_residual_quant']):>5}"
        )

    # Top 20 by compression (among configs scoring >= 0.7)
    good_configs = [r for r in configs if r["total_score"] >= 0.7]
    good_configs.sort(key=lambda r: r["compression"], reverse=True)

    print(f"\n{'='*80}")
    print("TOP 20 BY COMPRESSION (score >= 0.7)")
    print(f"{'='*80}")
    print(f"{'#':>3}  {'Score':>6}  {'PPL':>5}  {'Gen':>5}  {'Comp':>6}  {'K':>2}b  {'V':>2}b  {'Anchor':>6}  {'Window':>6}  {'ResQ':>4}")
    print("-" * 75)
    for i, r in enumerate(good_configs[:20]):
        c = r["config"]
        ppl_s = f"{r.get('ppl_score', 0):>5.3f}" if "ppl_score" in r else "  n/a"
        gen_s = f"{r.get('gen_score', 0):>5.3f}" if "gen_score" in r else "  n/a"
        print(
            f"{i+1:>3}  {r['total_score']:>6.4f}  {ppl_s}  {gen_s}  {r['compression']:>5.2f}x  "
            f"{c['key_bits']:>2}   {c['val_bits']:>2}   {c['anchor_interval']:>6}  "
            f"{c['fp16_window']:>6}  {str(c['use_residual_quant']):>5}"
        )

    # Pareto frontier
    frontier = []
    for r in configs:
        frontier = update_pareto(frontier, r["compression"], r["total_score"])

    print(f"\n{'='*80}")
    print("PARETO FRONTIER (compression vs quality)")
    print(f"{'='*80}")
    for comp, score in frontier:
        # Find the config
        match = [r for r in configs if abs(r["compression"] - comp) < 0.01 and abs(r["total_score"] - score) < 0.0001]
        if match:
            c = match[0]["config"]
            desc = f"K{c['key_bits']} V{c['val_bits']} A{c['anchor_interval']} W{c['fp16_window']} RQ={c['use_residual_quant']}"
        else:
            desc = "?"
        print(f"  {comp:>5.2f}x compression, {score:.4f} score -- {desc}")

    # Quality tiers
    tiers = {
        "Excellent (>= 0.9)": [r for r in configs if r["total_score"] >= 0.9],
        "Good (>= 0.8)": [r for r in configs if 0.8 <= r["total_score"] < 0.9],
        "OK (>= 0.7)": [r for r in configs if 0.7 <= r["total_score"] < 0.8],
        "Poor (>= 0.5)": [r for r in configs if 0.5 <= r["total_score"] < 0.7],
        "Garbled (< 0.5)": [r for r in configs if r["total_score"] < 0.5],
    }

    print(f"\n{'='*80}")
    print("QUALITY TIER DISTRIBUTION")
    print(f"{'='*80}")
    for tier_name, tier_configs in tiers.items():
        if tier_configs:
            max_comp = max(r["compression"] for r in tier_configs)
            avg_comp = sum(r["compression"] for r in tier_configs) / len(tier_configs)
            print(f"  {tier_name}: {len(tier_configs)} configs, max compression {max_comp:.2f}x, avg {avg_comp:.2f}x")
        else:
            print(f"  {tier_name}: 0 configs")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_autoresearch(
    model,
    tokenizer,
    max_rounds: int = 800,
    results_file: str = "autoresearch_results.jsonl",
    resume: bool = False,
    skip_needle: bool = False,
):
    """Run the autoresearch loop.

    Sweeps configurations in priority order, scores each using perplexity +
    generation quality, and saves results incrementally to JSONL file.

    Args:
        model: Loaded HuggingFace model.
        tokenizer: Loaded HuggingFace tokenizer.
        max_rounds: Max configurations to test.
        results_file: JSONL output path.
        resume: Skip already-tested configurations.
        skip_needle: Skip needle-in-haystack evaluation (faster).
    """
    all_configs = generate_config_list()

    # Resume support: skip already-tested configs
    if resume:
        completed = load_completed_keys(results_file)
        original_count = len(all_configs)
        all_configs = [c for c in all_configs if config_key(c) not in completed]
        print(f"Resume: {len(completed)} already tested, {len(all_configs)} remaining "
              f"(out of {original_count} total)")
    else:
        completed = set()

    # Apply max_rounds cap
    configs = all_configs[:max_rounds]
    total_configs = len(configs)

    if total_configs == 0:
        print("No configurations to test. All done!")
        return

    print(f"\nAutoresearch v2: testing {total_configs} configurations")
    print(f"Results file: {results_file}")
    print(f"Scoring: 60% perplexity + 40% generation quality")
    print(f"Needle-in-haystack: {'SKIP' if skip_needle else 'ON'}")
    print(f"Estimated time: {total_configs * 60 / 60:.0f}-{total_configs * 120 / 60:.0f} minutes")
    print()

    # Initialize benchmark runner
    benchmark_runner = BenchmarkRunner(model, tokenizer)

    # Compute FP16 baseline perplexity ONCE
    print("Computing FP16 baseline perplexity...")
    baseline_ppl = benchmark_runner.compute_model_perplexity(cache=None)
    print(f"FP16 baseline perplexity: {baseline_ppl:.2f}")

    # Compute FP16 baseline responses for self-judge comparison
    print("Computing FP16 baseline generation responses...")
    baseline_responses = benchmark_runner.compute_baseline_responses()
    print(f"Baseline responses computed for {len(baseline_responses)} prompts")
    print()

    best_score = 0.0
    best_config = None
    pareto_frontier: List[Tuple[float, float]] = []

    # Load existing Pareto frontier from results if resuming
    if resume and os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if not entry.get("config", {}).get("baseline"):
                        pareto_frontier = update_pareto(
                            pareto_frontier,
                            entry["compression"],
                            entry["total_score"],
                        )
                        if entry["total_score"] > best_score:
                            best_score = entry["total_score"]
                            best_config = entry["config"]
                except (json.JSONDecodeError, KeyError):
                    continue

    cumulative_time = 0.0

    for round_num, config in enumerate(configs):
        start_time = time.time()

        try:
            result_dict, compression = score_config_v2(
                benchmark_runner,
                config,
                baseline_ppl=baseline_ppl,
                baseline_responses=baseline_responses,
                run_needle=not skip_needle,
            )
            total_score = result_dict["total_score"]
        except Exception as e:
            # Don't let one bad config kill the loop
            elapsed = time.time() - start_time
            error_result = {
                "round": round_num,
                "config": config,
                "config_key": config_key(config),
                "total_score": 0.0,
                "ppl_score": 0.0,
                "gen_score": 0.0,
                "legacy_score": 0.0,
                "per_prompt": [],
                "compression": compute_compression_ratio(config),
                "elapsed_s": round(elapsed, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": f"{type(e).__name__}: {e}",
            }
            with open(results_file, "a") as f:
                f.write(json.dumps(error_result) + "\n")

            if config.get("quantizer_type") == "adaptive":
                err_desc = (
                    f"adaptive h{config['high_bits']}/l{config['low_bits']} "
                    f"v={config['val_bits']}b win={config['fp16_window']}"
                )
            else:
                err_desc = (
                    f"k={config['key_bits']}b v={config['val_bits']}b "
                    f"anchor={config['anchor_interval']} "
                    f"window={config['fp16_window']} "
                    f"resq={config['use_residual_quant']}"
                )
            print(
                f"[{round_num+1}/{total_configs}] ERROR "
                f"{err_desc} "
                f"-- {type(e).__name__}: {e}"
            )
            traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        elapsed = time.time() - start_time
        cumulative_time += elapsed

        # Log result -- merge benchmark result with config metadata
        log_entry = {
            "round": round_num,
            "config": config,
            "config_key": config_key(config),
            "compression": compression,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **result_dict,
        }

        with open(results_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Track best
        is_new_best = total_score > best_score
        if is_new_best:
            best_score = total_score
            best_config = config

        # Update Pareto frontier
        pareto_frontier = update_pareto(pareto_frontier, compression, total_score)

        # Progress indicator
        ppl_pct = result_dict.get("ppl_increase_pct", 0)
        if total_score >= 0.8:
            stars = "**"
        elif total_score >= 0.5:
            stars = "* "
        else:
            stars = ". "

        # Estimated time remaining
        avg_time = cumulative_time / (round_num + 1)
        remaining = (total_configs - round_num - 1) * avg_time
        eta_min = remaining / 60.0

        # Format config description for progress line
        if config.get("quantizer_type") == "adaptive":
            cfg_desc = (
                f"adaptive h{config['high_bits']}/l{config['low_bits']} "
                f"v={config['val_bits']}b  "
                f"win={config['fp16_window']}"
            )
        else:
            cfg_desc = (
                f"k={config['key_bits']}b v={config['val_bits']}b  "
                f"anchor={config['anchor_interval']}  "
                f"win={config['fp16_window']}  "
                f"resq={config['use_residual_quant']}"
            )

        print(
            f"[{round_num+1}/{total_configs}] {stars} "
            f"score={total_score:.4f}  "
            f"ppl={result_dict.get('ppl_score', 0):.3f}  "
            f"gen={result_dict.get('gen_score', 0):.3f}  "
            f"ppl+{ppl_pct:.1f}%  "
            f"comp={compression:.2f}x  "
            f"{cfg_desc}  "
            f"({elapsed:.1f}s, ETA {eta_min:.0f}m)"
        )

        if is_new_best:
            print(f"  >>> NEW BEST: {total_score:.4f} at {compression:.2f}x compression")

        # Periodic GC
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final report
    print("\n" + "=" * 80)
    print("AUTORESEARCH COMPLETE")
    print("=" * 80)
    print(f"Rounds tested:  {total_configs}")
    print(f"Total time:     {cumulative_time / 60:.1f} minutes")
    print(f"Avg per round:  {cumulative_time / max(total_configs, 1):.1f}s")
    print(f"Best score:     {best_score:.4f}")
    print(f"Best config:    {best_config}")
    print(f"Pareto points:  {len(pareto_frontier)}")
    for comp, score in pareto_frontier:
        print(f"  {comp:.2f}x compression, {score:.4f} score")
    print("=" * 80)

    # Full summary from file
    print_final_summary(results_file)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch: overnight TurboQuantDC configuration sweep",
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help=f"Model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=800,
        help="Max configurations to test (default: 800 = full sweep)",
    )
    parser.add_argument(
        "--results-file", default="autoresearch_results.jsonl",
        help="JSONL file to save results (default: autoresearch_results.jsonl)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-tested configurations (read from results file)",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Just print the summary from an existing results file, don't run",
    )
    parser.add_argument(
        "--skip-needle", action="store_true",
        help="Skip needle-in-haystack evaluation (faster per round)",
    )
    args = parser.parse_args()

    results_path = os.path.join(REPO_ROOT, args.results_file)

    if args.summary_only:
        print_final_summary(results_path)
        return

    # Load model ONCE
    print(f"Loading model: {args.model}")
    print(f"This takes ~30-60 seconds...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Model loaded on {next(model.parameters()).device}")

    # Run FP16 baseline using benchmark runner
    print("\n--- FP16 BASELINE (v2: perplexity + generation quality) ---")
    benchmark_runner = BenchmarkRunner(model, tokenizer)
    baseline_result_obj = benchmark_runner.evaluate_baseline()

    print(f"FP16 baseline perplexity: {baseline_result_obj.baseline_ppl:.2f}")
    print(f"FP16 baseline total score: {baseline_result_obj.total_score:.4f}")
    print(f"FP16 baseline legacy score: {baseline_result_obj.legacy_score:.4f}")
    print(f"FP16 needle score: {baseline_result_obj.needle_score:.4f}")
    for p in baseline_result_obj.per_prompt:
        print(f"  [{p['type']:>12}] legacy={p.get('legacy_score', 0):.2f}  {p['response'][:80]}")

    # Save baseline
    baseline_config = {
        "baseline": True, "key_bits": 16, "val_bits": 16,
        "anchor_interval": 0, "fp16_window": 0,
        "use_residual_quant": False, "mse_only": True,
    }
    baseline_entry = {
        "round": -1,
        "config": baseline_config,
        "config_key": "fp16_baseline",
        "compression": 1.0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **baseline_result_obj.to_dict(),
    }

    # Only write baseline if not resuming or if file doesn't exist
    if not args.resume or not os.path.exists(results_path):
        with open(results_path, "a") as f:
            f.write(json.dumps(baseline_entry) + "\n")

    print()

    # Run the loop
    run_autoresearch(
        model, tokenizer,
        max_rounds=args.max_rounds,
        results_file=results_path,
        resume=args.resume,
        skip_needle=args.skip_needle,
    )


if __name__ == "__main__":
    main()
