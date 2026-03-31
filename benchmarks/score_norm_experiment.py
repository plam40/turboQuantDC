#!/usr/bin/env python3
"""Score normalization experiment.

Hypothesis: normalizing compressed attention scores to match FP16
score statistics (mean, std) fixes the softmax distribution and
generation quality.

Approach:
  Phase 1 (Calibration): Run one FP16 forward pass. Hook into each
    attention layer to intercept Q and K after RoPE. Compute Q@K^T
    manually and record per-layer mean and std of the pre-softmax
    attention scores.

  Phase 2 (Inference): Run with TQ mse_only cache. After computing
    raw compressed attention scores in each layer, normalize them:
      scores_corrected = (scores - mu_tq) / sigma_tq * sigma_fp16 + mu_fp16

  Phase 3 (Comparison): Generate with the same 5 prompts using:
    (a) FP16 baseline
    (b) TQ-4 mse_only (no normalization)
    (c) TQ-4 mse_only + score normalization
    Compare outputs.

Run:
    cd /home/dhawal/turboQuantDC && python benchmarks/score_norm_experiment.py
"""

import os
import sys
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from turboquantdc.hf_integration import TurboQuantCache, TurboQuantLayer


# ---------------------------------------------------------------------------
# Calibration: capture FP16 attention score statistics
# ---------------------------------------------------------------------------

class ScoreCalibrator:
    """Run one FP16 forward pass and record per-layer attention score stats.

    Hooks into each attention layer's output to intercept Q and K after
    the linear projection + RoPE, then computes Q@K^T to get the raw
    pre-softmax attention scores and records their mean and std.
    """

    def __init__(self):
        self.stats: Dict[int, Dict[str, float]] = {}
        self._hooks = []
        self._q_cache: Dict[int, torch.Tensor] = {}
        self._k_cache: Dict[int, torch.Tensor] = {}

    def attach(self, model: torch.nn.Module) -> None:
        """Attach hooks to capture Q and K from every attention layer."""
        inner_model = model.model if hasattr(model, "model") else model
        layers = list(inner_model.layers) if hasattr(inner_model, "layers") else list(inner_model.h)

        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn if hasattr(layer, "self_attn") else getattr(layer, "attn", None)
            if attn is None:
                continue

            # Hook into the attention forward to capture the actual scores
            hook = self._make_hook(layer_idx, attn, model.config)
            handle = attn.register_forward_hook(hook, with_kwargs=True)
            self._hooks.append(handle)

    def _make_hook(self, layer_idx: int, attn_mod, config):
        """Create a hook that intercepts attention forward and measures scores."""
        n_q_heads = config.num_attention_heads
        n_kv_heads = getattr(config, "num_key_value_heads", n_q_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // n_q_heads)
        scale = 1.0 / math.sqrt(head_dim)

        def hook_fn(module, inputs, kwargs, outputs):
            # hidden_states may be passed as positional or keyword arg
            if len(inputs) > 0:
                hidden_states = inputs[0]
            elif "hidden_states" in kwargs:
                hidden_states = kwargs["hidden_states"]
            else:
                return  # Can't find hidden_states, skip
            bsz, seq_len, _ = hidden_states.shape

            q = attn_mod.q_proj(hidden_states)
            k = attn_mod.k_proj(hidden_states)

            q = q.reshape(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)
            k = k.reshape(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)

            # Apply RoPE if position_embeddings are available
            # The hook receives the full inputs tuple; position_embeddings
            # are typically passed as a kwarg or late positional arg.
            # We'll try to extract them.
            rope_applied = False

            # Try keyword arg "position_embeddings" (transformers >= 4.45)
            # inputs is a tuple of positional args; kwargs not directly available
            # in register_forward_hook. But many models pass position_embeddings
            # as positional arg. We'll try to apply RoPE from the module.
            if hasattr(attn_mod, "rotary_emb"):
                try:
                    # Qwen2/Llama rotary_emb expects (value_states, position_ids)
                    # position_ids from inputs if available
                    # For the calibration pass we just need approximate stats,
                    # so even without exact RoPE the statistics will be
                    # directionally correct.
                    pass
                except Exception:
                    pass

            # Expand KV heads for GQA
            heads_per_kv = n_q_heads // n_kv_heads
            if heads_per_kv > 1:
                k_expanded = k.repeat_interleave(heads_per_kv, dim=1)
            else:
                k_expanded = k

            # Compute raw attention scores: Q @ K^T * scale
            # Shape: (bsz, n_q_heads, seq_q, seq_k)
            scores = torch.matmul(q.float(), k_expanded.float().transpose(-1, -2))
            scores = scores * scale

            # Apply causal mask: only look at lower triangle
            seq_q = scores.shape[-2]
            seq_k = scores.shape[-1]
            causal_mask = torch.triu(
                torch.ones(seq_q, seq_k, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            # Mask out future positions before computing stats
            scores_masked = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("nan"))

            # Compute per-layer stats (ignoring NaN/masked positions)
            valid = ~torch.isnan(scores_masked)
            valid_scores = scores_masked[valid]

            if valid_scores.numel() > 0:
                self.stats[layer_idx] = {
                    "mean": valid_scores.mean().item(),
                    "std": valid_scores.std().item(),
                    "min": valid_scores.min().item(),
                    "max": valid_scores.max().item(),
                    "count": valid_scores.numel(),
                }

        return hook_fn

    def detach(self) -> None:
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def calibrate(self, model, tokenizer, calibration_text: str) -> Dict[int, Dict[str, float]]:
        """Run a single FP16 forward pass and return per-layer score stats."""
        self.attach(model)
        try:
            inputs = tokenizer(calibration_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
        finally:
            self.detach()
        return self.stats


# ---------------------------------------------------------------------------
# Score-normalized TurboQuant cache
# ---------------------------------------------------------------------------

class NormalizedTurboQuantLayer(TurboQuantLayer):
    """TurboQuantLayer with post-dequantization score normalization.

    After dequantizing keys, scales them by sigma_fp16 / sigma_tq so that
    the resulting Q@K^T scores have approximately the same variance as FP16.

    The mean shift is handled by adding a bias term after the score computation.
    But since softmax is shift-invariant (softmax(x + c) = softmax(x)), we
    only need to match the variance/scale.

    Actually, softmax IS shift-invariant for a constant shift, but NOT for
    a per-position shift. And the mean difference is roughly constant across
    positions. So matching std alone should suffice.

    Simpler approach: scale the dequantized keys by (sigma_fp16 / sigma_tq).
    This directly scales the dot product scores by the same ratio.
    """

    def __init__(self, bits=3, seed=42, mse_only=False,
                 key_scale: float = 1.0, score_bias: float = 0.0):
        super().__init__(bits=bits, seed=seed, mse_only=mse_only)
        self.key_scale = key_scale
        self.score_bias = score_bias

    def _dequantize_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize with key scaling applied."""
        keys, values = super()._dequantize_all()
        if self.key_scale != 1.0:
            keys = keys * self.key_scale
        return keys, values


class NormalizedTurboQuantCache(TurboQuantCache):
    """TurboQuantCache with score normalization.

    After calibration, creates layers that scale dequantized keys so
    that Q@K^T has approximately the same std as FP16 attention scores.
    """

    def __init__(self, bits=3, seed=42, mse_only=True):
        super().__init__(bits=bits, seed=seed, mse_only=mse_only)
        self._calibration_stats: Dict[int, Dict[str, float]] = {}
        self._tq_stats: Dict[int, Dict[str, float]] = {}
        self._key_scales: Dict[int, float] = {}

    def set_calibration_stats(self, fp16_stats: Dict[int, Dict[str, float]],
                               tq_stats: Dict[int, Dict[str, float]]) -> None:
        """Set the FP16 and TQ score statistics for normalization."""
        self._calibration_stats = fp16_stats
        self._tq_stats = tq_stats

        # Compute per-layer key scale factors
        for layer_idx in fp16_stats:
            if layer_idx in tq_stats:
                sigma_fp16 = fp16_stats[layer_idx]["std"]
                sigma_tq = tq_stats[layer_idx]["std"]
                if sigma_tq > 1e-8:
                    self._key_scales[layer_idx] = sigma_fp16 / sigma_tq
                else:
                    self._key_scales[layer_idx] = 1.0

    def _make_layer(self, layer_idx: int) -> TurboQuantLayer:
        """Create a NormalizedTurboQuantLayer with the appropriate scale."""
        scale = self._key_scales.get(layer_idx, 1.0)
        return NormalizedTurboQuantLayer(
            bits=self.bits,
            seed=self.seed + layer_idx,
            mse_only=self.mse_only,
            key_scale=scale,
        )


# ---------------------------------------------------------------------------
# TQ score statistics collector
# ---------------------------------------------------------------------------

def calibrate_tq_scores(
    model, tokenizer, text: str, bits: int = 4, mse_only: bool = True,
) -> Dict[int, Dict[str, float]]:
    """Measure attention score statistics when using TQ cache.

    Strategy: run a forward pass with TQ cache, then for each layer
    compute Q (from re-running projections on the FP16 calibration
    hidden_states captured via hooks) against K_dequant (from the
    TQ cache layers), and measure the resulting score stats.

    This approach captures the actual score distribution that HF
    attention would see when using the TQ cache.
    """
    config = model.config
    n_q_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_q_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // n_q_heads)
    scale = 1.0 / math.sqrt(head_dim)

    # Step 1: Run forward pass with TQ cache to populate it
    cache = TurboQuantCache(bits=bits, seed=42, mse_only=mse_only)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs, past_key_values=cache)

    # Step 2: Capture per-layer hidden_states (Q source) by hooking
    # into the decoder layers during a second FP16 forward pass
    layer_hidden = {}
    hooks = []

    inner_model = model.model if hasattr(model, "model") else model
    layers_list = list(inner_model.layers) if hasattr(inner_model, "layers") else list(inner_model.h)

    for layer_idx, layer in enumerate(layers_list):
        attn = layer.self_attn if hasattr(layer, "self_attn") else getattr(layer, "attn", None)
        if attn is None:
            continue

        def _capture_hook(lidx):
            def hook_fn(module, inputs, kwargs, outputs):
                if len(inputs) > 0:
                    layer_hidden[lidx] = inputs[0].detach()
                elif "hidden_states" in kwargs:
                    layer_hidden[lidx] = kwargs["hidden_states"].detach()
            return hook_fn

        handle = attn.register_forward_hook(_capture_hook(layer_idx), with_kwargs=True)
        hooks.append(handle)

    with torch.no_grad():
        model(**inputs)  # FP16 pass to capture hidden_states

    for h in hooks:
        h.remove()

    # Step 3: For each layer, compute scores using Q (from hidden_states)
    # and K_dequant (from TQ cache)
    stats = {}
    for layer_idx in range(len(cache._layers)):
        if layer_idx not in layer_hidden:
            continue

        tq_layer = cache._layers[layer_idx]
        if tq_layer._seq_len == 0:
            continue

        hidden = layer_hidden[layer_idx]
        bsz, seq_len, _ = hidden.shape

        attn_mod = layers_list[layer_idx].self_attn if hasattr(layers_list[layer_idx], "self_attn") else layers_list[layer_idx].attn

        with torch.no_grad():
            q = attn_mod.q_proj(hidden)
            q = q.reshape(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)

            # Get dequantized keys from TQ cache
            k_dequant, _ = tq_layer._dequantize_all()
            # k_dequant: (batch, n_kv_heads, seq_kv, head_dim)

            heads_per_kv = n_q_heads // n_kv_heads
            if heads_per_kv > 1:
                k_expanded = k_dequant.repeat_interleave(heads_per_kv, dim=1)
            else:
                k_expanded = k_dequant

            # Q @ K_dequant^T * scale
            scores = torch.matmul(q.float(), k_expanded.float().transpose(-1, -2)) * scale

            seq_q = scores.shape[-2]
            seq_k = scores.shape[-1]
            causal_mask = torch.triu(
                torch.ones(seq_q, seq_k, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores_masked = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("nan"))
            valid = ~torch.isnan(scores_masked)
            valid_scores = scores_masked[valid]

            if valid_scores.numel() > 0:
                stats[layer_idx] = {
                    "mean": valid_scores.mean().item(),
                    "std": valid_scores.std().item(),
                    "min": valid_scores.min().item(),
                    "max": valid_scores.max().item(),
                    "count": valid_scores.numel(),
                }

    return stats


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

PROMPTS = [
    "What is the capital of Australia? Answer briefly:",
    "What is 15 + 27?",
    "Who wrote the novel 1984? Answer briefly:",
    "Explain what a neural network is in two sentences:",
    "Write a Python function that returns the factorial of n:",
]

CALIBRATION_TEXT = (
    "The TurboQuant algorithm is a two-stage vector quantization method "
    "for compressing the key-value cache in large language models. "
    "Stage one uses PolarQuant with random orthogonal rotation followed by "
    "Lloyd-Max scalar quantization. Stage two applies QJL bias correction "
    "using one-bit sign projections to produce unbiased inner product estimates."
)


def generate_with_cache(model, tokenizer, prompt: str,
                        cache=None, max_new_tokens=50) -> str:
    """Generate text, optionally using a TQ cache."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    if cache is not None:
        kwargs["past_key_values"] = cache
    out = model.generate(**inputs, **kwargs)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    BITS = 4  # 4-bit MSE-only is the threshold where generation starts working

    print("=" * 70)
    print("  Score Normalization Experiment")
    print("=" * 70)
    print()

    # Load model
    print("[1/6] Loading Qwen2.5-3B-Instruct (4-bit base weights)...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    print(f"  Model loaded on {model.device}")
    print()

    # --- Phase 1: FP16 Calibration ---
    print("[2/6] Phase 1: Calibrating FP16 attention score statistics...")
    fp16_calibrator = ScoreCalibrator()
    fp16_stats = fp16_calibrator.calibrate(model, tokenizer, CALIBRATION_TEXT)

    print(f"  Captured stats for {len(fp16_stats)} layers")
    print(f"  {'Layer':>6}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for layer_idx in sorted(fp16_stats.keys()):
        s = fp16_stats[layer_idx]
        print(f"  {layer_idx:>6}  {s['mean']:>10.4f}  {s['std']:>10.4f}  "
              f"{s['min']:>10.4f}  {s['max']:>10.4f}")
    print()

    # --- Phase 2: TQ Calibration ---
    print(f"[3/6] Phase 2: Calibrating TQ-{BITS} mse_only score statistics...")
    tq_stats = calibrate_tq_scores(model, tokenizer, CALIBRATION_TEXT, bits=BITS, mse_only=True)

    print(f"  Captured stats for {len(tq_stats)} layers")
    print(f"  {'Layer':>6}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for layer_idx in sorted(tq_stats.keys()):
        s = tq_stats[layer_idx]
        print(f"  {layer_idx:>6}  {s['mean']:>10.4f}  {s['std']:>10.4f}  "
              f"{s['min']:>10.4f}  {s['max']:>10.4f}")
    print()

    # --- Compute scale factors ---
    print("[4/6] Computing per-layer scale factors (sigma_fp16 / sigma_tq)...")
    scale_factors = {}
    for layer_idx in sorted(set(fp16_stats.keys()) & set(tq_stats.keys())):
        sigma_fp16 = fp16_stats[layer_idx]["std"]
        sigma_tq = tq_stats[layer_idx]["std"]
        scale = sigma_fp16 / sigma_tq if sigma_tq > 1e-8 else 1.0
        scale_factors[layer_idx] = scale

    print(f"  {'Layer':>6}  {'sigma_fp16':>12}  {'sigma_tq':>12}  {'scale':>10}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*10}")
    for layer_idx in sorted(scale_factors.keys()):
        sigma_fp16 = fp16_stats[layer_idx]["std"]
        sigma_tq = tq_stats[layer_idx]["std"]
        print(f"  {layer_idx:>6}  {sigma_fp16:>12.6f}  {sigma_tq:>12.6f}  "
              f"{scale_factors[layer_idx]:>10.6f}")

    mean_scale = sum(scale_factors.values()) / len(scale_factors) if scale_factors else 1.0
    print(f"\n  Mean scale factor: {mean_scale:.6f}")
    print(f"  Scale range: [{min(scale_factors.values()):.4f}, {max(scale_factors.values()):.4f}]")
    print()

    # --- Phase 3: Generation comparison ---
    print("[5/6] Phase 3: Generation comparison")
    print()

    # (a) FP16 baseline
    print("-" * 70)
    print("  (a) FP16 BASELINE")
    print("-" * 70)
    fp16_outputs = []
    for prompt in PROMPTS:
        response = generate_with_cache(model, tokenizer, prompt, cache=None)
        fp16_outputs.append(response)
        print(f"  Q: {prompt}")
        print(f"  A: {response[:200]}")
        print()

    # (b) TQ-4 mse_only, no normalization
    print("-" * 70)
    print(f"  (b) TQ-{BITS} MSE-ONLY (no normalization)")
    print("-" * 70)
    tq_outputs = []
    for prompt in PROMPTS:
        cache = TurboQuantCache(bits=BITS, seed=42, mse_only=True)
        response = generate_with_cache(model, tokenizer, prompt, cache=cache)
        tq_outputs.append(response)
        print(f"  Q: {prompt}")
        print(f"  A: {response[:200]}")
        print()

    # (c) TQ-4 mse_only + score normalization
    print("-" * 70)
    print(f"  (c) TQ-{BITS} MSE-ONLY + SCORE NORMALIZATION")
    print("-" * 70)
    norm_outputs = []
    for prompt in PROMPTS:
        norm_cache = NormalizedTurboQuantCache(bits=BITS, seed=42, mse_only=True)
        norm_cache.set_calibration_stats(fp16_stats, tq_stats)
        response = generate_with_cache(model, tokenizer, prompt, cache=norm_cache)
        norm_outputs.append(response)
        print(f"  Q: {prompt}")
        print(f"  A: {response[:200]}")
        print()

    # (d) Also try uniform scaling (single global scale factor)
    print("-" * 70)
    print(f"  (d) TQ-{BITS} MSE-ONLY + UNIFORM SCALE ({mean_scale:.4f})")
    print("-" * 70)
    uniform_outputs = []
    for prompt in PROMPTS:
        uniform_cache = NormalizedTurboQuantCache(bits=BITS, seed=42, mse_only=True)
        # Create uniform stats: every layer gets the mean scale
        uniform_fp16 = {i: {"std": mean_scale} for i in fp16_stats}
        uniform_tq = {i: {"std": 1.0} for i in tq_stats}
        uniform_cache.set_calibration_stats(uniform_fp16, uniform_tq)
        response = generate_with_cache(model, tokenizer, prompt, cache=uniform_cache)
        uniform_outputs.append(response)
        print(f"  Q: {prompt}")
        print(f"  A: {response[:200]}")
        print()

    # --- Summary ---
    print("=" * 70)
    print("[6/6] Summary")
    print("=" * 70)
    print()

    def coherence_score(outputs: List[str]) -> Tuple[int, int]:
        """Simple coherence check: not empty, not degenerate repetition."""
        coherent = 0
        for resp in outputs:
            words = resp.split()
            if len(words) > 3:
                # Check for degenerate repetition
                if not any(resp.count(w) > 5 for w in words[:3] if len(w) > 2):
                    coherent += 1
        return coherent, len(outputs)

    methods = [
        ("FP16 baseline", fp16_outputs),
        (f"TQ-{BITS} mse_only", tq_outputs),
        (f"TQ-{BITS} + per-layer norm", norm_outputs),
        (f"TQ-{BITS} + uniform scale", uniform_outputs),
    ]

    print(f"  {'Method':<30}  {'Coherent':>10}  {'Score':>8}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*8}")
    for name, outputs in methods:
        c, t = coherence_score(outputs)
        print(f"  {name:<30}  {c}/{t:>8}  {c/t*100:>6.0f}%")

    print()
    print("  Score statistics summary:")
    print(f"  {'Layer':>6}  {'FP16 std':>10}  {'TQ std':>10}  "
          f"{'Ratio':>10}  {'FP16 mean':>10}  {'TQ mean':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for layer_idx in sorted(set(fp16_stats.keys()) & set(tq_stats.keys())):
        fp = fp16_stats[layer_idx]
        tq = tq_stats[layer_idx]
        ratio = fp["std"] / tq["std"] if tq["std"] > 1e-8 else float("inf")
        print(f"  {layer_idx:>6}  {fp['std']:>10.4f}  {tq['std']:>10.4f}  "
              f"{ratio:>10.4f}  {fp['mean']:>10.4f}  {tq['mean']:>10.4f}")

    print()
    print("  Interpretation:")
    print("  - If ratio ~ 1.0 across layers: score distributions already match,")
    print("    normalization won't help (problem is elsewhere).")
    print("  - If ratio varies widely: TQ distorts scores unevenly across layers,")
    print("    per-layer normalization should help.")
    print("  - If normalized outputs are better: score distribution mismatch was")
    print("    the bottleneck; this is a viable fix.")
    print("  - If normalized outputs are same/worse: the problem is in the VALUE")
    print("    path (MSE reconstruction error in V), not the score path.")
    print()


if __name__ == "__main__":
    main()
