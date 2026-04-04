#!/usr/bin/env python3
"""End-to-end generation quality comparison.

Generates 200 tokens with each compression method and compares against
the FP16 baseline:
  - Exact token match rate vs FP16
  - Perplexity on wikitext-2 subset
  - Tests with boundary layer protection ON/OFF
  - Tests with FP16 hot window ON/OFF

Run:
    python benchmarks/generation_quality.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from turboquantdc.generation_cache import GenerationCache
from turboquantdc.residual_quant import ResidualQuantCache

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MAX_NEW_TOKENS = 200

# Standard prompt for generation comparison
GENERATION_PROMPT = (
    "Explain the mathematical foundations of KV cache compression in "
    "transformer-based language models. Start from the attention mechanism "
    "and derive why quantization of key vectors requires careful treatment "
    "of inner product preservation:"
)

# Wikitext-2 excerpt for perplexity measurement (first ~500 tokens)
WIKITEXT_EXCERPT = (
    "Robert Boulter is an English film, television and theatre actor. He had "
    "a guest-making role on the television series The Bill in 2000. This was "
    "followed by a starring role in the play Herons written by Simon Stephens, "
    "which was performed in 2001 at the Royal Court Theatre. He had a guest role "
    "in the television series The Supply Teacher in 2003. In 2004, Boulter landed "
    "a role in the television series Judge John Deed. He also had roles in the "
    "films Nailing Vienna and ## Fiction. Boulter appeared in the television "
    "series Waterloo Road in 2006. He was nominated for an award at the Off West "
    "End Theatre Awards for his performance in the play Bentham in 2010. "
    "Valkyria Chronicles III is a tactical role-playing video game developed by "
    "Sega and Media.Vision for the PlayStation Portable. Released in January 2011 "
    "in Japan, it is the third game in the Valkyria series. Employing the same "
    "fusion of tactical and real-time gameplay as its predecessors, the story "
    "runs parallel to the first game and follows the Nameless, a penal military "
    "unit serving the nation of Gallia during the Second Europan War who perform "
    "secret black operations and are pitted against the Imperial unit Calamity "
    "Raven."
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """Load model with 4-bit quantization."""
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def generate_with_cache(
    model,
    tokenizer,
    prompt: str,
    cache: Optional[Any],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Tuple[str, List[int], float]:
    """Generate tokens with a given cache and return text, token IDs, and time.

    Args:
        model: HF causal LM.
        tokenizer: Matching tokenizer.
        prompt: Input text.
        cache: KV cache object (None for FP16 baseline).
        max_new_tokens: Number of tokens to generate.

    Returns:
        (generated_text, token_ids, generation_time_sec)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    torch.cuda.empty_cache()
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            past_key_values=cache,
            use_cache=True,
        )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Extract only the generated tokens (not the prompt)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return gen_text, gen_ids, elapsed


def compute_perplexity(
    model,
    tokenizer,
    text: str,
    cache_factory=None,
) -> float:
    """Compute perplexity on a text excerpt.

    Args:
        model: HF causal LM.
        tokenizer: Matching tokenizer.
        text: Input text for perplexity measurement.
        cache_factory: Callable that returns a fresh cache object, or None for FP16.

    Returns:
        Perplexity (exp of average cross-entropy loss).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(model.device)

    if input_ids.shape[1] < 2:
        return float("inf")

    cache = cache_factory() if cache_factory else None

    with torch.no_grad():
        outputs = model(
            input_ids,
            past_key_values=cache,
            use_cache=True,
            labels=input_ids,
        )
    loss = outputs.loss.item()
    return math.exp(loss)


def token_match_rate(baseline_ids: List[int], test_ids: List[int]) -> float:
    """Compute exact token match rate between two generation sequences."""
    min_len = min(len(baseline_ids), len(test_ids))
    if min_len == 0:
        return 0.0
    matches = sum(1 for a, b in zip(baseline_ids[:min_len], test_ids[:min_len]) if a == b)
    return matches / min_len


def first_divergence(baseline_ids: List[int], test_ids: List[int]) -> int:
    """Find the first token position where the sequences diverge."""
    for i, (a, b) in enumerate(zip(baseline_ids, test_ids)):
        if a != b:
            return i
    return min(len(baseline_ids), len(test_ids))


# ---------------------------------------------------------------------------
# Cache configurations
# ---------------------------------------------------------------------------

def make_cache_configs() -> List[Dict[str, Any]]:
    """Define all cache configurations to test."""
    configs = []

    # 1. FP16 baseline (no compression)
    configs.append({
        "name": "FP16 Baseline",
        "factory": lambda: None,
        "description": "No compression, standard DynamicCache",
    })

    # 2. ResidualQuant with boundary protection + FP16 window
    configs.append({
        "name": "ResidualQuant (full)",
        "factory": lambda: GenerationCache(
            key_bits=3, val_bits=3,
            fp16_window=64,
            anchor_strategy="boundary",
            num_layers=36,  # Qwen2.5-3B has 36 layers
            use_residual_quant=True,
            seed=SEED,
        ),
        "description": "3-bit RQ + boundary anchors + FP16 window",
    })

    # 3. ResidualQuant WITHOUT boundary protection
    configs.append({
        "name": "ResidualQuant (no boundary)",
        "factory": lambda: GenerationCache(
            key_bits=3, val_bits=3,
            fp16_window=64,
            anchor_interval=0,
            anchor_strategy="fixed",
            use_residual_quant=True,
            seed=SEED,
        ),
        "description": "3-bit RQ + no anchors + FP16 window",
    })

    # 4. ResidualQuant WITHOUT FP16 window
    configs.append({
        "name": "ResidualQuant (no window)",
        "factory": lambda: GenerationCache(
            key_bits=3, val_bits=3,
            fp16_window=0,
            anchor_strategy="boundary",
            num_layers=36,
            use_residual_quant=True,
            seed=SEED,
        ),
        "description": "3-bit RQ + boundary anchors + no FP16 window",
    })

    # 5. ResidualQuant bare (no boundary, no window)
    configs.append({
        "name": "ResidualQuant (bare)",
        "factory": lambda: GenerationCache(
            key_bits=3, val_bits=3,
            fp16_window=0,
            anchor_interval=0,
            anchor_strategy="fixed",
            use_residual_quant=False,  # MSE-only within GenerationCache
            seed=SEED,
        ),
        "description": "3-bit MSE-only in GenerationCache (no RQ, no anchors, no window)",
    })

    # 6. MSE-only via GenerationCache (with anchors + window but no RQ)
    configs.append({
        "name": "MSE-only (with infra)",
        "factory": lambda: GenerationCache(
            key_bits=3, val_bits=3,
            fp16_window=64,
            anchor_strategy="boundary",
            num_layers=36,
            use_residual_quant=False,
            seed=SEED,
        ),
        "description": "3-bit MSE-only + boundary anchors + FP16 window",
    })

    # 7. ResidualQuantCache (standalone, no anchors/window)
    configs.append({
        "name": "ResidualQuantCache (standalone)",
        "factory": lambda: ResidualQuantCache(bits=3, seed=SEED),
        "description": "Standalone ResidualQuantCache at 3-bit",
    })

    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def format_generation_markdown(results: Dict[str, Any]) -> str:
    """Format generation results as markdown."""
    lines = []
    lines.append("# Generation Quality Benchmark\n")
    lines.append(f"Model: {MODEL_NAME}")
    lines.append(f"Max new tokens: {MAX_NEW_TOKENS}")
    lines.append(f"Sampling: greedy (do_sample=False)")
    lines.append("")

    lines.append("## Token Match Rate vs FP16 Baseline\n")
    lines.append("| Configuration | Match Rate | First Diverge | Perplexity | Gen Time (s) |")
    lines.append("|---------------|-----------|---------------|------------|-------------|")

    for cfg in results.get("configs", []):
        name = cfg["name"]
        match = cfg.get("token_match_rate", 0)
        div = cfg.get("first_divergence", 0)
        ppl = cfg.get("perplexity", float("inf"))
        gen_time = cfg.get("generation_time", 0)
        ppl_str = f"{ppl:.2f}" if ppl < 1e6 else "N/A"

        if name == "FP16 Baseline":
            lines.append(f"| **{name}** | 1.0000 | N/A | {ppl_str} | {gen_time:.2f} |")
        else:
            lines.append(f"| {name} | {match:.4f} | {div} | {ppl_str} | {gen_time:.2f} |")

    lines.append("")

    # Ablation summary
    lines.append("## Ablation: Impact of Each Component\n")
    lines.append("| Component Removed | Match Rate Delta | Perplexity Delta |")
    lines.append("|-------------------|-----------------|-----------------|")

    ablations = results.get("ablations", {})
    for removed, delta in ablations.items():
        match_d = delta.get("match_delta", 0)
        ppl_d = delta.get("ppl_delta", 0)
        sign_m = "+" if match_d >= 0 else ""
        sign_p = "+" if ppl_d >= 0 else ""
        lines.append(f"| {removed} | {sign_m}{match_d:.4f} | {sign_p}{ppl_d:.2f} |")

    lines.append("")

    # Generated text samples
    lines.append("## Generated Text Samples\n")
    for cfg in results.get("configs", []):
        name = cfg["name"]
        text = cfg.get("generated_text", "")
        lines.append(f"### {name}\n")
        lines.append(f"```\n{text[:500]}\n```\n")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  Generation Quality Benchmark")
    print("=" * 70)
    print()

    model, tokenizer = load_model_and_tokenizer()
    configs = make_cache_configs()

    all_results: Dict[str, Any] = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt": GENERATION_PROMPT,
        "configs": [],
    }

    baseline_ids: Optional[List[int]] = None
    baseline_ppl: Optional[float] = None

    # Full config name -> result for ablation computation
    result_map: Dict[str, Dict[str, Any]] = {}

    for cfg in configs:
        name = cfg["name"]
        factory = cfg["factory"]
        desc = cfg["description"]

        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"  {desc}")
        print(f"{'='*50}")

        # 1. Generate tokens
        cache = factory()
        gen_text, gen_ids, gen_time = generate_with_cache(
            model, tokenizer, GENERATION_PROMPT, cache
        )
        print(f"  Generated {len(gen_ids)} tokens in {gen_time:.2f}s")
        print(f"  First 100 chars: {gen_text[:100]}...")

        # 2. Compute perplexity
        try:
            ppl = compute_perplexity(model, tokenizer, WIKITEXT_EXCERPT, factory)
            print(f"  Perplexity: {ppl:.2f}")
        except Exception as e:
            print(f"  Perplexity computation failed: {e}")
            ppl = float("inf")

        # 3. Compare to baseline
        if baseline_ids is None:
            baseline_ids = gen_ids
            baseline_ppl = ppl
            match_rate = 1.0
            div_pos = len(gen_ids)
        else:
            match_rate = token_match_rate(baseline_ids, gen_ids)
            div_pos = first_divergence(baseline_ids, gen_ids)

        print(f"  Token match rate vs FP16: {match_rate:.4f}")
        print(f"  First divergence at token: {div_pos}")

        cfg_result = {
            "name": name,
            "description": desc,
            "token_match_rate": match_rate,
            "first_divergence": div_pos,
            "perplexity": ppl,
            "generation_time": gen_time,
            "num_tokens": len(gen_ids),
            "generated_text": gen_text,
        }
        all_results["configs"].append(cfg_result)
        result_map[name] = cfg_result

    # Compute ablation deltas
    full_rq = result_map.get("ResidualQuant (full)", {})
    ablations = {}

    for removed, bare_name in [
        ("Boundary Anchors", "ResidualQuant (no boundary)"),
        ("FP16 Window", "ResidualQuant (no window)"),
        ("Residual Correction", "MSE-only (with infra)"),
        ("All Protection", "ResidualQuant (bare)"),
    ]:
        bare = result_map.get(bare_name, {})
        if full_rq and bare:
            ablations[removed] = {
                "match_delta": bare.get("token_match_rate", 0) - full_rq.get("token_match_rate", 0),
                "ppl_delta": bare.get("perplexity", 0) - full_rq.get("perplexity", 0),
            }

    all_results["ablations"] = ablations

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    json_path = results_dir / "generation_quality.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON results saved to: {json_path}")

    md_text = format_generation_markdown(all_results)
    md_path = results_dir / "generation_quality.md"
    with open(md_path, "w") as f:
        f.write(md_text)
    print(f"Markdown table saved to: {md_path}")

    print("\n" + md_text)
    return all_results


if __name__ == "__main__":
    main()
