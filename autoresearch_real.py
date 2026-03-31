#!/usr/bin/env python3
"""REAL AutoResearch for TurboQuantDC — AI modifies the algorithm itself.

Unlike the parameter sweep (autoresearch.py), this loop:
1. Reads the current compression code
2. Proposes a CODE modification
3. Tests generation quality
4. If improved: KEEPS the change, commits it
5. If worse: REVERTS
6. Repeats — each round builds on previous successes

Inspired by Karpathy's autoresearch, applied to KV cache compression.

Usage:
    python autoresearch_real.py [--max-rounds 100] [--model MODEL]
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# The file the AI modifies — a self-contained compression module
# ---------------------------------------------------------------------------

EVOLVING_FILE = os.path.join(os.path.dirname(__file__), "turboquantdc", "evolving_compressor.py")
EVOLVING_BACKUP = EVOLVING_FILE + ".backup"
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "autoresearch_real_results.jsonl")
LOG_FILE = os.path.join(os.path.dirname(__file__), "autoresearch_real.log")

# The seed code — the starting point for evolution
SEED_CODE = '''"""Evolving KV Cache Compressor — modified by autoresearch.

This file is the ONLY file the autoresearch loop modifies.
It contains a self-contained KV cache compressor that duck-types
the HuggingFace Cache protocol.

The autoresearch loop will propose modifications to improve
generation quality while maintaining compression. Each successful
modification is kept and built upon.

Current approach: MSE-only PolarQuant with norm correction.
"""

import math
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from .codebook import LloydMaxCodebook, gaussian_pdf
from .rotation import generate_rotation_matrix


class EvolvingLayer:
    """Single layer's compressed KV storage."""

    def __init__(self, bits: int = 4, seed: int = 42):
        self.bits = bits
        self.seed = seed
        self._seq_len = 0
        self._head_dim = None
        self._num_heads = None
        self._batch_size = None
        self._dtype = None
        self._device = None

        # Quantization components (lazy init)
        self._rotation = None
        self._codebook = None

        # Storage
        self._key_indices: List[torch.Tensor] = []
        self._key_norms: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._val_norms: List[torch.Tensor] = []

    def _lazy_init(self, key_states, value_states):
        self._batch_size = key_states.shape[0]
        self._num_heads = key_states.shape[1]
        self._head_dim = key_states.shape[3]
        self._dtype = key_states.dtype
        self._device = key_states.device
        d = self._head_dim
        device = str(self._device)

        # Rotation matrix
        self._rotation = generate_rotation_matrix(d, seed=self.seed, device=device)

        # Lloyd-Max codebook for keys (full bits)
        self._key_codebook = LloydMaxCodebook(
            bits=self.bits, d=d, pdf=gaussian_pdf, device=device
        )

        # Lloyd-Max codebook for values (2-bit — values tolerate aggressive compression)
        self._val_codebook = LloydMaxCodebook(
            bits=2, d=d, pdf=gaussian_pdf, device=device
        )

    def _quantize_vectors(self, vectors, codebook):
        """Quantize vectors: normalize, rotate, per-coord centroid lookup."""
        batch, heads, seq, d = vectors.shape
        flat = vectors.float().reshape(-1, d)

        # Store norms
        norms = flat.norm(dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-8)

        # Rotate
        rotated = normalized @ self._rotation

        # Quantize per coordinate using codebook boundaries
        indices = torch.bucketize(rotated, codebook.boundaries)
        indices = indices.clamp(0, codebook.centroids.shape[0] - 1)

        return indices.reshape(batch, heads, seq, d), norms.reshape(batch, heads, seq)

    def _dequantize_vectors(self, indices, norms, codebook):
        """Dequantize: centroid lookup, unrotate, rescale."""
        batch, heads, seq, d = indices.shape
        flat_idx = indices.reshape(-1, d)
        flat_norms = norms.reshape(-1)

        # Centroid lookup
        reconstructed = codebook.centroids[flat_idx]

        # Unrotate
        reconstructed = reconstructed @ self._rotation.T

        # Rescale
        reconstructed = reconstructed * flat_norms.unsqueeze(-1)

        return reconstructed.reshape(batch, heads, seq, d)

    def update(self, key_states, value_states):
        """Compress and store new KV, return dequantized full cache."""
        if self._rotation is None:
            self._lazy_init(key_states, value_states)

        # Compress keys (full bits)
        k_idx, k_norms = self._quantize_vectors(key_states, self._key_codebook)
        self._key_indices.append(k_idx)
        self._key_norms.append(k_norms)

        # Compress values (2-bit)
        v_idx, v_norms = self._quantize_vectors(value_states, self._val_codebook)
        self._val_indices.append(v_idx)
        self._val_norms.append(v_norms)

        self._seq_len += key_states.shape[2]

        # Return full dequantized cache
        return self._get_all()

    def _get_all(self):
        if self._seq_len == 0:
            empty = torch.zeros(1, 1, 0, self._head_dim or 1,
                                dtype=self._dtype, device=self._device)
            return empty, empty

        all_k_idx = torch.cat(self._key_indices, dim=2)
        all_k_norms = torch.cat(self._key_norms, dim=2)
        all_v_idx = torch.cat(self._val_indices, dim=2)
        all_v_norms = torch.cat(self._val_norms, dim=2)

        keys = self._dequantize_vectors(all_k_idx, all_k_norms, self._key_codebook)
        values = self._dequantize_vectors(all_v_idx, all_v_norms, self._val_codebook)

        return keys.to(self._dtype), values.to(self._dtype)

    @property
    def seq_len(self):
        return self._seq_len


class EvolvingCache:
    """HF-compatible cache using the evolving compressor."""

    is_compileable = False

    def __init__(self, bits: int = 4, seed: int = 42):
        self.bits = bits
        self.seed = seed
        self._layers: List[EvolvingLayer] = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        while len(self._layers) <= layer_idx:
            self._layers.append(EvolvingLayer(
                bits=self.bits, seed=self.seed + len(self._layers)
            ))
        return self._layers[layer_idx].update(key_states, value_states)

    def get_seq_length(self, layer_idx=0):
        if layer_idx < len(self._layers):
            return self._layers[layer_idx].seq_len
        return 0

    def get_max_cache_shape(self):
        return None

    def __len__(self):
        return len(self._layers)

    def __iter__(self):
        for layer in self._layers:
            yield layer._get_all()

    def __getitem__(self, idx):
        if idx < len(self._layers):
            return self._layers[idx]._get_all()
        return None, None

    def __contains__(self, idx):
        return idx < len(self._layers)

    def reset(self):
        self._layers.clear()

    @property
    def seen_tokens(self):
        if self._layers:
            return self._layers[0].seq_len
        return 0

    def crop(self, max_length):
        pass

    def reorder_cache(self, beam_idx):
        pass

    def batch_repeat_interleave(self, repeats):
        pass

    def batch_select_indices(self, indices):
        pass
'''

# ---------------------------------------------------------------------------
# Test suite — the metric we optimize
# ---------------------------------------------------------------------------

FILLER = (
    "The quarterly report showed steady growth across all divisions. "
    "Revenue increased moderately while operating costs remained stable. "
    "The research team achieved promising results in efficiency studies. "
    "Customer satisfaction scores improved over the previous quarter. "
) * 15  # ~300 tokens

TEST_PROMPTS = [
    {
        "prompt": FILLER + "\nWhat is the capital of Australia? Answer with just the city name:",
        "expected": ["Canberra"],
    },
    {
        "prompt": FILLER + "\nWhat is 15 + 27? Answer with just the number:",
        "expected": ["42"],
    },
    {
        "prompt": FILLER + "\nWho wrote the novel 1984? Answer briefly:",
        "expected": ["George Orwell", "Orwell"],
    },
    {
        "prompt": FILLER + "\nWhat is the largest planet in our solar system? Answer briefly:",
        "expected": ["Jupiter"],
    },
    {
        "prompt": FILLER + "\nWhat is the chemical formula for water?",
        "expected": ["H2O"],
    },
]


def score_generation(model, tokenizer, device="cuda") -> float:
    """Score the current evolving compressor on generation quality.

    Returns a score from 0 to 1.
    """
    # Import the current version of the evolving compressor
    # Force reimport to pick up changes
    import importlib
    if "turboquantdc.evolving_compressor" in sys.modules:
        importlib.reload(sys.modules["turboquantdc.evolving_compressor"])
    from turboquantdc.evolving_compressor import EvolvingCache

    total = 0.0
    for test in TEST_PROMPTS:
        try:
            cache = EvolvingCache(bits=4, seed=42)
            inputs = tokenizer(test["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=30,
                    past_key_values=cache, do_sample=False,
                )
            response = tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )

            # Factual accuracy
            found = sum(1 for e in test["expected"] if e.lower() in response.lower())
            accuracy = found / len(test["expected"])

            # Coherence — penalize repetition
            words = response.split()
            if len(words) > 3:
                unique = len(set(words)) / len(words)
                coherence = min(unique / 0.4, 1.0)
            else:
                coherence = 0.5

            total += 0.7 * accuracy + 0.3 * coherence

        except Exception as e:
            # If the code crashes, score is 0 for this prompt
            total += 0.0

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return round(total / len(TEST_PROMPTS), 4)


# ---------------------------------------------------------------------------
# The AI modification engine — uses Claude to propose code changes
# ---------------------------------------------------------------------------

def propose_modification(current_code: str, history: list, current_score: float) -> str:
    """Use Claude (via claude CLI) to propose a code modification.

    Sends the current code + history to Claude and asks for an improvement.
    Returns the modified code.
    """
    history_text = ""
    for h in history[-10:]:  # Last 10 rounds
        status = "KEPT" if h["kept"] else "REVERTED"
        history_text += (
            f"  Round {h['round']}: {status} | "
            f"score {h['prev_score']:.3f} -> {h['new_score']:.3f} | "
            f"change: {h['description']}\n"
        )

    prompt = f"""You are an AI researcher optimizing a KV cache compression algorithm for LLM inference.

CURRENT SCORE: {current_score:.3f} (out of 1.0)
The score measures generation quality on 5 factual Q&A prompts with ~300 tokens of context prefix.
Higher is better. FP16 baseline scores ~0.85 on this test.

HISTORY OF MODIFICATIONS:
{history_text if history_text else "  (none yet — this is round 0)"}

CURRENT CODE (the file you're modifying):
```python
{current_code}
```

YOUR TASK: Propose ONE modification to improve the generation quality score.

RULES:
1. Output ONLY the complete modified Python file. No explanations before or after.
2. The file must be valid Python that imports correctly.
3. Keep the EvolvingCache class with the same HF Cache protocol interface.
4. Keep the same __init__ signature (bits, seed).
5. Focus on the compression/decompression logic — that's where quality is determined.
6. Small, targeted changes are better than rewrites.

IDEAS TO TRY (pick one or invent your own):
- Add norm correction: store original_norm / reconstruction_norm instead of original_norm
- Smooth centroids by averaging nearby codebook entries
- Use different bit-widths for early vs late layers
- Add a small residual correction (store sign of residual per coordinate)
- Use Walsh-Hadamard transform instead of random QR rotation
- Weight the rotation matrix by learned importance per coordinate
- Apply exponential moving average to smooth reconstructed vectors
- Clip extreme quantization errors
- Use different codebooks for different heads
- Interpolate between two nearest centroids instead of picking one

Output the COMPLETE modified file now:"""

    # Call claude CLI to get the modification
    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "text"],
        capture_output=True, text=True, timeout=120,
        cwd=os.path.dirname(__file__),
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed: {result.stderr[:500]}")

    response = result.stdout.strip()

    # Extract code from response (handle markdown code blocks)
    if "```python" in response:
        code = response.split("```python", 1)[1]
        if "```" in code:
            code = code.split("```", 1)[0]
    elif "```" in response:
        code = response.split("```", 1)[1]
        if "```" in code:
            code = code.split("```", 1)[0]
    else:
        code = response

    return code.strip()


def describe_diff(old_code: str, new_code: str) -> str:
    """Generate a brief description of what changed."""
    old_lines = set(old_code.strip().splitlines())
    new_lines = set(new_code.strip().splitlines())
    added = new_lines - old_lines
    removed = old_lines - new_lines

    if len(added) == 0 and len(removed) == 0:
        return "no changes"

    # Try to summarize
    desc_parts = []
    if len(added) > 0:
        desc_parts.append(f"+{len(added)} lines")
    if len(removed) > 0:
        desc_parts.append(f"-{len(removed)} lines")

    # Look for key patterns in added lines
    for line in added:
        line = line.strip()
        if line.startswith("def "):
            desc_parts.append(f"new function: {line[:50]}")
        elif "correction" in line.lower() or "norm" in line.lower():
            desc_parts.append("norm/correction change")
        elif "residual" in line.lower():
            desc_parts.append("residual handling")
        elif "codebook" in line.lower() or "centroid" in line.lower():
            desc_parts.append("codebook modification")

    return "; ".join(desc_parts[:3]) or f"+{len(added)}/-{len(removed)} lines"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args):
    log_fh = open(LOG_FILE, "a")
    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_fh.write(line + "\n")
        log_fh.flush()

    log("=" * 60)
    log("REAL AUTORESEARCH — Algorithm Evolution")
    log("=" * 60)

    # Initialize the evolving file if it doesn't exist
    if not os.path.exists(EVOLVING_FILE):
        with open(EVOLVING_FILE, "w") as f:
            f.write(SEED_CODE)
        log(f"Created seed code at {EVOLVING_FILE}")

    # Load model ONCE
    log(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log("Model loaded")

    # Score the initial code
    log("Scoring initial code...")
    current_score = score_generation(model, tokenizer)
    log(f"Initial score: {current_score:.3f}")

    # Load history
    history = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                if line.strip():
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if history:
            current_score = history[-1].get("best_score", current_score)
            log(f"Resumed from round {len(history)}, best score: {current_score:.3f}")

    best_score = current_score
    start_round = len(history)

    for round_num in range(start_round, start_round + args.max_rounds):
        log(f"\n{'='*60}")
        log(f"ROUND {round_num}")
        log(f"{'='*60}")

        # Read current code
        with open(EVOLVING_FILE) as f:
            current_code = f.read()

        # Back up current code
        shutil.copy2(EVOLVING_FILE, EVOLVING_BACKUP)

        try:
            # Ask Claude to propose a modification
            log("Asking Claude for a modification...")
            t0 = time.time()
            new_code = propose_modification(current_code, history, current_score)
            propose_time = time.time() - t0
            log(f"Got proposal in {propose_time:.1f}s")

            # Describe the change
            description = describe_diff(current_code, new_code)
            log(f"Change: {description}")

            # Write the new code
            with open(EVOLVING_FILE, "w") as f:
                f.write(new_code)

            # Test it — does it even import?
            log("Testing import...")
            import importlib
            if "turboquantdc.evolving_compressor" in sys.modules:
                del sys.modules["turboquantdc.evolving_compressor"]
            try:
                from turboquantdc import evolving_compressor  # noqa
                importlib.reload(evolving_compressor)
                log("Import OK")
            except Exception as e:
                log(f"IMPORT FAILED: {e}")
                shutil.copy2(EVOLVING_BACKUP, EVOLVING_FILE)
                result = {
                    "round": round_num,
                    "prev_score": current_score,
                    "new_score": 0.0,
                    "best_score": best_score,
                    "kept": False,
                    "description": f"IMPORT FAILED: {str(e)[:100]}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                history.append(result)
                with open(RESULTS_FILE, "a") as f:
                    f.write(json.dumps(result) + "\n")
                continue

            # Score the new code
            log("Scoring new code...")
            t0 = time.time()
            new_score = score_generation(model, tokenizer)
            score_time = time.time() - t0
            log(f"New score: {new_score:.3f} (was {current_score:.3f}) in {score_time:.1f}s")

            # Keep or revert
            if new_score > current_score:
                log(f"IMPROVEMENT! {current_score:.3f} -> {new_score:.3f} (+{new_score - current_score:.3f})")
                log("KEEPING the change")
                kept = True
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    log(f"NEW BEST SCORE: {best_score:.3f}")
            elif new_score == current_score:
                # Same score — keep if it's simpler (fewer lines)
                if len(new_code.splitlines()) <= len(current_code.splitlines()):
                    log(f"Same score, simpler code — KEEPING")
                    kept = True
                else:
                    log(f"Same score, more complex — REVERTING")
                    shutil.copy2(EVOLVING_BACKUP, EVOLVING_FILE)
                    kept = False
            else:
                log(f"WORSE: {current_score:.3f} -> {new_score:.3f} ({new_score - current_score:.3f})")
                log("REVERTING")
                shutil.copy2(EVOLVING_BACKUP, EVOLVING_FILE)
                kept = False

            result = {
                "round": round_num,
                "prev_score": current_score if not kept else current_score - (new_score - current_score) if kept else current_score,
                "new_score": new_score,
                "best_score": best_score,
                "kept": kept,
                "description": description,
                "propose_time": propose_time,
                "score_time": score_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            history.append(result)
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(result) + "\n")

        except Exception as e:
            log(f"ERROR in round {round_num}: {e}")
            traceback.print_exc()
            # Revert on any error
            if os.path.exists(EVOLVING_BACKUP):
                shutil.copy2(EVOLVING_BACKUP, EVOLVING_FILE)
            result = {
                "round": round_num,
                "prev_score": current_score,
                "new_score": 0.0,
                "best_score": best_score,
                "kept": False,
                "description": f"ERROR: {str(e)[:200]}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            history.append(result)
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(result) + "\n")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final report
    log("\n" + "=" * 60)
    log("AUTORESEARCH COMPLETE")
    log(f"Rounds: {len(history)}")
    log(f"Best score: {best_score:.3f}")
    kept_count = sum(1 for h in history if h.get("kept", False))
    log(f"Kept: {kept_count}/{len(history)} modifications")
    log(f"Final code at: {EVOLVING_FILE}")
    log("=" * 60)

    log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real AutoResearch — Algorithm Evolution")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-rounds", type=int, default=50)
    args = parser.parse_args()
    run(args)
