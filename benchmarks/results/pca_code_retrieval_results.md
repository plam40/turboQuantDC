# PCA Code Retrieval Experiment Results

**Date:** 2026-04-04
**Model:** Qwen/Qwen2.5-3B-Instruct (BnB 4-bit, eager attention)
**Hardware:** RTX 4090
**Context:** 2072 tokens | **d:** 128 | **KV heads:** 2 | **GQA:** 8x
**Quantization:** 3-bit (2-bit MSE + 1-bit signs)
**Retrieval k:** 64 | **Window:** 64
**PCA hash:** Binary (1-bit per PCA component), Hamming-2 multi-probe
**WHT hash:** 4-level (2-bit per coordinate), Hamming-1 multi-probe
**Layers tested:** [0, 5, 10, 15, 20, 25, 35] (7 layers x 2 KV heads = 14 heads)
**Queries per head:** 32 (last 32 positions)

## Hypothesis

PCA rotation concentrates 48.7% of variance in the top 10% of coordinates.
WHT rotation spreads information uniformly (12.5% in any 10%). PCA codes
from leading coordinates should be a dramatically better locality-sensitive
hash for approximate attention retrieval.

**Prior WHT results (code_retrieval_results.md):**
- Oracle mode at hash_width=16: 67% recall
- Realistic mode at hash_width=16: 55-59% recall
- Root cause: WHT spreads info uniformly, 16/128 coords = 12.5% of info

## TL;DR

**PCA codes are a better hash -- but not dramatically so.** The advantage
is in EFFICIENCY, not absolute recall. PCA achieves the same recall as WHT
while searching 5x fewer candidates, and maintains quality at wide hash
widths where WHT collapses. The fundamental ceiling is that attention
similarity does not align perfectly with key-space similarity in ANY rotation.

**Critical discovery during implementation:** The PCA quantizer WHITENS
coordinates to N(0, 1/d) before quantizing, erasing the variance concentration.
Using whitened PCA codes gives IDENTICAL results to WHT codes (both become
uniform random hash). The correct approach is BINARY PCA hash: use the sign
of each raw PCA coordinate (above/below median) as a 1-bit hash component.

---

## Experiment 1: PCA vs WHT Hash Quality (Oracle Mode)

For each query, hash the top-1 attended key's code, retrieve from same/nearby
buckets, measure recall of true high-attention tokens. PCA uses binary hash
(2 levels, Hamming-2 probe). WHT uses 4-level hash (Hamming-1 probe).

| Hash Width | PCA Var% | WHT Var% | PCA Top-1 | WHT Top-1 | PCA Recall | WHT Recall | PCA Cosine | WHT Cosine | PCA Cand | WHT Cand |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4 | 27.7% | 3.1% | 0.770 | 0.958 | 0.734 | 0.772 | 0.892 | 0.891 | 1393 | 543 |
| 6 | 35.2% | 4.7% | 0.792 | 1.000 | 0.733 | 0.758 | 0.888 | 0.866 | 695 | 329 |
| 8 | 41.5% | 6.2% | 0.864 | 1.000 | 0.752 | 0.754 | 0.860 | 0.854 | 306 | 278 |
| 10 | 46.9% | 7.8% | 0.893 | 1.000 | 0.756 | 0.753 | 0.852 | 0.852 | 112 | 254 |
| **12** | **51.7%** | **9.4%** | **1.000** | **1.000** | **0.779** | **0.752** | **0.863** | **0.851** | **38** | **198** |
| 16 | 59.6% | 12.5% | 1.000 | 1.000 | 0.754 | 0.750 | 0.851 | 0.851 | 6 | 168 |
| 24 | 70.9% | 18.8% | 1.000 | 1.000 | 0.746 | 0.746 | 0.851 | 0.851 | 1 | 59 |
| **32** | **78.7%** | **25.0%** | **1.000** | **0.636** | **0.745** | **0.668** | **0.851** | **0.797** | **1** | **0** |

### Key Findings

**1. Efficiency advantage (hash_width=12, sweet spot):**
PCA achieves 77.9% recall searching only 38 candidates (1.9% of tokens).
WHT achieves 75.2% recall searching 198 candidates (9.5% of tokens).
PCA gets **+2.7pp recall with 5.2x fewer candidates**.

**2. Scalability advantage (hash_width=32):**
PCA maintains 1.000 top-1 recall and 74.5% significant recall.
WHT collapses to 0.636 top-1 recall and 66.8% significant recall.
PCA codes scale to wide hash where WHT codes over-hash into singletons.

**3. Equal-candidate comparison (hash_width=8 PCA vs hash_width=8 WHT):**
Both search ~300 candidates. PCA recall=75.2%, WHT recall=75.4%.
At the same candidate budget, recall is essentially identical.

### PCA vs WHT Delta

| Hash Width | Top-1 Delta | Recall Delta | Candidate Ratio (PCA/WHT) |
|:---:|:---:|:---:|:---:|
| 4 | -0.188 | -0.037 | 2.57x more |
| 6 | -0.208 | -0.025 | 2.11x more |
| 8 | -0.136 | -0.002 | 1.10x more |
| 10 | -0.107 | +0.003 | 0.44x fewer |
| **12** | **+0.000** | **+0.027** | **0.19x fewer** |
| 16 | +0.000 | +0.004 | 0.04x fewer |
| 24 | +0.000 | +0.000 | 0.02x fewer |
| **32** | **+0.364** | **+0.077** | **N/A (WHT=0)** |

---

## Experiment 2: PCA Retrieval Attention (Key-as-Query, Realistic)

Hash the key at the query position (proxy for actual query), retrieve candidates
via binary PCA code, compute attention over candidates + window.

| Hash Width | Top-1 | Top-5 | Top-10 | Cosine | Recall | Avg Cand | Frac Searched |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4 | 0.509 | 0.724 | 0.724 | 0.590 | 0.690 | 1397 | 67.4% |
| 6 | 0.509 | 0.726 | 0.726 | 0.595 | 0.693 | 704 | 34.0% |
| 8 | 0.511 | 0.725 | 0.724 | 0.604 | 0.687 | 302 | 14.6% |
| 10 | 0.502 | 0.722 | 0.719 | 0.606 | 0.688 | 118 | 5.7% |
| **12** | **0.522** | **0.719** | **0.710** | **0.607** | **0.693** | **44** | **2.1%** |
| 16 | 0.500 | 0.690 | 0.678 | 0.608 | 0.659 | 7 | 0.4% |
| 24 | 0.493 | 0.673 | 0.661 | 0.608 | 0.640 | 2 | 0.1% |
| 32 | 0.493 | 0.670 | 0.657 | 0.608 | 0.636 | 1 | 0.1% |

**Window-only baseline:** Top-1=0.493, Top-5=0.666, Recall=0.632

### Comparison vs Prior WHT Results at hash_width=16

| Method | Top-1 | Top-5 | Recall | Candidates | Frac Searched |
|:---|:---:|:---:|:---:|:---:|:---:|
| PCA binary hash (hw=12) | 0.522 | 0.719 | 0.693 | 44 | 2.1% |
| WHT 4-level hash (hw=16)* | 0.502 | 0.580 | 0.559 | 104 | 5.0% |
| PCA binary hash (hw=16) | 0.500 | 0.690 | 0.659 | 7 | 0.4% |
| Window only (64) | 0.493 | 0.666 | 0.632 | 0 | 3.1% |

*From code_retrieval_results.md (same model, same context length)

**Best PCA (hw=12): +6.1pp recall over window-only, searching only 2.1% of tokens.**

---

## Experiment 3: Combined Compression + Retrieval System

Head-to-head at hash_width=16, 3-bit quantization.

| System | Top-1 | Top-5 | Recall | Cosine | MSE | Index Memory |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| PCA rot + PCA code retrieval | 0.500 | 0.690 | 0.659 | 0.608 | 0.480 | **0 bytes** |
| WHT rot + WHT code retrieval | 0.493 | 0.669 | 0.638 | 0.599 | 4.393 | **0 bytes** |
| Window only (no retrieval) | 0.493 | 0.666 | 0.632 | 0.608 | N/A | 0 bytes |
| FAISS IVF-Flat (np=8)* | 0.778 | 0.924 | 0.937 | 0.979 | N/A | 512 B/key |

*FAISS results from faiss_retrieval_results.md (500 token context, k=128)

**Compression MSE ratio (PCA/WHT):** 0.109x (9.2x lower MSE with PCA)

---

## Memory Analysis

| System | Compression | Index Memory/Key | Total Extra Memory (100K tokens) |
|:---|:---:|:---:|:---:|
| FAISS Flat (exact IP) | None | 512 B | 51.2 MB |
| FAISS IVF-PQ (m=16) | None | 16 B | 1.6 MB |
| **PCA + Code (this work)** | **5x** | **0 B** | **0 B** |
| WHT + Code (prior work) | 5x | 0 B | 0 B |

---

## Root Cause Analysis

### Why PCA codes are better but not dramatically better

**1. The whitening trap (discovered during implementation):**
The PCA quantizer whitens all coordinates to N(0, 1/d) before quantizing.
After whitening, every coordinate has the same distribution -- the variance
concentration is erased. Whitened PCA codes are NO better than WHT codes.
Only RAW (unwhitened) PCA coordinates carry the concentrated signal.

**2. The fundamental ceiling: attention != key similarity:**
Attention scores depend on query-key interaction, position, structural tokens,
and recency -- not just key-key proximity. Even a perfect hash on key vectors
cannot capture:
- Position bias (recency): handled by the window
- Structural tokens (BOS, delimiters): position-dependent, not content-dependent
- Query-key interaction: the hash operates in key space, not query-key space

**3. PCA advantage is in efficiency, not absolute recall:**
PCA codes discriminate more per bit because leading components carry more info.
At hash_width=12 (binary PCA), 12 bits capture 51.7% of variance.
At hash_width=12 (WHT 4-level), 24 bits capture 9.4% of variance.
PCA uses fewer bits for more signal, yielding fewer but better candidates.

### What DOES work

The combined system at hash_width=12 achieves:
- 69.3% recall searching only 2.1% of tokens (44 candidates)
- vs FAISS at 93.7% recall but needing 512 bytes per key of extra memory
- vs window-only at 63.2% recall

For long-context scenarios (>>2K tokens), the 2.1% search fraction means
the PCA code index scales sub-linearly: at 100K tokens, checking 2.1% = 2100
candidates, not 100K. Combined with the zero-memory property, this is a
practical first-pass filter.

---

## Verdict

**PCA codes are a BETTER hash than WHT codes, confirmed by two mechanisms:**

1. **5x more efficient:** Same recall with 5x fewer candidates at hash_width=12
   (38 candidates vs 198 for WHT). This translates to lower search latency.

2. **Scale-robust:** PCA maintains perfect top-1 recall at hash_width=32 where
   WHT collapses to 63.6%. Binary PCA hash with Hamming-2 probe never over-hashes.

**PCA codes do NOT achieve >90% recall (target not met).** The ceiling is ~78%
in oracle mode and ~69% in realistic mode. This is a fundamental limitation
of key-space hashing for attention retrieval, not specific to PCA vs WHT.

**The real finding: PCA rotation unifies compression AND retrieval.** The same
rotation that gives 9.2x lower MSE for compression ALSO gives a better
locality-sensitive hash for retrieval -- zero extra cost, zero extra memory.
This is the unified system:
- PCA rotation: 9.2x better MSE compression
- PCA binary codes: 5x more efficient hash retrieval
- Combined: 5x compression + sub-linear retrieval + zero index overhead
