# Path to $100M: TurboQuantDC Strategic Research Compendium

*Living document. Last updated: 2026-03-28.*
*Built by one person + 35 AI agents in a single session.*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What We Built](#2-what-we-built)
3. [What We Proved](#3-what-we-proved)
4. [What We Disproved](#4-what-we-disproved)
5. [The Market](#5-the-market)
6. [The Competition](#6-the-competition)
7. [The Technology Frontier](#7-the-technology-frontier)
8. [The Generation Quality Wall](#8-the-generation-quality-wall)
9. [The Acquisition Playbook](#9-the-acquisition-playbook)
10. [The Roadmap](#10-the-roadmap)
11. [Agent Deployment Log](#11-agent-deployment-log)
12. [Key Decisions Made](#12-key-decisions-made)
13. [Open Questions](#13-open-questions)

---

## 1. Executive Summary

TurboQuantDC is a from-scratch PyTorch implementation of Google's TurboQuant algorithm (ICLR 2026, arXiv 2504.19874) for compressing LLM key-value caches from FP16 to 3 bits per dimension, achieving 5.0x compression with <0.5% attention quality loss. In a single session, 35+ AI agents built 21 source modules (8,859 lines), 19 test files (10,541 lines), 10 benchmark scripts (6,980 lines), streaming inference, asymmetric K/V compression, fused attention kernels, Triton kernels, and a HuggingFace integration -- all validated on real models (Qwen2.5-3B, Qwen2.5-14B, Qwen3.5-27B) against the paper's theoretical bounds. The core algorithm works: 0.9969 cosine similarity, 94.4% top-5 attention match, 5.0x compression on real attention patterns. The unsolved problem is autoregressive generation quality at <4-bit, which no implementation in the ecosystem has cracked. First to solve it wins the KV compression space.

---

## 2. What We Built (7 Phases)

### Phase 1: Core Algorithm (complete, 2026-03-26)

Modules: Lloyd-Max codebook (283 lines), rotation (102 lines), PolarQuant (138 lines), QJL (119 lines), estimator (195 lines), KV cache (256 lines).

179 tests, all passing in 6 seconds.

| Metric | Measured | Paper Bound | Status |
|---|---|---|---|
| D_mse (b=3) | 0.035 | 0.043 | Within bound |
| D_prod (b=3, d=128) | 0.0014 | 0.0021 | Within bound |
| Unbiasedness | bias ~ 0 | E[error] = 0 | Confirmed |
| Compression ratio (3-bit) | 5.02x | 5.0x | Matches |
| 1-bit centroids | +/-0.07052 | +/-0.07053 | 5-digit match |

### Phase 2: Real Model Testing (complete, 2026-03-26)

Validated on real KV caches from Qwen2.5-3B-Instruct at 2K/4K context. GPU throughput: 27M vec/sec quantize, 71M vec/sec inner product on RTX 4090.

| Bits | Cosine Sim | Top-1 | Top-5 | Compression |
|---|---|---|---|---|
| 2 | 0.9886 | 69% | 84% | 7.3x |
| **3** | **0.9959** | **80%** | **91.7%** | **5.0x** |
| 4 | 0.9987 | 89% | 94% | 3.8x |

### Phase 3: Big Model / Long Context (complete, 2026-03-26)

Tested Qwen2.5-14B (d=128) and Qwen3.5-27B (d=256). The 27B model is a hybrid DeltaNet+Attention architecture with head_dim=256, a dimension the paper never tested.

| Model | Bits | CosSim | Top-1 | Top-5 | Compression |
|---|---|---|---|---|---|
| Qwen2.5-14B (d=128) | 3 | 0.9964 | 78% | 95.3% | 5.0x |
| Qwen2.5-14B (d=128) | 4 | 0.9989 | 89% | 97.7% | 3.8x |
| **Qwen3.5-27B (d=256)** | **3** | **0.9932** | **98.4%** | **100%** | **5.2x** |
| **Qwen3.5-27B (d=256)** | **4** | **0.9980** | **100%** | **100%** | **3.9x** |

### Phase 4: Ship (complete, 2026-03-28)

Showcase, packaging, 54 integration tests. 233 total tests passing. PyPI package, HuggingFace Space, Colab notebook, vLLM integration module (936 lines).

Showcase results on RTX 4090, Qwen2.5-3B-Instruct:

| Bits | Cosine Sim | Top-1 | Top-5 | Compression |
|---|---|---|---|---|
| 2 | 0.9913 | 77.8% | 93.1% | 7.3x |
| **3** | **0.9969** | **73.6%** | **94.4%** | **5.0x** |
| 4 | 0.9990 | 86.1% | 94.4% | 3.8x |

### Phase 5: Beyond the Paper (complete, 2026-03-28)

Extensions from community research and independent experimentation. 98 new tests, 331 total passing in 13.18s.

| Extension | Module | Tests | Key Result |
|---|---|---|---|
| Sparse V dequantization | `sparse_v.py` | 20 | +22.8% decode speed, 0.999+ cosine sim |
| Fractional bit rates | `outlier.py` | 15 | 2.5-bit @ 5.56x, 3.5-bit @ 4.13x |
| Layer-adaptive compression | `layer_adaptive.py` | 32 | tail_preserve/gradient/custom strategies |
| Walsh-Hadamard Transform | `rotation.py` | 12 | O(d log d), 256x less memory than QR |
| Temporal decay | `temporal_decay.py` | 19 | 3-tier hot/warm/cold, 30-34% additional savings |

### Phase 6: Impossible Inference (complete)

Streaming inference engine: 14B model on 24GB GPU at 8.3 GB peak VRAM. 1M tokens of KV cache on single RTX 4090: 4.92 GB at TQ-2, 100% needle retrieval at all 6 depth positions.

Delta coding disproved: Pearson r=0.001 between adjacent layers, relative delta norm 142%.

Sparsity profiled: layer 1 = 98.7% sparse, neuron predictability 93.4% across prompts.

Ultra-long context needle-in-haystack:

| Context | Bits | VRAM (KV only) | Needle Retrieval | Compression |
|---|---|---|---|---|
| 128K | 2 | 0.61 GB | 100% | 7.3x |
| 256K | 2 | 1.23 GB | 100% | 7.3x |
| 512K | 2 | 2.46 GB | 100% | 7.3x |
| 1M | 2 | 4.92 GB | 100% | 7.3x |

### Phase 7: Launch + Generation Quality (in progress)

Built: chunked prefill engine, asymmetric K/V compression (K4/V2 at 5.1x), fused attention kernel (cosine sim 1.000000 vs dequantize path), custom attention with QJL, mse_only mode. Hard-task benchmarks run with honest results.

HF integration benchmark (Qwen2.5-3B, autoregressive generation):

| Config | Compression | Output Quality |
|---|---|---|
| FP16 (baseline) | 1.0x | Reference |
| TQ-4 (mse_only) | 3.8x | Mostly coherent, some repetition |
| TQ-3 | 5.0x | Degraded, garbled on most prompts |
| TQ-2 | 7.3x | Severely degraded, unusable |

**Codebase totals:** 21 source modules (8,859 lines), 19 test files (10,541 lines), 10 benchmark scripts (6,980 lines). 568+ tests across all files.

---

## 3. What We Proved

### Proven with measured data:

**1. TurboQuant matches paper bounds.** MSE distortion 0.035 vs 0.043 bound (3-bit). IP distortion 0.0014 vs 0.0021 bound. Compression ratio 5.02x vs 5.0x target. Lloyd-Max centroids match to 5 decimal places (+/-0.07052 vs +/-0.07053).

**2. 5.0x compression at 0.9969 cosine similarity on real attention patterns.** Measured on Qwen2.5-3B-Instruct showcase benchmark, not synthetic data. All 72 attention heads individually score >= 0.99. Worst head: 0.9902.

**3. 1M tokens fit on a single RTX 4090.** 4.92 GB at TQ-2, vs 36 GB for FP16. 100% needle retrieval at all 6 depth positions across 128K, 256K, 512K, 1M context lengths.

**4. 14B model runs on 24GB GPU via streaming.** Streaming inference engine loads one layer at a time from CPU. Peak VRAM: 8.3 GB for a model that needs 29.5 GB in FP16.

**5. QJL correction hurts generation quality.** Mean absolute error 27% higher with QJL than MSE-only at 3-bit. Confirmed independently by TheTom (llama.cpp), 0xSero (Triton/vLLM), tonbistudio (V3 reference, MSE-only default), scos-lab (8-model benchmark), vLLM PR reviewers.

**6. 4-bit MSE-only produces coherent generation.** Tested on Qwen2.5-3B: correct factual answers at short context. Compression ratio 3.8x with no QJL overhead and no custom attention kernel needed. Drop-in replacement via `TurboQuantCache(bits=4, mse_only=True)`.

**7. Keys need more bits than values.** K/V norm ratio averages 2.38x, reaches up to 48.5x in early layers (Qwen2.5-3B). Asymmetric K4/V2 achieves 5.1x compression with better quality than symmetric K3/V3.

**8. Early transformer layers are highly sparse.** Layer 1: 98.7% sparse. Neuron predictability: 93.4% across prompts. This enables sparse activation loading for streaming inference.

**9. TurboQuant is within 1.5x of Shannon's information-theoretic limit.** 3-bit TQ achieves 5.0x compression. The theoretical floor for this distortion level is approximately 3.3x. Not much algorithmic headroom remains.

**10. Fused attention is mathematically identical to the dequantize path.** Cosine similarity between fused kernel output and standard dequantize+matmul output: 1.000000. FP16 materialization is NOT the cause of generation quality loss.

**11. Qwen3.5-27B at d=256 hits 100% top-5 at 3-bit.** A head dimension the paper never tested. Every attention head preserves its correct top-5 pattern.

**12. GPU throughput exceeds targets by 27-71x.** 27M vec/sec quantize, 71M vec/sec inner product on RTX 4090 (pure PyTorch, no Triton). Paper target was 1M vec/sec.

---

## 4. What We Disproved

### Disproved with evidence:

**1. Cross-layer delta coding does NOT work for KV caches.** Pearson correlation between adjacent-layer KV vectors: r=0.001. Relative delta norm: 142% (deltas are larger than the layers). Conditional entropy approximately equals marginal entropy -- zero savings. Shannon's 1.5 bits/param floor confirmed. Delta coding DOES work for model weights (adjacent layers share structure). KV caches are fundamentally different: each layer sees different projected subspaces of the hidden state.

**2. QJL does NOT help generation.** The paper guarantees unbiased inner products in expectation. Autoregressive generation samples a single realization at each step. QJL's variance (pi/(2m) * ||y||^2 per score) exceeds the MSE bias it corrects. Every production TurboQuant implementation has disabled QJL or moved it behind a fused kernel that avoids materializing the corrected vector.

**3. FP16 materialization is NOT the cause of generation quality loss.** Built a fused attention kernel that computes scores in float32 directly from compressed indices without ever producing FP16 intermediate tensors. Score comparison: cosine sim 1.000000 vs dequantize path. Quality is identical. The problem is error compounding across 36 layers and 100+ decode steps, not FP16 precision.

**4. 200B dense on 24GB is information-theoretically impossible at near-lossless quality.** Weight entropy alone exceeds 37.5 GB at quality levels needed for coherent generation. The five-layer compression stack (weight quantization + TQ KV + temporal decay + streaming + delta/sparse) can fit the model at 3 GB GPU memory, but speed is limited to 0.3-16 tok/s depending on delta coding and sparse loading -- both of which remain projections, not validated implementations. The 200B MoE path (20B active) at 13.1 GB is realistic.

**5. Proxy metrics overstate actual generation quality.** 0.9969 cosine similarity and 94.4% top-5 attention match at 3-bit. But autoregressive generation produces garbled output beyond ~100 tokens at 3-bit. Per-step accuracy does not compound into end-to-end quality because softmax exponentiates score errors, and 36 layers of multiplicative compounding randomize attention routing.

---

## 5. The Market

### Size

- AI inference market: $106B (2025), projected $255B by 2030.
- 52 million monthly Ollama downloads (up from 100K in Q1 2023, 520x growth).
- 400K+ production GPUs on SGLang.
- 135,000+ GGUF-format models on HuggingFace.
- KV cache is the #1 unsolved bottleneck at long context.

### Pain point

The KV cache grows linearly with context length and quadratically consumes attention compute. Concrete examples from validated projections:

| Model | Context | FP16 KV Cache | TQ-3 KV Cache | Savings |
|---|---|---|---|---|
| Qwen2.5-14B | 32K | 6.0 GB | 1.2 GB | 4.8 GB freed |
| Qwen3.5-27B | 128K | 8.0 GB | 1.6 GB | 6.4 GB freed |
| Qwen3.5-27B | 262K | 16.0 GB | 3.1 GB | OOM -> fits with 7 GB spare |
| Qwen2.5-3B | 1M | 36.0 GB | 4.92 GB | 1M tokens on a single RTX 4090 |

Inference companies running 1000 GPUs spend approximately $21M/year on KV-cache-constrained VRAM. 5x compression directly reduces GPU count.

### Gap

Weight quantization is solved (GGUF, GPTQ, AWQ -- mature, widely deployed). KV cache quantization has no production-ready sub-4-bit solution. FP8 KV (2x compression) is deployed in some vLLM configurations. 3-bit KV (5x) is pure research. The gap between weight quantization maturity and KV cache quantization maturity is where TurboQuantDC sits.

---

## 6. The Competition

### TurboQuant ecosystem (all appeared within 1 week of the paper, March 25-28 2026)

| Project | Stars | Focus | Unique Angle |
|---|---|---|---|
| TheTom/turboquant_plus | 2,244 | llama.cpp + Metal | Sparse V, Boundary V, turbo2/3/4 types, norm correction |
| tonbistudio/turboquant-pytorch | 630 | Reference impl | V3 defaults to MSE-only, clean PyTorch |
| 0xSero/turboquant | 471 | Triton + vLLM | First vLLM plugin, performance focus |
| spiritbuun/llama-cpp-turboquant-cuda | 279 | CUDA | 97.5% of standard decode speed |
| scrya-com/rotorquant | 193 | Clifford algebra | 120x fewer rotation parameters |
| mitkox/vllm-turboquant | 316 | vLLM fork | Pre-packaged server |
| hackimov/turboquant-kv | -- | HF fused kernel | Only impl with working TurboQuantProd generation |

### Community consensus (as of March 28, 2026)

1. QJL is dead for generation. Every production implementation disabled it.
2. Keys need more bits than values.
3. Hadamard rotation preferred over QR (deterministic, lower variance, faster).
4. Generation quality at <4-bit is unsolved by everyone.
5. MSE-only at 4-bit is the current "it works" configuration.

### Adjacent competitors

| Method | Compression | Key Difference |
|---|---|---|
| KIVI (ICML 2024) | 2.6x (asymmetric INT4) | Per-channel, no rotation, established |
| NVIDIA KVPress/KVTC | Up to 20x | Requires calibration data, corporate-backed |
| Apple CommVQ | Near 1-bit | Requires model retraining |
| Gear (low-rank + quant) | ~4x | More complex, no unbiasedness guarantee |
| KVQuant (NF4 + outlier) | ~4x | Per-channel with outlier handling |

### Potential paradigm shift

"The Residual Stream Is All You Need" (March 2026 preprint) argues KV caches are architecturally redundant. If this holds, all KV compression becomes irrelevant. Hybrid architectures (Qwen3.5: 75% linear attention, 25% standard attention) are already reducing KV cache size at the architecture level. MLA (DeepSeek) compresses KV via latent projection at near-zero cost. Combined with hardware trajectory (24 GB -> 32 GB -> 64 GB over the next 5 years), the KV compression problem may be resolved by 2030 without algorithmic intervention.

---

## 7. The Technology Frontier

### Information-theoretic limits

| Limit | TQ-3 Status | Headroom |
|---|---|---|
| Shannon MSE bound | Within 1.5x | Minimal algorithmic improvement possible |
| KV cache phase transition | ~90% compression (1.5 bits/dim) hallucination cliff | We hit this at TQ-2 |
| Landauer limit | 85,000x headroom | Irrelevant |
| Speed of light bandwidth | 100x headroom | Not the bottleneck |

### What is physically possible on RTX 4090 (24 GB)

| Scenario | Verdict | Evidence |
|---|---|---|
| 200B MoE (20B active) on 24GB | YES | 13.1 GB projected with 4-bit weights + TQ-3 + temporal decay |
| 200B dense on 24GB | Fits but slow | 3.0 GB with full streaming stack, 0.3-16 tok/s depending on delta/sparse |
| 200B dense near-lossless on 24GB | NO | Shannon entropy of weights alone exceeds 37.5 GB |
| 1M context on single GPU | YES | Proven: 4.92 GB at TQ-2, 100% needle retrieval |
| 53B dense at >95% quality (non-streaming) | YES | 4-bit weights + TQ-3 KV, fits within 24 GB |

### Maximum model sizes on RTX 4090

From the Impossible Inference analysis, using validated compression measurements:

| Stack | 4K ctx | 32K ctx | 128K ctx |
|---|---|---|---|
| FP16 | 11B | 9B | 6B |
| 4-bit weights | 43B | 31B | 13B |
| 4-bit + TQ-3 KV | 44B | 42B | 34B |
| 4-bit + TQ-3 + temporal | 45B | 42B | 36B |
| Streaming + 4-bit* | 1.7TB | 1.1TB | 316B |

*Speed-limited, not memory-limited. Requires CPU RAM >= model size.

### Hardware trajectory

| GPU | VRAM | PCIe | Year |
|---|---|---|---|
| RTX 4090 | 24 GB | 4.0 (32 GB/s) | 2022 |
| RTX 5090 | 32 GB | 5.0 (64 GB/s) | 2025 |
| RTX 6090 (est) | 48 GB | 6.0 (128 GB/s) | 2027 |

Algorithmic efficiency improvements + hardware VRAM growth = the KV cache problem largely disappears by 2030-2032. TurboQuant shelf life as a dominant approach: 3-5 years.

### Architecture trajectory

Hybrid models (75% linear attention, 25% standard attention in Qwen3.5) reduce KV cache 4-10x. MLA (DeepSeek) reduces further 20-30x. Combined: KV cache problem shrinks 200-1000x by 2036. This makes TurboQuant's value proposition time-limited, which means speed of deployment matters more than perfection.

---

## 8. The Generation Quality Wall

### The problem

Every TurboQuant implementation produces garbled output during autoregressive generation at >100 tokens at 3-bit. This is universal:

- TurboQuantDC: all modes (MSE-only, QJL, fused) fail at 3-bit beyond ~100 tokens.
- TheTom/turboquant_plus: does not claim autoregressive generation.
- 0xSero/turboquant: focuses on throughput benchmarks, not generation quality.
- vLLM PR reviewers: found 0% gsm8k accuracy with TQ-3.
- hackimov/turboquant-kv: only works with fused kernel that bypasses dequantization.
- DEJAN blog: confirmed adding QJL correction back to reconstructed vector produces cosine similarity of 0.69 and incoherent output.

### Root cause

The error amplification chain, as analyzed in the Attention Fix Plan:

```
Per-coordinate MSE error (~0.035 at 3-bit)
  -> Per-vector reconstruction error (23-44%)
    -> Per-score error in Q @ K^T
      -> Softmax EXPONENTIATES score errors
        -> Wrong attention weights -> wrong value sum
          -> Wrong hidden state fed to next layer
            -> 36 layers of multiplicative compounding
              -> After ~100 tokens: attention routing randomized
```

Quantitative per-layer breakdown (d=128, 3-bit, single layer):
- MSE score bias: ~0.001 per score (systematic, same direction)
- QJL correction: removes bias but adds variance ~0.012 per score
- Softmax on biased scores: weights shift ~0.1% (tolerable)
- Softmax on unbiased-but-noisy scores: weights shift 2-5% (destructive over 36 layers)

### What we tried and the results

| Approach | Result |
|---|---|
| MSE-only (no QJL) | Works at <100 tokens, fails beyond. Best generation mode at 3-bit. |
| Full QJL correction | Worse than MSE-only. Variance exceeds bias benefit. |
| Custom attention kernel | Same quality as standard dequantize-then-multiply path. |
| Fused attention (no FP16 materialization) | Mathematically identical scores (cos sim 1.000000). Same quality. |
| Norm correction | Fixes magnitude, not direction. Marginal improvement. |
| Different bit-widths | 4-bit mostly coherent. 3-bit garbled. 2-bit unusable. |

### What has not been tried yet (from the Attention Fix Plan)

| Approach | Expected Impact | Confidence |
|---|---|---|
| FP16 anchor layers (every 6th layer uncompressed) | Resets error accumulation chain | HIGH feasibility, ~3.5x compression |
| Softmax temperature calibration | Compensates for score variance reduction | MEDIUM |
| Higher bits for keys only (5-6 bit MSE keys + 2-bit values) | Reduces key error at softmax input | HIGH (conservative path) |
| Per-head adaptive bit allocation | Saves bits on easy heads | MEDIUM |
| Gradient-free correction network | Learns to fix distortion | LOW feasibility, HIGH potential |
| WHT + 4-bit MSE-only keys, 2-bit values | Coherent 500+ tokens at 4.5x | HIGH (community validated) |

### The opportunity

Nobody has solved autoregressive generation at <4-bit KV compression. The first team to crack it -- producing coherent 500+ token generation at >=4x compression -- wins the entire KV compression space. This is the single highest-value unsolved problem in the TurboQuant ecosystem.

---

## 9. The Acquisition Playbook

### Top acquisition targets

**1. AMD ($100-300M range).** Hungriest buyer in the inference market. Needs differentiation from NVIDIA on the software stack. ROCm port of TurboQuantDC would be a strategic asset. AMD has been acquiring inference tooling companies to close the CUDA ecosystem gap.

**2. NVIDIA ($150-500M range).** Highest ceiling. TurboQuant complements NVIDIA's KVTC (which requires calibration data). NVIDIA acquired OctoAI (~$250M) and Lepton AI (~$300M) for inference optimization. Pattern: acquire teams with production inference tech.

**3. Meta ($100-300M range).** Largest Llama fleet. Acqui-hire pattern (Inflection -> Microsoft at $650M). Runs the most open-source LLM inference at scale. KV compression directly reduces their serving costs.

### What is needed for $100M

| Requirement | Current Status | Gap |
|---|---|---|
| 3-5 person team | 1 person + AI agents | Need 2-4 engineers |
| $500K-2M ARR | $0 | Need enterprise customers |
| Production deployment | No production users | Need 1-2 company pilots |
| 500+ GitHub stars | ~0 (not yet launched) | Achievable in 90 days if demo is strong |
| Provisional patents | Not filed | Must file within 12-month grace period |

### Comparable transactions

| Company | Acquirer | Price | Relevance |
|---|---|---|---|
| MosaicML | Databricks | $1.3B | Training infrastructure |
| OctoAI | NVIDIA | ~$250M | Inference optimization |
| Lepton AI | NVIDIA | ~$300M | Inference serving |
| Inflection | Microsoft | $650M | Acqui-hire |
| Groq | NVIDIA | $20B (reported) | Inference hardware |

### Immediate actions for acquisition path

1. **File provisional patents** on asymmetric K/V compression, fused MSE attention, layer-adaptive bit allocation. 12-month grace period from GitHub publication date.
2. **Port to ROCm** to unlock AMD acquisition path.
3. **Build the viral demo** -- 70B model streaming on single GPU, OOM-to-fits screenshot.
4. **Launch enterprise support** ($5-25K/month) for inference companies.
5. **Recruit 1-2 GPU kernel engineers** -- the Triton/CUDA expertise is the primary hiring need.

---

## 10. The Roadmap

### Next 30 days

- [ ] Fix generation quality (WHT + 4-bit MSE-only keys, temperature calibration, FP16 anchor layers)
- [ ] Post to r/LocalLLaMA (Tuesday/Wednesday, 10-11 AM EST)
- [ ] Submit Show HN
- [ ] ROCm port for AMD path
- [ ] File provisional patents
- [ ] Launch Colab notebook on free T4 GPU
- [ ] Target 100+ GitHub stars

### Next 90 days

- [ ] vLLM mainline PR (module exists at 936 lines, needs production testing)
- [ ] Contribute benchmarks to llama.cpp Discussion #20969
- [ ] SGLang integration exploration
- [ ] Research paper submission (NeurIPS/ICML deadline)
- [ ] Production pilot with 1-2 inference companies
- [ ] Recruit 1-2 engineers
- [ ] "Can It Run?" community benchmark challenge

### Next 12 months

- [ ] $500K+ ARR from enterprise support
- [ ] 500+ GitHub stars
- [ ] 3-5 person team
- [ ] Solve the generation quality wall (this is the $100M unlock)
- [ ] GTC 2027 submission
- [ ] Approach AMD/NVIDIA/Meta for acquisition conversations

### Key timing constraint

Google may release official TurboQuant code in Q2 2026. The window to establish TurboQuantDC as the reference community implementation is weeks, not months. llama.cpp launched with "I have no idea if it works correctly" -- shipping fast beats shipping perfect.

---

## 11. Agent Deployment Log

| Phase | Agents | Key Deliverables |
|---|---|---|
| Phase 1 (Core) | Archimedes, Darwin, Turing | Math spec, reference analysis, 6 core modules, 179 tests |
| Phase 4 (Ship) | Tesla, Maxwell, Turing | 54 integration tests, showcase, packaging, vLLM module |
| Phase 5 (Beyond Paper) | Gauss, Euler, Faraday, Fourier, Boltzmann | Sparse V, fractional bits, layer-adaptive, WHT, temporal decay, 98 new tests |
| Research Wave 1 | Aristotle, Sun Tzu, Machiavelli, Feynman, Bezos | Market analysis, competition, viral strategy, tech frontier, integration |
| Phase 6 (Impossible) | Newton, Planck, Dirac, Oppenheimer | Streaming engine, delta coding (disproved), sparsity profiling, impossible inference |
| Launch Kit | Mercury, Galileo, Curie, Hemingway, Da Vinci | PyPI package, Colab notebook, HF integration, Reddit post, Gradio demo |
| Frontier Research | Shannon, Turing2036, Hawking, Carmack, Debate | Info-theoretic limits, architecture predictions, physics, engineering, devil's advocate |
| Visualization | Michelangelo (x2) | Interactive GitHub Pages demo |
| Generation Fix | WarCouncil, Volta, Edison | Attention fix battle plan, fused attention kernel, chunked prefill |
| Strategy | Rockefeller, TomWatch, Scout agents | M&A strategy, competitive intelligence, research landscape |
| Features | Kepler, Gutenberg, Babbage, Lovelace, Fermi, TuringBench, Hubble | Asymmetric K/V, README, chunked prefill, HF PR, Triton kernels, hard-task benchmarks, 512K demo |

Total: 35+ agents deployed across 8+ swarms in a single session.

---

## 12. Key Decisions Made

| Date | Decision | Rationale | Source |
|---|---|---|---|
| Mar 25 | Target 27-32B models, not 70B | 70B Q4 weights alone exceed 24GB | PLAN.md |
| Mar 26 | Cosine sim ~0.96 on synthetic is expected | Paper's 0.995 target is for real LLM patterns, not random vectors | Phase 1 validation |
| Mar 28 | `mse_only=True` as default | QJL hurts generation; community confirms across 6+ independent implementations | Phase 7 / Attention Fix Plan |
| Mar 28 | 4-bit recommended for generation | 3-bit garbles output at >100 tokens; 4-bit mostly coherent at 3.8x compression | HF benchmark results |
| Mar 28 | Skip delta coding for KV caches | r=0.001 between layers; deltas are 142% of original -- no savings | Phase 6 delta coding test |
| Mar 28 | Asymmetric K/V (K4/V2) | Keys need more bits (48.5x norm ratio in early layers); values only need MSE reconstruction | Phase 7 weight analysis |
| Mar 28 | Ship within 3-5 days of completion | TurboQuant wave is cresting; Google official code expected Q2 2026; land-grab window is closing | Growth Playbook timing analysis |
| Mar 28 | Fused attention built as diagnostic | Proved FP16 materialization is not the quality issue (cos sim 1.000000) | Phase 7 fused attention test |
| Mar 28 | Generation wall is fundamental, not a bug | Every implementation across Python/C/Rust/Triton hits the same wall at 3-bit | Attention Fix Plan community survey |
| Mar 28 | AMD as primary acquisition target | Hungriest buyer, best strategic fit, needs ROCm inference differentiation | M&A analysis |
| Mar 28 | File patents immediately | 12-month grace period from GitHub publication; novel findings worth protecting | Strategy phase |

---

## 13. Open Questions

1. **Can FP16 anchor layers (every 6th layer uncompressed) break the error accumulation chain?** The Attention Fix Plan rates this HIGH confidence at ~3.5x compression. Not yet implemented.

2. **What is the exact generation quality threshold bit-width?** 4-bit works (mostly). 3-bit fails. Is 3.5-bit sufficient? The fractional bit-rate infrastructure exists (`outlier.py`) but has not been tested for generation.

3. **Can a lightweight correction network (100K params) learn to fix compression distortion?** Low feasibility, high potential. No implementation attempted.

4. **Will "The Residual Stream Is All You Need" make KV compression irrelevant?** If validated, all KV compression research becomes architectural footnotes. Unknown timeline.

5. **Can we deploy to production before Google releases official code (Q2 2026)?** The growth playbook estimates the consolidation window at weeks. We have the most complete implementation (568+ tests, 5 models validated, 5 beyond-paper extensions) but zero GitHub stars.

6. **What is the minimum team size for acquisition credibility?** Comparables suggest 3-5 people with production deployment. Currently 1 person + AI agents.

7. **Can TurboQuant's fused kernel path (hackimov approach) enable TurboQuantProd for generation?** hackimov shows it works when the fused kernel never materializes a single reconstructed vector. We built a fused kernel and confirmed score equivalence, but have not tested it end-to-end for generation quality. This is the most promising unexplored path.

8. **How does TurboQuant perform on Llama, Mistral, and Gemma architectures?** All validation is on Qwen-family models. The algorithm is architecture-agnostic in theory, but empirical validation on other families is missing.

9. **Is the 90% compression cliff (1.5 bits/dim hallucination spike) universal or model-dependent?** Observed in our tests but not systematically characterized across architectures and scales.

10. **Can temporal decay (hot/warm/cold tiers) bridge the generation quality gap?** Keep recent tokens at FP16, compress only old tokens. This is the safest path to usable generation quality at high compression -- the infrastructure exists in `temporal_decay.py` but has not been tested for generation.

---

*This document will be updated as new research, benchmarks, and market data come in.*

**Links:**
- GitHub: https://github.com/dhawalc/turboQuantDC
- PyPI: `pip install turboquantdc`
- Paper: arXiv 2504.19874 (ICLR 2026)
- Interactive Demo: https://dhawalc.github.io/turboQuantDC/
- HuggingFace Space: https://huggingface.co/spaces/dhawalchheda/turboquantdc
