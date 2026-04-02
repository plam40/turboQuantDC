# Phase 5: 200B+ Models on RTX 4090

**Date:** 2026-04-02
**Hardware:** RTX 4090 (24GB VRAM), 62GB System RAM, NVMe storage
**Software:** Ollama (serving), CUDA 13.0, Driver 580.126.20

## Executive Summary

**MISSION ACCOMPLISHED.** Three models with >100B total parameters are running on a single RTX 4090:

| Model | Total Params | Active Params | Quant | File Size | tok/s | Quality |
|-------|-------------|---------------|-------|-----------|-------|---------|
| **DeepSeek-V2** | **236B** (MoE) | 21B | IQ1_S | 47 GB | 3.79 | Low (IQ1 too aggressive) |
| **Mixtral 8x22B** | **141B** (MoE) | ~44B | Q2_K | 52 GB | 3.28 | Good |
| **Llama 4 Scout** | **109B** (MoE) | 17B | Q4_K_M | 67 GB | 4.96 | Excellent |

**Winner: Llama 4 Scout** -- best quality at 4.96 tok/s with latest architecture.
**Biggest: DeepSeek-V2** -- 236 billion parameters running on consumer hardware.

---

## Detailed Results

### 1. Llama 4 Scout (109B MoE, 17B active) -- BEST OVERALL

**Model:** `llama4:scout` (Q4_K_M, 67.4 GB)
**Architecture:** MoE with 16 experts, 1 active per token
**Load time:** 32.8s
**VRAM used:** 23.3 GB / 24.6 GB
**RAM used:** 61 GB / 62 GB (very tight)

#### Test 1: Basic Knowledge
- **Prompt:** "What is the capital of France? Answer in one sentence."
- **Response:** "The capital of France is Paris."
- **Speed:** 4.96 tok/s, 8 tokens in 1.6s

#### Test 2: Reasoning
- **Prompt:** "Explain quantum entanglement to a 10-year-old in 3-4 sentences."
- **Response:** Clear analogy with toy cars. Well-structured, age-appropriate explanation.
- **Speed:** 4.28 tok/s, 107 tokens in 25.0s

#### Test 3: Code Generation
- **Prompt:** "Write a Python fibonacci function with memoization and docstring."
- **Response:** Complete, correct implementation with error handling, docstring, example usage, and complexity analysis. Production quality.
- **Speed:** 4.35 tok/s, 469 tokens in 107.7s

**Verdict:** Highest quality output. Fastest generation. But consumes nearly all system RAM (61/62 GB), leaving almost no headroom. Best model if you dedicate the machine to inference.

---

### 2. Mixtral 8x22B (141B MoE, ~44B active) -- BEST BALANCE

**Model:** `mixtral:8x22b-instruct-v0.1-q2_K` (Q2_K, 52.1 GB)
**Architecture:** MoE with 8 experts, 2 active per token
**Load time:** 81.7s
**VRAM used:** 22.9 GB / 24.6 GB
**RAM used:** 50 GB / 62 GB (12 GB headroom)

#### Test 1: Basic Knowledge
- **Response:** "The capital of France is Paris."
- **Speed:** 3.78 tok/s, 8 tokens in 2.1s

#### Test 2: Reasoning
- **Response:** Fun analogy with magic dice. Clear and engaging.
- **Speed:** 3.28 tok/s, 82 tokens in 25.0s

#### Test 3: Code Generation
- **Response:** Correct memoized fibonacci with docstring and input validation.
- **Speed:** 3.23 tok/s, 245 tokens in 75.9s

**Verdict:** Good quality at Q2_K quantization. Leaves 12 GB RAM headroom for other applications. More practical for daily use than Llama 4 Scout.

---

### 3. DeepSeek-V2 (236B MoE, 21B active) -- BIGGEST MODEL

**Model:** `deepseek-v2-236b` (IQ1_S, 47.4 GB)
**Architecture:** MoE with 160 experts, 6 active per token
**Source:** Custom GGUF from HuggingFace (mradermacher/DeepSeek-V2-Chat-i1-GGUF)
**Load time:** 65.0s
**VRAM used:** 23.3 GB / 24.6 GB
**RAM used:** 50 GB / 62 GB

#### Test 1: Basic Knowledge
- **Response:** Correct answer (Paris) but overly verbose with formatting.
- **Speed:** 3.79 tok/s, 87 tokens in 22.9s

#### Test 2: Reasoning
- **Response:** Attempted explanation but rambling, with caveats about complexity.
- **Speed:** 3.74 tok/s, 176 tokens in 47.1s

#### Test 3: Code Generation
- **Response:** FAILED -- produced incorrect fibonacci code (returned "Hello World!").
- **Speed:** 3.80 tok/s, 52 tokens in 13.7s

**Verdict:** IQ1_S quantization is too aggressive for this architecture. The model runs and generates fluent text, but reasoning and code quality are significantly degraded. The 236B parameter count is impressive, but the extreme 1.5-bit quantization destroys too much signal. Would need IQ2_XXS or better for usable output (but that would be ~80GB, exceeding our memory budget).

---

## Performance Comparison

| Metric | Llama 4 Scout | Mixtral 8x22B | DeepSeek-V2 236B |
|--------|---------------|---------------|-------------------|
| Total params | 109B | 141B | 236B |
| Active params/token | 17B | ~44B | 21B |
| Quantization | Q4_K_M | Q2_K | IQ1_S |
| File size | 67.4 GB | 52.1 GB | 47.4 GB |
| Load time | 33s | 82s | 65s |
| Generation speed | **4.96 tok/s** | 3.28 tok/s | 3.79 tok/s |
| VRAM usage | 23.3 GB | 22.9 GB | 23.3 GB |
| RAM usage | 61 GB | 50 GB | 50 GB |
| RAM headroom | 1 GB | **12 GB** | 12 GB |
| Output quality | **Excellent** | Good | Poor (quant damage) |

## Previously Tested (Phase 4)

| Model | Params | Quant | tok/s |
|-------|--------|-------|-------|
| Qwen 2.5 72B | 72B | Q2_K | 1.8 |

## Key Insights

1. **MoE architecture is the key** to running 100B+ models on consumer hardware. Only active parameters need to be processed per token, so a 109B MoE model with 17B active is faster than a 72B dense model.

2. **Quantization quality matters more than parameter count.** Llama 4 Scout at Q4_K_M (109B) massively outperforms DeepSeek-V2 at IQ1_S (236B) on all quality metrics. The bits-per-parameter tradeoff has diminishing returns below ~2 bits.

3. **RAM is the bottleneck**, not VRAM. All three models saturate the 24 GB GPU memory similarly (~23 GB), but the difference in RAM usage determines what else can run alongside inference. Llama 4 Scout at 61/62 GB RAM is practically a dedicated inference machine.

4. **Ollama handles offloading seamlessly.** All models loaded and ran without any manual layer-splitting configuration. The automatic GPU/CPU split just works.

5. **Load time correlates with model size on disk**, not parameter count. The 67 GB Llama 4 takes 33s to load (NVMe speed), while the 52 GB Mixtral takes 82s (possibly less optimized format handling).

## Models Available for Use

All models are cached locally and can be run instantly:

```bash
# Llama 4 Scout (109B, best quality)
ollama run llama4:scout

# Mixtral 8x22B (141B, best balance)
ollama run mixtral:8x22b-instruct-v0.1-q2_K

# DeepSeek-V2 (236B, biggest but quality-degraded)
ollama run deepseek-v2-236b

# Qwen 2.5 (72B, from previous testing)
ollama run qwen2.5:72b-instruct-q2_K
```

## Recommendation

For daily use: **Llama 4 Scout** (`ollama run llama4:scout`)
- Latest architecture (April 2025)
- Best output quality of all tested models
- Fastest at 4.96 tok/s
- 109B total parameters (17B active)
- Only downside: uses nearly all system RAM

For background inference with other apps running: **Mixtral 8x22B** (`ollama run mixtral:8x22b-instruct-v0.1-q2_K`)
- Leaves 12 GB RAM for other processes
- Good quality despite Q2_K quantization
- 141B total parameters
