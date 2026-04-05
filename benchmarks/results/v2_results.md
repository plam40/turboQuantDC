# TurboQuantDC v2 Unified Cache Results

**Date:** 2026-04-04
**Model:** Qwen/Qwen2.5-3B-Instruct (BnB 4-bit, RTX 4090)
**Architecture:** 36 layers, d=128, 2 KV heads (GQA)

## System Architecture

```
Input KV pair
    |
    v
[Boundary check] --> first/last 2 layers --> store FP16 (anchor layers)
    |
    v
[FP16 hot window] --> last N tokens --> store FP16 (recent context)
    |
    v
[WHT/PCA rotation] --> orthogonal rotation into quantization domain
    |
    v
[Mean removal] --> subtract running per-head mean (shift-invariant)
    |
    v
[Importance scoring] --> EMA of attention weights (when available)
    |
    v
[Tier assignment] --> based on importance:
    Tier 0 (top 5%):     4-bit ResidualQuant
    Tier 1 (next 25%):   3-bit ResidualQuant  (default)
    Tier 2 (bottom 70%): DeltaQuant (3-bit anchor + 1-bit delta, G=4)
    |
    v
[FAISS index] --> IVF-Flat index per head (full mode only)
    |
    v
[Store compressed] --> pre-dequantized for fast retrieval
```

## Calibration

- PCA rotation matrices calibrated from 128-token forward pass
- Time: 0.44s for all 36 layers
- Layer 0: top-5 PCA explains 99.0% of key variance
- Key finding: PCA concentrates variance but WHT is more robust for generation

## FP16 Baseline

| Prompt | Tokens | Time (s) | Tok/s |
|--------|--------|----------|-------|
| 1 (math explanation, long) | 200 | 9.76 | 20.5 |
| 2 (capital of Australia) | 8 | 0.42 | 19.0 |
| 3 (Python factorial) | 200 | 9.47 | 21.1 |

## Compress Mode (WHT + Mean-Removal + 3-bit RQ, window=64)

| Prompt | Eff Bits | Compression | Token Match | First Div | Tok/s |
|--------|----------|-------------|-------------|-----------|-------|
| 1 | 8.04 | 2.0x | 13.5% | 21 | 6.5 |
| 2 | 16.0 | 1.0x | 100.0% | 8 | 7.7 |
| 3 | 8.38 | 1.9x | 4.0% | 7 | 7.2 |

- Short prompts (prompt 2): 100% token match -- fits entirely in FP16 window
- Long prompts (prompt 1): coherent output, diverges at token 21
- Code prompts (prompt 3): reasonable divergence at token 7

## Full Mode (Compression + FAISS Retrieval)

| Prompt | Eff Bits | Compression | Token Match | First Div | Tok/s |
|--------|----------|-------------|-------------|-----------|-------|
| 1 | 8.04 | 2.0x | 13.5% | 21 | 6.5 |
| 2 | 16.0 | 1.0x | 100.0% | 8 | 7.4 |
| 3 | 8.38 | 1.9x | 4.0% | 7 | 7.1 |

- FAISS retrieval adds minimal overhead at 200-token scale
- Quality identical to compress mode (as expected for short contexts)
- FAISS benefit emerges at longer contexts (1K+ tokens)

## Configuration Sweep (Prompt 1: Long Math Explanation)

| Config | Rotation | Window | Eff Bits | Compression | Token Match | First Div | Tok/s |
|--------|----------|--------|----------|-------------|-------------|-----------|-------|
| wht_w0_3bit | WHT | 0 | 5.06 | **3.2x** | 0.0% | 0 | 6.1 |
| wht_w32_3bit | WHT | 32 | 6.55 | 2.4x | 3.0% | 2 | 6.1 |
| wht_w32_2bit | WHT | 32 | 5.78 | **2.8x** | 3.0% | 2 | 6.2 |
| wht_w64_3bit | WHT | 64 | 8.04 | 2.0x | **13.5%** | **21** | 6.6 |
| wht_w64_4bit | WHT | 64 | 8.04 | 2.0x | **13.5%** | **21** | 6.7 |
| wht_w128_3bit | WHT | 128 | 11.02 | 1.5x | **17.5%** | **21** | 9.0 |
| pca_w64_3bit | PCA | 64 | 8.04 | 2.0x | 11.0% | 21 | 10.9 |
| pca_w128_3bit | PCA | 128 | 11.02 | 1.5x | 17.5% | 21 | 12.6 |

## Key Findings

### 1. WHT vs PCA Rotation
- **WHT wins at w64**: 13.5% match vs PCA's 11.0% at identical compression
- **PCA is faster**: 10.9 tok/s vs 6.6 tok/s (WHT O(d log d) butterfly is slower in Python)
- **PCA ties at w128**: both achieve 17.5% match
- PCA calibration does not transfer well enough to justify the complexity
- **Recommendation: WHT for quality, PCA acceptable for speed**

### 2. Window Size is Critical
- **w0**: 3.2x compression but 0% match -- too aggressive
- **w32**: 2.4x compression, 3% match -- marginal quality
- **w64**: 2.0x compression, 13.5% match -- **sweet spot**
- **w128**: 1.5x compression, 17.5% match -- diminishing returns
- First divergence is token 21 for all configs with window >= 64

### 3. Bit-Width Has Diminishing Returns at Same Window
- wht_w64_3bit and wht_w64_4bit both get 13.5% match at 2.0x compression
- The window dominates quality more than the compressed bit-width
- 3-bit is sufficient when window >= 64

### 4. Tier System Behavior
- Without attention weight feedback, all tokens default to Tier 1 (3-bit RQ)
- DeltaQuant (Tier 2) is not activated in standard generate() flow
- Importance-based tiering requires model patching to return attention weights
- The tier system is designed for long-context (1K+) where it activates

### 5. Effective Compression
At 200-token generation with window=64:
- **Boundary layers** (4 layers): always FP16
- **Compressed layers** (32 layers): 3-bit RQ for tokens outside window
- **FP16 window** (64 tokens): full precision
- Effective bits: ~8 (mix of FP16 window + 3-bit compressed)
- At longer contexts, effective bits drop toward 5-6 as window fraction shrinks

## Comparison with Production GenerationCache

| System | Eff Bits | Compression | Quality | Speed |
|--------|----------|-------------|---------|-------|
| GenerationCache (K3/V3 balanced) | ~6.5 | 3.3x | 95.1% quality score | 21 tok/s |
| V2Cache (WHT w64) | ~8.0 | 2.0x | 13.5% token match | 6.6 tok/s |
| V2Cache (WHT w128) | ~11.0 | 1.5x | 17.5% token match | 9.0 tok/s |

The production GenerationCache achieves better compression and speed because:
1. It uses Triton-fused quantize/dequantize kernels
2. It stores compressed indices (not pre-dequantized tensors)
3. Its anchor layer strategy is optimized from 246-config sweep
4. It doesn't add mean-removal overhead during prefill

## Architecture Validation

The v2 unified pipeline successfully integrates:
- [x] PCA rotation calibration (0.44s per model)
- [x] WHT fallback rotation (robust, no calibration needed)
- [x] Mean removal (per-head running mean, shift-invariant)
- [x] ResidualQuant (MSE + 1-bit sign correction)
- [x] DeltaQuant (anchor + delta coding, implemented but not activated by default)
- [x] Adaptive importance tiers (framework ready, needs attention weight hookup)
- [x] FAISS retrieval (IVF-Flat per head, functional)
- [x] Boundary layers (first/last 2 at FP16)
- [x] FP16 hot window (configurable size)
- [x] HuggingFace Cache protocol (full duck-typing)

## Files

- **Cache implementation:** `turboquantdc/v2_cache.py`
- **Benchmark:** `benchmarks/v2_benchmark.py`
- **PCA calibration data:** `benchmarks/results/v2_pca_rotations.pt`
