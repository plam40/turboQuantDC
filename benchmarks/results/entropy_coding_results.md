# Entropy Coding Analysis Results

**Model:** Qwen/Qwen2.5-3B-Instruct
**Context length:** 2057 tokens
**Layers:** 36 | **KV heads:** 2 | **head_dim:** 128
**Date:** 2026-04-04 12:22

## Summary: Free Compression Available

| Bits | Allocated | Theory H | Empirical H | Savings | ANS bps | Zlib bps | LZMA bps |
|------|-----------|----------|-------------|---------|---------|----------|----------|
| 2 | 2.0 b | 1.911 b | 1.920 b | **4.0%** | 1.919 | 1.986 | 1.460 |
| 3 | 3.0 b | 2.825 b | 2.829 b | **5.7%** | 2.831 | 3.047 | 2.464 |
| 4 | 4.0 b | 3.765 b | 3.761 b | **6.0%** | 3.767 | 4.123 | 3.558 |

## Interpretation

- **Best entropy savings: 6.0%** (lossless, zero quality loss)
- **2-bit:** entropy = 1.920 bits (vs 2.0 allocated) = **4.0% free compression**
- **3-bit:** entropy = 2.829 bits (vs 3.0 allocated) = **5.7% free compression**
- **4-bit:** entropy = 3.761 bits (vs 4.0 allocated) = **6.0% free compression**

## Effective Compression Ratios (with entropy coding)

Baseline TurboQuant compression = 16 / (b + overhead). With entropy coding, the effective bits are lower.

| Bits | Without EC | With ANS | With LZMA | Improvement |
|------|-----------|----------|-----------|-------------|
| 2 | 8.00x | 8.34x | 10.96x | +4.2% |
| 3 | 5.33x | 5.65x | 6.49x | +6.0% |
| 4 | 4.00x | 4.25x | 4.50x | +6.2% |

## WHT vs QR Rotation

| Bits | WHT Entropy | QR Entropy | Difference |
|------|-------------|------------|------------|
| 2 | 1.9045 | 1.9084 | +0.0038 |
| 3 | 2.8276 | 2.8144 | -0.0132 |
| 4 | 3.7464 | 3.7424 | -0.0040 |

## Sequential Correlations

- **2-bit:** correlation gain = 0.0320 bits (1.60% of allocated) -> negligible
- **3-bit:** correlation gain = 0.0667 bits (2.22% of allocated) -> EXPLOITABLE
- **4-bit:** correlation gain = 0.1265 bits (3.16% of allocated) -> EXPLOITABLE

## Run-Length Analysis

- **2-bit:** avg run = 1.35 (expected random: 1.52) -> RLE not useful
- **3-bit:** avg run = 1.15 (expected random: 1.23) -> RLE not useful
- **4-bit:** avg run = 1.07 (expected random: 1.12) -> RLE not useful

## Per-Coordinate Entropy

| Bits | Mean | Std | Min | Max | Range |
|------|------|-----|-----|-----|-------|
| 2 | 0.1403 | 0.2531 | -0.0000 | 0.9915 | 0.9915 |
| 3 | 0.4793 | 0.3346 | -0.0000 | 0.9998 | 0.9998 |
| 4 | 0.9316 | 0.2046 | 0.3120 | 1.1767 | 0.8647 |

## Per-Layer Entropy Detail

### 2-bit

| Layer | Avg Entropy | Savings |
|-------|-------------|--------|
| 0 | 1.8842 b | 5.79% |
| 1 | 1.8976 b | 5.12% |
| 2 | 1.9182 b | 4.09% |
| 3 | 1.9243 b | 3.78% |
| 4 | 1.9147 b | 4.26% |
| 5 | 1.9231 b | 3.84% |
| 6 | 1.9182 b | 4.09% |
| 7 | 1.9156 b | 4.22% |
| 8 | 1.9236 b | 3.82% |
| 9 | 1.9251 b | 3.75% |
| 10 | 1.9180 b | 4.10% |
| 11 | 1.9268 b | 3.66% |
| 12 | 1.9238 b | 3.81% |
| 13 | 1.9269 b | 3.66% |
| 14 | 1.9284 b | 3.58% |
| 15 | 1.9174 b | 4.13% |
| 16 | 1.9190 b | 4.05% |
| 17 | 1.9153 b | 4.23% |
| 18 | 1.9316 b | 3.42% |
| 19 | 1.9108 b | 4.46% |
| 20 | 1.9383 b | 3.09% |
| 21 | 1.9220 b | 3.90% |
| 22 | 1.9149 b | 4.25% |
| 23 | 1.9192 b | 4.04% |
| 24 | 1.9117 b | 4.41% |
| 25 | 1.9144 b | 4.28% |
| 26 | 1.9236 b | 3.82% |
| 27 | 1.9331 b | 3.35% |
| 28 | 1.9245 b | 3.77% |
| 29 | 1.9313 b | 3.43% |
| 30 | 1.9150 b | 4.25% |
| 31 | 1.9242 b | 3.79% |
| 32 | 1.9221 b | 3.89% |
| 33 | 1.9119 b | 4.40% |
| 34 | 1.9207 b | 3.96% |
| 35 | 1.9225 b | 3.88% |

### 3-bit

| Layer | Avg Entropy | Savings |
|-------|-------------|--------|
| 0 | 2.8112 b | 6.29% |
| 1 | 2.8194 b | 6.02% |
| 2 | 2.8284 b | 5.72% |
| 3 | 2.8298 b | 5.67% |
| 4 | 2.8269 b | 5.77% |
| 5 | 2.8295 b | 5.68% |
| 6 | 2.8288 b | 5.71% |
| 7 | 2.8303 b | 5.66% |
| 8 | 2.8309 b | 5.64% |
| 9 | 2.8311 b | 5.63% |
| 10 | 2.8276 b | 5.75% |
| 11 | 2.8306 b | 5.65% |
| 12 | 2.8338 b | 5.54% |
| 13 | 2.8317 b | 5.61% |
| 14 | 2.8330 b | 5.57% |
| 15 | 2.8279 b | 5.74% |
| 16 | 2.8309 b | 5.64% |
| 17 | 2.8293 b | 5.69% |
| 18 | 2.8280 b | 5.73% |
| 19 | 2.8208 b | 5.97% |
| 20 | 2.8235 b | 5.88% |
| 21 | 2.8306 b | 5.65% |
| 22 | 2.8276 b | 5.75% |
| 23 | 2.8298 b | 5.67% |
| 24 | 2.8280 b | 5.73% |
| 25 | 2.8260 b | 5.80% |
| 26 | 2.8318 b | 5.61% |
| 27 | 2.8222 b | 5.93% |
| 28 | 2.8310 b | 5.63% |
| 29 | 2.8328 b | 5.57% |
| 30 | 2.8285 b | 5.72% |
| 31 | 2.8311 b | 5.63% |
| 32 | 2.8326 b | 5.58% |
| 33 | 2.8278 b | 5.74% |
| 34 | 2.8321 b | 5.60% |
| 35 | 2.8342 b | 5.53% |

### 4-bit

| Layer | Avg Entropy | Savings |
|-------|-------------|--------|
| 0 | 3.6519 b | 8.70% |
| 1 | 3.7624 b | 5.94% |
| 2 | 3.7622 b | 5.94% |
| 3 | 3.7643 b | 5.89% |
| 4 | 3.7660 b | 5.85% |
| 5 | 3.7648 b | 5.88% |
| 6 | 3.7654 b | 5.87% |
| 7 | 3.7655 b | 5.86% |
| 8 | 3.7654 b | 5.87% |
| 9 | 3.7656 b | 5.86% |
| 10 | 3.7654 b | 5.87% |
| 11 | 3.7643 b | 5.89% |
| 12 | 3.7656 b | 5.86% |
| 13 | 3.7645 b | 5.89% |
| 14 | 3.7643 b | 5.89% |
| 15 | 3.7650 b | 5.88% |
| 16 | 3.7657 b | 5.86% |
| 17 | 3.7665 b | 5.84% |
| 18 | 3.7541 b | 6.15% |
| 19 | 3.7626 b | 5.94% |
| 20 | 3.7475 b | 6.31% |
| 21 | 3.7663 b | 5.84% |
| 22 | 3.7656 b | 5.86% |
| 23 | 3.7658 b | 5.85% |
| 24 | 3.7660 b | 5.85% |
| 25 | 3.7650 b | 5.87% |
| 26 | 3.7656 b | 5.86% |
| 27 | 3.7467 b | 6.33% |
| 28 | 3.7651 b | 5.87% |
| 29 | 3.7602 b | 6.00% |
| 30 | 3.7666 b | 5.84% |
| 31 | 3.7655 b | 5.86% |
| 32 | 3.7657 b | 5.86% |
| 33 | 3.7668 b | 5.83% |
| 34 | 3.7656 b | 5.86% |
| 35 | 3.7638 b | 5.91% |


## Actual Compressed Sizes (per layer, head 0)

| Bits | Raw (packed) | ANS | Zlib | LZMA | ANS/packed | Zlib/packed | LZMA/packed |
|------|-------------|-----|------|------|-----------|------------|------------|
| 2 | 2,369,664 B | 2,273,907 B | 2,353,353 B | 1,730,044 B | 1.042x | 1.007x | 1.370x |
| 3 | 3,554,496 B | 3,354,397 B | 3,610,566 B | 2,919,808 B | 1.060x | 0.984x | 1.217x |
| 4 | 4,739,328 B | 4,463,628 B | 4,885,080 B | 4,215,184 B | 1.062x | 0.970x | 1.124x |

## Recommendation

The 3-bit sweet spot shows **5.7% free compression** from entropy coding alone. This is lossless — identical quality, fewer bits.

**RECOMMEND:** 5-10% free compression is meaningful at scale. Implement entropy coding for memory-constrained deployments. Use zlib as fast backend, ANS for maximum compression.

Sequential correlations provide an additional 0.067 bits/symbol opportunity. Consider context-adaptive coding (FSE with context) for a second-order gain.
