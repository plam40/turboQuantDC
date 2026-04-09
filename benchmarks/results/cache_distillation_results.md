# Cache Distillation + TurboQuant Compression Results

**Date:** 2026-04-09 16:42
**Model:** Qwen/Qwen2.5-3B-Instruct
**Sequence length:** 1086 tokens
**Head dim:** 128, KV heads: 2, Q heads: 16
**TurboQuant:** K3/V3
**Distillation:** 50 steps, lr=0.01

## Distillation Quality

| Layer | Ratio | N->M | Attn Cosine | Top-5 | Distill x | Quant x | Total x | Post-Quant Cos |
|-------|-------|------|-------------|-------|-----------|---------|---------|----------------|
|     0 |     4x | 1086->271 | 0.9996 | 0.981 | 4.0x | 4.9x | 19.7x | 0.8113 |
|     0 |     8x | 1086->135 | 0.9979 | 0.954 | 8.0x | 4.9x | 39.6x | 0.7804 |
|     0 |    16x | 1086->67 | 0.9950 | 0.935 | 16.2x | 4.9x | 79.8x | 0.7230 |
|     7 |     4x | 1086->271 | 0.9984 | 0.954 | 4.0x | 4.9x | 19.7x | 0.7165 |
|     7 |     8x | 1086->135 | 0.9964 | 0.927 | 8.0x | 4.9x | 39.6x | 0.7124 |
|     7 |    16x | 1086->67 | 0.9934 | 0.905 | 16.2x | 4.9x | 79.8x | 0.4419 |
|    15 |     4x | 1086->271 | 0.9992 | 0.971 | 4.0x | 4.9x | 19.7x | 0.8360 |
|    15 |     8x | 1086->135 | 0.9975 | 0.950 | 8.0x | 4.9x | 39.6x | 0.8600 |
|    15 |    16x | 1086->67 | 0.9949 | 0.927 | 16.2x | 4.9x | 79.8x | 0.8562 |
|    23 |     4x | 1086->271 | 0.9984 | 0.959 | 4.0x | 4.9x | 19.7x | 0.8459 |
|    23 |     8x | 1086->135 | 0.9943 | 0.925 | 8.0x | 4.9x | 39.6x | 0.8471 |
|    23 |    16x | 1086->67 | 0.9865 | 0.898 | 16.2x | 4.9x | 79.8x | 0.8038 |
|    35 |     4x | 1086->271 | 0.9993 | 0.950 | 4.0x | 4.9x | 19.7x | 0.9372 |
|    35 |     8x | 1086->135 | 0.9981 | 0.925 | 8.0x | 4.9x | 39.6x | 0.8933 |
|    35 |    16x | 1086->67 | 0.9956 | 0.892 | 16.2x | 4.9x | 79.8x | 0.9248 |

## Baseline: 3-bit TurboQuant Only (5.0x)

| Layer | Cosine | Key Distortion |
|-------|--------|----------------|
|     0 | 0.7227 | 0.072442 |
|     7 | 0.8843 | 0.064831 |
|    15 | 0.9018 | 0.062849 |
|    23 | 0.9372 | 0.072522 |
|    35 | 0.9245 | 0.045865 |

## Compressibility: Distilled vs Real Tokens

Key hypothesis: distilled tokens live in a lower-dimensional subspace
and should be MORE compressible (lower entropy, lower effective rank).

| Layer | Ratio | Real Entropy | Distilled Entropy | Real EffRank | Distilled EffRank | Real VarRatio | Distilled VarRatio |
|-------|-------|-------------|-------------------|-------------|-------------------|--------------|-------------------|
|     0 |     4x | 7.08 | 7.07 | 19.8 | 15.2 | 1.5 | 1.9 |
|     0 |     8x | 7.08 | 7.06 | 19.8 | 11.3 | 1.5 | 2.1 |
|     0 |    16x | 7.08 | 7.05 | 19.8 | 6.5 | 1.5 | 3.3 |
|     7 |     4x | 6.84 | 7.05 | 99.6 | 85.4 | 1.9 | 2.3 |
|     7 |     8x | 6.84 | 7.01 | 99.6 | 69.8 | 1.9 | 3.2 |
|     7 |    16x | 6.84 | 7.22 | 99.6 | 40.3 | 1.9 | 3.8 |
|    15 |     4x | 7.05 | 7.16 | 90.4 | 78.2 | 2.2 | 2.7 |
|    15 |     8x | 7.05 | 7.31 | 90.4 | 64.2 | 2.2 | 2.8 |
|    15 |    16x | 7.05 | 7.25 | 90.4 | 39.2 | 2.2 | 3.8 |
|    23 |     4x | 7.02 | 7.14 | 99.4 | 87.5 | 2.5 | 2.8 |
|    23 |     8x | 7.02 | 7.24 | 99.4 | 73.8 | 2.5 | 3.1 |
|    23 |    16x | 7.02 | 7.30 | 99.4 | 44.7 | 2.5 | 4.4 |
|    35 |     4x | 7.26 | 7.30 | 90.8 | 78.8 | 2.0 | 2.3 |
|    35 |     8x | 7.26 | 7.32 | 90.8 | 65.0 | 2.0 | 2.7 |
|    35 |    16x | 7.26 | 7.45 | 90.8 | 36.9 | 2.0 | 3.8 |

## Aggregated Results (Mean Across Layers)

### 4x Distillation
- **Attention cosine:** 0.9990
- **Top-5 match:** 0.963
- **Post-quant cosine:** 0.8294
- **Total compression:** 19.7x
- **Distillation time:** 0.53s per head
- **Entropy (real/distilled):** 7.05 / 7.14
- **Effective rank (real/distilled):** 80.0 / 69.0

### 8x Distillation
- **Attention cosine:** 0.9968
- **Top-5 match:** 0.936
- **Post-quant cosine:** 0.8186
- **Total compression:** 39.6x
- **Distillation time:** 0.25s per head
- **Entropy (real/distilled):** 7.05 / 7.19
- **Effective rank (real/distilled):** 80.0 / 56.8

### 16x Distillation
- **Attention cosine:** 0.9931
- **Top-5 match:** 0.912
- **Post-quant cosine:** 0.7499
- **Total compression:** 79.8x
- **Distillation time:** 0.15s per head
- **Entropy (real/distilled):** 7.05 / 7.25
- **Effective rank (real/distilled):** 80.0 / 33.5

### Baseline (3-bit TurboQuant only)
- **Cosine:** 0.8741
- **Compression:** ~5.0x

## Key Findings

1. **Distilled tokens ARE more compressible** than real tokens (effective rank: 69.0 vs 80.0)
2. **4x distillation quality:** 0.9990 attention cosine
3. **Maximum compression:** 80x total at 0.7499 post-quant cosine
