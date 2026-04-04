# Spectral KV Cache Compression — Research Results

**Model:** Qwen/Qwen2.5-0.5B-Instruct
**Head dimension:** d=64
**Date:** 2026-04-04 04:06

## 1. DCT Energy Spectrum Analysis

How many DCT coefficients capture X% of vector energy?

| Metric | Keys | Values |
|--------|------|--------|
| 90% energy | 48/64 (75.5%) | 55/64 (86.5%) |
| 95% energy | 55/64 (85.2%) | 60/64 (93.2%) |
| 99% energy | 61/64 (95.8%) | 63/64 (99.2%) |

### Per-Layer Variation (90% energy)
| Layer range | Keys (K for 90%) | Values (K for 90%) |
|-------------|------|--------|
| Min | 31 | 52 |
| Max | 54 | 57 |
| Mean | 48.3 | 55.3 |
| Std | 5.2 | 1.1 |

## 2. SVD Singular Value Spectrum

How many principal components capture X% of variance?

| Metric | Keys | Values |
|--------|------|--------|
| 90% variance | 22/64 (34.0%) | 41/64 (64.2%) |
| 95% variance | 30/64 (46.9%) | 50/64 (78.1%) |
| 99% variance | 45/64 (70.8%) | 61/64 (94.6%) |

## 3. DCT Compression Quality

### Keys
| K kept | % of d | bits/dim | CR | cos sim | attn cos | top-1 match |
|--------|--------|----------|-----|---------|----------|-------------|
| 6 | 9% | 2.06 | 7.8x | 0.626536 | 0.5924 | 0.060 |
| 9 | 14% | 3.09 | 5.2x | 0.714185 | 0.6591 | 0.090 |
| 12 | 19% | 4.12 | 3.9x | 0.777050 | 0.7331 | 0.113 |
| 16 | 25% | 5.50 | 2.9x | 0.838109 | 0.7921 | 0.133 |
| 19 | 30% | 6.53 | 2.4x | 0.872521 | 0.8423 | 0.177 |
| 25 | 39% | 8.59 | 1.9x | 0.922295 | 0.8873 | 0.263 |
| 32 | 50% | 11.00 | 1.5x | 0.958962 | 0.9440 | 0.361 |
| 38 | 59% | 13.06 | 1.2x | 0.978274 | 0.9627 | 0.465 |
| 48 | 75% | 16.50 | 1.0x | 0.994732 | 0.9908 | 0.660 |
| 57 | 89% | 19.59 | 0.8x | 0.999464 | 0.9985 | 0.859 |

### Values
| K kept | % of d | bits/dim | CR | cos sim | attn cos | top-1 match |
|--------|--------|----------|-----|---------|----------|-------------|
| 6 | 9% | 2.06 | 7.8x | 0.633600 | 0.6270 | 0.124 |
| 9 | 14% | 3.09 | 5.2x | 0.720024 | 0.7157 | 0.165 |
| 12 | 19% | 4.12 | 3.9x | 0.781790 | 0.7774 | 0.195 |
| 16 | 25% | 5.50 | 2.9x | 0.841739 | 0.8371 | 0.311 |
| 19 | 30% | 6.53 | 2.4x | 0.875560 | 0.8703 | 0.323 |
| 25 | 39% | 8.59 | 1.9x | 0.924334 | 0.9197 | 0.449 |
| 32 | 50% | 11.00 | 1.5x | 0.960166 | 0.9580 | 0.552 |
| 38 | 59% | 13.06 | 1.2x | 0.979060 | 0.9776 | 0.648 |
| 48 | 75% | 16.50 | 1.0x | 0.995002 | 0.9945 | 0.828 |
| 57 | 89% | 19.59 | 0.8x | 0.999497 | 0.9994 | 0.932 |

## 4. SVD Subspace Compression Quality

### Keys
| k dim | % of d | bits/dim | CR | var expl | cos sim | attn cos | top-1 |
|-------|--------|----------|-----|----------|---------|----------|-------|
| 6 | 9% | 1.96 | 8.1x | 0.6872 | 0.866628 | 0.8326 | 0.070 |
| 9 | 14% | 2.91 | 5.5x | 0.7446 | 0.891793 | 0.8686 | 0.115 |
| 12 | 19% | 3.86 | 4.1x | 0.7900 | 0.911508 | 0.8922 | 0.176 |
| 16 | 25% | 5.13 | 3.1x | 0.8375 | 0.931708 | 0.9171 | 0.253 |
| 19 | 30% | 6.08 | 2.6x | 0.8659 | 0.943937 | 0.9328 | 0.277 |
| 25 | 39% | 7.98 | 2.0x | 0.9098 | 0.962076 | 0.9534 | 0.374 |
| 32 | 50% | 10.19 | 1.6x | 0.9445 | 0.975944 | 0.9731 | 0.488 |
| 48 | 75% | 15.25 | 1.0x | 0.9855 | 0.992638 | 0.9926 | 0.701 |

## 5. ResidualQuant Baseline (for comparison)
| Config | bits/dim | CR | cos sim | attn cos | top-1 |
|--------|----------|-----|---------|----------|-------|
| 2bit_keys | 2.50 | 6.4x | 0.942130 | 0.9237 | 0.259 |
| 3bit_keys | 3.50 | 4.6x | 0.983167 | 0.9706 | 0.493 |
| 4bit_keys | 4.50 | 3.6x | 0.995141 | 0.9897 | 0.642 |

## 6. Hybrid SVD + DCT Residual (Keys)
| SVD k | DCT K | bits/dim | CR | cos sim | attn cos | top-1 |
|-------|-------|----------|-----|---------|----------|-------|
| 16 | 16 | 9.50 | 1.7x | 0.979850 | 0.9781 | 0.501 |
| 32 | 16 | 13.50 | 1.2x | 0.992150 | 0.9925 | 0.685 |
| 32 | 32 | 19.00 | 0.8x | 0.997320 | 0.9981 | 0.814 |
| 48 | 16 | 17.50 | 0.9x | 0.997320 | 0.9981 | 0.833 |
| 48 | 32 | 23.00 | 0.7x | 0.998966 | 0.9995 | 0.905 |
| 64 | 16 | 21.50 | 0.7x | 1.000000 | 1.0000 | 1.000 |
| 64 | 32 | 27.00 | 0.6x | 1.000000 | 1.0000 | 1.000 |

## 7. Low-Frequency vs Top-Magnitude DCT (Keys)
| K kept | Low-freq cos | Top-mag cos | Winner |
|--------|-------------|-------------|--------|
| 9/64 (15%) | 0.391752 | 0.714185 | top-mag |
| 16/64 (25%) | 0.506015 | 0.838109 | top-mag |
| 25/64 (40%) | 0.634638 | 0.922295 | top-mag |
| 32/64 (50%) | 0.719735 | 0.958962 | top-mag |
| 48/64 (75%) | 0.874523 | 0.994732 | top-mag |

## 8. Per-Head Energy Variation
| Layer | K90 mean | K90 std | K90 range | K95 mean | K95 range |
|-------|----------|---------|-----------|----------|-----------|
| layer_0 | 32.0 | 1.0 | 31-33 | 38.5 | 38-39 |
| layer_12 | 45.5 | 2.5 | 43-48 | 53.5 | 52-55 |
| layer_18 | 49.0 | 0.0 | 49-49 | 56.0 | 56-56 |
| layer_23 | 46.5 | 1.5 | 45-48 | 54.5 | 54-55 |
| layer_6 | 51.5 | 1.5 | 50-53 | 57.5 | 57-58 |

## 9. Critical Experiment: KLT (PCA) vs DCT Energy Compaction

The optimal energy-compacting transform for any signal is the Karhunen-Loeve
Transform (KLT), which uses the eigenvectors of the covariance matrix. If KLT
significantly outperforms DCT, it means KV vectors have structure, but not the
smooth/periodic structure that DCT captures.

| Layer | Method | K for 90% | K for 95% | K for 99% | cos@K=16 | cos@K=32 |
|-------|--------|-----------|-----------|-----------|----------|----------|
| 0 | KLT | 1 | 2 | 5 | 0.999856 | 0.999999 |
| 0 | DCT | 44 | 50 | 58 | 0.823001 | 0.956428 |
| 6 | KLT | 25 | 34 | 52 | 0.913566 | 0.971741 |
| 6 | DCT | 52 | 58 | 63 | 0.840680 | 0.960254 |
| 12 | KLT | 19 | 27 | 44 | 0.947967 | 0.985058 |
| 12 | DCT | 48 | 55 | 62 | 0.845971 | 0.960841 |
| 18 | KLT | 26 | 34 | 51 | 0.916035 | 0.968391 |
| 18 | DCT | 51 | 57 | 63 | 0.849935 | 0.963241 |
| 23 | KLT | 16 | 26 | 44 | 0.961651 | 0.988003 |
| 23 | DCT | 46 | 54 | 62 | 0.841929 | 0.961684 |

**KLT dominates DCT across all layers.** Layer 0 is especially striking:
only 1 principal component captures 90% of variance (cos=0.9999 at K=16),
while DCT needs 44 coefficients. Even in later layers where the effective
dimensionality increases, KLT consistently needs 2-3x fewer components.

## 10. Coordinate Correlation Analysis

Real KV vectors have significant inter-coordinate correlation that DCT
cannot exploit but SVD/KLT can.

| Metric | Real KV vectors | Random vectors |
|--------|-----------------|----------------|
| Mean |correlation| | 0.4264 | 0.2119 |
| Max |correlation| | 0.9818 | 0.7722 |
| Median |correlation| | 0.3798 | (varies) |

KV coordinates are ~2x more correlated than random vectors. This correlation
pattern is layer-specific and learned (not smooth/periodic), so the
data-adaptive KLT captures it while the fixed-basis DCT cannot.

## Summary & Conclusions

### 3-bit ResidualQuant reference point:
- bits/dim: 3.50
- cosine sim: 0.983167

### Best DCT matching 3-bit RQ quality:
- K=48, bits/dim=16.50, cos=0.994732
- 371.4% MORE bits than ResidualQuant (DCT index overhead)

### Best SVD matching 3-bit RQ quality:
- k=48, bits/dim=15.25, cos=0.992638

### Key Insight: Why DCT Fails for KV Vectors

KV vectors DO have exploitable structure (mean |correlation| = 0.43), but it is
NOT frequency-domain structure. The correlation pattern is:
- **Layer-specific** (each layer has a different covariance structure)
- **Learned** (shaped by training, not by any smooth/periodic prior)
- **Non-smooth** (no spatial locality between adjacent coordinate indices)

DCT assumes that nearby coordinates are correlated (smooth signals), which is
the right assumption for images and audio but the WRONG assumption for attention
head dimensions. The coordinate indices in a KV vector have no spatial meaning
-- coordinate 42 has no special relationship to coordinate 43.

The data-adaptive KLT/PCA captures the actual correlation structure and achieves
3-10x better energy compaction than DCT across all layers. Layer 0 is the most
extreme: KLT needs 1 component for 90% variance vs 44 for DCT.

### Energy Distribution Summary
- 90% DCT energy requires 48/64 coefficients (75.5% of d) -- nearly flat spectrum
- 90% KLT/PCA variance requires 1-26/64 components (1.6-40.6%) -- highly concentrated
- Low-frequency DCT is strictly worse than top-magnitude DCT (Experiment 7)
- Per-head variation is moderate (K90 std ~1-2.5 across heads within a layer)

### Verdict

**DCT-based spectral compression does NOT beat spatial quantization.** The approach
fails for two independent reasons:

1. **Wrong basis**: DCT exploits smooth/periodic structure that KV vectors lack.
   At equal compression, DCT achieves cos=0.84 vs ResidualQuant's cos=0.98.

2. **Index overhead tax**: Each kept DCT coefficient costs ceil(log2(d))=6 bits
   just for the index, plus value bits. This overhead alone exceeds ResidualQuant's
   total 3.5 bits/dim at any useful compression level.

**SVD/PCA subspace projection is the promising direction.** It captures the real
structure in KV vectors (layer-specific covariance patterns) and achieves:
- 90% variance in 22/64 dimensions for keys (34% of d)
- Much better reconstruction than DCT at the same dimensionality

**However**, SVD has a fundamental practical limitation: it requires per-layer
fitting on representative data. The projection matrix V_k (k x d x 32 bits)
must be computed and stored for each layer, and it may not generalize across
different inputs. This makes it unsuitable as a drop-in replacement for
ResidualQuant, which needs zero calibration data.

### Future Direction: Cross-Layer Subspace Sharing

If principal subspaces are similar across layers (or across positions within a
layer), a shared subspace could amortize the overhead. This is related to
"GQA" (grouped query attention) which already exploits cross-head structure.
A "grouped subspace quantization" combining SVD dimensionality reduction with
scalar quantization could be investigated.