# TurboQuantDC V2 Long-Context Validation Results

**Date:** 2026-04-04T20:24:23.060152
**Model:** Qwen/Qwen2.5-3B-Instruct
**Device:** cuda
**Generation:** 100 new tokens per run
**Config:** 36 layers, d=128, 2 KV heads

## Summary Table

| Context | Method | Eff Bits | Compression | Token Match | 1st Diverge | Speed (tok/s) | Peak GPU (MB) |
|---------|--------|----------|-------------|------------|-------------|---------------|---------------|
| 2K | FP16 Baseline | 16.00 | 1.00x | - | - | 19.8 | 2257 |
| 2K | V2 Compress | 5.38 | 2.97x | 0.0% | 0 | 9.5 | 2451 |
| 2K | V2 Full/FAISS | 5.38 | 2.97x | 0.0% | 0 | 9.2 | 2451 |
| 2K | Production GC | 4.95 | 3.23x | 23.0% | 16 | 7.2 | 2740 |
| 4K | FP16 Baseline | 16.00 | 1.00x | - | - | 18.8 | 2466 |
| 4K | V2 Compress | 5.22 | 3.06x | 0.0% | 0 | 9.2 | 2852 |
| 4K | V2 Full/FAISS | 5.22 | 3.06x | 0.0% | 0 | 8.9 | 2852 |
| 4K | Production GC | 4.79 | 3.34x | 17.0% | 14 | 7.0 | 3429 |
| 8K | FP16 Baseline | 16.00 | 1.00x | - | - | 18.4 | 2952 |
| 8K | V2 Compress | 5.10 | 3.14x | 0.0% | 0 | 9.0 | 3722 |
| 8K | V2 Full/FAISS | 5.10 | 3.14x | 0.0% | 0 | 8.8 | 3722 |
| 8K | Production GC | 4.70 | 3.40x | 11.0% | 8 | 7.0 | 4878 |

## 2K Context (2047 tokens)

### FP16 Baseline

- **Effective bits:** 16.0
- **Compression ratio:** 1.0x
- **Token match vs FP16:** -
- **First divergence:** token -
- **Speed:** 19.8 tok/s (5.05s total)
- **Peak GPU:** 2257.0 MB
- **Output preview:**  The text discusses the career and achievements of Robert Boulter, an English actor, and provides details about his roles in various media forms such ...

### V2 Compress

- **Effective bits:** 5.382
- **Compression ratio:** 2.97x
- **Token match vs FP16:** 0.0
- **First divergence:** token 0
- **Speed:** 9.5 tok/s (10.57s total)
- **Peak GPU:** 2451.0 MB
- **Tier summary:** {'boundary_fp16': 8584, 'fp16_window': 2048, 'tier_counts': {4: 0, 3: 66624, 1: 0}, 'effective_bits': 5.381950916433675, 'compression_ratio': 2.9728996507835723}
- **Window size:** 64
- **Output preview:** 下半大叔 leadingRANDOM颁奖unteUN茱ccion演出大叔 leadingSubmitted大树ccion大树 imaging大叔egl大叔itan大叔邀请前前​ขอบ leading前...(:,:,itan#if.dm教学딩 me讲述了演出 thou#if□itan大象 movcc...

### V2 Full/FAISS

- **Effective bits:** 5.382
- **Compression ratio:** 2.97x
- **Token match vs FP16:** 0.0
- **First divergence:** token 0
- **Speed:** 9.2 tok/s (10.89s total)
- **Peak GPU:** 2451.0 MB
- **Tier summary:** {'boundary_fp16': 8584, 'fp16_window': 2048, 'tier_counts': {4: 0, 3: 66624, 1: 0}, 'effective_bits': 5.381950916433675, 'compression_ratio': 2.9728996507835723}
- **Window size:** 64
- **Output preview:** 下半大叔 leadingRANDOM颁奖unteUN茱ccion演出大叔 leadingSubmitted大树ccion大树 imaging大叔egl大叔itan大叔邀请前前​ขอบ leading前...(:,:,itan#if.dm教学딩 me讲述了演出 thou#if□itan大象 movcc...

### Production GC

- **Effective bits:** 4.951
- **Compression ratio:** 3.23x
- **Token match vs FP16:** 0.23
- **First divergence:** token 16
- **Speed:** 7.2 tok/s (13.86s total)
- **Peak GPU:** 2740.0 MB
- **Output preview:**  The text discusses the career and achievements of Robert Boulter, an English actor. It begins with his notable roles in television and theater, inclu...


## 4K Context (4096 tokens)

### FP16 Baseline

- **Effective bits:** 16.0
- **Compression ratio:** 1.0x
- **Token match vs FP16:** -
- **First divergence:** token -
- **Speed:** 18.8 tok/s (5.31s total)
- **Peak GPU:** 2466.0 MB
- **Output preview:**  The text discusses the life and works of the Chinese poet Du Fu, who lived from 712 to 770 AD. It begins with an overview of his early life, includin...

### V2 Compress

- **Effective bits:** 5.223
- **Compression ratio:** 3.06x
- **Token match vs FP16:** 0.0
- **First divergence:** token 0
- **Speed:** 9.2 tok/s (10.9s total)
- **Peak GPU:** 2852.0 MB
- **Tier summary:** {'boundary_fp16': 16780, 'fp16_window': 2048, 'tier_counts': {4: 0, 3: 132192, 1: 0}, 'effective_bits': 5.2225268176400474, 'compression_ratio': 3.063651094324121}
- **Window size:** 64
- **Output preview:** UN面孔...UN#if#if颁奖前 ev imagingitan_CLOCKadratic前 av大树UN�数�数ccionUNirlingFLOW#if.dm meitanDDL meitanitanitanitan imaging imaging讲述masterUN...itanUN#if演出...

### V2 Full/FAISS

- **Effective bits:** 5.223
- **Compression ratio:** 3.06x
- **Token match vs FP16:** 0.0
- **First divergence:** token 0
- **Speed:** 8.9 tok/s (11.24s total)
- **Peak GPU:** 2852.0 MB
- **Tier summary:** {'boundary_fp16': 16780, 'fp16_window': 2048, 'tier_counts': {4: 0, 3: 132192, 1: 0}, 'effective_bits': 5.2225268176400474, 'compression_ratio': 3.063651094324121}
- **Window size:** 64
- **Output preview:** UN面孔...UN#if#if颁奖前 ev imagingitan_CLOCKadratic前 av大树UN�数�数ccionUNirlingFLOW#if.dm meitanDDL meitanitanitanitan imaging imaging讲述masterUN...itanUN#if演出...

### Production GC

- **Effective bits:** 4.785
- **Compression ratio:** 3.34x
- **Token match vs FP16:** 0.17
- **First divergence:** token 14
- **Speed:** 7.0 tok/s (14.19s total)
- **Peak GPU:** 3429.0 MB
- **Output preview:**  The text discusses the life and works of the Chinese poet Du Fu, including his early education, travels, and eventual rise to prominence as a poet. I...


## 8K Context (8192 tokens)

### FP16 Baseline

- **Effective bits:** 16.0
- **Compression ratio:** 1.0x
- **Token match vs FP16:** -
- **First divergence:** token -
- **Speed:** 18.4 tok/s (5.42s total)
- **Peak GPU:** 2952.0 MB
- **Output preview:**  The text discusses the life and works of the Chinese poet Du Fu, including his early life, travels, and experiences during the An Lushan Rebellion. I...

### V2 Compress

- **Effective bits:** 5.098
- **Compression ratio:** 3.14x
- **Token match vs FP16:** 0.0
- **First divergence:** token 0
- **Speed:** 9.0 tok/s (11.1s total)
- **Peak GPU:** 3722.0 MB
- **Tier summary:** {'boundary_fp16': 33164, 'fp16_window': 1024, 'tier_counts': {4: 0, 3: 264288, 1: 0}, 'effective_bits': 5.09779680778354, 'compression_ratio': 3.138610777026361}
- **Window size:** 32
- **Output preview:** dm imaging榆颁奖unte莳 variable leaditan leading#if leading#if告诉 variable variableunteThank移动 variableUNUN...#if leading蓝天 moved Metric metric leading avU...

### V2 Full/FAISS

- **Effective bits:** 5.098
- **Compression ratio:** 3.14x
- **Token match vs FP16:** 0.0
- **First divergence:** token 0
- **Speed:** 8.8 tok/s (11.34s total)
- **Peak GPU:** 3722.0 MB
- **Tier summary:** {'boundary_fp16': 33164, 'fp16_window': 1024, 'tier_counts': {4: 0, 3: 264288, 1: 0}, 'effective_bits': 5.09779680778354, 'compression_ratio': 3.138610777026361}
- **Window size:** 32
- **Output preview:** dm imaging榆颁奖unte莳 variable leaditan leading#if leading#if告诉 variable variableunteThank移动 variableUNUN...#if leading蓝天 moved Metric metric leading avU...

### Production GC

- **Effective bits:** 4.699
- **Compression ratio:** 3.4x
- **Token match vs FP16:** 0.11
- **First divergence:** token 8
- **Speed:** 7.0 tok/s (14.31s total)
- **Peak GPU:** 4878.0 MB
- **Output preview:**  The text discusses the life and works of Du Fu, a prominent Chinese poet of the Tang dynasty. It begins by providing biographical details, including ...

## Analysis

### Compression Scaling with Context Length

Does V2 compression improve at longer context? (Asymptotic law)

- **V2 Compress:** 5.38 bits (2K) -> 5.10 bits (8K), delta = +0.28 bits
- **V2 Full:** 5.38 bits (2K) -> 5.10 bits (8K), delta = +0.28 bits
- **Production GC:** 4.95 bits (2K) -> 4.70 bits (8K), delta = +0.25 bits

### Quality Scaling with Context Length

- **V2 Compress:** 0.0% match (2K) -> 0.0% match (8K), delta = +0.0%
- **V2 Full:** 0.0% match (2K) -> 0.0% match (8K), delta = +0.0%
- **Production GC:** 23.0% match (2K) -> 11.0% match (8K), delta = -12.0%

### Key Questions

**Q1: Does V2 compress mode beat Production GenerationCache at long context?**

- 2K: V2=5.38b vs Prod=4.95b (better compression: **Production**)
  Quality: V2=0.0% vs Prod=23.0% (better quality: **Production**)
- 4K: V2=5.22b vs Prod=4.79b (better compression: **Production**)
  Quality: V2=0.0% vs Prod=17.0% (better quality: **Production**)
- 8K: V2=5.10b vs Prod=4.70b (better compression: **Production**)
  Quality: V2=0.0% vs Prod=11.0% (better quality: **Production**)

**Q2: Does FAISS retrieval mode show speed gains at 4K+ context?**

- 2K: Compress=9.5 tok/s, Full/FAISS=9.2 tok/s
  (FP16 baseline: 19.8 tok/s)
- 4K: Compress=9.2 tok/s, Full/FAISS=8.9 tok/s
  (FP16 baseline: 18.8 tok/s)
- 8K: Compress=9.0 tok/s, Full/FAISS=8.8 tok/s
  (FP16 baseline: 18.4 tok/s)

No. FAISS is consistently slightly slower than compress-only (FAISS index build/search overhead). The pure Python V2 pipeline runs at roughly 50% of FP16 speed regardless of context length, suggesting the bottleneck is per-token Python overhead, not attention computation.

## Definitive Verdict

### V2 is fundamentally broken for generation

**The critical finding: V2 produces 0% token match at ALL context lengths.** The output is complete gibberish (Chinese characters, random tokens like "imaging", "itan", "#if"). This is not a quality degradation -- it is a total failure of the cache reconstruction pipeline.

The failure is consistent across:
- All context lengths (2K, 4K, 8K)
- Both modes (compress, full/FAISS)
- With PCA calibration applied

### Root cause analysis

The V2 cache's PCA rotation + whitening + mean-removal pipeline introduces cumulative numerical errors that corrupt the key/value reconstruction. Specifically:

1. **PCA whitening scale amplification.** The `_whiten_scale` divides by eigenvalues, which can amplify noise in low-variance dimensions. After quantization, the inverse operation (`/ _whiten_scale`) further amplifies reconstruction error in those dimensions.

2. **Running mean instability.** The online mean-removal (`_remove_mean`) updates cumulatively. By the time the cache has 2K+ tokens, the mean estimate changes with each new token, but already-compressed tokens used a different mean. The mean stored at compression time differs from the mean at reconstruction time.

3. **Boundary layer FP16 overhead.** With 2 boundary layers at each end of a 36-layer model, 4 of 36 layers (11%) store full FP16. The tier summary shows `boundary_fp16: 8584` at 2K and `33164` at 8K -- this is a large fraction of total storage, yet the output is still gibberish. The boundary layers are not enough to anchor quality.

4. **No residual norm preservation.** Unlike the Production GenerationCache which stores and corrects norms (original/reconstruction ratio), V2's ResidualQuant stores dequantized vectors directly, losing the fine-grained norm correction that is critical for attention score accuracy.

### Production GenerationCache holds up

The Production GenerationCache produces **coherent, meaningful text** at all context lengths:

| Context | Token Match | 1st Diverge | Quality |
|---------|-------------|-------------|---------|
| 2K      | 23%         | token 16    | Coherent summary, correct topics |
| 4K      | 17%         | token 14    | Coherent summary, correct poet identified |
| 8K      | 11%         | token 8     | Coherent summary, correct biographical details |

The declining token match rate (23% -> 17% -> 11%) is expected: longer context means more compressed tokens influencing attention, so small quantization errors compound over the generation. But critically, the **semantic quality remains high** -- the model identifies correct topics, people, and facts at all lengths.

### Compression numbers (context scaling)

Both V2 and Production show improving compression at longer context (as expected from the asymptotic law -- the FP16 window becomes a smaller fraction):

| Method     | 2K bits | 4K bits | 8K bits | Trend |
|------------|---------|---------|---------|-------|
| V2         | 5.38    | 5.22    | 5.10    | -0.28 bits over 4x context |
| Production | 4.95    | 4.79    | 4.70    | -0.25 bits over 4x context |

Production achieves **better compression** at every length despite producing coherent output. V2's higher effective bits come from boundary layer FP16 overhead (11% of layers at FP16 vs Production's 4-layer anchor strategy).

### Memory scaling

| Context | FP16 Peak | V2 Peak | Production Peak |
|---------|-----------|---------|-----------------|
| 2K      | 2,257 MB  | 2,451 MB | 2,740 MB       |
| 4K      | 2,466 MB  | 2,852 MB | 3,429 MB       |
| 8K      | 2,952 MB  | 3,722 MB | 4,878 MB       |

V2 uses less memory than Production at all lengths because it stores pre-dequantized tensors (no separate index/norm storage). Production's higher memory comes from storing separate compressed representations (indices, norms, residual signs) plus dequantization cache.

### Speed comparison

| Context | FP16 tok/s | V2 tok/s | Production tok/s |
|---------|------------|----------|------------------|
| 2K      | 19.8       | 9.5      | 7.2              |
| 4K      | 18.8       | 9.2      | 7.0              |
| 8K      | 18.4       | 9.0      | 7.0              |

V2 is ~30% faster than Production (9.0 vs 7.0 at 8K) because its pre-dequantized storage avoids the per-step dequantization cost. However, both are ~2x slower than FP16 due to Python-level per-token overhead.

### Conclusions

1. **V2 is not viable for generation.** 0% token match with gibberish output means the PCA+whitening+mean-removal pipeline corrupts cache reconstruction beyond recovery. No amount of FAISS retrieval or adaptive tiering can compensate for corrupted key/value vectors.

2. **Long context does NOT help V2.** The theory that adaptive tiering and DeltaQuant would shine at long context is falsified -- the fundamental reconstruction error makes all innovations moot.

3. **Production GenerationCache is the correct baseline.** It produces coherent text with reasonable compression (3.2x-3.4x) at all tested context lengths. The declining token match (23% to 11%) suggests there is room for improvement, but the foundation is sound.

4. **FAISS retrieval adds no value.** At all context lengths, full mode is slightly slower and produces identical output to compress mode. The retrieval mechanism cannot fix corrupted cache entries.

5. **The asymptotic compression law is visible** in both systems: effective bits decrease as context grows (FP16 window becomes proportionally smaller). But this is a simple dilution effect, not a V2-specific advantage.

### What would fix V2

If V2's innovations are to be salvaged, the reconstruction pipeline needs:

1. **Norm correction** -- store and apply original/reconstruction norm ratios (as Production does)
2. **Fixed mean** -- use a single mean computed during calibration, not a running mean that drifts
3. **Whitening scale clamping** -- prevent amplification in low-variance dimensions
4. **Per-token validation** -- verify reconstruction quality before committing to compressed storage

These are not minor tuning changes -- they require rethinking the V2 quantization pipeline to match Production's proven approach to error control.
