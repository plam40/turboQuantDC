# TurboQuantDC — Session Handoff (April 2-9, 2026)

## What This Is
From-scratch PyTorch implementation of Google's TurboQuant (ICLR 2026) for KV cache compression, extended with 7 novel research breakthroughs. 108 commits, 662+ tests, MIT license.

**Live showcase:** https://dhawalc.github.io/turboQuantDC/
**Repo:** https://github.com/dhawalc/turboQuantDC
**PR to llama.cpp:** https://github.com/TheTom/llama-cpp-turboquant/pull/45#issuecomment-4181916947

## Current State: v0.3.0

### What Works (Ship-Ready)
- **GenerationCache** — Production KV cache with mean-removal, ResidualQuant, boundary anchors, FP16 hot window. Validated 3B-72B. Coherent generation, <1% PPL cost.
- **Mean-removal** — Integrated into production. One line: `keys -= keys.mean()`. 3-bit beats 4-bit quality.
- **ResidualQuant** — Beats QJL by 19% on attention match. Tom (TurboQuant author) confirmed.
- **PCA-adaptive rotation** — 13x lower MSE than WHT. 100% top-5 at 3-bit. Calibrate 128 tokens, transfers across prompts.
- **CUDA kernels** — 29x faster dequantize at d=256. WHT extended to d=2048.
- **DeltaQuant** — 6.1x at identical 3-bit quality via cross-token delta coding.
- **Adaptive bits** — Power-law exploitation: top 10% tokens get 80% attention. 7.8x near-lossless.
- **Entropy coding** — 6% free lossless via ANS on top of quantization.
- **pip install turboquantdc[all]** — Builds clean, 107 exports.
- **llama.cpp PR** — `GGML_TYPE_RQ3_0 = 46` on branch `feat/residualquant-rq3` in `/home/dhawal/tom-llama-cpp`. CPU-only, compiles with CUDA. Ready to submit.

### What's Experimental (Bugs to Fix)
- **TurboQuantV2Cache** (`v2_cache.py`) — Garbled output. PCA whitening amplifies noise in low-variance dimensions. Fix: clamp whitening scale, sync running mean.
- **TurboRetrievalCache** (`turbo_retrieval_cache.py`) — Works at 1.5K (needle found). Fails at 3K+ (FAISS undertrained, sliding-window prefill loses distant tokens). Fix: full-attention prefill, better nlist scaling, dtype optimization (int64→int8 for indices = 8x RAM savings).

### Research Findings (Publishable)
1. **Asymptotic compression law** — Gini 0.60→0.85 (128→2K tokens). Min bits/token = O(1/n). Longer context = better compression.
2. **Mean-removal free bit** — 57% of key variance is invisible to softmax.
3. **PCA enables both compression AND retrieval** — WHT can't. Dual-mode PCA.
4. **ResidualQuant > QJL** — Lower variance beats unbiasedness for autoregressive generation.
5. **Infrastructure > algorithm** — Boundary + hot window worth 40 PPL points; algorithm choice worth 1.

### Dead Ends (Proven Negative Results)
- Cross-head delta: GQA kills correlation (cosine=0.12)
- Cross-layer prediction: layers independent (deltas 2.6x larger)
- Temporal delta: error accumulation O(sqrt(T))
- Spectral/DCT: flat energy spectrum
- Sign prediction: WHT decorrelation is complete

## Benchmark Headlines

| Model | Context Extension | Speed | Key Result |
|-------|-------------------|-------|------------|
| Gemma 4 26B MoE | 196K→262K (full native) | 150 tok/s | FP16 OOMs, turbo3 runs |
| Gemma 4 E4B | — | — | 0.999994 cosine, 100% top-5 |
| Gemma 3 27B | 49K→115K (2.3x) | 44 tok/s | Zero speed degradation |
| Llama 3.1 70B | 4K→16K (4x) | 2.8 tok/s | FP16 OOMs at 8K |
| Llama 3.1 8B | 48K→100K (2.1x) | 86 tok/s | turbo3 FASTER than FP16 |
| Qwen 2.5 72B | — | — | 100% top-1, 100% top-5 |
| Perplexity (8B) | — | — | +0.67% PPL (turbo3 vs FP16) |

## Key Files

| File | What |
|------|------|
| `turboquantdc/generation_cache.py` | Re-export wrapper (split into 4 modules) |
| `turboquantdc/generation_core.py` | Production GenerationCache |
| `turboquantdc/generation_layers.py` | _CompressedLayer, _FP16Layer |
| `turboquantdc/residual_quant.py` | ResidualQuant (with mean-removal) |
| `turboquantdc/learned_rotation.py` | PCA rotation |
| `turboquantdc/delta_quant.py` | DeltaQuant cross-token coding |
| `turboquantdc/adaptive_bits.py` | Importance scoring + adaptive tiers |
| `turboquantdc/retrieval_cache.py` | FAISS index wrapper |
| `turboquantdc/turbo_retrieval_cache.py` | TurboRetrievalCache (experimental) |
| `turboquantdc/v2_cache.py` | Unified V2 (experimental, has bug) |
| `turboquantdc/cuda/dequantize.cu` | Raw CUDA dequantize kernel |
| `turboquantdc/cuda/wht.cu` | CUDA WHT to d=2048 |
| `tools/residualquant_reference.c` | C99 reference for llama.cpp PR |
| `docs/RESIDUALQUANT_LLAMA_CPP_SPEC.md` | GGML_TYPE_RQ3_0 spec |
| `docs/RESEARCH_LANDSCAPE.md` | 40-paper competitive analysis |
| `visualization/index.html` | GitHub Pages (deployed) |

## Immediate Next Steps

1. **Benchmark Qwen 3.5 35B MoE at 262K** — 2 KV heads, lightest KV cache of any large model
2. **Fix TurboRetrievalCache** — dtype optimization + full-attention prefill
3. **Fix V2Cache** — clamp PCA whitening scale
4. **Submit llama.cpp PR** to Tom's repo
5. **Enter Gemma 4 Good Hackathon** — $200K prize pool, deadline May 18

## External Validation
- Tom Turney (@no_stp_on_snek): "we killed QJL early for the same reason. the stacking numbers speak for themselves."
- PR #45 comment posted with full RTX 4090 benchmark data
- GitHub Pages live: https://dhawalc.github.io/turboQuantDC/
