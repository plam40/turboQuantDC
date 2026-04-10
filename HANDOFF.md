# TurboQuantDC — Session Handoff (April 2-9, 2026)

## Critical: Tom's Review Needs Response

Tom (@no_stp_on_snek) reviewed our work and raised 6 valid points. Reply drafted at `~/Downloads/TOM_REPLY.md`. Key actions before tweeting anything else:

1. **Run PPL on Qwen2.5-7B and Llama 3.1 8B** vs production WHT (Tom's ask)
2. **Run NIAH at 32K** at start/mid/end positions (Tom's ask)
3. **Correct the baseline comparison** — our "+89% over WHT" was vs internal pipeline, not production TQ+
4. **Mark mean-removal as 3-bit-symmetric only** — hurts at 4-bit with block rotations

## What's Real (Adversarial Validated)

| Technique | Claim | Validated? | Honest Number |
|-----------|-------|------------|---------------|
| WHT + mean-removal 3-bit | 0.985-0.998 cosine | YES (std<0.001, 3B+14B, 5 prompts, 3 seeds) | Ship it |
| Mean-removal | Never hurts | YES at 3-bit symmetric | 3-bit symmetric ONLY, hurts at 4-bit+block |
| Cayley learned rotation | 0.974 cosine | INFLATED (layer 0 anomaly) | +0.002-0.006 on typical layers |
| Expected Attention | 10x at 0.978 | OVERSTATED | Spearman 0.37 real data, BREAKS on topic shift |
| Triple stack | 59.8x at 0.90 | OVERSTATED | 20-40x at ~0.89 honestly |
| KVSculpt distillation | 0.999 pre-quant | REAL | Distillation itself is near-lossless |
| Block rotation + mean | Beats RotorQuant 16.6% | MISLEADING | Beats on attention cosine proxy, not PPL |

## What's Broken (Must Fix)

1. **Expected Attention on topic shifts**: Spearman -0.035 (ANTI-correlated). Needs shift-detection guard.
2. **TurboRetrievalCache > 2K tokens**: FAISS undertrained, sliding-window loses distant tokens. Fix: full-attention prefill, dtype optimization (int64→int8).
3. **V2Cache**: PCA whitening amplifies noise. Fix: clamp low-variance dimensions.
4. **Layer 0**: Always needs FP16 anchor. No quantization scheme works well on it.

## Immediate Next Actions (Priority Order)

1. **Reply to Tom** — post the reply from ~/Downloads/TOM_REPLY.md
2. **Run PPL benchmark** — Qwen2.5-7B + Llama 3.1 8B, WHT+mean-removal vs production WHT, wikitext-2 perplexity
3. **Run NIAH at 32K** — start/mid/end positions, with our GenerationCache
4. **Prep mean-removal patch for Tom** — one-liner he can test on Metal/Pascal
5. **Update GitHub Pages** — correct overstated claims
6. **Update tweets** — qualify the RotorQuant comparison

## What to Ship (v0.3.1)

- WHT + mean-removal as new default (proven)
- Block rotation as optional (`rotation_type="givens"`)
- Cayley as optional calibration (frame as "modest per-layer improvement")
- Expected Attention with shift-detection guard
- KVSculpt distillation (standalone, not in generation pipeline yet)
- Fix get_mask_sizes for transformers 5.5+

## What NOT to Ship Yet

- TurboRetrievalCache (broken > 2K)
- V2Cache (PCA whitening bug)
- Triple stack as "60x" (needs honest framing)
- Any "beats RotorQuant" claim without PPL backing

## Codebase State

- **120+ commits**, 662+ tests, v0.3.0 on PyPI (needs bump to 0.3.1)
- GitHub Pages live: https://dhawalc.github.io/turboQuantDC/
- llama.cpp PR ready: feat/residualquant-rq3 on /home/dhawal/tom-llama-cpp
- PR #45 comment posted: https://github.com/TheTom/llama-cpp-turboquant/pull/45

## Key Files

| File | What | Status |
|------|------|--------|
| turboquantdc/generation_core.py | Production GenerationCache | STABLE, ship |
| turboquantdc/residual_quant.py | ResidualQuant + mean-removal | STABLE, ship |
| turboquantdc/block_rotation.py | Givens/Quaternion rotation | STABLE, optional |
| turboquantdc/cayley_quant.py | Learned full rotation | EXPERIMENTAL |
| turboquantdc/expected_attention.py | EA pruning | NEEDS shift guard |
| turboquantdc/cache_distillation.py | KVSculpt | STANDALONE only |
| turboquantdc/turbo_retrieval_cache.py | FAISS + compressed KV | BROKEN > 2K |
| turboquantdc/v2_cache.py | Unified V2 | BROKEN (whitening bug) |
| benchmarks/adversarial_validation.py | Honest validation | REFERENCE |
| ~/Downloads/TOM_REPLY.md | Reply to Tom | POST THIS |

## Research Dead Ends (Published)

Cross-head delta (GQA kills it), cross-layer prediction (independent layers), temporal delta (error accumulation), spectral/DCT (flat energy), XQuant rematerialization (worse for GQA), sign prediction (WHT decorrelation complete).

## The Real Moat

After adversarial validation, the defensible innovations are:
1. **Mean-removal** — one line, proven across models/prompts/seeds at 3-bit
2. **Asymptotic compression law** — Gini increases with context, publishable finding
3. **KVSculpt distillation** — near-lossless token synthesis, novel combination
4. **Expected Attention** — works when attention is stable (needs guard for shifts)
5. **Comprehensive benchmark data** — 20+ experiments, honest dead ends published

The "60x compression" and "beats RotorQuant" claims need correction. The mean-removal insight and the honest science are what build real credibility with Tom and the community.
