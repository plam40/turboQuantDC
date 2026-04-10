# TurboQuantDC — Session Handoff (April 2-9, 2026)

## What Happened Today (April 9)

The single most important finding of the entire project:

Mean-removal is not an optimization. It fixes catastrophic PPL failure.
- Qwen2.5-7B WHT 3-bit WITHOUT mean-removal: PPL 9,410
- Qwen2.5-7B WHT 3-bit WITH mean-removal: PPL 7.90 (+0.38 vs FP16 7.52)
- NIAH at 8K: FAIL without, PASS at all positions with mean-removal
- Matches Tom's +62.95 finding on Qwen2.5-3B (same root cause)

Tom reviewed our work, raised 6 valid points. We ran the benchmarks he asked for and sent results. Reply posted. Follow-up with PPL/NIAH numbers ready at ~/Downloads/TOM_FOLLOWUP.txt.

Also ran adversarial validation across 14B model, 5 prompts, 3 seeds. Corrected all overstated claims. Ran deep strategic + technical novelty research.

## What's Real (Final Assessment)

SHIP IT (proven, production-ready):
- WHT + mean-removal at 3-bit: PPL +0.30-0.38. NIAH passes. std<0.001.
- Mean-removal: the fix for catastrophic failure on Qwen models, not just an optimization
- GenerationCache: works 3B-72B with boundary anchors + FP16 hot window
- CUDA kernels: 29x faster at d=256

GENUINELY NOVEL (publishable, no prior art):
- Asymptotic compression law: Gini ~ 0.08 * ln(n), O(1/n) min bits/token
- Triple-stack pipeline: eviction + distillation + quant, 37.9x at 0.93 (honest)
- Attention-KL rotation objective (Cayley): novel objective, modest practical gain

NOT NOVEL (prior art: NSNQuant May 2025):
- Mean-removal technique itself (but connecting it to TQ failure IS new)

OVERSTATED (corrected):
- "Beats RotorQuant 16.6%" — proxy metric, no PPL backing
- "59.8x at 0.90" — honest: 20-40x at ~0.89
- Cayley "breakthrough" — +0.002-0.006 on typical layers

BROKEN (must fix before shipping):
- Expected Attention on topic shifts: ANTI-correlated (-0.035 Spearman)
- TurboRetrievalCache > 2K tokens
- V2Cache PCA whitening bug
- Layer 0 always needs FP16

## Immediate Next Actions

1. POST Tom follow-up (~/Downloads/TOM_FOLLOWUP.txt) with PPL + NIAH numbers
2. GET HF token for Llama 3.1 8B and run same PPL benchmark
3. ENTER Gemma 4 Good hackathon (deadline May 18, $200K pool)
   - Build an APPLICATION (medical doc analysis on consumer hardware)
   - Not more benchmarks
4. SUBMIT asymptotic compression law as workshop paper (ICML 2026 deadlines ~May)
5. UPSTREAM mean-removal to Tom's TQ+ (one-liner C patch, already shared)
6. UPDATE GitHub Pages with honest numbers (PPL, not attention cosine)

## Strategic Direction

TurboQuantDC is a research contribution and credential, not a standalone product.
Best paths (ranked by expected value):
1. Upstream novel techniques to llama.cpp/vLLM (52M+ monthly users)
2. Use codebase as portfolio for inference engineering roles ($300-600K)
3. Gemma 4 hackathon ($10-50K near-term)
4. Publish asymptotic law paper (career capital)

## Codebase State

- 120+ commits, v0.3.0, 1796+ tests, MIT license
- GitHub Pages live: https://dhawalc.github.io/turboQuantDC/
- llama.cpp PR branch: feat/residualquant-rq3 on /home/dhawal/tom-llama-cpp
- All results pushed to GitHub

## Key Files

| File | Status |
|------|--------|
| benchmarks/results/ppl_for_tom.md | PPL numbers for Tom |
| benchmarks/results/niah_for_tom.md | NIAH results for Tom |
| benchmarks/results/adversarial_validation.md | Honest validation |
| ~/Downloads/TOM_FOLLOWUP.txt | Reply to post |
| ~/Downloads/STRATEGIC_ANALYSIS.md | Business strategy |
| ~/Downloads/WHAT_IS_GENUINELY_NOVEL.md | Novelty map vs literature |
| turboquantdc/generation_core.py | Production cache (STABLE) |
| turboquantdc/expected_attention.py | EA pruning (needs shift guard) |
| turboquantdc/cache_distillation.py | KVSculpt (standalone) |
| turboquantdc/cayley_quant.py | Learned rotation (experimental) |
