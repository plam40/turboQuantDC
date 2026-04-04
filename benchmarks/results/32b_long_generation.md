# 32B Long Generation Stress Test

**Date:** 2026-04-02
**Model:** Qwen/Qwen2.5-32B-Instruct (BnB NF4, 4-bit weights)
**Hardware:** RTX 4090 24GB
**KV Cache:** 3-bit ResidualQuant (K3/V3, boundary anchors [first 2 + last 2 layers FP16], FP16 window=64)
**Decoding:** Greedy (do_sample=False)
**Prompt:** "Write a detailed essay about the history of artificial intelligence, covering the key milestones from the 1950s to 2025."

## Summary

| Target | Actual | Match Rate | First Diverge | Divergences | FP16 (s) | RQ3 (s) | FP16 tok/s | RQ3 tok/s |
|--------|--------|-----------|---------------|-------------|----------|---------|------------|-----------|
| 200 | 200 | 0.2600 | 52 | 148 | 11.7 | 45.4 | 17.1 | 4.4 |
| 500 | 500 | 0.1120 | 52 | 444 | 28.2 | 103.6 | 17.7 | 4.8 |
| 1000 | N/A | N/A | N/A | OOM | 54.6 | OOM | 18.3 | OOM |

## Key Finding: Previous 50-Token Match Was a Coincidence

The prior validation (50 tokens, 100% match) stopped 2 tokens before the first divergence point. **At token 52, the model diverges and never recovers.** This means:

- Tokens 0-51: **IDENTICAL** (100% match)
- Token 52: FP16 produces "risks", RQ3 produces "of"
- Tokens 53+: **Cascading divergence** (once one token differs, all subsequent tokens differ)

The previous "IDENTICAL generation" claim at 50 tokens was technically correct but deeply misleading. The compression was already degrading attention scores, and one more token pushed it past the greedy decoding threshold.

## 200-Token Generation

**DIVERGENCE at token 52** (148 total divergences out of 200 tokens)
- Match rate: 0.2600 (52/200)
- After divergence: 0/148 tokens match (0% recovery)

### First Divergence Point

| | Token | Token ID |
|---|---|---|
| FP16 | `risks` | 15276 |
| RQ3 | `of` | 315 |

**FP16 context around divergence:**
> ... Conclude with predictions for the future development and potential risks of AI technology.
The history of artificial intelligence (

**RQ3 context around divergence:**
> ... Conclude with predictions for the future development and potential of AI technology.
The history of artificial intelligence (AI

### Text Comparison (first 500 chars)

**FP16:**
```
 Conclude with predictions for the future development and potential risks of AI technology.

The history of artificial intelligence (AI) is a fascinating tale of human ingenuity, scientific breakthroughs, and technological advancements that spans over seven decades. From its conceptual beginnings in the 1950s to the transformative developments of the 2020s, AI has evolved from a theoretical curiosity into one of the most powerful and pervasive technologies in the world. This essay explores the key mileston
```

**RQ3:**
```
 Conclude with predictions for the future development and potential of AI technology.

The history of artificial intelligence (AI) is a story that stretches across more than seven decades, from the earliest theoretical musings to the cutting-edge innovations that are reshaping the world today. This essay will explore the major milestones in AI's development, from the pioneering work of the 1950s to the transformative breakthroughs of the 2020s.

### The Birth of AI: The 1950s

The origins of artificial intell
```

## 500-Token Generation

**DIVERGENCE at token 52** (444 total divergences out of 500 tokens)
- Match rate: 0.1120 (56/500)
- Same divergence point as 200-token test (consistent)

### Text Comparison (first 500 chars)

**FP16:**
```
 Conclude with predictions for the future development and potential risks of AI technology.

The history of artificial intelligence (AI) is a fascinating tale of human ingenuity, scientific breakthroughs, and technological advancements that spans over seven decades. From its conceptual beginnings in the 1950s to the transformative developments of the 2020s, AI has evolved from a theoretical curiosity into one of the most powerful and pervasive technologies in the world. This essay explores the key mileston
```

**RQ3:**
```
 Conclude with predictions for the future development and potential of AI technology.

The history of artificial intelligence (AI) is a story that stretches across more than seven decades, from the earliest theoretical musings to the cutting-edge innovations that are reshaping the world today. This essay will explore the major milestones in AI's development, from the pioneering work of the 1950s to the transformative breakthroughs of the 2020s.

### The Birth of AI: The 1950s

The origins of artificial intell
```

## 1000-Token Generation

**OOM:** CUDA out of memory during RQ3 generation at 1000 tokens.
FP16 baseline completed successfully (1000 tokens in 54.6s, 18.3 tok/s).

The OOM is caused by the GenerationCache's dequantization overhead combined with the 32B model already consuming ~20GB of 24GB VRAM. With 1000 tokens of KV cache per layer across 64 layers, the dequantized tensors exceed remaining memory.

## Analysis

### Why Divergence Occurs at Token 52

The prompt generates 30 input tokens. The model first generates ~22 tokens of "instruction framing" text (the model reformulates the prompt as part of its response pattern). These initial tokens are relatively high-confidence, so even slightly perturbed attention scores produce the same argmax.

At token 52 (about 22 tokens of actual essay content), the model reaches a point where two candidate tokens ("risks" vs "of") have nearly identical logit scores. The quantization error in the attention scores tips the balance from one to the other. Once diverged, the different context means all subsequent tokens are different.

### Implications for the "IDENTICAL Generation" Claim

The previous result of 100% token match at 50 tokens on this model was **NOT indicative of lossless compression**. The 3-bit ResidualQuant cache with boundary anchoring and FP16 window produces attention scores that are very close but not identical to FP16, and this gap manifests as divergence at the first low-confidence token boundary.

### Comparison with Smaller Models

| Model | Params | Previous Match at 50 tokens | Actual Divergence Point |
|-------|--------|---------------------------|------------------------|
| Qwen2.5-3B | 3B | 100% | Not yet tested at longer lengths |
| Qwen2.5-32B | 32B | 100% | Token 52 |

The 32B model has 64 layers (vs 36 for 3B), meaning quantization error accumulates through more layers. With boundary anchoring, only layers 0-1 and 62-63 are FP16, leaving 60 layers compressed.

### Throughput

RQ3 generation is ~3.7-4x slower than FP16 on this model:
- FP16: 17-18 tok/s
- RQ3: 4.4-4.8 tok/s

The overhead comes from per-token dequantization (index lookup + rotation + residual correction) across 60 compressed layers.

## Verdict

3-bit ResidualQuant KV cache **does NOT produce identical generation** to FP16 at lengths beyond ~50 tokens on Qwen2.5-32B-Instruct. The first divergence occurs at **token 52**, after which the sequences completely diverge.

The match at 50 tokens was a coincidence of the generation length stopping just before the first low-confidence token decision. This is an important correction to the prior validation results.

### Recommendations

1. **Report honestly**: 3-bit compression produces near-FP16 attention quality (cosine sim >0.94 on 32B) but NOT identical generation. The "IDENTICAL" claim from 50-token tests should be retracted for 32B.
2. **Increase FP16 coverage**: To maintain identical generation at longer sequences, consider increasing fp16_window (currently 64) or adding more anchor layers.
3. **Test 3B model at longer lengths**: The previous "IDENTICAL" result on Qwen2.5-3B at 200 tokens may also be fragile.
4. **Memory optimization**: The 1000-token OOM suggests the dequantization path needs memory optimization for deployment on memory-constrained devices.
