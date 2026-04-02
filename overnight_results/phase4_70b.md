# Phase 4: 70B+ Model Results

**Date:** 2026-04-02
**GPU:** NVIDIA GeForce RTX 4090
**VRAM:** 23.5 GB total (~1.4 GB used by other processes = ~22.1 GB available)
**RAM:** 62 GB total (~44 GB available)
**PyTorch:** 2.11.0+cu130

## Config
- **Target models:** Qwen2.5-72B-Instruct, Llama-3.3-70B-Instruct
- **Downloaded:** Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4 (39 GB, pre-quantized)

## Result: NOT FEASIBLE on Current Hardware/Software

### Analysis

A 70B+ model at 4-bit quantization requires ~40 GB of storage, which must be split across GPU and CPU. This requires CPU offload via `accelerate`, which has a critical compatibility issue with the current library versions:

| Library | Version | Issue |
|---------|---------|-------|
| transformers | 5.3.0 | Sets `_is_hf_initialized` attr on params |
| accelerate | 1.13.0 | Passes `param.__dict__` (including `_is_hf_initialized`) to BnB param constructors |
| bitsandbytes | 0.49.2 | `Params4bit.__new__()` and `Int8Params.__new__()` reject `_is_hf_initialized` kwarg |

Additionally, BnB 4-bit params cannot be moved from `meta` device because `quant_state.code` (a sub-tensor) becomes a meta tensor and loses its data. This makes the standard accelerate offload mechanism incompatible with BnB 4-bit quantization.

### Approaches Attempted (from Phase 3 debugging)

1. **BnB 4-bit + CPU offload** -- `_is_hf_initialized` TypeError (patched), then meta tensor `NotImplementedError` (fundamentally broken)
2. **BnB INT8 + CPU offload** -- Same `_is_hf_initialized` issue (patched), then OOM moving offloaded INT8 layers to GPU
3. **GPTQ-Int4 pre-quantized** -- Downloaded Qwen2.5-72B-Instruct-GPTQ-Int4 (39GB), but `auto_gptq` failed to build (`setuptools` QiGen kernel gen failure) and `optimum[gptq]` alone cannot load GPTQ models for inference
4. **FP16 + CPU offload** -- Would require loading 131GB+ of weights (not downloaded), impractical

### VRAM Budget Analysis

For a 70B model to work on an RTX 4090 (24GB):
- Model weights at 4-bit: ~40 GB (must split across GPU/CPU)
- GPU portion: ~14 GB (max ~20-25 layers)
- Activation buffer: ~300 MB per layer forward pass
- KV cache: ~2-8 GB depending on sequence length
- **Required:** Working CPU offload (broken) or 2x GPU (multi-GPU)

### What Would Enable This

1. **Fix library compatibility:** Downgrade transformers to ~4.44 or wait for bitsandbytes 0.50+ to accept `_is_hf_initialized`
2. **Build auto_gptq:** Fix the QiGen kernel generation issue to use pre-quantized GPTQ models
3. **Use vLLM/llama.cpp:** These frameworks have their own quantization that doesn't depend on BnB
4. **Multi-GPU setup:** Two RTX 4090s would have 48GB total -- enough for 70B at 4-bit

## Context: What TurboQuantDC Would Provide at 70B

Based on validated results at 3B and 14B:
- **KV cache compression:** 5.0-5.1x at 3-bit (from 2x16 bits/head/token to ~4 bits effective)
- **Output quality:** 5/5 exact match with FP16 on both 3B and 14B models
- **Needle-in-haystack:** 100% success rate across all tested configurations

For a 70B model at 32K context:
- FP16 KV cache: ~8 GB (80 layers x 8 KV heads x 128 dim x 32K tokens x 2 bytes x 2 K+V)
- TurboQuantDC 3-bit: ~1.6 GB (5x compression)
- **Savings:** ~6.4 GB -- critical for fitting on a single GPU

The algorithm is validated. The blocker is purely the model loading infrastructure (BnB + accelerate version compatibility), not TurboQuantDC.

## Verdict: ~~BLOCKED~~ RESOLVED

**UPDATE (2026-03-31):** The HF stack blocker was bypassed by using Ollama + llama.cpp with GGUF models. Qwen2.5-72B-Instruct (Q2_K, 29GB) runs successfully on the RTX 4090 at 1.8 tok/s with 5/5 output quality and needle-in-haystack PASSED. See `phase4_70b_llama_cpp.md` for full results.

**Original verdict:** 70B models could not run via HF transformers + accelerate + bitsandbytes due to version incompatibility. This was an infrastructure issue, not a TurboQuantDC issue. Resolved by switching to llama.cpp.
