# PPL Benchmark: Mean-Removal vs Production WHT

Date: 2026-04-09 21:42
Dataset: wikitext-2 test (max 4096 tokens)
Window: 512 tokens, stride 256
Compression: anchor=0, fp16_window=0, RQ=True, V3-bit
Context: Tom's TQ 3-bit gets +62.95 PPL on Qwen2.5-3B (catastrophic)

## Qwen2.5-7B-Instruct

| Config | PPL | Delta vs FP16 | Time |
|--------|-----|---------------|------|
| FP16 baseline | 7.5225 | baseline | 3.2s |
| WHT 3-bit | 9410.4876 | +9402.97 | 20.2s |
| WHT 3-bit + mean-removal | 7.9029 | +0.38 | 23.0s |
| WHT 4-bit | 1048.9915 | +1041.47 | 32.5s |
| WHT 4-bit + mean-removal | 7.7583 | +0.24 | 30.5s |

## Llama-3.1-8B-Instruct

FAILED: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct.
401 Client Error. (Request ID: Root=1-69d87fd1-6450263f3b563f686e3d310a;7ac85fc4-3524-4e0a-80a4-0846dc16174c)

Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json.
Access to model meta-llama/Llama-3.1-8B-Instruct is restricted. You must have access to it and be authenticated to access it. Please log in.

## Analysis

- Qwen2.5-7B-Instruct 3-bit: mean-removal helps? YES (delta 9402.97 -> 0.38, improvement 9402.58)
- Qwen2.5-7B-Instruct 4-bit: mean-removal helps? YES (delta 1041.47 -> 0.24, improvement 1041.23)