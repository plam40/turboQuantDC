# NIAH Benchmark: Mean-Removal vs Production WHT

Date: 2026-04-09 21:48
Model: Qwen/Qwen2.5-7B-Instruct
Context: 8192 tokens
Needle: "The secret code is PINEAPPLE-77."
Expected answer: PINEAPPLE-77
Compression: anchor=0, fp16_window=0, RQ=True, V3-bit

| Position | Config | Pass/Fail | Generated (truncated) | Time |
|----------|--------|-----------|-----------------------|------|
| 10% | FP16 baseline | PASS | PINEAPPLE-77 | 1.8s |
| 10% | WHT 3-bit | FAIL | 0000., .  numberWith); .0..0.. I.. The0 the, .,. | 5.7s |
| 10% | WHT 3-bit + mean-removal | PASS | PINEAPPLE-77 | 2.7s |
| 50% | FP16 baseline | PASS | PINEAPPLE-77 | 1.7s |
| 50% | WHT 3-bit | FAIL | 001 9111 2 mathematics 9 of mathematics 0. 1.  (1 10 mathema | 5.2s |
| 50% | WHT 3-bit + mean-removal | PASS | PINEAPPLE-77 | 2.9s |
| 90% | FP16 baseline | PASS | PINEAPPLE-77 | 1.2s |
| 90% | WHT 3-bit | FAIL | 1 0 0000000 strugg0 mathematics. 00.52 1.0.0, . | 5.6s |
| 90% | WHT 3-bit + mean-removal | PASS | PINEAPPLE-77 | 3.1s |

## Summary by Config

- FP16 baseline: 3/3 positions recalled
- WHT 3-bit: 0/3 positions recalled
- WHT 3-bit + mean-removal: 3/3 positions recalled

## Analysis

- FP16 baseline: 3/3 (reference)
- WHT 3-bit: 0/3
- WHT 3-bit + mean-removal: 3/3
- CONCLUSION: Mean-removal HELPS needle recall at 3-bit