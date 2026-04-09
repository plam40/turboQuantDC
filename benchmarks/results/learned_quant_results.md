# Learned Quantization Results

## Differentiable Attention-Optimal KV Cache Compression

**Model:** Qwen2.5-3B-Instruct (BnB 4-bit)
**Bits:** 3
**Head dim:** 128

## Main Comparison (3-bit)

| Config | Cosine | KL Div | Top-1 | Top-5 | Spearman |
|--------|--------|--------|-------|-------|----------|
| random_givens | 0.7635 | 3.773248 | 0.7176 | 0.7961 | 0.8539 |
| givens_mean_removal | 0.8580 | 0.611013 | 0.8102 | 0.9310 | 0.9647 |
| wht_mean_removal | 0.9351 | 0.187348 | 0.8706 | 0.9725 | 0.9764 |
| learned_rotation | 0.8263 | 1.412170 | 0.7702 | 0.9051 | 0.8500 |
| learned_rotation_mean | 0.8921 | 0.414147 | 0.8408 | 0.9435 | 0.9719 |
| learned_full | 0.9081 | 0.325671 | 0.8463 | 0.9616 | 0.9611 |

## Calibration Steps Sweep

| Steps | Cosine | KL Div | Top-5 | Cal Time (ms) |
|-------|--------|--------|-------|---------------|
| 10_steps | 0.4551 | 2.539709 | 0.6431 | 19 |
| 25_steps | 0.5231 | 2.124424 | 0.6157 | 45 |
| 50_steps | 0.5353 | 1.823677 | 0.7255 | 86 |
| 100_steps | 0.5264 | 1.502535 | 0.7843 | 178 |

## Calibration Tokens Sweep

| Tokens | Cosine | KL Div | Top-5 | Cal Time (ms) |
|--------|--------|--------|-------|---------------|
| 32_tokens | 0.9918 | 0.031496 | 1.0000 | 87 |
| 64_tokens | 0.9926 | 0.028429 | 1.0000 | 89 |
| 128_tokens | 0.9917 | 0.030390 | 1.0000 | 96 |
| 256_tokens | 0.9937 | 0.026530 | 1.0000 | 90 |

## Transfer Test

Calibrate on prompt A, evaluate on prompt B.

| Layer | No Cal Cosine | Transfer Cosine | Direct Cosine |
|-------|---------------|-----------------|---------------|
| layer_0 | 0.7865 | 0.6610 | 0.8006 |
| layer_8 | 0.9882 | 0.9864 | 0.9907 |
| layer_16 | 0.9829 | 0.9802 | 0.9862 |
| layer_35 | 0.8927 | 0.8925 | 0.9009 |
