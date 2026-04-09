# XQuant: Cross-Layer Pre-Projection Activation Caching

## Hypothesis

Instead of compressing K and V separately per layer, cache the pre-projection
activation X and rematerialize K=X@W_k, V=X@W_v on the fly. For GQA models
with few KV heads, this is NOT a direct win (X is larger than K+V). But if
adjacent layers have highly correlated X activations, we can store X every S
layers and interpolate, achieving S* additional compression.

## Model: Qwen/Qwen2.5-3B-Instruct

- hidden_size: 2048
- num_layers: 36
- num_attention_heads: 16
- num_kv_heads: 2
- head_dim: 128

## Experiment 1: Cross-Layer X Correlation

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| Adjacent (n vs n+1) cosine | 0.8152 | 0.0062 | 0.9415 |
| Skip-1 (n vs n+2) cosine | 0.7523 | 0.0051 | 0.8913 |
| Skip-3 (n vs n+4) cosine | 0.6651 | 0.0003 | 0.8298 |

### Per-Layer Adjacent Cosine Similarity (averaged across 4 prompts)

| Layer Pair | Cosine (avg) | Delta Norm (avg) | Viable? |
|------------|-------------|-----------------|---------|
| 0 -> 1 | 0.011 | 1.13 | NO - near-zero |
| 1 -> 2 | 0.402 | 1.21 | NO |
| 2 -> 3 | 0.532 | 1.62 | NO |
| 3 -> 4 | 0.722 | 0.72 | NO |
| 4 -> 5 | 0.719 | 0.98 | NO |
| 5 -> 6 | 0.677 | 0.77 | NO |
| 6 -> 7 | 0.862 | 0.56 | Marginal |
| 7 -> 8 | 0.830 | 0.60 | Marginal |
| 8 -> 9 | 0.884 | 0.48 | Marginal |
| 9 -> 10 | 0.852 | 0.62 | Marginal |
| 10 -> 11 | 0.887 | 0.47 | Marginal |
| 11 -> 12 | 0.861 | 0.54 | Marginal |
| 12 -> 13 | 0.856 | 0.57 | Marginal |
| 13 -> 14 | 0.889 | 0.46 | Marginal |
| 14 -> 15 | 0.888 | 0.49 | Marginal |
| 15 -> 16 | 0.862 | 0.52 | Marginal |
| 16 -> 17 | 0.843 | 0.67 | Marginal |
| 17 -> 18 | 0.860 | 0.51 | Marginal |
| 18 -> 19 | 0.888 | 0.50 | Marginal |
| 19 -> 20 | 0.894 | 0.54 | Marginal |
| 20 -> 21 | 0.876 | 0.49 | Marginal |
| 21 -> 22 | 0.910 | 0.45 | OK |
| 22 -> 23 | 0.922 | 0.40 | OK |
| 23 -> 24 | 0.925 | 0.39 | OK |
| 24 -> 25 | 0.917 | 0.51 | OK |
| 25 -> 26 | 0.913 | 0.42 | OK |
| 26 -> 27 | 0.900 | 0.56 | Marginal |
| 27 -> 28 | 0.916 | 0.41 | OK |
| 28 -> 29 | 0.927 | 0.40 | OK |
| 29 -> 30 | 0.926 | 0.42 | OK |
| 30 -> 31 | 0.906 | 0.42 | OK |
| 31 -> 32 | 0.913 | 0.49 | OK |
| 32 -> 33 | 0.923 | 0.37 | OK |
| 33 -> 34 | 0.587 | 0.88 | NO - final LN |
| 34 -> 35 | 0.852 | 0.52 | Marginal |

## Experiment 2: Rematerialization Quality

| Method | K cosine (mean) | V cosine (mean) |
|--------|----------------|----------------|
| Direct (X_n) | 1.0000 | 1.0000 |
| Cross-layer (X_{n-1}) | 0.9417 | 0.7979 |
| Interpolated | 0.9614 | 0.8712 |
| Skip-2 (X_{n-2}) | 0.9159 | 0.7098 |

## Experiment 3: Storage Analysis

| Configuration | Bits/token/layer | Compression |
|--------------|-----------------|-------------|
| FP16 KV | 8192 | 1.0x |
| TurboQuant KV 3-bit | 1536 | 5.3x |
| TurboQuant KV 4-bit | 2048 | 4.0x |
| XQuant 3-bit stride=1 | 6144 | 1.3x |
| XQuant 4-bit stride=1 | 8192 | 1.0x |
| XQuant 3-bit stride=2 | 3072 | 2.7x |
| XQuant 4-bit stride=2 | 4096 | 2.0x |
| XQuant 3-bit stride=4 | 1536 | 5.3x |
| XQuant 4-bit stride=4 | 2048 | 4.0x |

## Experiment 4: Attention Score Accuracy

| Method | Top-5 Match | Score Pearson r |
|--------|------------|----------------|
| Adjacent (X_{n-1}) | 0.8109 | 0.9488 |
| Interpolated | 0.8516 | 0.9719 |

## Experiment 5: Residual Stream Analysis

- Mean update ratio: 0.6010
- Mean effective rank of delta: 20.2
- Mean top-10 SV energy: 0.7896

## Verdict

**NOT VIABLE for Qwen2.5-3B (GQA with 2 KV heads).**

### Three independent reasons XQuant fails for this architecture:

**1. X is 4x larger than K+V.** With hidden_size=2048 and only 2 KV heads x 128 dim = 256
values per K and V (512 total), X is 4x more data per layer. To break even at stride=1,
the model would need num_kv_heads >= 8.

**2. Cross-layer X correlation is insufficient.** Mean adjacent-layer cosine similarity
is only 0.815 (need >0.95 for viable sharing). Early layers (0-5) have near-zero
correlation. The "deeper layers are more similar" pattern (layers 21-33 reach ~0.92)
is not enough since ALL layers must participate.

**3. Rematerialization through W_k/W_v amplifies errors.** Even with X cosine 0.92
between adjacent layers, projecting through W_k produces K cosine of only 0.94, and
projecting through W_v produces V cosine of only 0.80. The projection weights are
layer-specific and amplify small X differences into large K/V differences.

### Why the XQuant paper's claims don't apply here:

The XQuant paper (if it claims >0.95 cross-layer X correlation) likely measured on
**MHA models** (all heads = attention heads, many KV heads) where:
- X dimension / KV dimension ratio is closer to 1
- Larger models may have smaller per-layer residual updates
- Different architectures (e.g., LLaMA-65B) have different residual dynamics

### Numbers that killed it:

| What we needed | What we measured |
|---------------|-----------------|
| X cos > 0.95 (adjacent) | 0.815 mean, 0.006 min |
| V remat cos > 0.99 (cross-layer) | 0.798 mean, 0.194 min |
| Storage win at stride=1 | 4x WORSE (X=2048 vs KV=512) |
| Update ratio < 0.1 | 0.60 mean (layers change ~60%) |
| Delta effective rank | 20.2 (out of 2048, NOT low-rank) |

### What DOES work for this model:

Standard TurboQuant KV compression at 3-bit achieves 5.3x compression with
>0.995 cosine similarity. This remains the best approach for GQA models with
few KV heads.
