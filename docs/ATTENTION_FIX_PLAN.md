# ATTENTION FIX PLAN: Making Autoregressive Generation Work

*War Council output -- 2026-03-28*
*Problem: Coherent generation at <100 tokens, garbled beyond that with compressed KV cache*

---

## 0. Root Cause Analysis

### Why generation fails

The failure is not a single bug. It is a fundamental mismatch between what TurboQuant
optimizes (unbiased inner products in expectation) and what autoregressive generation
needs (low-variance attention scores at every individual step).

**The error amplification chain:**

```
Per-coordinate MSE error (small, ~0.035 at 3-bit)
  -> Per-vector reconstruction error (23-44% -- EXPECTED, paper says this is fine)
    -> Per-score error in Q @ K^T (small per score, but...)
      -> Softmax EXPONENTIATES score errors (e^{score + error} != e^{score} * e^{error})
        -> Wrong attention weights -> wrong weighted value sum
          -> Wrong hidden state fed to NEXT layer
            -> 36 layers of multiplicative error compounding
              -> After ~100 tokens: attention routing is randomized
```

**The softmax amplification problem:**

Softmax is exp-normalize. A score error of epsilon becomes a weight multiplicative
factor of e^epsilon. For high-attention positions (large scores), even small absolute
errors shift disproportionate probability mass. QJL's variance of pi/(2m) * ||y||^2
translates to attention weight perturbations that compound across layers.

**Quantitative breakdown (d=128, 3-bit, single layer):**
- MSE score bias: ~0.001 per score (systematic, same direction)
- QJL correction: removes bias (E[error]=0) but adds variance ~0.012 per score
- Softmax on biased scores: weights shift by ~0.1% from true (tolerable)
- Softmax on unbiased-but-noisy scores: weights shift by ~2-5% (destructive over 36 layers)

**Community validation:** 6+ independent teams (Python, C, Rust) confirm that
MSE-only (biased, low variance) beats MSE+QJL (unbiased, high variance) for
autoregressive generation. The paper's theoretical guarantee (unbiased inner products)
does not translate to correct autoregressive behavior because the guarantee is about
expectations, not individual samples, and each decode step is a single sample.

---

## 1. Ranked Approaches (Feasibility x Impact)

| Rank | Approach | Feasibility | Impact | Score | Risk |
|------|----------|-------------|--------|-------|------|
| 1 | WHT + Higher-bit MSE-only keys | HIGH | HIGH | 9/10 | LOW |
| 2 | Fused attention from compressed indices | MEDIUM | HIGH | 8/10 | MEDIUM |
| 3 | Norm correction | HIGH | MEDIUM | 7/10 | LOW |
| 4 | Softmax temperature / score normalization | MEDIUM | MEDIUM | 6/10 | LOW |
| 5 | Variance-reduced QJL (median-of-means) | MEDIUM | MEDIUM | 5/10 | MEDIUM |
| 6 | Deterministic residual correction (Hadamard) | MEDIUM | LOW-MED | 4/10 | MEDIUM |
| 7 | Periodic FP16 recalibration layers | LOW | MEDIUM | 3/10 | HIGH |
| 8 | Gradient-free correction network | LOW | HIGH | 3/10 | HIGH |

---

## 2. The Attack Plan

### Wave 1: SHIP IT (parallel, 1-2 days)

**These three approaches are independent and can be built/tested simultaneously.**

#### 1A. WHT + MSE-Only Keys at 4-5 bits [HIGHEST PRIORITY]

Community consensus is clear: this works. hackimov/turboquant-kv, DEJAN, scos-lab,
llama.cpp discussion -- everyone converges on the same answer.

**Why it works:** Walsh-Hadamard Transform reduces coordinate variance before
quantization (deterministic, no random noise). Higher bit-width for keys gives
more centroids, reducing per-coordinate error. MSE-only avoids QJL variance.
Softmax sees low-variance scores. Generation stays coherent.

**Specific config (from community benchmarks):**

| Config | Key Bits | Val Bits | Compression | Quality |
|--------|----------|----------|-------------|---------|
| Conservative | 4 MSE-only | 3 MSE-only | 3.4x | Near-lossless |
| Balanced | 4 MSE-only | 2 MSE-only | 4.5x | Good |
| Aggressive | 3 MSE-only + WHT | 2 MSE-only | 5.8x | Acceptable |

#### 1B. Norm Correction [LOW-HANGING FRUIT]

Store `original_norm / ||reconstruction||` instead of raw norm. This corrects the
magnitude distortion introduced by quantization at zero decode cost.

TheTom's llama.cpp Metal implementation shows this makes turbo3 perplexity beat q8_0
on CUDA (-1.17%). It is a one-line change to the compression path.

#### 1C. Fused MSE Attention Kernel [SPEED + QUALITY]

Compute Q @ K^T directly from compressed indices without materializing FP16 keys.
The algebraic trick (from DEJAN): since R is orthogonal,
`<q, R^T * centroids[idx]> = <R*q, centroids[idx]>`. Pre-rotate queries once,
then do centroid table lookups and dot products per key position.

This avoids the quantize->dequantize->matmul path where rounding errors accumulate.
The inner product is computed exactly on the quantized representation.

### Wave 2: REFINE (sequential, 2-3 days, after Wave 1 results)

#### 2A. Softmax Temperature Calibration

If Wave 1 gets 90% of the way but generation still drifts at very long context
(>4K tokens), calibrate the softmax temperature to compensate for score variance.

The idea: quantized scores have lower dynamic range than FP16 scores. The softmax
temperature should be adjusted so that the quantized score distribution matches
the FP16 score distribution in terms of entropy.

#### 2B. Variance-Reduced QJL (only if 4-bit MSE-only is insufficient)

If we need to go below 4-bit for keys and MSE-only quality degrades, try fixing
QJL rather than abandoning it:

- **Median-of-means:** Split m projections into k groups, compute correction per
  group, take median. Reduces variance from O(1/m) to O(1/m * log(1/delta)).
- **WHT as QJL basis:** Replace random Gaussian S with structured Hadamard-based
  projection. Deterministic, lower variance, O(d log d) compute.

### Wave 3: RESEARCH (if Wave 1+2 insufficient)

#### 3A. Layer-Adaptive with FP16 Anchors

Keep every 6th layer at FP16. The FP16 layers act as "anchors" that reset the
error accumulation. Combined with 4-bit MSE-only on compressed layers, this gives
~3.5x compression with near-perfect quality.

#### 3B. Per-Head Adaptive Bit Allocation

Some attention heads are more sensitive than others. Measure per-head attention
entropy on a calibration pass; heads with low entropy (concentrated attention)
tolerate lower bits. Heads with high entropy (distributed attention) need more bits.

---

## 3. Implementation Specs

### Spec A: WHT + MSE-Only Asymmetric Cache (Wave 1A)

**Goal:** Replace the current TurboQuantEstimator-based key path with WHT-rotated,
MSE-only PolarQuant at configurable bit-width. No QJL for keys.

**File changes:**

1. **`turboquantdc/wht_mse_cache.py`** (NEW, ~300 lines)

   Core class: `WHTMSECache` -- a drop-in replacement for `TurboQuantKVCache` that
   uses WHT rotation instead of QR and MSE-only quantization for keys.

   ```python
   class WHTMSECache:
       """KV cache using WHT rotation + MSE-only quantization.

       Keys: WHT rotation + b-bit Lloyd-Max (MSE-only, no QJL)
       Values: WHT rotation + v-bit Lloyd-Max (MSE-only)

       WHT rotation is deterministic (no random noise), reducing coordinate
       variance before quantization. This produces lower-variance attention
       scores than QR rotation + QJL correction.
       """
       def __init__(
           self,
           d_key: int,
           d_value: int,
           key_bits: int = 4,      # Higher bits for keys
           val_bits: int = 2,      # Lower bits for values
           seed: int = 42,
           device: str = "cpu",
       ):
           # Key path: WHT rotation + MSE-only PolarQuant at key_bits
           # Value path: WHT rotation + MSE-only PolarQuant at val_bits
           # NO QJL stage for either path

       def compress_keys(self, keys: torch.Tensor) -> Dict:
           """Compress keys using WHT + MSE-only.

           Steps:
           1. Store ||k|| as FP16
           2. Normalize: k_norm = k / ||k||
           3. Apply WHT rotation: y = WHT(D * k_norm) / sqrt(d)
           4. Quantize each coordinate: idx = nearest_centroid(y)
           5. Store (idx, ||k||)

           Returns dict with 'indices' (uint8) and 'norms' (fp16).
           """

       def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
           """Compute Q @ K^T from compressed keys.

           Two paths:
           a) Dequantize path (simple):
              k_recon = unrotate(centroids[idx]) * norm
              scores = queries @ k_recon^T

           b) Fused path (fast, Wave 1C):
              q_rot = WHT(D * q) / sqrt(d)
              scores[i,j] = norm[j] * sum_c(q_rot[c] * centroids[idx[j,c]])
           """
   ```

2. **`turboquantdc/hf_integration.py`** -- Modify `TurboQuantLayer`

   Add `mse_only=True` mode that skips QJL entirely and uses full `bits` for
   MSE (not `bits-1`). This mode should be the DEFAULT for the HF integration
   path because it is what works for generation.

   Key change in `_lazy_init`:
   ```python
   if self.mse_only:
       # Full bits to MSE, no QJL
       self._key_pq = PolarQuant(d=head_dim, bits=self.bits, ...)
       self._key_est = None  # No estimator needed
   else:
       # Original: (bits-1) MSE + 1 QJL
       self._key_est = TurboQuantEstimator(d=head_dim, bits=self.bits, ...)
   ```

   Key change in `update` (when `mse_only=True`):
   ```python
   # Compress keys with MSE-only PolarQuant
   key_norms = keys_flat.norm(dim=-1, keepdim=True)
   keys_normalized = keys_flat / (key_norms + 1e-8)
   key_indices = self._key_pq.quantize(keys_normalized)
   # Store: indices + norms only. No QJL signs, no residual norms.
   ```

   Key change in `_dequantize_all` (when `mse_only=True`):
   ```python
   # Dequantize keys from MSE path
   key_recon = self._key_pq.dequantize(all_key_indices)
   key_recon = key_recon * all_key_norms.unsqueeze(-1)
   # Apply norm correction (Wave 1B):
   # recon_norms = key_recon.norm(dim=-1, keepdim=True)
   # key_recon = key_recon * (all_key_norms.unsqueeze(-1) / (recon_norms + 1e-8))
   ```

3. **`turboquantdc/rotation.py`** -- Add WHT-based PolarQuant rotation

   The WHT rotation functions (`fast_wht`, `generate_wht_rotation`,
   `apply_wht_rotation`) already exist. Create a `WHTRotatedPolarQuant` class
   that uses WHT instead of QR decomposition:

   ```python
   class WHTRotatedPolarQuant(nn.Module):
       """PolarQuant with Walsh-Hadamard rotation instead of QR.

       WHT is deterministic (no random matrix), reducing coordinate variance.
       O(d log d) vs O(d^2) for rotation. 256x less memory than QR.
       """
       def __init__(self, d, bits, seed=42, device="cpu"):
           # d must be power of 2 (128, 256 are fine)
           self.wht_params = generate_wht_rotation(d, seed=seed, device=device)
           self.codebook = LloydMaxCodebook(d, bits)

       def rotate(self, x):
           return apply_wht_rotation(x, self.wht_params, inverse=False)

       def unrotate(self, y):
           return apply_wht_rotation(y, self.wht_params, inverse=True)

       def quantize(self, x):
           y = self.rotate(x)
           return self.codebook.quantize(y)

       def dequantize(self, indices):
           y_hat = self.codebook.centroids[indices]
           return self.unrotate(y_hat)
   ```

4. **Default configuration change:**

   In `TurboQuantCache.__init__` and `AsymmetricTurboQuantCache.__init__`,
   change the default to `mse_only=True` for the HF integration path.
   The `TurboQuantEstimator` (MSE+QJL) remains available for benchmarking
   and for any future fused-attention path that can use the two-part
   representation directly.

**Testing plan:**

- Unit tests: WHT rotation preserves norms, codebook quantize/dequantize roundtrip
- Integration test: Generate 200 tokens with Qwen2.5-3B using WHT+MSE-only cache
  at 4-bit keys, 2-bit values. Compare output coherence to FP16 baseline.
- Quality metrics: cosine similarity of attention scores, top-1/top-5 match,
  perplexity on a held-out passage.

**Kill criteria:**
- If 4-bit MSE-only keys produce garbled output beyond 200 tokens -> escalate to
  5-bit or FP16 keys (Wave 2/3)
- If WHT rotation gives worse quality than QR rotation at same bit-width -> fall
  back to QR (the variance reduction claim may not hold for all architectures)
- Time limit: 1 day for implementation + testing. If not producing coherent
  200-token generation in 1 day, move to Wave 2.

---

### Spec B: Norm Correction (Wave 1B)

**Goal:** Fix the magnitude distortion in MSE reconstruction at zero decode cost.

**The problem:** When we dequantize `k_hat = unrotate(centroids[idx])`, the resulting
vector has a different norm than the original normalized vector (||k_hat|| != 1.0).
We then multiply by the stored original norm: `k_recon = k_hat * stored_norm`. But
since `||k_hat|| != 1`, the final vector has the wrong magnitude.

**The fix:** Instead of storing `||k_original||`, store `||k_original|| / ||k_hat||`.
Then `k_recon = k_hat * corrected_norm` has the correct magnitude.

**Implementation:** One-line change in the compression path.

**File: `turboquantdc/polarquant.py`** (or wherever norms are stored)

```python
# BEFORE (current):
key_norms = keys_flat.norm(dim=-1, keepdim=True)
keys_normalized = keys_flat / (key_norms + 1e-8)
key_indices = pq.quantize(keys_normalized)
# Store key_norms

# AFTER (with norm correction):
key_norms = keys_flat.norm(dim=-1, keepdim=True)
keys_normalized = keys_flat / (key_norms + 1e-8)
key_indices = pq.quantize(keys_normalized)
# Compute reconstruction norm for correction
key_recon = pq.dequantize(key_indices)
recon_norms = key_recon.norm(dim=-1, keepdim=True)
corrected_norms = key_norms / (recon_norms + 1e-8)
# Store corrected_norms instead of key_norms
```

**Cost analysis:**
- Compression: One extra dequantize + norm computation per vector (negligible,
  happens once during prefill/update)
- Decompression: Zero extra cost (same multiply-by-norm as before)
- Storage: Same (one FP16 scalar per vector)

**Testing:**
- Measure perplexity before/after on Qwen2.5-3B at 3-bit and 4-bit
- Expected: -1% to -2% perplexity improvement (per llama.cpp findings)

**Kill criteria:**
- If perplexity does not improve or worsens -> revert (1 hour experiment)

---

### Spec C: Fused MSE Attention Kernel (Wave 1C)

**Goal:** Compute attention scores directly from compressed key indices without
materializing FP16 key tensors. This eliminates the dequantize->matmul path
where rounding errors from centroid lookup + inverse rotation accumulate.

**The algebraic trick:**

```
score = <q, k_recon>
      = <q, unrotate(centroids[idx]) * norm>
      = norm * <q, R^T * centroids[idx]>        (unrotate = multiply by R^T for QR,
                                                   or inverse WHT)
      = norm * <R * q, centroids[idx]>           (orthogonality: <Ax, y> = <x, A^Ty>)
      = norm * sum_j (R*q)_j * centroids[idx_j]  (dot product decomposition)
```

So: pre-rotate queries ONCE, then for each key position, the score is just
`norm * sum(rotated_query * centroids[indices])`. This is a table lookup + dot
product, no inverse rotation needed per key.

**File: `turboquantdc/fused_mse_attention.py`** (NEW, ~200 lines)

```python
def fused_mse_attention(
    query_states: torch.Tensor,       # (batch, n_heads, seq_q, d)
    key_indices: torch.Tensor,        # (batch, n_kv_heads, seq_kv, d) uint8
    key_norms: torch.Tensor,          # (batch, n_kv_heads, seq_kv) fp16
    value_states: torch.Tensor,       # (batch, n_kv_heads, seq_kv, d) fp16
    rotation_matrix: torch.Tensor,    # (d, d) orthogonal
    centroids: torch.Tensor,          # (n_levels,) fp32
    scale: float = None,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Compute attention with scores derived directly from compressed keys.

    Instead of: dequantize keys -> Q @ K^T -> softmax -> @ V
    Does:       rotate Q -> gather centroids -> fused dot -> softmax -> @ V

    This avoids inverse rotation error and is more memory efficient
    (uint8 indices instead of fp16 keys in the inner loop).
    """
    batch, n_q_heads, seq_q, d = query_states.shape
    n_kv_heads = key_indices.shape[1]
    seq_kv = key_indices.shape[2]
    heads_per_kv = n_q_heads // n_kv_heads

    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # Step 1: Pre-rotate all queries (one matmul, amortized)
    # q_rot = q @ R^T  (for QR rotation)
    # or q_rot = WHT(D * q) / sqrt(d)  (for WHT rotation)
    q_rot = torch.matmul(query_states.float(), rotation_matrix.T)

    # Step 2: Gather centroids for all key positions
    # key_centroids[b, h, s, j] = centroids[key_indices[b, h, s, j]]
    key_centroids = centroids[key_indices.long()]  # (batch, n_kv, seq_kv, d)

    # Step 3: Compute scores via dot product in rotated space
    if heads_per_kv > 1:
        key_centroids = key_centroids.repeat_interleave(heads_per_kv, dim=1)
        key_norms_exp = key_norms.repeat_interleave(heads_per_kv, dim=1)
    else:
        key_norms_exp = key_norms

    # (batch, n_q_heads, seq_q, d) @ (batch, n_q_heads, d, seq_kv)
    scores = torch.matmul(q_rot, key_centroids.float().transpose(-1, -2))

    # Scale by key norms and attention scale
    scores = scores * key_norms_exp.float().unsqueeze(2) * scale

    # Step 4: Mask + softmax + value weighted sum (standard)
    if attention_mask is not None:
        scores = scores + attention_mask.float()

    weights = F.softmax(scores, dim=-1, dtype=torch.float32)

    if heads_per_kv > 1:
        v_exp = value_states.repeat_interleave(heads_per_kv, dim=1)
    else:
        v_exp = value_states

    output = torch.matmul(weights.to(v_exp.dtype), v_exp)
    return output
```

**Triton kernel version** (for performance):

The inner loop (centroid gather + dot) is memory-bound and benefits from Triton
fusion. The existing `_inner_product_kernel` in `triton_kernels.py` already does
this for the MSE+QJL case. Strip the QJL terms for an MSE-only variant:

```python
@triton.jit
def _mse_attention_kernel(
    q_rot_ptr,          # (n_queries, d) pre-rotated queries
    indices_ptr,        # (n_keys, d) uint8/int32 codebook indices
    centroids_ptr,      # (n_centroids,) centroid values
    norms_ptr,          # (n_keys,) key norms (corrected)
    output_ptr,         # (n_queries, n_keys) attention scores
    attn_scale,         # 1/sqrt(d)
    n_keys,
    d: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (n_queries, ceil(n_keys / BLOCK_K))
    q_idx = tl.program_id(0)
    k_block = tl.program_id(1)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < d
    q_rot = tl.load(q_rot_ptr + q_idx * d + d_offs, mask=d_mask, other=0.0)

    for ki in range(BLOCK_K):
        k_idx = k_block * BLOCK_K + ki
        if k_idx < n_keys:
            key_idx = tl.load(indices_ptr + k_idx * d + d_offs, mask=d_mask, other=0)
            key_cents = tl.load(centroids_ptr + key_idx, mask=d_mask, other=0.0)
            ip_mse = tl.sum(key_cents * q_rot, axis=0)
            norm_val = tl.load(norms_ptr + k_idx)
            result = ip_mse * norm_val * attn_scale
            tl.store(output_ptr + q_idx * n_keys + k_idx, result)
```

**Testing:**
- Verify fused scores match dequantize->matmul scores to FP32 precision
- Benchmark: should be faster than dequantize path (fewer memory reads)
- Integration: wire into `patch_model_attention` as an alternative to
  `turboquant_attention`

**Kill criteria:**
- If fused kernel gives DIFFERENT scores than dequantize->matmul (beyond FP32
  rounding) -> debug the centroid gather logic
- If no speed improvement -> still useful for correctness (eliminates inverse
  rotation rounding error), keep it

---

## 4. Expected Outcomes

### Wave 1 (parallel, 1-2 days)

| Approach | Expected Quality | Expected Compression | Confidence |
|----------|-----------------|---------------------|------------|
| 1A: WHT+4bit-MSE keys, 2bit values | Coherent 500+ tokens | 4.5x | HIGH (community validated) |
| 1A: WHT+3bit-MSE keys, 2bit values | Coherent 200+ tokens | 5.8x | MEDIUM |
| 1B: Norm correction | -1% perplexity | Same | HIGH (llama.cpp validated) |
| 1C: Fused MSE attention | Same quality, 2x speed | Same | MEDIUM |
| **Combined 1A+1B+1C** | **Coherent 500+ tokens** | **4.5x** | **HIGH** |

### Wave 2 (after Wave 1, 2-3 days)

| Approach | Expected Quality | Expected Compression | Confidence |
|----------|-----------------|---------------------|------------|
| 2A: Temperature calibration | +5-10% top-k match | Same | MEDIUM |
| 2B: Variance-reduced QJL | Coherent at 3-bit | 5.0x | LOW-MEDIUM |

### Wave 3 (research, if needed)

| Approach | Expected Quality | Expected Compression | Confidence |
|----------|-----------------|---------------------|------------|
| 3A: FP16 anchor layers | Near-perfect | 3.5x | HIGH |
| 3B: Per-head adaptive bits | +2-5% quality | 4.5-5.5x | MEDIUM |

### Target quality levels

| Metric | Minimum Acceptable | Good | Excellent |
|--------|-------------------|------|-----------|
| Coherent tokens | 200 | 500 | 1000+ |
| Cosine similarity | 0.993 | 0.996 | 0.999 |
| Top-5 attention match | 85% | 92% | 97% |
| Perplexity increase | <5% | <2% | <0.5% |

---

## 5. Kill Criteria

### Per-approach kill criteria

| Approach | Kill if... | Time limit |
|----------|-----------|------------|
| 1A: WHT+MSE-only | 4-bit keys still garbled at 200 tokens | 1 day |
| 1B: Norm correction | Perplexity does not improve | 2 hours |
| 1C: Fused attention | Scores differ from dequant path by >0.1% | 1 day |
| 2A: Temperature | No entropy match improvement after 4 configs | 1 day |
| 2B: VR-QJL | Variance still >50% of standard QJL | 2 days |
| 3A: FP16 anchors | Compression ratio drops below 2.5x | 1 day |
| 3B: Adaptive bits | Calibration pass takes >30s per model | 1 day |

### Project-level kill criteria

- If Wave 1 (all three approaches combined) does not produce coherent 200-token
  generation at >=4x compression, escalate to "higher bits everywhere" (5-6 bit
  keys, 3-4 bit values) which is guaranteed to work but at lower compression.

- If no approach achieves coherent 500-token generation at >=3.5x compression
  within 1 week, the conclusion is that TurboQuant at this bit-width is not
  suitable for drop-in autoregressive generation, and the project should pivot
  to prefill-only compression (compress historical KV cache, keep recent tokens
  at FP16) -- which is the temporal decay approach already implemented.

---

## 6. What hackimov/turboquant-kv Does Differently

From the web research, hackimov's repo achieves working generation by:

1. **Uses TurboQuantProd (MSE+QJL) but with a FUSED attention kernel** that feeds
   the two-part representation (indices + signs) directly into attention score
   computation. Never materializes a single reconstructed key vector.

2. **Mandatory fused backend** -- without the fused kernel, it falls back to stock
   HF attention (which means dequantize, which means MSE-only effectively).

3. **Architecture-specific fallbacks** -- sliding-window layers and DeepSeek MLA
   fall back to standard attention automatically.

4. **Calibration mechanisms** -- `calibrate_turboquant_from_tensor` adapts
   quantization parameters to actual data distributions.

The key insight: hackimov shows that TurboQuantProd (MSE+QJL) CAN work for
generation IF AND ONLY IF the attention kernel consumes the compressed
representation directly. The problem is not QJL itself -- it is the
dequantize-then-multiply path that destroys the unbiasedness guarantee.

**Implication for our plan:** Wave 1C (fused attention from compressed indices) is
more important than it might seem. It is not just a speed optimization -- it is
potentially a correctness fix. If we can compute scores directly from the compressed
representation (as hackimov does), QJL might work after all.

However, the simpler path (MSE-only at higher bits) is more robust and does not
require a custom attention kernel, so it remains the primary recommendation.

---

## 7. What DEJAN's Blog Found

DEJAN's "From Paper to Triton Kernel in One Session" discovered:

1. **TurboQuant_mse works for drop-in cache** (dequantize to FP16, standard attention)
2. **TurboQuant_prod works ONLY with a custom attention kernel** that uses the
   two-part representation directly
3. **Adding QJL correction back to the reconstructed vector produces garbage** --
   cosine similarity drops to 0.69, model output is incoherent
4. **The fused Triton kernel** pre-rotates queries and does centroid table lookups,
   achieving 17.8x speedup on microbenchmarks

This confirms our analysis: the QJL correction is mathematically correct but
operationally destructive when materialized as a single vector. It must be consumed
as a separate correction term in the attention score computation.

---

## 8. Builder Agent Task Assignments

### Agent 1: WHT + MSE-Only Cache (Wave 1A)

**Input:** This plan, existing `rotation.py` (WHT functions), `polarquant.py`,
`asymmetric.py` (for the asymmetric K/V pattern)

**Output:**
- `turboquantdc/wht_mse_cache.py` -- New WHT-rotated MSE-only cache
- Modified `turboquantdc/hf_integration.py` -- `mse_only=True` default
- Tests in `tests/test_wht_mse_cache.py`
- Generation test: 200 tokens with Qwen2.5-3B

**Acceptance:** Coherent 200-token generation at 4-bit keys, 2-bit values

### Agent 2: Norm Correction (Wave 1B)

**Input:** This plan, existing `hf_integration.py`, `asymmetric.py`

**Output:**
- Modified compression path in `TurboQuantLayer.update()` and
  `AsymmetricTurboQuantLayer.update()`
- Perplexity comparison test

**Acceptance:** Measurable perplexity improvement (any amount)

### Agent 3: Fused MSE Attention (Wave 1C)

**Input:** This plan, existing `custom_attention.py`, `triton_kernels.py`

**Output:**
- `turboquantdc/fused_mse_attention.py` -- PyTorch reference implementation
- Modified `_mse_attention_kernel` in `triton_kernels.py` -- Triton version
- Score comparison test (fused vs dequantize path)

**Acceptance:** Fused scores match dequantize scores to 4+ significant figures

---

## 9. Decision Matrix

After Wave 1, use this matrix to decide next steps:

```
IF coherent at 200 tokens AND >= 4x compression:
    -> SUCCESS. Optimize (Waves 2-3) for more tokens and higher compression.

IF coherent at 200 tokens BUT < 4x compression:
    -> Partial success. Try Wave 2B (VR-QJL) to reduce key bits.
    -> Try Wave 3B (per-head adaptive) to save bits on easy heads.

IF NOT coherent at 200 tokens:
    -> Wave 1C fused attention is critical. Does fused path work?
        IF YES: QJL works when consumed directly. Build fused-only path.
        IF NO:  Escalate to 5-6 bit keys. If still fails, pivot to
                temporal-decay-only (compress old tokens, keep recent FP16).

IF coherent at 500+ tokens AND >= 4x compression:
    -> Declare victory. Ship it. Write the blog post.
```

---

## Appendix: Community Evidence Summary

| Source | Finding | Relevance |
|--------|---------|-----------|
| [llama.cpp #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) | MSE-only beats MSE+QJL; WHT > QR rotation; norm correction -1.17% PPL | PRIMARY |
| [hackimov/turboquant-kv](https://github.com/hackimov/turboquant-kv) | Fused kernel makes Prod work; mandatory fused backend | KEY INSIGHT |
| [DEJAN blog](https://dejan.ai/blog/turboquant/) | QJL back-to-vector = garbage; fused kernel = 17.8x speedup | CONFIRMS |
| [0xSero/turboquant](https://github.com/0xSero/turboquant) | Hybrid decode path doesn't use fused kernels yet; value quality is bottleneck | VALIDATES |
| [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) | Reference implementation, complete working code | REFERENCE |
| [scos-lab/turboquant](https://github.com/scos-lab/turboquant) | 8-model benchmark; K/V norm ratios guide bit allocation | DATA |
| [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) | Layer-adaptive, sparse V, norm correction on Metal | TECHNIQUES |
| [vLLM PR #38280](https://github.com/vllm-project/vllm/pull/38280) | Production integration attempt, Phase 1 working | VALIDATES |
| [sglang #21618](https://github.com/sgl-project/sglang/issues/21618) | Feature request with engineering analysis | CONTEXT |
| [TurboQuant.net](https://turboquant.net/) | Independent analysis and benchmarks | VALIDATES |
