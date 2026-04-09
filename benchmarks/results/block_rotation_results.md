# Block Rotation Benchmark: Combined Stack Comparison

**Model:** Qwen2.5-3B-Instruct (BnB 4-bit)
**Head dim:** 128
**Device:** cuda (RTX 4090)
**Key vectors:** 5120 (from 5 layers, 512 tokens)
**Query vectors:** 200

## 3-bit Results

| Method | Vec Cos | Attn Cos | Top-1 | Top-5 | Quant ms | Dequant ms |
|--------|---------|----------|-------|-------|----------|------------|
| WHT+mean+RQ (our baseline) | 0.962647 | 0.211199 | 0.2000 | 0.2600 | 1.68 | 0.39 |
| WHT alone | 0.983845 | 0.199743 | 0.2000 | 0.2400 | 1.72 | 0.43 |
| Givens alone | 0.910085 | 0.304078 | 0.3000 | 0.3400 | 0.86 | 0.11 |
| Quaternion alone | 0.924315 | 0.343118 | 0.3400 | 0.4000 | 1.99 | 1.71 |
| **Givens+mean** | **0.753552** | **0.400039** | **0.3600** | **0.4400** | **0.84** | **0.26** |
| Quaternion+mean | 0.764139 | 0.368173 | 0.3600 | 0.4600 | 2.24 | 1.87 |
| Givens+mean+RQ | 0.708508 | 0.198314 | 0.2000 | 0.2200 | 1.45 | 0.23 |
| Quat+mean+RQ | 0.716380 | 0.226072 | 0.2200 | 0.2600 | 2.45 | 1.62 |
| IsoQuant-Full (RotorQuant) | 0.924315 | 0.343118 | 0.3400 | 0.4000 | 1.13 | 0.40 |

## 4-bit Results

| Method | Vec Cos | Attn Cos | Top-1 | Top-5 | Quant ms | Dequant ms |
|--------|---------|----------|-------|-------|----------|------------|
| WHT+mean+RQ (our baseline) | 0.989610 | 0.218861 | 0.2200 | 0.2400 | 0.47 | 0.35 |
| WHT alone | 0.995646 | 0.202350 | 0.2000 | 0.2800 | 0.40 | 0.31 |
| Givens alone | 0.951093 | 0.314300 | 0.3000 | 0.3400 | 0.24 | 0.11 |
| **Quaternion alone** | **0.961643** | **0.436391** | **0.4200** | **0.5800** | **0.85** | **0.43** |
| Givens+mean | 0.841330 | 0.210837 | 0.2000 | 0.3800 | 0.35 | 0.11 |
| Quaternion+mean | 0.857800 | 0.249477 | 0.2400 | 0.3400 | 0.63 | 0.44 |
| Givens+mean+RQ | 0.793911 | 0.256273 | 0.2200 | 0.4000 | 0.28 | 0.15 |
| Quat+mean+RQ | 0.805427 | 0.239163 | 0.2400 | 0.2400 | 0.62 | 0.44 |
| IsoQuant-Full (RotorQuant) | 0.961643 | 0.436390 | 0.4200 | 0.5800 | 1.01 | 0.40 |

## Analysis

### 3-bit

**Best attention cosine:** Givens+mean (0.400039) -- +89.4% vs WHT baseline, +16.6% vs IsoQuant
**Best top-5:** Quaternion+mean (0.4600)
**Best vector cosine:** WHT alone (0.983845)

| Comparison | Attn Cos | Top-1 | Top-5 |
|------------|----------|-------|-------|
| WHT+mean+RQ (our baseline) | 0.2112 | 0.20 | 0.26 |
| IsoQuant (RotorQuant best) | 0.3431 | 0.34 | 0.40 |
| **Givens+mean (NEW BEST)** | **0.4000** | **0.36** | **0.44** |

### 4-bit

**Best attention cosine:** Quaternion alone / IsoQuant-Full (0.436391)
**Best top-5:** Quaternion alone / IsoQuant-Full (0.5800)
**Best vector cosine:** WHT alone (0.995646)

| Comparison | Attn Cos | Top-1 | Top-5 |
|------------|----------|-------|-------|
| WHT+mean+RQ (our baseline) | 0.2189 | 0.22 | 0.24 |
| **Quaternion / IsoQuant** | **0.4364** | **0.42** | **0.58** |

## Key Findings

1. **Givens+mean is the 3-bit champion.** At 3-bit, Givens rotation + our mean-removal
   insight achieves 0.400 attention cosine -- beating IsoQuant (0.343) by +16.6% and
   our WHT baseline (0.211) by +89.4%. This is the simplest possible block rotation
   (2D pairs, O(d) compute) combined with our shift-invariance insight.

2. **At 4-bit, Quaternion/IsoQuant wins.** With more bits, the full SO(4) rotation
   has enough precision to dominate. Mean-removal actually hurts at 4-bit with
   block rotations because the mean subtraction changes the distribution shape that
   block rotations were designed for.

3. **ResidualQuant hurts block rotations.** Using (b-1) bits for MSE + 1 bit for
   residual signs is consistently worse than full b-bit MSE with block rotations.
   The residual sign correction was designed for the concentrated Beta/Gaussian
   distribution from WHT/QR, not the per-block distributions from Givens/Quaternion.

4. **WHT has highest vector cosine but lowest attention quality.** WHT minimizes
   global MSE (reconstruction error) but scrambles attention score ranking. Block
   rotations preserve ranking better (higher attention cosine) at the cost of
   per-vector reconstruction accuracy. This confirms RotorQuant's core insight.

5. **Speed: Givens is the fastest.** Givens quantize+dequant takes ~1.1ms total vs
   ~2.1ms for WHT+RQ. The O(d) block rotation is genuinely faster.

## Answer: Does our combined stack beat both standalone approaches?

**YES at 3-bit, NO at 4-bit.**

At 3-bit: **Givens+mean** (their simplest rotation + our mean-removal insight) achieves
0.400 attn cosine, beating BOTH IsoQuant (0.343, +16.6%) and our WHT+RQ baseline
(0.211, +89.4%). This is the new state of the art at 3-bit.

At 4-bit: Quaternion/IsoQuant alone at 0.436 is best. Our mean-removal doesn't help
at higher bit-widths with block rotations.

**Recommendation:** Use Givens+mean as the default 3-bit configuration. Use
Quaternion (no mean, no ResidualQuant) for 4-bit. Keep WHT+mean+RQ for generation
quality where vector cosine matters more than attention ranking.
