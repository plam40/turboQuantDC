# TurboQuantDC Speed Benchmark

Measures quantize and dequantize throughput comparing Triton fused kernels vs
Python/PyTorch paths across various batch sizes and bit-widths.

**Hardware:** NVIDIA RTX 4090 (24 GB, SM 89)
**CUDA:** 12.8.93
**PyTorch:** 2.10.0+cu128
**Date:** 2026-03-31

---

## Quantize / Dequantize Throughput (d=128)

| bits | n_vectors | Python quantize (vec/s) | Python dequant (vec/s) | Triton quantize (vec/s) | Triton speedup |
|------|-----------|------------------------:|----------------------:|------------------------:|---------------:|
| 3    | 100       |               ~327,000  |              ~335,000  |            ~1,720,000   |       **5.3x** |
| 3    | 1000      |             ~2,987,000  |            ~3,601,000  |           ~16,388,000   |       **5.5x** |
| 4    | 100       |               ~327,000  |              ~341,000  |            ~1,440,000   |       **4.4x** |
| 4    | 1000      |             ~2,920,000  |            ~3,644,000  |           ~14,199,000   |       **4.9x** |

**Key finding:** Triton fused kernels deliver **4–6x throughput** over the Python path.
Speedup is most pronounced at large batch sizes (n=1000) where kernel launch
overhead is amortized — Triton reaches **16 M vec/s** at 3-bit / n=1000.

---

## GenerationCache Full Update Cycle (d=128, 4 heads, key_bits=3, val_bits=3)

| Sequence length | Mean latency | p50 latency |
|-----------------|-------------:|------------:|
| seq=10          |     0.03 ms  |    0.03 ms  |
| seq=100         |     0.05 ms  |    0.05 ms  |
| seq=500         |     0.19 ms  |    0.19 ms  |

The GenerationCache update latency (quantize + store one new KV pair) scales
nearly linearly with context length. At seq=500 it stays under 0.2 ms —
well within the budget for real-time generation on an RTX 4090.

---

## Methodology

```
Python path:   PolarQuant.quantize / PolarQuant.dequantize
               (WHT rotation + Lloyd-Max brute-force nearest centroid)

Triton path:   TritonTurboQuant.quantize
               (fused rotate + boundary search + residual + QJL sign,
                single Triton kernel launch per batch)

Warmup:        10 iterations (kernel compilation / GPU warm-up)
Measurement:   100 repeats, wall-clock via time.perf_counter()
               torch.cuda.synchronize() before/after timing window
```

---

## Raw Output

```
TurboQuantDC Speed Benchmark
============================================================
d=128 bits=3 n=100:
  Python quantize:       327,484 vec/s
  Python dequant:        334,742 vec/s
  Triton quantize:  1,720,578 vec/s
d=128 bits=3 n=1000:
  Python quantize:     2,986,825 vec/s
  Python dequant:      3,600,926 vec/s
  Triton quantize:  16,388,391 vec/s
d=128 bits=4 n=100:
  Python quantize:       327,008 vec/s
  Python dequant:        340,512 vec/s
  Triton quantize:   1,440,094 vec/s
d=128 bits=4 n=1000:
  Python quantize:     2,919,560 vec/s
  Python dequant:      3,644,349 vec/s
  Triton quantize:  14,198,832 vec/s

GenerationCache (full update cycle):
  seq=10:  0.03ms mean, 0.03ms p50
  seq=100: 0.05ms mean, 0.05ms p50
  seq=500: 0.19ms mean, 0.19ms p50
```
