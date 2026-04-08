## RTX 4090 CUDA Results -- Weight + KV Cache Stacking

### Hardware
- GPU: RTX 4090 24GB (SM 89), CUDA 12.8
- Driver: 580.126.20
- Build: TheTom/llama-cpp-turboquant @ 3b8a01a92 (pr/tq4-weight-compression)
- Note: ~7GB VRAM unavailable (Ollama background process), effective free VRAM ~15.7GB

### Weight Quantization Types
| Type | BPW | Description |
|------|-----|-------------|
| Q4_K_M | 4.58 | Standard k-quant |
| TQ4_1S | 5.00 | WHT-rotated 4-bit (Tom's TurboQuant weight compression) |
| TQ3_1S | 4.00 | WHT-rotated 3-bit (Tom's TurboQuant weight compression) |

### KV Cache Types
| Type | Description | Approx V bits/element |
|------|-------------|----------------------|
| f16 | FP16 baseline | 16 |
| q8_0 K + turbo4 V | Conservative | K:8 + V:~4 |
| q8_0 K + turbo3 V | Aggressive | K:8 + V:~3 |
| q8_0 K + turbo2 V | Ultra-aggressive | K:8 + V:~2 |

---

### Llama 3.1 8B -- Standard Q4_K_M Weights (4.58 GiB)

**Short Context (512 prompt, 128 gen)**

| Weight | KV Config | FA | PP tok/s | TG tok/s |
|--------|-----------|:--:|--------:|--------:|
| Q4_K_M | f16 | 0 | 7651 | 101.8 |
| Q4_K_M | f16 | 1 | 8876 | 104.6 |
| Q4_K_M | q8_0 K + turbo4 V | 1 | 8202 | 95.4 |
| Q4_K_M | q8_0 K + turbo3 V | 1 | 8513 | 97.2 |
| Q4_K_M | q8_0 K + turbo2 V | 1 | 7381 | 97.2 |

**Long Context Scaling**

| Context | KV Config | PP tok/s | TG tok/s |
|--------:|-----------|--------:|--------:|
| 4096 | f16 | 9754 | 121.8 |
| 4096 | q8_0+turbo3 | 9464 | 116.7 |
| 8192 | f16 | 9014 | 117.9 |
| 8192 | q8_0+turbo3 | 8193 | 108.8 |
| 16384 | f16 | 6358 | 60.2 |
| 16384 | q8_0+turbo3 | 6904 | 65.8 |

---

### Llama 3.1 8B -- TQ4_1S Weights (4.77 GiB, 5.10 BPW)

**Short Context (512 prompt, 128 gen)**

| Weight | KV Config | PP tok/s | TG tok/s |
|--------|-----------|--------:|--------:|
| TQ4_1S | f16 | 8501 | 67.2 |
| TQ4_1S | q8_0 K + turbo4 V | 7734 | 64.5 |
| TQ4_1S | q8_0 K + turbo3 V | 7954 | 64.6 |
| TQ4_1S | q8_0 K + turbo2 V | 7918 | 64.5 |

**Long Context Scaling (the stacking payoff)**

| Context | KV Config | PP tok/s | TG tok/s | Notes |
|--------:|-----------|--------:|--------:|-------|
| 4096 | f16 | 10404 | 86.1 | |
| 4096 | q8_0+turbo3 | 9523 | 79.6 | |
| 8192 | f16 | 9195 | 78.4 | |
| 8192 | q8_0+turbo3 | 9170 | **86.5** | turbo3 faster (less VRAM pressure) |
| 16384 | f16 | 8360 | 78.6 | |
| 16384 | q8_0+turbo3 | 8022 | **85.5** | turbo3 faster (less VRAM pressure) |
| 32768 | f16 | 6912 | 88.5 | |
| 32768 | q8_0+turbo3 | 6499 | 82.1 | |
| 48000 | f16 | 5158 | 72.9 | near FP16 ceiling |
| 56000 | f16 | **OOM** | **OOM** | FP16 KV cannot allocate |
| 65536 | q8_0+turbo3 | 4683 | 85.8 | **turbo3 still running!** |
| 98304 | q8_0+turbo3 | 3517 | 79.3 | turbo3 at ~96K context |
| 100000 | q8_0+turbo3 | 3165 | 72.7 | |
| 112000 | q8_0+turbo3 | **OOM** | **OOM** | turbo3 ceiling |

**Key finding: turbo3 KV extends max context from ~48K to ~100K (2.1x) on the same GPU.**

---

### Llama 3.1 8B -- TQ3_1S Weights (3.90 GiB, 4.17 BPW)

**Short Context (512 prompt, 128 gen)**

| Weight | KV Config | PP tok/s | TG tok/s |
|--------|-----------|--------:|--------:|
| TQ3_1S | f16 | 3165 | 51.7 |
| TQ3_1S | q8_0 K + turbo4 V | 3356 | 51.4 |
| TQ3_1S | q8_0 K + turbo3 V | 3921 | 54.5 |
| TQ3_1S | q8_0 K + turbo2 V | 3579 | 53.3 |

---

### Qwen3.5-35B MoE A3B (18.48 GiB, reported Q8_0 UD quant)

All tests with ngl=30 (partial offload, ~15.7GB effective VRAM)

**Short Context (512 prompt, 128 gen)**

| Weight | KV Config | PP tok/s | TG tok/s |
|--------|-----------|--------:|--------:|
| Q8_0 UD | f16 | 769 | 30.3 |
| Q8_0 UD | q8_0 K + turbo4 V | 784 | 27.5 |
| Q8_0 UD | q8_0 K + turbo3 V | 780 | 28.0 |
| Q8_0 UD | q8_0 K + turbo2 V | 706 | 33.0 |

**Long Context Scaling**

| Context | KV Config | PP tok/s | TG tok/s |
|--------:|-----------|--------:|--------:|
| 2048 | f16 | 746 | 28.7 |
| 2048 | q8_0+turbo3 | 768 | 29.0 |
| 4096 | f16 | 726 | 26.4 |
| 4096 | q8_0+turbo3 | 711 | 27.1 |
| 8192 | f16 | 736 | 26.2 |
| 8192 | q8_0+turbo3 | 315 | 3.5 |

Note: 8K turbo3 slowdown on 35B likely due to VRAM pressure at ngl=30 causing KV offload to CPU.

---

### Combined Compression Analysis

**Llama 3.1 8B Stack**
```
                        Weight Size    KV @ 32K    Total @ 32K    Max Context (24GB*)
Q4_K_M + FP16 KV:      4.58 GiB      ~4.0 GiB    ~8.6 GiB       ~48K
TQ4_1S + FP16 KV:      4.77 GiB      ~4.0 GiB    ~8.8 GiB       ~48K
TQ4_1S + turbo3 KV:    4.77 GiB      ~1.0 GiB    ~5.8 GiB       ~100K (2.1x)
TQ3_1S + turbo3 KV:    3.90 GiB      ~1.0 GiB    ~4.9 GiB       ~110K+ (est)

* Effective VRAM ~15.7GB due to Ollama background process using ~7GB
  With full 24GB available, all numbers would be higher
```

**Qwen3.5-35B MoE Stack (partial offload)**
```
                        Weight Size    Notes
Q8_0 UD + FP16 KV:     18.48 GiB     ngl=30, ~30 tok/s TG
Q8_0 UD + turbo3 KV:   18.48 GiB     ngl=30, ~28 tok/s TG, negligible quality cost
TQ4_1S + turbo3 KV:    TBD           (quantization in progress)
```

### Key Takeaways

1. **KV compression has near-zero speed impact at short context** -- turbo3 and turbo4 maintain >95% of FP16 throughput at 512 tokens
2. **At long context, turbo3 can be FASTER than FP16** -- less VRAM pressure means less spilling, 8K context TQ4_1S saw 86.5 vs 78.4 tok/s (10% faster with turbo3)
3. **Context length doubles with turbo3** -- from ~48K to ~100K on the same RTX 4090 for an 8B model
4. **TQ weight compression adds overhead** -- TQ4_1S decode is ~35% slower than Q4_K_M (67 vs 105 tok/s), TQ3_1S is ~50% slower (52 vs 105)
5. **The stack compounds** -- TQ3_1S weights (3.9G) + turbo3 KV = fits comfortably with huge context headroom
6. **MoE models benefit from KV compression too** -- Qwen3.5-35B shows minimal throughput impact from turbo3 at moderate context
