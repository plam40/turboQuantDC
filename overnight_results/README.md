# TurboQuantDC — Overnight Results

Benchmark scripts and result reports for validating TurboQuantDC KV cache
compression on progressively larger models.

## Experimental Setup

### Hardware
- GPU: NVIDIA RTX 4090 (24 GB VRAM, SM 89, CUDA 12.8)
- CPU RAM: ~64 GB
- PyTorch 2.10.0+cu128 / Python 3.12

### Compression scheme
TurboQuantDC applies two-stage vector quantization to transformer KV caches:

| Stage | What it does |
|-------|-------------|
| Stage 1 — PolarQuant | WHT rotation + Lloyd-Max scalar quantization at b bits/coord |
| Stage 2 — ResidualQuant | 1-bit direct residual sign correction for unbiased inner products |
| Norm correction | Scales reconstructed vectors by original/reconstruction ratio |
| FP16 hot window | Last N tokens stored at full precision to protect recent context |

The default configuration (K3/V3, fp16_window=64, ResidualQuant=on) achieves
~5.1x compression vs FP16 with generation quality matching uncompressed output.

### Key insight for 200B+ models
Models that exceed GPU VRAM are loaded via accelerate's `device_map="auto"` with
`max_memory={0: "Xgib", "cpu": "Ygib"}`. Accelerate handles CPU<->GPU layer
transfers automatically — the model processes one layer at a time, shuttling
weights to GPU and back. TurboQuantDC further reduces VRAM pressure by
compressing the KV cache so the small on-GPU budget is not eaten by KV storage.

---

## Scripts

### `run_phase1.py` — 7B smoke test
Validates TurboQuantDC on Qwen2.5-7B. Fits entirely on a single 4090.

### `run_14b.py` — 14B validation
Runs Qwen2.5-14B with multiple cache configurations and a needle-in-haystack
long-context test. Uses `max_memory={0: "12GiB", "cpu": "32GiB"}`.

### `run_200b.py` — 200B+ ultra-streaming run
Runs any model with most weights on CPU, a tiny GPU slice, and TurboQuantDC
KV compression. Full CLI documented below.

---

## `run_200b.py` — Usage

```bash
# Default: Qwen2.5-72B-Instruct, 6 GB GPU budget, eviction cache
python overnight_results/run_200b.py

# Adjust GPU budget (more GPU = faster, needs free VRAM)
python overnight_results/run_200b.py --gpu-budget 10.0

# Boundary cache instead of eviction
python overnight_results/run_200b.py --kv-strategy boundary

# Different model
python overnight_results/run_200b.py --model Qwen/Qwen2.5-32B-Instruct

# Full options
python overnight_results/run_200b.py --help
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-72B-Instruct` | HuggingFace model ID |
| `--gpu-budget` | `6.0` | GiB of VRAM reserved for model weights |
| `--cpu-budget` | `48.0` | GiB of RAM for CPU-offloaded layers |
| `--kv-strategy` | `eviction` | `boundary` (GenerationCache) or `eviction` (EvictionCache) |
| `--key-bits` | `3` | Bits for key quantization |
| `--val-bits` | `3` | Bits for value quantization |
| `--fp16-window` | `64` | Recent tokens kept at FP16 (hot tier) |
| `--max-new-tokens` | `100` | Tokens to generate per prompt |
| `--seed` | `42` | Random seed |

---

## Model Size Matrix

| Model | Params | 4-bit size | Notes |
|-------|--------|-----------|-------|
| `Qwen/Qwen2.5-72B-Instruct` | 72B | ~40 GB | Primary target, no gating |
| `meta-llama/Llama-3.3-70B-Instruct` | 70B | ~39 GB | Needs `HF_TOKEN` |
| `Qwen/Qwen2.5-32B-Instruct` | 32B | ~18 GB | Fallback / faster iteration |
| `deepseek-ai/DeepSeek-V2-Lite` | 16B | ~9 GB | MoE architecture path |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~4 GB | Quick smoke test |

For genuine 200B+ models (e.g. Qwen2.5-Max, DeepSeek-V3-671B): the script is
architecture-agnostic. Pass `--model <name>` once access/download is available.
The `device_map="auto"` + `max_memory` strategy scales to any size as long as
total CPU RAM + GPU VRAM exceeds the 4-bit model footprint.

---

## Cache Strategy Comparison

| Strategy | Class | Compression | Best for |
|----------|-------|------------|---------|
| `boundary` | `GenerationCache` | ~5.1x | Highest quality, full sequence retention |
| `eviction` | `EvictionCache` | 6-8x | Long contexts, memory pressure |

The eviction cache uses a hybrid recency + structural importance score:
- Prompt tokens (first 20% of sequence) are never evicted
- High-norm keys (attention magnets) are retained
- Low-attention middle tokens are evicted after exceeding `max_warm_tokens=1024`

---

## Expected Output

```
======================================================================
TurboQuantDC — 200B Run Script
======================================================================
Model:       Qwen/Qwen2.5-72B-Instruct
GPU budget:  6.0 GiB
CPU budget:  48.0 GiB
KV strategy: eviction (K3/V3, fp16_window=64)

GPU:         NVIDIA GeForce RTX 4090
VRAM:        23.5 GB free / 24.0 GB total

[1/3] Loading Qwen/Qwen2.5-72B-Instruct...
  num_hidden_layers = 80
  Device placement: {'cuda:0': N, 'cpu': M}
  VRAM after load: 5.8 GB

[2/3] Creating eviction KV cache...
[3/3] Generating ...

============================================================
Prompt 1: What is the capital of France?
Response:  Paris is the capital of France.
Speed:     X.X tok/s
VRAM:      X.XX GB current / X.XX GB peak
```

---

## Results

Generated reports are saved to `run_200b_results.md` after each run.
The 14B validation report is in `phase1_7b.md`.
