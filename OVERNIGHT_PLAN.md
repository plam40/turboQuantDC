# Overnight Experiment Plan — 200B on RTX 4090

## Hardware
- GPU: RTX 4090 24GB (18GB free)
- RAM: 62GB (44GB available)
- Disk: 279GB free
- CUDA 12.8, PyTorch 2.10, transformers 5.3.0

## Phase 1: Validate at 7B (30 min)
- Model: Qwen2.5-7B-Instruct (4-bit BnB, ~4GB)
- KV: boundary K3/V3 + self-correcting cache
- Target: 128K context, measure quality + speed
- Success: coherent generation, >90% of FP16 quality

## Phase 2: Validate at 14B (45 min)
- Model: Qwen2.5-14B-Instruct (4-bit BnB, ~8GB)
- KV: boundary K3/V3 + eviction
- Target: 64K context, measure quality + speed

## Phase 3: Push to 32B (60 min)
- Model: Qwen3-32B (4-bit BnB, ~19GB) or Qwen2.5-32B
- KV: boundary K3/V3 + eviction (only ~3-4GB for KV)
- Target: 32K context minimum

## Phase 4: 70B with offload (90 min)
- Model: Llama-3.3-70B-Instruct (4-bit, partial GPU offload)
- ~50% layers on GPU, rest on CPU
- KV: boundary + eviction (10x effective)
- Target: 32K+ context at >5 tok/s

## Phase 5: 100B+ Push (120 min)
- Model: Llama-4-Scout (109B MoE, 17B active) or Qwen2.5-72B
- Expert/layer offloading + TurboQuantDC KV
- Target: get it running, any speed

## Phase 6: 200B (remaining time)
- Model: Llama-4-Maverick (400B/17B active) or similar
- Full streaming from CPU + compressed KV
- Target: generate at least one coherent response
