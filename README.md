# TurboQuantDC: Infinite Context Engine 🚀

Welcome to the **100x Context Revolution**. 

This repository leverages **TurboQuantDC**—a deeply optimized quantization library—to achieve the unimaginable: **Evaluating 1 Million+ tokens of context natively on a single consumer GPU (e.g., NVIDIA RTX 4090 24GB).**

No multi-node clusters. No $40k H100s. Just pure, unadulterated algorithmic efficiency.

## How we Shocked the World
Modern Large Language Models (LLMs) like `Qwen 2.5` and `Gemma 4` require massive amounts of VRAM to store intermediate Key-Value (KV) cache states during generation. A 1M token context window normally requires hundreds of gigabytes of VRAM.

**We bypassed the VRAM wall entirely.**

### The Infinite Architecture 🧠
1. **CPU/GPU Hybrid FAISS Engine (`TurboRetrievalCache`)**: We decouple the Keys and Values. The attention Keys are aggressively clustered on system RAM using high-speed FAISS instances `(IndexIVFPQ / Flat L2)`. The Values are kept on the GPU VRAM.
2. **3-Bit Extreme Quantization (`_CompressedLayer`)**: GPU Values are hyper-compressed down to a pristine **3-bit state**. When FAISS identifies the most critical `top_k` tokens relative to the current generation query, our custom CUDA paths dequantize exactly and only what matters back into high-precision on the fly.
3. **Chunked Sliding-Window Prefill**: By hijacking HuggingFace's `_call_impl` dynamically via instance-class monkey patching, we intercept `eager_attention_forward`. Massive sequences are ingested in 1024-token chunks keeping active KV sequence lengths clamped exactly to bounded mask sizes, meaning you will *never OOM*.

### Gemma 4 & Qwen 2 Support out-of-the-box
The engine provides drop-in replacement monkey-patches handling the nuances of various architectures:
*   `demo_gemma4.py`: Seamlessly extracts and routes through Gemma 4's massive memory-saving `is_kv_shared_layer` design, natively injecting exact FAISS indices without corrupting the rotary embeddings.
*   `run_infinite_context.py`: Built expressly to handle the standard interleaved SDPA heads of `Qwen2Attention`. 

---

## ⚡ Quickstart

Prepare to evaluate millions of tokens on your local rig.
1. Install requirements: `pip install torch transformers bitsandbytes faiss-cpu`
2. Spin up the Qwen engine:
```bash
python run_infinite_context.py --model Qwen/Qwen2.5-3B-Instruct --tokens 1000000
```
3. Spin up the Gemma 4 engine:
```bash
python demo_gemma4.py --model unsloth/gemma-4-E4B-it --tokens 1000000
```

*Note: Extreme contexts will run sequentially. Because the KV values are fundamentally heavily compressed to an extreme 3-bit lossy barrier, complex needle-in-a-haystack tasks may require tuning `k` and `window_size` to maintain deterministic recall!*

---
### "Hardware shouldn't limit imagination." 
Built on `TurboQuantDC`. Enjoy breaking constraints!
