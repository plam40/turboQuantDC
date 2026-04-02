"""Speed benchmark: Triton kernels vs Python for TurboQuantDC.

Measures quantize and dequantize throughput at various:
- Sequence lengths (1, 10, 100, 1000 tokens)
- Head dimensions (64, 128, 256)
- Bit widths (2, 3, 4)
"""
import torch, time, sys
sys.path.insert(0, ".")

from turboquantdc.codebook import LloydMaxCodebook
from turboquantdc.polarquant import PolarQuant
from turboquantdc.rotation import generate_rotation_matrix

def benchmark_quantize(d=128, bits=3, n_vectors=1000, warmup=10, repeats=100):
    """Benchmark quantize throughput."""
    pq = PolarQuant(d, bits, seed=42, device="cuda")
    x = torch.randn(n_vectors, d, device="cuda")

    # Warmup
    for _ in range(warmup):
        pq.quantize(x)
    torch.cuda.synchronize()

    # Measure
    t0 = time.perf_counter()
    for _ in range(repeats):
        pq.quantize(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    vectors_per_sec = n_vectors * repeats / elapsed
    return vectors_per_sec

def benchmark_dequantize(d=128, bits=3, n_vectors=1000, warmup=10, repeats=100):
    """Benchmark dequantize throughput."""
    pq = PolarQuant(d, bits, seed=42, device="cuda")
    x = torch.randn(n_vectors, d, device="cuda")
    indices = pq.quantize(x)

    # Warmup
    for _ in range(warmup):
        pq.dequantize(indices)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        pq.dequantize(indices)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    vectors_per_sec = n_vectors * repeats / elapsed
    return vectors_per_sec

def benchmark_generation_cache(d=128, key_bits=3, val_bits=3, seq_len=100, n_heads=4):
    """Benchmark full GenerationCache update+dequantize cycle."""
    from turboquantdc.generation_cache import GenerationCache

    cache = GenerationCache(
        key_bits=key_bits, val_bits=val_bits,
        anchor_strategy="boundary", fp16_window=0,
        use_residual_quant=True, num_layers=4,
    )

    # Fill cache to seq_len
    for t in range(seq_len):
        k = torch.randn(1, n_heads, 1, d, device="cuda")
        v = torch.randn(1, n_heads, 1, d, device="cuda")
        cache.update(k, v, layer_idx=0)

    # Benchmark one more update (quantize + dequantize all)
    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        k = torch.randn(1, n_heads, 1, d, device="cuda")
        v = torch.randn(1, n_heads, 1, d, device="cuda")
        t0 = time.perf_counter()
        cache.update(k, v, layer_idx=0)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "p50_ms": sorted(times)[len(times)//2] * 1000,
        "p99_ms": sorted(times)[int(len(times)*0.99)] * 1000,
    }

# Also try the Triton path if available
def benchmark_triton_quantize(d=128, bits=3, n_vectors=1000, warmup=10, repeats=100):
    try:
        from turboquantdc.triton_kernels import TritonTurboQuant
        ttq = TritonTurboQuant(d=d, bits=bits, qjl_dim=d, seed=42)
        x = torch.randn(n_vectors, d, device="cuda")

        for _ in range(warmup):
            ttq.quantize(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(repeats):
            ttq.quantize(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return n_vectors * repeats / elapsed
    except Exception as e:
        return f"FAILED: {e}"

if __name__ == "__main__":
    print("TurboQuantDC Speed Benchmark")
    print("=" * 60)

    for d in [128]:
        for bits in [3, 4]:
            for n in [100, 1000]:
                py_q = benchmark_quantize(d, bits, n)
                py_dq = benchmark_dequantize(d, bits, n)
                tri_q = benchmark_triton_quantize(d, bits, n)
                print(f"d={d} bits={bits} n={n}:")
                print(f"  Python quantize:  {py_q:>12,.0f} vec/s")
                print(f"  Python dequant:   {py_dq:>12,.0f} vec/s")
                print(f"  Triton quantize:  {tri_q}")

    print()
    print("GenerationCache (full update cycle):")
    for seq in [10, 100, 500]:
        stats = benchmark_generation_cache(seq_len=seq)
        print(f"  seq={seq}: {stats['mean_ms']:.2f}ms mean, {stats['p50_ms']:.2f}ms p50")
