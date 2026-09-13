"""
crossover_roofline.py

Sweeps batch sizes M = [1, 32, 128, 256] across matrix sizes to produce
the data for Tables 1-2 and the roofline classification in Sections 4.1-4.2
of the paper. For each (M, K, N) combination, benchmarks all four
implementations (FP16 dense, INT8, INT4, ternary tensor-core kernel) and
computes achieved GFLOP/s, achieved GB/s, arithmetic intensity, and
compute-bound vs memory-bound classification.

RTX 2060 theoretical peaks (Turing TU106, 6 GB GDDR6 variant):
  Memory bandwidth: 336 GB/s (GDDR6 14 Gbps x 192-bit bus)
  FP16 tensor-core throughput (FP32 accumulate): 25.8 TFLOP/s
"""

import torch
import bitsandbytes as bnb

from ternary_matmul_dot import ternary_matmul_dot

PEAK_BANDWIDTH_GBPS = 336.0
PEAK_COMPUTE_TFLOPS = 25.8
RIDGE_POINT = (PEAK_COMPUTE_TFLOPS * 1e3) / PEAK_BANDWIDTH_GBPS


def benchmark(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def roofline_stats(M, K, N, time_ms, x_bytes_per_el, w_bytes_per_el, y_bytes_per_el):
    flops = 2 * M * K * N
    bytes_moved = (M * K * x_bytes_per_el) + (K * N * w_bytes_per_el) + (M * N * y_bytes_per_el)

    time_s = time_ms / 1000.0
    achieved_gflops = (flops / time_s) / 1e9
    achieved_gbps = (bytes_moved / time_s) / 1e9
    arithmetic_intensity = flops / bytes_moved

    bound_type = "compute-bound" if arithmetic_intensity > RIDGE_POINT else "memory-bound"
    return achieved_gflops, achieved_gbps, arithmetic_intensity, bound_type


def run_one_size(M, K, N, device="cuda"):
    results = {}

    # FP16 dense
    x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device=device, dtype=torch.float16)
    t = benchmark(lambda: x_fp16 @ w_fp16)
    results["fp16_dense"] = (t, roofline_stats(M, K, N, t, 2, 2, 2))

    # INT8
    layer_int8 = bnb.nn.Linear8bitLt(K, N, has_fp16_weights=False, bias=False).to(device)
    x_i8 = torch.randn(M, K, device=device, dtype=torch.float16)
    t = benchmark(lambda: layer_int8(x_i8))
    results["bnb_int8"] = (t, roofline_stats(M, K, N, t, 2, 1, 2))

    # INT4
    layer_int4 = bnb.nn.Linear4bit(K, N, bias=False, compute_dtype=torch.float16).to(device)
    x_i4 = torch.randn(M, K, device=device, dtype=torch.float16)
    t = benchmark(lambda: layer_int4(x_i4))
    results["bnb_int4"] = (t, roofline_stats(M, K, N, t, 2, 0.5, 2))

    # Ternary (tl.dot / tensor-core)
    w_ternary = torch.randint(-1, 2, (K, N), dtype=torch.int8, device=device)
    scale = torch.tensor(0.02, device=device)
    x_t = torch.randn(M, K, device=device, dtype=torch.float16)
    t = benchmark(lambda: ternary_matmul_dot(x_t, w_ternary, scale))
    results["ternary_dot"] = (t, roofline_stats(M, K, N, t, 2, 1, 2))

    return results


if __name__ == "__main__":
    assert torch.cuda.is_available()

    m_values = [1, 32, 128, 256]
    sizes = [
        (128, 128),
        (768, 768),
        (768, 3072),
        (3072, 768),
        (2048, 2048),
        (3584, 3584),
        (4096, 4096),
        (6144, 6144),
    ]

    print(f"Ridge point (compute/memory-bound boundary): {RIDGE_POINT:.2f} FLOPs/byte\n")

    for M in m_values:
        print(f"\n{'='*80}")
        print(f"  M = {M}")
        print(f"{'='*80}")

        header = (f"{'K':>6} {'N':>6} {'impl':>12} | "
                  f"{'ms':>8} {'GFLOP/s':>10} {'GB/s':>8} {'AI':>8} {'bound':>13}")
        print(header)
        print("-" * len(header))

        for K, N in sizes:
            torch.cuda.empty_cache()
            r = run_one_size(M, K, N)
            for impl_name, (t_ms, (gflops, gbps, ai, bound)) in r.items():
                print(f"{K:>6} {N:>6} {impl_name:>12} | "
                      f"{t_ms:>8.4f} {gflops:>10.1f} {gbps:>8.1f} {ai:>8.2f} {bound:>13}")
            print()
