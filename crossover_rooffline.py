"""
crossover_roofline.py

Two things in one script, since they're both needed before writing
Results/Discussion:

1. EXTENDED SWEEP: fills in sizes around the region where ternary and INT8
   appeared to cross over (2048, 3584, 6144 added to your original five),
   so the crossover point can be pinned down precisely rather than guessed
   at between two widely-spaced data points.

2. ROOFLINE-STYLE ANALYSIS: for each implementation and size, computes
   achieved GFLOP/s and achieved memory bandwidth (GB/s), then compares
   against your GPU's theoretical peaks to determine whether each run is
   memory-bound or compute-bound. This turns "tensor cores matter" from a
   hypothesis into a grounded, numeric explanation for Discussion.

IMPORTANT: the PEAK_* constants below are commonly cited RTX 2060 specs,
but you should confirm the exact numbers for your card before quoting them
in the paper -- run `nvidia-smi -q` or check NVIDIA's official spec sheet.
Getting a peer-reviewable hardware spec wrong is an easy, avoidable mistake.


"""

import torch
import bitsandbytes as bnb

from ternary_matmulKernel import ternary_matmul_fp16
from ternary_matmul_dot import ternary_matmul_dot

# ---------------------------------------------------------------------------
# RTX 2060 theoretical peaks -- VERIFY THESE before using in the paper.
# Commonly cited figures (Turing TU106, 6GB variant):
#   Memory bandwidth: ~336 GB/s
#   FP16 tensor-core compute (FP16 accumulate): ~51.6 TFLOP/s
#   FP16 tensor-core compute (FP32 accumulate, what tl.dot with acc=fp32 uses): ~25.8 TFLOP/s
# Using the FP32-accumulate figure since that's what your kernel actually does.
# ---------------------------------------------------------------------------
PEAK_BANDWIDTH_GBPS = 336.0
PEAK_COMPUTE_TFLOPS = 25.8
RIDGE_POINT = (PEAK_COMPUTE_TFLOPS * 1e3) / PEAK_BANDWIDTH_GBPS  # FLOPs/byte where compute-bound begins


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
    return start.elapsed_time(end) / iters  # ms/call


def roofline_stats(M, K, N, time_ms, x_bytes_per_el, w_bytes_per_el, y_bytes_per_el):
    """Returns (achieved_gflops, achieved_gbps, arithmetic_intensity, bound_type)."""
    flops = 2 * M * K * N  # one multiply + one add per MAC
    bytes_moved = (M * K * x_bytes_per_el) + (K * N * w_bytes_per_el) + (M * N * y_bytes_per_el)

    time_s = time_ms / 1000.0
    achieved_gflops = (flops / time_s) / 1e9
    achieved_gbps = (bytes_moved / time_s) / 1e9
    arithmetic_intensity = flops / bytes_moved  # FLOPs per byte

    bound_type = "compute-bound" if arithmetic_intensity > RIDGE_POINT else "memory-bound"
    return achieved_gflops, achieved_gbps, arithmetic_intensity, bound_type


def run_one_size(M, K, N, device="cuda"):
    results = {}

    # ---- FP16 dense ----
    x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device=device, dtype=torch.float16)
    t = benchmark(lambda: x_fp16 @ w_fp16)
    results["fp16_dense"] = (t, roofline_stats(M, K, N, t, 2, 2, 2))

    # ---- INT8 ----
    layer_int8 = bnb.nn.Linear8bitLt(K, N, has_fp16_weights=False, bias=False).to(device)
    x_i8 = torch.randn(M, K, device=device, dtype=torch.float16)
    t = benchmark(lambda: layer_int8(x_i8))
    results["bnb_int8"] = (t, roofline_stats(M, K, N, t, 2, 1, 2))  # weight ~1 byte/el

    # ---- INT4 ----
    layer_int4 = bnb.nn.Linear4bit(K, N, bias=False, compute_dtype=torch.float16).to(device)
    x_i4 = torch.randn(M, K, device=device, dtype=torch.float16)
    t = benchmark(lambda: layer_int4(x_i4))
    results["bnb_int4"] = (t, roofline_stats(M, K, N, t, 2, 0.5, 2))  # weight ~0.5 byte/el

    # ---- Ternary (tl.dot / tensor-core version) ----
    w_ternary = torch.randint(-1, 2, (K, N), dtype=torch.int8, device=device)
    scale = torch.tensor(0.02, device=device)
    x_t = torch.randn(M, K, device=device, dtype=torch.float16)
    t = benchmark(lambda: ternary_matmul_dot(x_t, w_ternary, scale))
    results["ternary_dot"] = (t, roofline_stats(M, K, N, t, 2, 1, 2))  # weight stored as int8 = 1 byte/el

    return results


if __name__ == "__main__":
    assert torch.cuda.is_available()

    M = 256
    sizes = [
        (128, 128),
        (768, 768),
        (768, 3072),
        (3072, 768),
        (2048, 2048),   # new: fills the gap before the crossover region
        (3584, 3584),   # new: brackets the suspected crossover near 4096
        (4096, 4096),
        (6144, 6144),   # new: confirms whether ternary keeps losing to INT8 as size grows further
    ]

    print(f"Ridge point (compute/memory-bound boundary): {RIDGE_POINT:.2f} FLOPs/byte\n")

    header = f"{'K':>6} {'N':>6} {'impl':>12} | {'ms':>8} {'GFLOP/s':>10} {'GB/s':>8} {'AI':>6} {'bound':>13}"
    print(header)
    print("-" * len(header))

    for K, N in sizes:
        r = run_one_size(M, K, N)
        for impl_name, (t_ms, (gflops, gbps, ai, bound)) in r.items():
            print(f"{K:>6} {N:>6} {impl_name:>12} | {t_ms:>8.4f} {gflops:>10.1f} {gbps:>8.1f} {ai:>6.2f} {bound:>13}")
        print()