"""


measures peak GPU memory usage for each
implementation, across the same M values and matrix sizes already used in
crossover_and_roofline.py. 

Uses torch.cuda.max_memory_allocated(), reset before each measurement so
each number reflects only that implementation's peak usage, not a running
total contaminated by previous implementations in the same process.


"""

import torch
import bitsandbytes as bnb

from ternary_matmul_dot import ternary_matmul_dot


def measure_peak_memory_mb(fn, warmup=3):
    """Runs fn a few times to reach steady state, then measures peak memory
    for one additional call in isolation."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated()
    return peak_bytes / (1024 ** 2)  # MB


def run_one_size(M, K, N, device="cuda"):
    results = {}

    # ---- FP16 dense ----
    x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device=device, dtype=torch.float16)
    results["fp16_dense"] = measure_peak_memory_mb(lambda: x_fp16 @ w_fp16)

    # ---- INT8 ----
    layer_int8 = bnb.nn.Linear8bitLt(K, N, has_fp16_weights=False, bias=False).to(device)
    x_i8 = torch.randn(M, K, device=device, dtype=torch.float16)
    results["bnb_int8"] = measure_peak_memory_mb(lambda: layer_int8(x_i8))

    # ---- INT4 ----
    layer_int4 = bnb.nn.Linear4bit(K, N, bias=False, compute_dtype=torch.float16).to(device)
    x_i4 = torch.randn(M, K, device=device, dtype=torch.float16)
    results["bnb_int4"] = measure_peak_memory_mb(lambda: layer_int4(x_i4))

    # ---- Ternary (tl.dot version) ----
    w_ternary = torch.randint(-1, 2, (K, N), dtype=torch.int8, device=device)
    scale = torch.tensor(0.02, device=device)
    x_t = torch.randn(M, K, device=device, dtype=torch.float16)
    results["ternary_dot"] = measure_peak_memory_mb(lambda: ternary_matmul_dot(x_t, w_ternary, scale))

    return results


if __name__ == "__main__":
    assert torch.cuda.is_available()

    sizes = [
        (128, 128), (768, 768), (768, 3072), (3072, 768),
        (2048, 2048), (3584, 3584), (4096, 4096), (6144, 6144),
    ]
    m_values = [1, 32, 128, 256]  # matches the sweep already run for speed/roofline

    for M in m_values:
        print(f"\n=== M = {M} ===")
        header = f"{'K':>6} {'N':>6} | {'fp16 (MB)':>10} {'int8 (MB)':>10} {'int4 (MB)':>10} {'ternary (MB)':>13}"
        print(header)
        print("-" * len(header))

        for K, N in sizes:
            # Fresh process-level peak tracking per size to avoid one huge
            # earlier allocation making later, smaller ones look identical.
            torch.cuda.empty_cache()
            r = run_one_size(M, K, N)
            print(f"{K:>6} {N:>6} | {r['fp16_dense']:>10.2f} {r['bnb_int8']:>10.2f} "
                  f"{r['bnb_int4']:>10.2f} {r['ternary_dot']:>13.2f}")