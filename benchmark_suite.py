"""
cCompares four matmul implementations across a range of matrix sizes:
  1. Dense FP16 (plain torch.matmul)
  2. bitsandbytes INT8 (Linear8bitLt)
  3. bitsandbytes INT4 (Linear4bit)
  4. Custom ternary Triton kernel (ternary_matmulKernel.py)

  Measures average wall clock time per call using CUDA events 

  all 4 paths run on fp 16 i/O, essentially "ternary vs multiply" is what is being measured

"""


import torch
import bitsandbytes as bnb
 
from ternary_matmul_dot import ternary_matmul_dot  # your verified fp16 kernel
 
 
def benchmark(fn, warmup=10, iters=50):
    """Times a zero-arg callable on GPU using CUDA events. Returns avg ms/call."""
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
 
    total_ms = start.elapsed_time(end)
    return total_ms / iters
 
 
def make_ternary_weight(K, N, device):
    """Random ternary weight matrix for benchmarking (correctness already
    verified separately -- here we only care about timing, so random values
    of the right shape/dtype are fine)."""
    w = torch.randint(-1, 2, (K, N), dtype=torch.int8, device=device)
    scale = torch.tensor(0.02, device=device)  # representative magnitude, doesn't affect timing
    return w, scale
 
 
def run_one_size(M, K, N, device="cuda"):
    results = {}
 
    # ---- 1. Dense FP16 ----
    x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device=device, dtype=torch.float16)
    results["fp16_dense"] = benchmark(lambda: x_fp16 @ w_fp16)
 
    # ---- 2. bitsandbytes INT8 ----
    layer_int8 = bnb.nn.Linear8bitLt(K, N, has_fp16_weights=False, bias=False).to(device)
    x_int8_input = torch.randn(M, K, device=device, dtype=torch.float16)
    results["bnb_int8"] = benchmark(lambda: layer_int8(x_int8_input))
 
    # ---- 3. bitsandbytes INT4 ----
    layer_int4 = bnb.nn.Linear4bit(K, N, bias=False, compute_dtype=torch.float16).to(device)
    x_int4_input = torch.randn(M, K, device=device, dtype=torch.float16)
    results["bnb_int4"] = benchmark(lambda: layer_int4(x_int4_input))
 
    # ---- 4. Ternary kernel (fp16 variant, for a fair comparison) ----
    w_ternary, scale = make_ternary_weight(K, N, device)
    x_ternary = torch.randn(M, K, device=device, dtype=torch.float16)
    results["ternary_kernel"] = benchmark(lambda: ternary_matmul_dot(x_ternary, w_ternary, scale))
 
    return results
 
 
if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA GPU required."
 
    # Matrix sizes to sweep. M is fixed (representative batch/sequence chunk);
    # K, N vary to include your real Pythia-160m layer shapes plus a couple
    # of extra points on each side, to help locate a crossover.
    M = 32
    sizes = [
        (128, 128),
        (768, 768),
        (768, 3072),   # matches dense_h_to_4h
        (3072, 768),   # matches dense_4h_to_h
        (4096, 4096),
    ]
 
    print(f"{'K':>6} {'N':>6} | {'fp16 (ms)':>10} {'int8 (ms)':>10} {'int4 (ms)':>10} {'ternary (ms)':>13}")
    print("-" * 65)
 
    for K, N in sizes:
        r = run_one_size(M, K, N)
        print(f"{K:>6} {N:>6} | {r['fp16_dense']:>10.4f} {r['bnb_int8']:>10.4f} "
              f"{r['bnb_int4']:>10.4f} {r['ternary_kernel']:>13.4f}")