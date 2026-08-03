"""
ternary_matmul_dot.py

Diagnostic kernel: same ternary weights, same overall matmul, but combines
X and W using tl.dot() instead of the tl.where broadcast-and-select trick.
tl.dot uses tensor cores -- this reintroduces "real" multiplication on
purpose, specifically so we can measure how much of the current slowness is
"no tensor cores" vs. something else in the kernel structure.

This is NOT meant to replace your multiplication-free kernel. It's a
control condition for isolating hypothesis 1 (tensor core usage) from
everything else.

Run:
    D:\\Ternary-QuantEdge> python ternary_matmul_dot.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def ternary_matmul_dot_kernel(
    x_ptr, w_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + (offs_k[None, :] + k0) * stride_xk)
        w_ptrs = w_ptr + ((offs_k[:, None] + k0) * stride_wk + offs_n[None, :] * stride_wn)

        x_mask = (offs_m[:, None] < M) & ((offs_k[None, :] + k0) < K)
        w_mask = ((offs_k[:, None] + k0) < K) & (offs_n[None, :] < N)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)          # fp16
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0).to(tl.float16)  # ternary values, cast to fp16 so tl.dot can use tensor cores

        acc = tl.dot(x_tile, w_tile, acc)  # real multiply, tensor-core path -- this is the control condition

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def ternary_matmul_dot(x: torch.Tensor, w_ternary: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16, "x must be fp16 for tl.dot / tensor cores"
    M, K = x.shape
    K2, N = w_ternary.shape
    assert K == K2

    y = torch.empty((M, N), device=x.device, dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    ternary_matmul_dot_kernel[grid](
        x, w_ternary, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_ternary.stride(0), w_ternary.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return (y.float() * scale).half()


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


if __name__ == "__main__":
    from ternary_matmulKernel import ternary_matmul_fp16  # your existing tl.where version

    device = "cuda"
    data = torch.load("ternary_weight_sample.pt")
    w_ternary = data["w_ternary"].to(device).t().contiguous()  # (K, N)
    scale = data["scale"].to(device)
    K, N = w_ternary.shape

    torch.manual_seed(0)

    # ---- correctness check for the new tl.dot version ----
    M = 17
    x16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_dense16 = (w_ternary.float() * scale).half()
    y_ref = x16 @ w_dense16
    y_dot = ternary_matmul_dot(x16, w_ternary, scale)
    rel_err = (y_dot.float() - y_ref.float()).norm() / y_ref.float().norm()
    print(f"[tl.dot version] Relative error vs dense reference: {rel_err.item():.6f}  "
          f"{'PASS' if rel_err < 1e-2 else 'FAIL'}")

    # ---- speed comparison: tl.where kernel vs tl.dot kernel, same sizes as benchmark_suite.py ----
    print(f"\n{'K':>6} {'N':>6} | {'tl.where (ms)':>14} {'tl.dot (ms)':>12}")
    print("-" * 40)

    sizes = [(128, 128), (768, 768), (768, 3072), (3072, 768), (4096, 4096)]
    M = 32
    for K_, N_ in sizes:
        w_t = torch.randint(-1, 2, (K_, N_), dtype=torch.int8, device=device)
        s = torch.tensor(0.02, device=device)
        x = torch.randn(M, K_, device=device, dtype=torch.float16)

        t_where = benchmark(lambda: ternary_matmul_fp16(x, w_t, s))
        t_dot = benchmark(lambda: ternary_matmul_dot(x, w_t, s))

        print(f"{K_:>6} {N_:>6} | {t_where:>14.4f} {t_dot:>12.4f}")