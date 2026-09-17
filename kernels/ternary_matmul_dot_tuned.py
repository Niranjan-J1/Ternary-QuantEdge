"""
ternary_matmul_dot_tuned.py

Same tl.dot-based ternary kernel as ternary_matmul_dot.py, but with
triton.autotune sweeping block sizes and launch parameters, since the
roofline analysis showed the untuned version achieving only ~7.5% of peak
memory bandwidth at 4096x4096 vs. cuBLAS's ~45% -- clear sign of
implementation inefficiency, not a fundamental limit.

triton.autotune works by trying every listed config on the first call for a
given input shape, timing each, and caching the winner for subsequent calls
with that same shape. This means the FIRST call per shape will be slow
(it's running the whole sweep) -- exclude it from any timing you report,
same warm-up discipline as before, just more of it.

"""

import torch
import triton
import triton.language as tl

from ternary_matmul_dot import ternary_matmul_dot  # untuned baseline, for comparison


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],  # re-tune (and re-cache) whenever the problem shape changes
)
@triton.jit
def ternary_matmul_dot_tuned_kernel(
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

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0).to(tl.float16)

        acc = tl.dot(x_tile, w_tile, acc)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def ternary_matmul_dot_tuned(x: torch.Tensor, w_ternary: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    M, K = x.shape
    K2, N = w_ternary.shape
    assert K == K2

    y = torch.empty((M, N), device=x.device, dtype=torch.float16)
    # No BLOCK_* passed here -- autotune supplies them based on the configs above.
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    ternary_matmul_dot_tuned_kernel[grid](
        x, w_ternary, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_ternary.stride(0), w_ternary.stride(1),
        y.stride(0), y.stride(1),
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
    device = "cuda"
    data = torch.load("ternary_weight_sample.pt")
    w_ternary = data["w_ternary"].to(device).t().contiguous()
    scale = data["scale"].to(device)
    K, N = w_ternary.shape

    torch.manual_seed(0)

    # ---- correctness check ----
    M = 17
    x16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_dense16 = (w_ternary.float() * scale).half()
    y_ref = x16 @ w_dense16
    y_tuned = ternary_matmul_dot_tuned(x16, w_ternary, scale)
    rel_err = (y_tuned.float() - y_ref.float()).norm() / y_ref.float().norm()
    print(f"[tuned] Relative error vs dense reference: {rel_err.item():.6f}  "
          f"{'PASS' if rel_err < 1e-2 else 'FAIL'}")
    print(f"[diagnostic] Winning config for shape ({M},{K},{N}): "
          f"{ternary_matmul_dot_tuned_kernel.best_config}")

    # ---- untuned vs tuned, same sizes used in the roofline sweep ----
    print(f"\n{'K':>6} {'N':>6} | {'untuned (ms)':>13} {'tuned (ms)':>11} {'speedup':>8}")
    print("-" * 48)

    sizes = [(128, 128), (768, 768), (768, 3072), (3072, 768),
              (2048, 2048), (3584, 3584), (4096, 4096), (6144, 6144)]
    M = 32
    for K_, N_ in sizes:
        w_t = torch.randint(-1, 2, (K_, N_), dtype=torch.int8, device=device)
        s = torch.tensor(0.02, device=device)
        x = torch.randn(M, K_, device=device, dtype=torch.float16)

        # Note: first call per shape includes the autotune sweep itself --
        # call once, discarded, before timing, so we're not measuring
        # autotune overhead as if it were steady-state performance.
        _ = ternary_matmul_dot_tuned(x, w_t, s)
        torch.cuda.synchronize()

        print(f"[diagnostic] shape ({M},{K_},{N_}) winning config: "
              f"{ternary_matmul_dot_tuned_kernel.best_config}")

        # Explicit call-by-call timing (not averaged) to check whether
        # autotune is re-tuning on every call instead of caching.
        individual_times = []
        for i in range(5):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = ternary_matmul_dot_tuned(x, w_t, s)
            end.record()
            torch.cuda.synchronize()
            individual_times.append(start.elapsed_time(end))
        print(f"[diagnostic] first 5 individual call times (ms): "
              f"{[f'{t:.4f}' for t in individual_times]}")

        t_untuned = benchmark(lambda: ternary_matmul_dot(x, w_t, s))
        t_tuned = benchmark(lambda: ternary_matmul_dot_tuned(x, w_t, s))

        print(f"{K_:>6} {N_:>6} | {t_untuned:>13.4f} {t_tuned:>11.4f} {t_untuned/t_tuned:>7.2f}x")
