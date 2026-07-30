import torch
import triton
import triton.language as tl


@triton.jit
def ternary_matmul_kernel(
    x_ptr, w_ptr, y_ptr,
    M, N, K, ## matmul dimensions
    stride_xm, stride_xk, # for each tenspr how much memory posn u jump as you jump in dimension
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
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0).to(tl.int32)

        x_b = x_tile[:, :, None]
        w_b = w_tile[None, :, :]
        contrib = tl.where(w_b == 1, x_b, tl.where(w_b == -1, -x_b, 0.0))
        acc += tl.sum(contrib, axis=1)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


def ternary_matmul(x: torch.Tensor, w_ternary: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    K2, N = w_ternary.shape
    assert K == K2

    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    ternary_matmul_kernel[grid](
        x, w_ternary, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_ternary.stride(0), w_ternary.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return y * scale


def ternary_matmul_fp16(x: torch.Tensor, w_ternary: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    K2, N = w_ternary.shape
    assert K == K2

    y = torch.empty((M, N), device=x.device, dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    ternary_matmul_kernel[grid](
        x, w_ternary, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_ternary.stride(0), w_ternary.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return y * scale

if __name__ == "__main__":
    device = "cuda"
    data = torch.load("ternary_weight_sample.pt")
    w_ternary = data["w_ternary"].to(device).t().contiguous()  # (K, N)
    scale = data["scale"].to(device)
    K, N = w_ternary.shape

    M = 17
    torch.manual_seed(0)
    x = torch.randn(M, K, device=device, dtype=torch.float32)

    w_dense = w_ternary.float() * scale
    y_ref = x @ w_dense
    y_kernel = ternary_matmul(x, w_ternary, scale)

    rel_err = (y_kernel - y_ref).norm() / y_ref.norm()
    print(f"Relative error vs dense reference: {rel_err.item():.6f}")
    print("PASS" if rel_err < 1e-4 else "FAIL")