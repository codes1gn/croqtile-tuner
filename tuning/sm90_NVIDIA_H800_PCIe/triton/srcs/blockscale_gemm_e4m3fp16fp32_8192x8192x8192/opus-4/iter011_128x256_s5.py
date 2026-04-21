#!/usr/bin/env python3
"""
iter011_128x256_s5.py — Triton FP8 e4m3 GEMM with 128x256, 5 stages
Shape: 8192x8192x8192
Input: e4m3, Output: fp16, Accumulator: fp32
More pipeline stages for better latency hiding
"""
import torch
import triton
import triton.language as tl

M, N, K = 8192, 8192, 8192
BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 64
GROUP_M = 8
num_stages = 5
num_warps = 8


@triton.jit
def blockscale_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    a_scale = tl.load(a_scale_ptr + offs_am, mask=offs_am < M, other=1.0)
    b_scale = tl.load(b_scale_ptr + offs_bn, mask=offs_bn < N, other=1.0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc * a_scale[:, None] * b_scale[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def blockscale_gemm(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M_dim, K_dim = a.shape
    K_dim2, N_dim = b.shape
    c = torch.empty((M_dim, N_dim), device=a.device, dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M_dim, META['BLOCK_M']) * triton.cdiv(N_dim, META['BLOCK_N']),)
    blockscale_gemm_kernel[grid](
        a, b, c,
        a_scale, b_scale,
        M_dim, N_dim, K_dim,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_stages=num_stages, num_warps=num_warps,
    )
    return c


def verify():
    torch.manual_seed(0)
    a_fp32 = torch.randn((M, K), device='cuda', dtype=torch.float32) * 0.1
    b_fp32 = torch.randn((K, N), device='cuda', dtype=torch.float32) * 0.1

    a_scale = torch.rand((M,), device='cuda', dtype=torch.float32) * 2 + 0.5
    b_scale = torch.rand((N,), device='cuda', dtype=torch.float32) * 2 + 0.5

    a_fp8 = a_fp32.to(torch.float8_e4m3fn)
    b_fp8 = b_fp32.to(torch.float8_e4m3fn)

    triton_out = blockscale_gemm(a_fp8, b_fp8, a_scale, b_scale)

    raw_out = torch.matmul(a_fp8.float(), b_fp8.float())
    torch_out = (raw_out * a_scale[:, None] * b_scale[None, :]).half()

    diff = (triton_out - torch_out).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    nonzero_mask = torch_out.abs() > 0.001
    if nonzero_mask.any():
        rel_err = diff[nonzero_mask] / torch_out[nonzero_mask].abs()
        max_rtol = rel_err.max().item()
        mean_rtol = rel_err.mean().item()
    else:
        max_rtol = 0.0
        mean_rtol = 0.0

    passed = max_err < 0.5 and mean_rtol < 0.02
    if passed:
        print(f"VERIFY: PASS max_abs_err={max_err:.6f} mean_err={mean_err:.6f} mean_rtol={mean_rtol:.4f}")
        return True
    else:
        print(f"VERIFY: FAIL max_abs_err={max_err:.6f} mean_err={mean_err:.6f} mean_rtol={mean_rtol:.4f}")
        return False


def bench(warmup=10, iters=50):
    torch.manual_seed(0)
    a_fp32 = torch.randn((M, K), device='cuda', dtype=torch.float32) * 0.1
    b_fp32 = torch.randn((K, N), device='cuda', dtype=torch.float32) * 0.1

    a_scale = torch.rand((M,), device='cuda', dtype=torch.float32) * 2 + 0.5
    b_scale = torch.rand((N,), device='cuda', dtype=torch.float32) * 2 + 0.5

    a_fp8 = a_fp32.to(torch.float8_e4m3fn)
    b_fp8 = b_fp32.to(torch.float8_e4m3fn)

    for _ in range(warmup):
        _ = blockscale_gemm(a_fp8, b_fp8, a_scale, b_scale)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = blockscale_gemm(a_fp8, b_fp8, a_scale, b_scale)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    flops = 2 * M * N * K
    tflops = flops / elapsed_ms / 1e9
    print(f"TFLOPS: {tflops:.2f}   time_ms: {elapsed_ms:.3f}")
    return tflops


if __name__ == "__main__":
    if verify():
        bench()
