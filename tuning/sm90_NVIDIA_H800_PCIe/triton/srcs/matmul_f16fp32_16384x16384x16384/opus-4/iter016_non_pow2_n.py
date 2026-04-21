#!/usr/bin/env python3
"""
iter016_non_pow2_n: Try BLOCK_N=192 for different memory alignment
Shape: 16384x16384x16384
Config: BLOCK_M=128, BLOCK_N=192, BLOCK_K=64, num_stages=3, num_warps=8
Hypothesis: Non-power-of-2 tile might avoid bank conflicts or L2 conflicts
"""

import torch
import triton
import triton.language as tl

M, N, K = 16384, 16384, 16384
BLOCK_M = 128
BLOCK_N = 192
BLOCK_K = 64
GROUP_SIZE_M = 8
num_stages = 3
num_warps = 8


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.shape[1] == B.shape[0]
    M_dim, K_dim = A.shape
    K_dim, N_dim = B.shape
    C = torch.empty((M_dim, N_dim), device=A.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M_dim, META['BLOCK_M']) * triton.cdiv(N_dim, META['BLOCK_N']),)
    matmul_kernel[grid](
        A, B, C,
        M_dim, N_dim, K_dim,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return C


def verify():
    torch.manual_seed(42)
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    C_triton = matmul_triton(A, B)
    C_ref = torch.matmul(A.float(), B.float())
    
    max_abs_err = (C_triton - C_ref).abs().max().item()
    rel_err = max_abs_err / C_ref.abs().max().item()
    
    tol = 1e-2
    if max_abs_err < tol or rel_err < 1e-3:
        print(f"VERIFY: PASS max_abs_err={max_abs_err:.6f} rel_err={rel_err:.6f}")
        return True
    else:
        print(f"VERIFY: FAIL max_abs_err={max_abs_err:.6f} rel_err={rel_err:.6f}")
        return False


def bench(warmup=10, iters=50):
    torch.manual_seed(42)
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    for _ in range(warmup):
        _ = matmul_triton(A, B)
    
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        _ = matmul_triton(A, B)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iters
    tflops = 2 * M * N * K / elapsed_ms / 1e9
    print(f"TFLOPS: {tflops:.2f}   time_ms: {elapsed_ms:.3f}")
    return tflops, elapsed_ms


if __name__ == "__main__":
    print(f"Triton matmul FP16->FP32: {M}x{N}x{K}")
    print(f"Config: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
    print(f"        num_stages={num_stages}, num_warps={num_warps}, GROUP_SIZE_M={GROUP_SIZE_M}")
    
    if verify():
        bench()
