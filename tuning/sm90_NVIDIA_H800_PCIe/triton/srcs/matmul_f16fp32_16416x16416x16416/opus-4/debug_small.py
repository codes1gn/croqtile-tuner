#!/usr/bin/env python3
import torch
import triton
import triton.language as tl

M_small, N_small, K_small = 1024, 1024, 1024
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
GROUP_M = 8
num_stages = 3
num_warps = 4


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def matmul(a, b):
    M_dim, K_dim = a.shape
    K_dim2, N_dim = b.shape
    c = torch.empty((M_dim, N_dim), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M_dim, META['BLOCK_M']) * triton.cdiv(N_dim, META['BLOCK_N']),)
    matmul_kernel[grid](
        a, b, c,
        M_dim, N_dim, K_dim,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_stages=num_stages, num_warps=num_warps,
    )
    return c


torch.manual_seed(0)
print(f"Testing small matrix: {M_small}x{N_small}x{K_small}")
a = torch.randn((M_small, K_small), device='cuda', dtype=torch.float16)
b = torch.randn((K_small, N_small), device='cuda', dtype=torch.float16)
print(f"a range: [{a.min():.4f}, {a.max():.4f}]")
print(f"b range: [{b.min():.4f}, {b.max():.4f}]")
torch.cuda.synchronize()

print("Running triton kernel...")
triton_out = matmul(a, b)
torch.cuda.synchronize()
print(f"triton_out shape: {triton_out.shape}")

print("Running torch reference...")
torch_out = torch.matmul(a.float(), b.float()).half()
torch.cuda.synchronize()

diff = (triton_out - torch_out).abs()
print(f'triton range: [{triton_out.min():.4f}, {triton_out.max():.4f}]')
print(f'torch range: [{torch_out.min():.4f}, {torch_out.max():.4f}]')
print(f'diff max: {diff.max():.6f}, mean: {diff.mean():.6f}')
print("SUCCESS" if diff.max() < 1.0 else "FAILURE")
