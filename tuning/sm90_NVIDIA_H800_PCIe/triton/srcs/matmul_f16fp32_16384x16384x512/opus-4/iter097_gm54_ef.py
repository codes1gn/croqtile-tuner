#!/usr/bin/env python3
import torch, triton, triton.language as tl
M, N, K = 16384, 16384, 512
BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
GROUP_M, num_stages, num_warps = 54, 3, 8

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m, num_pid_n = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
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
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0, eviction_policy="evict_first")
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N), other=0.0, eviction_policy="evict_first")
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def matmul(a, b):
    M_dim, K_dim = a.shape
    _, N_dim = b.shape
    c = torch.empty((M_dim, N_dim), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M_dim, META['BLOCK_M']) * triton.cdiv(N_dim, META['BLOCK_N']),)
    matmul_kernel[grid](a, b, c, M_dim, N_dim, K_dim, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M, num_stages=num_stages, num_warps=num_warps)
    return c

def verify():
    torch.manual_seed(0)
    a, b = torch.randn((M, K), device='cuda', dtype=torch.float16), torch.randn((K, N), device='cuda', dtype=torch.float16)
    diff = (matmul(a, b) - torch.matmul(a.float(), b.float()).half()).abs()
    passed = diff.max().item() < 0.1 and diff.mean().item() < 1e-3
    print(f"VERIFY: {'PASS' if passed else 'FAIL'} max_abs_err={diff.max().item():.6f} mean_err={diff.mean().item():.6f}")
    return passed

def bench(warmup=10, iters=50):
    torch.manual_seed(0)
    a, b = torch.randn((M, K), device='cuda', dtype=torch.float16), torch.randn((K, N), device='cuda', dtype=torch.float16)
    for _ in range(warmup): matmul(a, b)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iters
    tflops = 2 * M * N * K / elapsed_ms / 1e9
    print(f"TFLOPS: {tflops:.2f}   time_ms: {elapsed_ms:.3f}")
    return tflops

if __name__ == "__main__":
    if verify(): bench()
