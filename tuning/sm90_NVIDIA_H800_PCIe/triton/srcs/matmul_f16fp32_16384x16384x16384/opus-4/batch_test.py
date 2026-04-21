#!/usr/bin/env python3
"""Batch test multiple configurations"""
import torch
import triton
import triton.language as tl
import sys

M, N, K = 16384, 16384, 16384

@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M); num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n; group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M; group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
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
        acc = tl.dot(a, b, acc); a_ptrs += BLOCK_K * stride_ak; b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)

def bench_config(BM, BN, BK, stages, warps, group_size, warmup=5, iters=20):
    torch.manual_seed(42)
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    def matmul_fn():
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_SIZE_M=group_size, num_stages=stages, num_warps=warps)
        return C
    
    try:
        for _ in range(warmup): matmul_fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters): matmul_fn()
        end.record(); torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / iters
        tflops = 2 * M * N * K / elapsed_ms / 1e9
        return tflops
    except Exception as e:
        return 0.0

if __name__ == "__main__":
    configs = [
        (128, 256, 64, 3, 8, 6),   # GROUP_SIZE_M=6
        (128, 256, 64, 3, 8, 7),   # GROUP_SIZE_M=7
        (128, 256, 64, 3, 8, 9),   # GROUP_SIZE_M=9
        (128, 256, 64, 3, 8, 11),  # GROUP_SIZE_M=11
        (128, 256, 64, 3, 8, 14),  # GROUP_SIZE_M=14
        (128, 256, 64, 4, 8, 8),   # 4 stages
        (128, 256, 32, 3, 8, 8),   # K=32
        (128, 256, 64, 2, 8, 8),   # 2 stages
        (128, 256, 64, 3, 8, 2),   # GROUP_SIZE_M=2
        (128, 256, 64, 3, 8, 32),  # GROUP_SIZE_M=32
    ]
    
    results = []
    for cfg in configs:
        BM, BN, BK, stages, warps, group = cfg
        tflops = bench_config(BM, BN, BK, stages, warps, group)
        results.append((cfg, tflops))
        print(f"{BM}x{BN}x{BK} s{stages}w{warps}g{group}: {tflops:.2f} TFLOPS")
    
    print("\n=== Best configs ===")
    for cfg, tflops in sorted(results, key=lambda x: -x[1])[:5]:
        print(f"{cfg}: {tflops:.2f} TFLOPS")
