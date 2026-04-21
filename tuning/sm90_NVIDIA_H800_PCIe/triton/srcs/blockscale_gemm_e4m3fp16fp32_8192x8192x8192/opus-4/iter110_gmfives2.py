#!/usr/bin/env python3
"""iter110_gmfives2.py — 256x64 BK=256 GM=5 W=8 S=2"""
import torch, triton, triton.language as tl
M, N, K = 8192, 8192, 8192
BLOCK_M, BLOCK_N, BLOCK_K = 256, 64, 256
GROUP_M, num_stages, num_warps = 5, 2, 8

@triton.jit
def kernel(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0); num_pid_m = tl.cdiv(M, BLOCK_M); num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n; group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M; group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m); pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    a_scale = tl.load(a_scale_ptr + offs_am, mask=offs_am < M, other=1.0); b_scale = tl.load(b_scale_ptr + offs_bn, mask=offs_bn < N, other=1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32); a_ptrs += BLOCK_K * stride_ak; b_ptrs += BLOCK_K * stride_bk
    acc = acc * a_scale[:, None] * b_scale[None, :]
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def gemm(a, b, a_scale, b_scale):
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(a.shape[0], META['BLOCK_M']) * triton.cdiv(b.shape[1], META['BLOCK_N']),)
    kernel[grid](a, b, c, a_scale, b_scale, a.shape[0], b.shape[1], a.shape[1], a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M, num_stages=num_stages, num_warps=num_warps)
    return c

if __name__ == "__main__":
    torch.manual_seed(0)
    a = (torch.randn((M, K), device='cuda', dtype=torch.float32) * 0.1).to(torch.float8_e4m3fn)
    b = (torch.randn((K, N), device='cuda', dtype=torch.float32) * 0.1).to(torch.float8_e4m3fn)
    a_scale = torch.rand((M,), device='cuda', dtype=torch.float32) * 2 + 0.5
    b_scale = torch.rand((N,), device='cuda', dtype=torch.float32) * 2 + 0.5
    out = gemm(a, b, a_scale, b_scale)
    ref = (torch.matmul(a.float(), b.float()) * a_scale[:, None] * b_scale[None, :]).half()
    max_err = (out - ref).abs().max().item()
    passed = max_err < 0.5
    if passed:
        for _ in range(10): gemm(a, b, a_scale, b_scale)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50): gemm(a, b, a_scale, b_scale)
        end.record(); torch.cuda.synchronize()
        tflops = 2 * M * N * K / (start.elapsed_time(end) / 50) / 1e9
        print(f"VERIFY: PASS max_err={max_err:.4f} TFLOPS: {tflops:.2f}")
    else: print(f"VERIFY: FAIL max_err={max_err:.4f}")
