#!/usr/bin/env python3
"""
iter003_occupancy_tune.py — TileLang FP16 GEMM with 128x192x64 tiles
Shape: 8192x8192x8192
Input: fp16, Output: fp16, Accumulator: fp32

Changed from 128x128x64 to 128x192x64 for better tile occupancy.
Slightly larger N tile improves WGMMA utilization.
"""
import torch
import tilelang
import tilelang.language as T

M, N, K = 8192, 8192, 8192
BLOCK_M = 128
BLOCK_N = 192
BLOCK_K = 64
SCALE_BLOCK_K = 128
num_stages = 3


def gemm_tilelang(M, N, K, block_M, block_N, block_K, num_stages):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            
            T.use_swizzle(panel_size=10, enable=True)
            
            T.clear(C_local)
            
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return main


def get_kernel():
    func = gemm_tilelang(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages)
    jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda")
    return jit_kernel


_kernel_cache = None


def gemm(a, b):
    global _kernel_cache
    if _kernel_cache is None:
        _kernel_cache = get_kernel()
    c = _kernel_cache(a, b)
    return c


def blockscale_preprocess(a_fp8, b_fp8, a_scale, b_scale):
    num_scale_k = a_scale.shape[1]
    scale_block_k = a_fp8.shape[1] // num_scale_k
    
    a_f16 = a_fp8.to(torch.float16)
    b_f16 = b_fp8.to(torch.float16)
    
    for i in range(num_scale_k):
        k_start = i * scale_block_k
        k_end = k_start + scale_block_k
        a_f16[:, k_start:k_end] = (a_f16[:, k_start:k_end].float() * a_scale[:, i:i+1]).half()
        b_f16[k_start:k_end, :] = (b_f16[k_start:k_end, :].float() * b_scale[:, i:i+1].T).half()
    
    return a_f16, b_f16


def blockscale_gemm(a_fp8, b_fp8, a_scale, b_scale):
    a_f16, b_f16 = blockscale_preprocess(a_fp8, b_fp8, a_scale, b_scale)
    return gemm(a_f16, b_f16)


def verify():
    torch.manual_seed(0)
    
    a_fp32 = torch.randn((M, K), device='cuda', dtype=torch.float32) * 0.1
    b_fp32 = torch.randn((K, N), device='cuda', dtype=torch.float32) * 0.1
    
    num_scale_k = K // SCALE_BLOCK_K
    a_scale = torch.rand((M, num_scale_k), device='cuda', dtype=torch.float32) * 2 + 0.5
    b_scale = torch.rand((N, num_scale_k), device='cuda', dtype=torch.float32) * 2 + 0.5
    
    a_fp8 = a_fp32.to(torch.float8_e4m3fn)
    b_fp8 = b_fp32.to(torch.float8_e4m3fn)
    
    tilelang_out = blockscale_gemm(a_fp8, b_fp8, a_scale, b_scale)
    
    a_scaled_ref = a_fp8.float()
    b_scaled_ref = b_fp8.float()
    
    for i in range(num_scale_k):
        k_start = i * SCALE_BLOCK_K
        k_end = k_start + SCALE_BLOCK_K
        a_scaled_ref[:, k_start:k_end] *= a_scale[:, i:i+1]
        b_scaled_ref[k_start:k_end, :] *= b_scale[:, i:i+1].T
    
    torch_out = torch.matmul(a_scaled_ref, b_scaled_ref).half()
    
    diff = (tilelang_out - torch_out).abs()
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
    
    passed = max_err < 1.0 and mean_rtol < 0.1
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
    
    num_scale_k = K // SCALE_BLOCK_K
    a_scale = torch.rand((M, num_scale_k), device='cuda', dtype=torch.float32) * 2 + 0.5
    b_scale = torch.rand((N, num_scale_k), device='cuda', dtype=torch.float32) * 2 + 0.5
    
    a_fp8 = a_fp32.to(torch.float8_e4m3fn)
    b_fp8 = b_fp32.to(torch.float8_e4m3fn)
    
    a_f16, b_f16 = blockscale_preprocess(a_fp8, b_fp8, a_scale, b_scale)
    
    for _ in range(warmup):
        _ = gemm(a_f16, b_f16)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = gemm(a_f16, b_f16)
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
