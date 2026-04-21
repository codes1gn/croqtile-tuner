#!/usr/bin/env python3
"""
iter001_base_blockscale.py — TileLang FP8 e4m3 blockscale GEMM
Shape: 8192x8192x8192
Input: e4m3, Output: fp16, Accumulator: fp32
Per-block scaling with 128-element groups along K

Block-scaled GEMM: For each (M, K_block) tile of A and (K_block, N) tile of B,
we have scaling factors that must be applied before accumulation.
"""
import torch
import tilelang
import tilelang.language as T

M, N, K = 8192, 8192, 8192
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
SCALE_BLOCK_K = 128
num_stages = 3


def blockscale_gemm_tilelang(M, N, K, block_M, block_N, block_K, scale_block_K, num_stages):
    num_scale_k = K // scale_block_K
    
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float8_e4m3"),
        B: T.Tensor((K, N), "float8_e4m3"),
        A_scale: T.Tensor((M, num_scale_k), "float32"),
        B_scale: T.Tensor((N, num_scale_k), "float32"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            
            A_scale_local = T.alloc_fragment((block_M,), "float32")
            B_scale_local = T.alloc_fragment((block_N,), "float32")
            
            T.clear(C_local)
            
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                k_base = ko * block_K
                scale_idx = k_base // scale_block_K
                
                for i in T.Parallel(block_M):
                    A_scale_local[i] = A_scale[by * block_M + i, scale_idx]
                
                for j in T.Parallel(block_N):
                    B_scale_local[j] = B_scale[bx * block_N + j, scale_idx]
                
                for i, k in T.Parallel(block_M, block_K):
                    a_val = T.cast(A[by * block_M + i, k_base + k], "float32")
                    a_scaled = a_val * A_scale_local[i]
                    A_shared[i, k] = T.cast(a_scaled, "float16")
                
                for k, j in T.Parallel(block_K, block_N):
                    b_val = T.cast(B[k_base + k, bx * block_N + j], "float32")
                    b_scaled = b_val * B_scale_local[j]
                    B_shared[k, j] = T.cast(b_scaled, "float16")
                
                T.gemm(A_shared, B_shared, C_local)
            
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return main


def get_kernel():
    func = blockscale_gemm_tilelang(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, SCALE_BLOCK_K, num_stages)
    jit_kernel = tilelang.compile(func, out_idx=[4], target="cuda")
    return jit_kernel


_kernel_cache = None


def blockscale_gemm(a, b, a_scale, b_scale):
    global _kernel_cache
    if _kernel_cache is None:
        _kernel_cache = get_kernel()
    c = _kernel_cache(a, b, a_scale, b_scale)
    return c


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
