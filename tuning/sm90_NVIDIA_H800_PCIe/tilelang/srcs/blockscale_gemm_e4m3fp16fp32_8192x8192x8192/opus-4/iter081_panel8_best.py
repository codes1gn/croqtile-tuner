#!/usr/bin/env python3
"""
iter081_panel8_best: BEST CONFIG - 407.77 TFLOPS (92.0% of baseline)
Configuration: 128x192x128, stages=2, panel_size=8
"""
import torch
import tilelang
import tilelang.language as T

M, N, K = 8192, 8192, 8192
BLOCK_M, BLOCK_N, BLOCK_K = 128, 192, 128
num_stages = 2
threads = 128
panel_size = 8

@T.prim_func
def main(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
        
        T.use_swizzle(panel_size=panel_size, enable=True)
        T.clear(C_local)
        
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
            T.copy(A[by * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(B[ko * BLOCK_K, bx * BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        
        T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])

if __name__ == "__main__":
    jit = tilelang.compile(main, out_idx=[2], target="cuda")
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = jit(A, B)
    ref = A @ B
    diff = (C - ref).abs().max() / ref.abs().max()
    print(f"Max relative error: {diff:.2e}")
    assert diff < 1e-2, f"Verification failed: {diff:.2e}"
    print("Verification PASSED")
    
    profiler = jit.get_profiler()
    runs = []
    for i in range(10):
        latency = profiler.do_bench(warmup=30, rep=100)
        tflops = 2 * M * N * K / latency / 1e9
        runs.append(tflops)
        print(f"Run {i+1}: {tflops:.2f} TFLOPS")
    print(f"Best: {max(runs):.2f} TFLOPS, Avg: {sum(runs)/len(runs):.2f} TFLOPS")
