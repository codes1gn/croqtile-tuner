#!/usr/bin/env python3
import torch
import tilelang
import tilelang.language as T

M, N, K = 8192, 8192, 8192

@T.prim_func
def main(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"), C: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(N, 192), T.ceildiv(M, 128), threads=128) as (bx, by):
        A_shared = T.alloc_shared((128, 128), "float16")
        B_shared = T.alloc_shared((128, 192), "float16")
        C_local = T.alloc_fragment((128, 192), "float32")
        T.use_swizzle(panel_size=17, enable=True)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, 128), num_stages=2):
            T.copy(A[by * 128, ko * 128], A_shared)
            T.copy(B[ko * 128, bx * 192], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * 128, bx * 192])

if __name__ == "__main__":
    jit = tilelang.compile(main, out_idx=[2], target="cuda")
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = jit(A, B)
    ref = A @ B
    diff = (C - ref).abs().max() / ref.abs().max()
    assert diff < 1e-2, f"Verification failed: {diff:.2e}"
    profiler = jit.get_profiler()
    runs = [2 * M * N * K / profiler.do_bench(warmup=20, rep=80) / 1e9 for _ in range(8)]
    print(f"panel=17: best={max(runs):.2f}T avg={sum(runs)/len(runs):.2f}T")
