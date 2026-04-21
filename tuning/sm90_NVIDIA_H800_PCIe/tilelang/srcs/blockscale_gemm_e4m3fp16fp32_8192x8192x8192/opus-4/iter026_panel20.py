#!/usr/bin/env python3
import torch
import tilelang
import tilelang.language as T

M, N, K = 8192, 8192, 8192

@T.prim_func
def main(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"), C: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(N, 192), T.ceildiv(M, 128), threads=128) as (bx, by):
        A_s = T.alloc_shared((128, 128), "float16")
        B_s = T.alloc_shared((128, 192), "float16")
        C_l = T.alloc_fragment((128, 192), "float32")
        T.use_swizzle(panel_size=20, enable=True)
        T.clear(C_l)
        for ko in T.Pipelined(T.ceildiv(K, 128), num_stages=2):
            T.copy(A[by * 128, ko * 128], A_s)
            T.copy(B[ko * 128, bx * 192], B_s)
            T.gemm(A_s, B_s, C_l)
        T.copy(C_l, C[by * 128, bx * 192])

if __name__ == "__main__":
    jit = tilelang.compile(main, out_idx=[2], target="cuda")
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = jit(A, B)
    ref = A @ B
    diff = (C - ref).abs().max() / ref.abs().max()
    assert diff < 1e-2
    profiler = jit.get_profiler()
    runs = [2 * M * N * K / profiler.do_bench(warmup=20, rep=80) / 1e9 for _ in range(6)]
    print(f"Best: {max(runs):.2f}T, Avg: {sum(runs)/len(runs):.2f}T")
