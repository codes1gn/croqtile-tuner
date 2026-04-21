#!/usr/bin/env python3
import torch
import tilelang
import tilelang.language as T

M, N, K = 16384, 16384, 512
BLOCK_M, BLOCK_N, BLOCK_K = 192, 128, 64
num_stages = 3
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

def verify():
    jit = tilelang.compile(main, out_idx=[2], target="cuda")
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = jit(A, B)
    ref = (A.float() @ B.float()).half()
    diff = (C.float() - ref.float()).abs().max() / ref.float().abs().max()
    if diff < 1e-2:
        print(f"VERIFY: PASS max_rel_err={diff:.2e}")
        return True, jit, A, B
    else:
        print(f"VERIFY: FAIL max_rel_err={diff:.2e}")
        return False, jit, A, B

def bench(jit, A, B, warmup=10, iters=50):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        jit(A, B)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        jit(A, B)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iters
    tflops = 2 * M * N * K / elapsed_ms / 1e9
    print(f"TFLOPS: {tflops:.2f}   time_ms: {elapsed_ms:.3f}")
    return tflops, elapsed_ms

if __name__ == "__main__":
    passed, jit, A, B = verify()
    if passed:
        tflops, elapsed_ms = bench(jit, A, B)
