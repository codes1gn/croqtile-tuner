#!/usr/bin/env python3
"""iter031_b256x128_panel9: Tall tiles 256x128 with panel_size=9."""
import torch, tilelang, tilelang.language as T
M, N, K = 16416, 16416, 16416
DTYPE, ACCUM_DTYPE = "float16", "float"
BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, NUM_THREADS = 256, 128, 32, 4, 256
def matmul_kernel():
    @T.prim_func
    def main(A: T.Tensor((M, K), DTYPE), B: T.Tensor((K, N), DTYPE), C: T.Tensor((M, N), ACCUM_DTYPE)):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=NUM_THREADS) as (bx, by):
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), DTYPE)
            B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), DTYPE)
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), ACCUM_DTYPE)
            T.use_swizzle(panel_size=9, enable=True)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
                T.copy(A[by * BLOCK_M, ko * BLOCK_K], A_shared)
                T.copy(B[ko * BLOCK_K, bx * BLOCK_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])
    return main
def verify():
    func = matmul_kernel()
    jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda")
    torch.manual_seed(42)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = jit_kernel(a, b)
    ref_c = torch.matmul(a.float(), b.float())
    max_abs_err = (c - ref_c).abs().max().item()
    rel_err = max_abs_err / ref_c.abs().max().item()
    if max_abs_err < 1e-2 or rel_err < 1e-3:
        print(f"VERIFY: PASS max_abs_err={max_abs_err:.6f} rel_err={rel_err:.6f}")
        return True
    print(f"VERIFY: FAIL max_abs_err={max_abs_err:.6f} rel_err={rel_err:.6f}")
    return False
def bench(warmup=10, iters=50):
    func = matmul_kernel()
    jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda")
    torch.manual_seed(42)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    for _ in range(warmup): _ = jit_kernel(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): _ = jit_kernel(a, b)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iters
    tflops = 2.0 * M * N * K / elapsed_ms / 1e9
    print(f"TFLOPS: {tflops:.2f}   time_ms: {elapsed_ms:.3f}")
    return tflops, elapsed_ms
if __name__ == "__main__":
    print(f"256x128 panel_size=9")
    if verify(): bench()
    else: exit(1)
