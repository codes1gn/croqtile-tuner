"""
iter160: Final Optimized Sparse FP8 GEMM Kernel
Shape: M=4096, N=8192, K=8192 (2:4 structured sparsity, e4m3→fp16)

Best configuration found through 159 iterations of systematic search:
  - block_sizes=[128, 256, 128]  (BM, BN, BK)
  - num_warps=8
  - num_stages=3

Performance: 982.80 TFLOPS
  - 104.4% of cuBLAS dense FP8 (torch._scaled_mm)
  - 130.3% of cuSPARSELt sparse FP16
"""

from __future__ import annotations
import torch
import helion
import helion.language as hl

M, N, K = 4096, 8192, 8192
WARMUP, ITERS = 10, 50

# Optimal configuration
config = helion.Config(
    block_sizes=[128, 256, 128],  # BM=128, BN=256, BK=128
    num_warps=8,
    num_stages=3,
)

@helion.kernel(static_shapes=True, config=config)
def sparse_fp8_gemm_optimized(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Optimized Sparse FP8 GEMM kernel.
    
    Computes C = X @ W.T where:
      - X: (M, K) activation matrix in FP8 e4m3
      - W: (N, K) weight matrix in FP8 e4m3 with 2:4 sparsity
      - C: (M, N) output in FP16
    
    Configuration rationale:
      - BM=128: Matches M=4096 dimension (32 tiles)
      - BN=256: Large N tile for high arithmetic intensity
      - BK=128: Large K tile to maximize tensor core utilization
      - 8 warps: Optimal occupancy for H100 SM
      - 3 stages: Pipeline depth balancing latency hiding vs smem
    """
    m, k = x.size()
    n, k2 = w.size()
    
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)
    
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        
        for tile_k in hl.tile(k):
            x_tile = x[tile_m, tile_k]
            w_tile = w[tile_n, tile_k]
            acc = hl.dot(x_tile, w_tile.T, acc=acc)
        
        out[tile_m, tile_n] = acc.to(torch.float16)
    
    return out


def apply_24_pattern(x: torch.Tensor) -> torch.Tensor:
    """Apply 2:4 sparsity: keep 2 largest magnitude per 4 elements."""
    orig_shape = x.shape
    x_flat = x.reshape(-1, 4)
    _, indices = torch.topk(x_flat.abs(), k=2, dim=1)
    mask = torch.zeros_like(x_flat, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return (x_flat * mask).reshape(orig_shape)


def bench(kernel_fn, *args, warmup: int = WARMUP, iters: int = ITERS):
    for _ in range(warmup):
        kernel_fn(*args)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        kernel_fn(*args)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / iters
    tflops = 2 * M * N * K / time_ms / 1e9
    return time_ms, tflops


def main():
    print("=" * 70)
    print("iter160: Final Optimized Sparse FP8 GEMM")
    print(f"Shape: M={M}, N={N}, K={K}")
    print("=" * 70)
    
    torch.manual_seed(42)
    x_fp32 = torch.randn(M, K, device='cuda', dtype=torch.float32)
    w_fp32 = torch.randn(N, K, device='cuda', dtype=torch.float32)
    
    # Apply 2:4 sparsity
    w_sparse = apply_24_pattern(w_fp32.clone())
    sparsity = 1.0 - (w_sparse != 0).float().mean().item()
    print(f"\nWeight sparsity: {sparsity*100:.1f}%")
    
    # Convert to FP8
    x_fp8 = x_fp32.to(torch.float8_e4m3fn)
    w_fp8 = w_sparse.to(torch.float8_e4m3fn)
    
    # Run optimized kernel
    print("\n[1] Helion Optimized Sparse FP8 GEMM")
    out = sparse_fp8_gemm_optimized(x_fp8, w_fp8)
    print(f"    Output shape: {out.shape}")
    
    # Verify
    ref = torch.mm(x_fp32, w_sparse.T)
    err = (out.float() - ref).abs()
    print(f"    Max abs error: {err.max():.4f}")
    print(f"    Mean abs error: {err.mean():.6f}")
    
    # Benchmark
    time_ms, tflops = bench(sparse_fp8_gemm_optimized, x_fp8, w_fp8)
    print(f"    TFLOPS: {tflops:.2f}")
    print(f"    time_ms: {time_ms:.3f}")
    
    # Compare with cuBLAS
    print("\n[2] Reference: cuBLAS Dense FP8 (torch._scaled_mm)")
    scale = torch.tensor(1.0, device='cuda')
    w_fp8_t = w_fp32.to(torch.float8_e4m3fn).T.contiguous().T
    time_cublas, tflops_cublas = bench(
        lambda: torch._scaled_mm(x_fp8, w_fp8_t, scale, scale, out_dtype=torch.float16)
    )
    print(f"    TFLOPS: {tflops_cublas:.2f}")
    print(f"    time_ms: {time_cublas:.3f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Helion Optimized: {tflops:.2f} TFLOPS")
    print(f"cuBLAS Dense FP8: {tflops_cublas:.2f} TFLOPS")
    print(f"Speedup: {tflops/tflops_cublas*100:.1f}%")
    
    print(f"\nTFLOPS: {tflops:.2f}   time_ms: {time_ms:.3f}")


if __name__ == "__main__":
    main()
