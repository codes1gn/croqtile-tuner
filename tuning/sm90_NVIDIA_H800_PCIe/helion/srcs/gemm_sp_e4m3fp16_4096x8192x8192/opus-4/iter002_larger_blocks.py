"""
iter002: Larger block sizes for better compute intensity
Shape: M=4096, N=8192, K=8192

Hypothesis: Larger block sizes (256x128 instead of 128x128) will improve
tensor core utilization by increasing arithmetic intensity per block.
"""

from __future__ import annotations
import torch
import helion
import helion.language as hl

M, N, K = 4096, 8192, 8192
WARMUP, ITERS = 10, 50

config = helion.Config(
    block_sizes=[256, 128, 64],  # BM=256 (was 128), BN=128, BK=64
    num_warps=8,  # More warps for larger blocks
    num_stages=4,  # More pipeline stages
)

@helion.kernel(static_shapes=True, config=config)
def dense_fp8_gemm_helion(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
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


def verify(out: torch.Tensor, ref: torch.Tensor, tol: float = 5.0) -> bool:
    """Verify with FP8-appropriate tolerance."""
    err = (out.float() - ref.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    ref_scale = ref.abs().mean().item()
    rel_err = max_err / (ref_scale + 1e-6)
    
    if rel_err < tol:
        print(f"VERIFY: PASS max_abs_err={max_err:.4f} rel_err={rel_err:.4f}")
        return True
    else:
        print(f"VERIFY: FAIL max_abs_err={max_err:.4f} rel_err={rel_err:.4f} (tol={tol})")
        return False


def bench(kernel_fn, *args, warmup: int = WARMUP, iters: int = ITERS) -> tuple[float, float]:
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
    print(f"iter002: Larger blocks BM=256, BN=128, warps=8, stages=4")
    print(f"Shape: M={M}, N={N}, K={K}")
    print("=" * 70)
    
    torch.manual_seed(42)
    x_fp32 = torch.randn(M, K, device='cuda', dtype=torch.float32)
    w_fp32 = torch.randn(N, K, device='cuda', dtype=torch.float32)
    x_fp8 = x_fp32.to(torch.float8_e4m3fn)
    w_fp8 = w_fp32.to(torch.float8_e4m3fn)
    
    ref = torch.mm(x_fp32, w_fp32.T)
    
    print("\n[1] Helion FP8 GEMM with larger blocks")
    try:
        out = dense_fp8_gemm_helion(x_fp8, w_fp8)
        print(f"    Output shape: {out.shape}")
        verify(out, ref)
        time_ms, tflops = bench(dense_fp8_gemm_helion, x_fp8, w_fp8)
        print(f"    TFLOPS: {tflops:.2f}   time_ms: {time_ms:.3f}")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        tflops, time_ms = 0.0, float('inf')
    
    print(f"\nTFLOPS: {tflops:.2f}   time_ms: {time_ms:.3f}")


if __name__ == "__main__":
    main()
