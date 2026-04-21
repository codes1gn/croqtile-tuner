"""
iter003: Systematic block size sweep for optimal configuration
Shape: M=4096, N=8192, K=8192

Strategy: Try multiple block size combinations to find optimal for this shape.
"""

from __future__ import annotations
import torch
import helion
import helion.language as hl

M, N, K = 4096, 8192, 8192
WARMUP, ITERS = 5, 30

def create_kernel(bm, bn, bk, warps, stages):
    config = helion.Config(
        block_sizes=[bm, bn, bk],
        num_warps=warps,
        num_stages=stages,
    )
    
    @helion.kernel(static_shapes=True, config=config)
    def kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
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
    
    return kernel


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
    print(f"iter003: Block size sweep")
    print(f"Shape: M={M}, N={N}, K={K}")
    print("=" * 70)
    
    torch.manual_seed(42)
    x = torch.randn(M, K, device='cuda', dtype=torch.float32).to(torch.float8_e4m3fn)
    w = torch.randn(N, K, device='cuda', dtype=torch.float32).to(torch.float8_e4m3fn)
    
    configs = [
        # (BM, BN, BK, warps, stages) - exploring different configurations
        (256, 128, 64, 8, 4),   # iter002 baseline
        (256, 128, 32, 8, 5),   # Smaller BK, more stages
        (256, 128, 128, 8, 3),  # Larger BK, fewer stages
        (128, 256, 64, 8, 4),   # Swapped BM/BN
        (256, 256, 64, 8, 3),   # Square, larger tiles
        (128, 128, 128, 8, 4),  # Larger BK
        (256, 64, 64, 4, 5),    # Narrow BN
        (64, 256, 64, 4, 5),    # Narrow BM
    ]
    
    results = []
    best_tflops = 0
    best_config = None
    
    for bm, bn, bk, warps, stages in configs:
        tag = f"BM{bm}_BN{bn}_BK{bk}_W{warps}_S{stages}"
        try:
            kernel = create_kernel(bm, bn, bk, warps, stages)
            time_ms, tflops = bench(kernel, x, w)
            results.append((tag, tflops, time_ms))
            print(f"{tag}: {tflops:.2f} TFLOPS ({time_ms:.3f} ms)")
            
            if tflops > best_tflops:
                best_tflops = tflops
                best_config = (bm, bn, bk, warps, stages)
        except Exception as e:
            print(f"{tag}: FAILED - {e}")
            results.append((tag, 0, float('inf')))
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for tag, tflops, time_ms in sorted(results, key=lambda x: -x[1]):
        print(f"  {tag}: {tflops:.2f} TFLOPS")
    
    print(f"\nBest config: BM={best_config[0]}, BN={best_config[1]}, BK={best_config[2]}, warps={best_config[3]}, stages={best_config[4]}")
    print(f"TFLOPS: {best_tflops:.2f}   time_ms: {results[configs.index(best_config)][2]:.3f}")


if __name__ == "__main__":
    main()
