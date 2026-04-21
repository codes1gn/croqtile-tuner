"""
iter005: Search around new best config BM=128, BN=256, BK=128
Shape: M=4096, N=8192, K=8192

Best so far: 980.16 TFLOPS (104.1% of cuBLAS)
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
    print(f"iter005: Search around BM=128, BN=256, BK=128")
    print(f"Shape: M={M}, N={N}, K={K}")
    print("=" * 70)
    
    torch.manual_seed(42)
    x = torch.randn(M, K, device='cuda', dtype=torch.float32).to(torch.float8_e4m3fn)
    w = torch.randn(N, K, device='cuda', dtype=torch.float32).to(torch.float8_e4m3fn)
    
    configs = [
        # Variations around BM=128, BN=256, BK=128, W=8, S=3
        (128, 256, 128, 8, 3),   # Baseline best
        (128, 256, 128, 8, 2),   # Fewer stages
        (128, 256, 128, 8, 4),   # More stages
        (128, 256, 128, 4, 3),   # Fewer warps
        (128, 256, 128, 16, 3),  # More warps
        (128, 256, 64, 8, 4),    # Smaller BK
        (128, 256, 256, 8, 2),   # Larger BK
        (64, 256, 128, 8, 4),    # Smaller BM
        (128, 512, 64, 8, 3),    # Larger BN
        (64, 512, 64, 8, 4),     # Very large BN
        (128, 128, 256, 8, 3),   # Different balance
        (64, 256, 256, 8, 2),    # Small M, large K tiles
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
            marker = " ***" if tflops > 980 else ""
            print(f"{tag}: {tflops:.2f} TFLOPS ({time_ms:.3f} ms){marker}")
            
            if tflops > best_tflops:
                best_tflops = tflops
                best_config = (bm, bn, bk, warps, stages)
        except Exception as e:
            print(f"{tag}: FAILED - {e}")
            results.append((tag, 0, float('inf')))
    
    print("\n" + "=" * 70)
    print("TOP 5 RESULTS")
    print("=" * 70)
    for tag, tflops, time_ms in sorted(results, key=lambda x: -x[1])[:5]:
        pct = tflops / 941.34 * 100
        print(f"  {tag}: {tflops:.2f} TFLOPS ({pct:.1f}% of cuBLAS)")
    
    print(f"\nBest: BM={best_config[0]}, BN={best_config[1]}, BK={best_config[2]}, warps={best_config[3]}, stages={best_config[4]}")
    best_time = [r[2] for r in results if r[0] == f"BM{best_config[0]}_BN{best_config[1]}_BK{best_config[2]}_W{best_config[3]}_S{best_config[4]}"][0]
    print(f"TFLOPS: {best_tflops:.2f}   time_ms: {best_time:.3f}")


if __name__ == "__main__":
    main()
