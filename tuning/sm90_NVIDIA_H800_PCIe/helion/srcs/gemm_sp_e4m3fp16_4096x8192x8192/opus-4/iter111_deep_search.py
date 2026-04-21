"""
iter111-160: Deep search around best config (BM=128, BN=256, BK=128)
Shape: M=4096, N=8192, K=8192

Current best: 982.82 TFLOPS (iter006)
Target: Find configs that exceed 1000 TFLOPS
"""

from __future__ import annotations
import torch
import helion
import helion.language as hl
import json

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


def apply_24_pattern(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x_flat = x.reshape(-1, 4)
    _, indices = torch.topk(x_flat.abs(), k=2, dim=1)
    mask = torch.zeros_like(x_flat, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return (x_flat * mask).reshape(orig_shape)


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
    print(f"iter111-160: Deep search around best configuration")
    print(f"Shape: M={M}, N={N}, K={K}")
    print("=" * 70)
    
    torch.manual_seed(42)
    x = torch.randn(M, K, device='cuda', dtype=torch.float32)
    w = torch.randn(N, K, device='cuda', dtype=torch.float32)
    w_sparse = apply_24_pattern(w.clone())
    x_fp8 = x.to(torch.float8_e4m3fn)
    w_fp8 = w_sparse.to(torch.float8_e4m3fn)
    
    # Deep search around known good configs
    configs = [
        # Around BM=128, BN=256, BK=128 (best so far)
        (128, 256, 128, 8, 3),
        (128, 256, 128, 8, 4),
        (128, 256, 128, 8, 5),
        (128, 256, 128, 8, 6),
        (128, 256, 64, 8, 3),
        (128, 256, 64, 8, 4),
        
        # Try larger BN
        (64, 256, 128, 8, 3),
        (64, 256, 128, 8, 4),
        (64, 256, 256, 8, 2),
        (64, 256, 256, 8, 3),
        
        # Try swapped dimensions
        (256, 128, 128, 8, 3),
        (256, 128, 128, 8, 4),
        (256, 128, 256, 8, 2),
        (256, 128, 256, 8, 3),
        
        # High warp counts
        (128, 256, 128, 16, 2),
        (128, 256, 128, 16, 3),
        (128, 128, 128, 16, 3),
        (128, 128, 128, 16, 4),
        
        # Low warp counts
        (128, 256, 128, 4, 3),
        (128, 256, 128, 4, 4),
        (128, 256, 256, 4, 2),
        
        # Other promising combinations
        (64, 128, 256, 8, 3),
        (64, 128, 256, 8, 4),
        (128, 64, 256, 8, 3),
        (256, 64, 128, 8, 4),
        
        # Square tiles
        (128, 128, 128, 8, 3),
        (128, 128, 128, 8, 4),
        (128, 128, 128, 8, 5),
        (256, 256, 128, 4, 2),
        
        # Additional variations
        (64, 512, 64, 8, 3),
        (128, 512, 32, 8, 4),
        (256, 512, 32, 4, 3),
        (512, 128, 32, 8, 4),
        (512, 64, 64, 8, 3),
        
        # More stage variations
        (128, 256, 128, 8, 2),
        (128, 256, 64, 8, 5),
        (128, 256, 64, 8, 6),
        (128, 128, 256, 8, 2),
        (128, 128, 256, 8, 3),
        
        # Additional combinations
        (64, 256, 64, 8, 3),
        (64, 256, 64, 8, 4),
        (64, 256, 64, 8, 5),
        (64, 256, 64, 4, 4),
        (64, 256, 64, 4, 5),
        
        (256, 256, 64, 4, 2),
        (256, 256, 64, 4, 3),
        (256, 256, 64, 8, 2),
        
        (128, 128, 64, 4, 4),
        (128, 128, 64, 8, 4),
    ]
    
    print(f"Testing {len(configs)} configurations...")
    
    results = []
    best_tflops = 0
    best_config = None
    
    for i, (bm, bn, bk, warps, stages) in enumerate(configs):
        iter_num = 111 + i
        tag = f"iter{iter_num:03d}_BM{bm}_BN{bn}_BK{bk}_W{warps}_S{stages}"
        
        try:
            kernel = create_kernel(bm, bn, bk, warps, stages)
            time_ms, tflops = bench(kernel, x_fp8, w_fp8)
            
            decision = "KEEP" if tflops > best_tflops else "DISCARD"
            if tflops > best_tflops:
                best_tflops = tflops
                best_config = (bm, bn, bk, warps, stages)
            
            results.append({
                "iter": iter_num,
                "tag": tag,
                "tflops": tflops,
                "time_ms": time_ms,
                "decision": decision,
            })
            
            marker = " ***" if tflops > 980 else (" ++" if tflops > 950 else "")
            print(f"[{i+1}/{len(configs)}] {tag}: {tflops:.2f} TFLOPS ({time_ms:.3f} ms){marker}")
            
        except Exception as e:
            results.append({
                "iter": iter_num,
                "tag": tag,
                "tflops": 0,
                "time_ms": float('inf'),
                "decision": "COMPILE_FAIL",
            })
            print(f"[{i+1}/{len(configs)}] {tag}: FAILED - {str(e)[:40]}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TOP 10 RESULTS")
    print("=" * 70)
    sorted_results = sorted([r for r in results if r['tflops'] > 0], key=lambda x: -x['tflops'])
    for r in sorted_results[:10]:
        pct = r['tflops'] / 941.34 * 100
        print(f"  {r['tag']}: {r['tflops']:.2f} TFLOPS ({pct:.1f}% of cuBLAS)")
    
    print(f"\nTotal iterations: {len(configs)}")
    print(f"Successful: {len([r for r in results if r['tflops'] > 0])}")
    print(f"Best: {best_config} -> {best_tflops:.2f} TFLOPS")
    
    # Save results
    output_file = "tuning/sm90_NVIDIA_H800_PCIe/helion/logs/gemm_sp_e4m3fp16_4096x8192x8192/opus-4/deep_search_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nTFLOPS: {best_tflops:.2f}   time_ms: {sorted_results[0]['time_ms']:.3f}")


if __name__ == "__main__":
    main()
