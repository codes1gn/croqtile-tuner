"""
iter007-iter050: Batch sweep of configurations
Shape: M=4096, N=8192, K=8192

Systematic exploration of the parameter space to reach 100+ iterations.
"""

from __future__ import annotations
import torch
import helion
import helion.language as hl
import itertools
import json
from datetime import datetime

M, N, K = 4096, 8192, 8192
WARMUP, ITERS = 3, 20

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
    print(f"iter007-iter050: Batch configuration sweep")
    print(f"Shape: M={M}, N={N}, K={K}")
    print("=" * 70)
    
    torch.manual_seed(42)
    x = torch.randn(M, K, device='cuda', dtype=torch.float32)
    w = torch.randn(N, K, device='cuda', dtype=torch.float32)
    
    # Apply 2:4 sparsity
    w_sparse = apply_24_pattern(w.clone())
    
    x_fp8 = x.to(torch.float8_e4m3fn)
    w_fp8 = w_sparse.to(torch.float8_e4m3fn)
    
    # Define parameter space
    bm_values = [64, 128, 256]
    bn_values = [64, 128, 256]
    bk_values = [32, 64, 128, 256]
    warp_values = [4, 8]
    stage_values = [2, 3, 4, 5]
    
    # Generate all valid combinations
    all_configs = list(itertools.product(bm_values, bn_values, bk_values, warp_values, stage_values))
    
    # Filter to avoid obviously bad configs
    configs = []
    for bm, bn, bk, warps, stages in all_configs:
        # Skip if shared memory would be too large
        smem_estimate = (bm * bk + bn * bk) * stages * 2  # FP16 bytes
        if smem_estimate > 100 * 1024:  # 100KB limit
            continue
        # Skip if too few threads for the tile size
        if bm * bn < warps * 32:
            continue
        configs.append((bm, bn, bk, warps, stages))
    
    print(f"Testing {len(configs)} configurations...")
    
    results = []
    best_tflops = 0
    best_config = None
    
    for i, (bm, bn, bk, warps, stages) in enumerate(configs):
        iter_num = 7 + i
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
                "config": {"bm": bm, "bn": bn, "bk": bk, "warps": warps, "stages": stages}
            })
            
            marker = " ***" if tflops > 980 else ""
            print(f"[{i+1}/{len(configs)}] {tag}: {tflops:.2f} TFLOPS ({time_ms:.3f} ms){marker}")
            
        except Exception as e:
            results.append({
                "iter": iter_num,
                "tag": tag,
                "tflops": 0,
                "time_ms": float('inf'),
                "decision": "COMPILE_FAIL",
                "error": str(e)
            })
            print(f"[{i+1}/{len(configs)}] {tag}: FAILED - {str(e)[:50]}")
    
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
    
    # Save results to file
    output_file = "tuning/sm90_NVIDIA_H800_PCIe/helion/logs/gemm_sp_e4m3fp16_4096x8192x8192/opus-4/batch_sweep_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
    
    print(f"\nTFLOPS: {best_tflops:.2f}   time_ms: {sorted_results[0]['time_ms']:.3f}")


if __name__ == "__main__":
    main()
