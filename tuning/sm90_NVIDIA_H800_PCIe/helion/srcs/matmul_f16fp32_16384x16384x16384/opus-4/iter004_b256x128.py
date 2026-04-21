#!/usr/bin/env python3
"""Helion matmul f16->f32: iter004_b256x128
Shape: 16384x16384x16384
Config: block_sizes=[256,128,64], num_warps=8, num_stages=4
Hypothesis: Larger M tile for better A matrix reuse
"""
from __future__ import annotations

import torch
import helion
import helion.language as hl

M, N, K = 16384, 16384, 16384
WARMUP = 10
ITERS = 50

cfg = helion.Config(
    block_sizes=[256, 128, 64],
    num_warps=8,
    num_stages=4,
)

@helion.kernel(
    config=cfg,
    settings=helion.Settings(autotune_effort="none", static_shapes=True),
)
def matmul_f16fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    k2, n = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return out


def verify() -> bool:
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    out = matmul_f16fp32(a, b)
    ref = torch.matmul(a.float(), b.float())
    max_err = (out - ref).abs().max().item()
    tol = 5e-2
    if max_err < tol:
        print(f"VERIFY: PASS max_abs_err={max_err:.6f}")
        return True
    else:
        print(f"VERIFY: FAIL max_abs_err={max_err:.6f} > tol={tol}")
        return False


def bench() -> float:
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    
    for _ in range(WARMUP):
        _ = matmul_f16fp32(a, b)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        _ = matmul_f16fp32(a, b)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / ITERS
    tflops = 2 * M * N * K / elapsed_ms / 1e9
    print(f"TFLOPS: {tflops:.2f}   time_ms: {elapsed_ms:.3f}")
    return tflops


def profile_only() -> None:
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    for _ in range(3):
        _ = matmul_f16fp32(a, b)
    torch.cuda.synchronize()
    out = matmul_f16fp32(a, b)
    torch.cuda.synchronize()
    print("PROFILE: done")


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--profile":
        profile_only()
    elif verify():
        bench()


if __name__ == "__main__":
    main()
