#!/usr/bin/env python3
from __future__ import annotations
import torch
import helion
import helion.language as hl

M, N, K = 16416, 16416, 16416
WARMUP = 10
ITERS = 50

cfg = helion.Config(
    block_sizes=[64, 512, 64],
    num_warps=16,
    num_stages=3,
    indexing="block_ptr",
)

@helion.kernel(config=cfg, settings=helion.Settings(autotune_effort="none", static_shapes=True))
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
    if max_err < 5e-2:
        print(f"VERIFY: PASS max_abs_err={max_err:.6f}")
        return True
    else:
        print(f"VERIFY: FAIL max_abs_err={max_err:.6f}")
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

def main() -> None:
    if verify():
        bench()

if __name__ == "__main__":
    main()
