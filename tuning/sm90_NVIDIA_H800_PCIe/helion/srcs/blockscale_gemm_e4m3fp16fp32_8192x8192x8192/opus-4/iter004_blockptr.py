#!/usr/bin/env python3
"""
iter004: iter002 with indexing block_ptr (TMA-friendly loads).
"""
from __future__ import annotations

import torch

import helion
import helion.language as hl

GROUP = 32

M, N, K = 8192, 8192, 8192


def _ref_blockscale(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
) -> torch.Tensor:
    m, k1 = a.shape
    k2, n = b.shape
    assert k1 == k2
    g = GROUP
    a_f = a.float()
    b_f = b.float()
    for i in range(k1 // g):
        ks, ke = i * g, (i + 1) * g
        sa = torch.pow(2.0, a_s[:, i].float() - 127.0).unsqueeze(-1)
        sb = torch.pow(2.0, b_s[:, i].float() - 127.0).unsqueeze(0)
        a_f[:, ks:ke] *= sa
        b_f[ks:ke, :] *= sb
    return torch.matmul(a_f, b_f)


def verify() -> bool:
    torch.manual_seed(0)
    vm, vn, vk = 256, 256, 256

    a = (torch.randn(vm, vk, device="cuda", dtype=torch.float32) * 0.02).to(
        torch.float8_e4m3fn
    )
    b = (torch.randn(vk, vn, device="cuda", dtype=torch.float32) * 0.02).to(
        torch.float8_e4m3fn
    )
    a_s = torch.randint(124, 128, (vm, vk // GROUP), device="cuda", dtype=torch.uint8)
    b_s = torch.randint(124, 128, (vn, vk // GROUP), device="cuda", dtype=torch.uint8)

    cfg_v = helion.Config(
        block_sizes=[128, 128],
        num_warps=4,
        num_stages=1,
        indexing="block_ptr",
    )

    @helion.kernel(
        config=cfg_v,
        settings=helion.Settings(autotune_effort="none", static_shapes=True),
    )
    def k_small(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
    ) -> torch.Tensor:
        m, kd = a.shape
        _k2, n = b.shape
        out = torch.empty((m, n), dtype=torch.float16, device=a.device)
        for t_m, t_n in hl.tile([m, n]):
            acc = hl.zeros([t_m, t_n], dtype=torch.float32)
            for t_k in hl.tile(kd, block_size=32):
                kb = t_k.begin // 32
                acc = hl.dot_scaled(
                    a[t_m, t_k],
                    a_s[t_m, kb : kb + 1],
                    "e4m3",
                    b[t_k, t_n],
                    b_s[t_n, kb : kb + 1],
                    "e4m3",
                    acc=acc,
                    out_dtype=torch.float32,
                )
            out[t_m, t_n] = acc.to(torch.float16)
        return out

    out = k_small(a, a_s, b, b_s)
    ref = _ref_blockscale(a, a_s, b, b_s)
    max_err = float((out.float() - ref).abs().max().item())
    ok = max_err < 0.25
    print(f"VERIFY: {'PASS' if ok else 'FAIL'} max_abs_err={max_err:.6f}")
    return ok


def bench(warmup: int = 10, iters: int = 50) -> float:
    cfg = helion.Config(
        block_sizes=[128, 128],
        num_warps=4,
        num_stages=1,
        indexing="block_ptr",
    )

    @helion.kernel(
        config=cfg,
        settings=helion.Settings(autotune_effort="none", static_shapes=True),
    )
    def blockscale_gemm_kernel(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
    ) -> torch.Tensor:
        m, kd = a.shape
        _k2, n = b.shape
        out = torch.empty((m, n), dtype=torch.float16, device=a.device)
        for t_m, t_n in hl.tile([m, n]):
            acc = hl.zeros([t_m, t_n], dtype=torch.float32)
            for t_k in hl.tile(kd, block_size=32):
                kb = t_k.begin // 32
                acc = hl.dot_scaled(
                    a[t_m, t_k],
                    a_s[t_m, kb : kb + 1],
                    "e4m3",
                    b[t_k, t_n],
                    b_s[t_n, kb : kb + 1],
                    "e4m3",
                    acc=acc,
                    out_dtype=torch.float32,
                )
            out[t_m, t_n] = acc.to(torch.float16)
        return out

    torch.manual_seed(1)
    a = (torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.1).to(torch.float8_e4m3fn)
    b = (torch.randn(K, N, device="cuda", dtype=torch.float32) * 0.1).to(torch.float8_e4m3fn)
    a_s = torch.randint(120, 132, (M, K // GROUP), device="cuda", dtype=torch.uint8)
    b_s = torch.randint(120, 132, (N, K // GROUP), device="cuda", dtype=torch.uint8)

    for _ in range(warmup):
        _ = blockscale_gemm_kernel(a, a_s, b, b_s)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = blockscale_gemm_kernel(a, a_s, b, b_s)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    flops = 2.0 * M * N * K
    tflops = flops / elapsed_ms / 1e9
    print(f"TFLOPS: {tflops:.2f}   time_ms: {elapsed_ms:.3f}")
    return tflops


if __name__ == "__main__":
    if verify():
        bench()
