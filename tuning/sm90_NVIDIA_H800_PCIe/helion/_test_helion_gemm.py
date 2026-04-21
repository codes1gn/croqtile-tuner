#!/usr/bin/env python3
"""Temporary probe: minimal Helion GEMM — delete after session stable."""
from __future__ import annotations

import torch

import helion
import helion.language as hl

M, N, K = 128, 128, 128


def main() -> None:
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    cfg = helion.Config(
        block_sizes=[32, 32, 32],
        num_warps=4,
        num_stages=2,
    )

    @helion.kernel(
        config=cfg,
        settings=helion.Settings(autotune_effort="none", static_shapes=True),
    )
    def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.empty((M, N), dtype=torch.float32, device=a.device)
        for m in hl.tile(M):
            for n in hl.tile(N):
                acc = hl.zeros([m, n], dtype=torch.float32)
                for k in hl.tile(K):
                    acc = hl.dot(a[m, k], b[k, n], acc=acc)
                out[m, n] = acc
        return out

    out = gemm(a, b)
    ref = torch.matmul(a.float(), b.float())
    print("max err", (out - ref).abs().max().item())


if __name__ == "__main__":
    main()
