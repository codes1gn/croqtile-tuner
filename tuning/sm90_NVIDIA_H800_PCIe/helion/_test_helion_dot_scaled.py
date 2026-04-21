#!/usr/bin/env python3
"""hl.dot_scaled: triton-style reference vs Helion."""
from __future__ import annotations

import torch

import helion
import helion.language as hl

M, N, K = 128, 128, 128
GROUP = 32


def ref_triton_style(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    *,
    k_dim: int,
    g: int,
) -> torch.Tensor:
    """Blockscale matching repo Triton iter001 reference."""
    a_f = a.float()
    b_f = b.float()
    for i in range(k_dim // g):
        ks, ke = i * g, (i + 1) * g
        sa = torch.pow(2.0, a_s[:, i].float() - 127.0).unsqueeze(-1)
        sb = torch.pow(2.0, b_s[:, i].float() - 127.0).unsqueeze(0)
        a_f[:, ks:ke] *= sa
        b_f[ks:ke, :] *= sb
    return torch.matmul(a_f, b_f)


def main() -> None:
    torch.manual_seed(0)
    a = (torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.02).to(
        torch.float8_e4m3fn
    )
    b = (torch.randn(K, N, device="cuda", dtype=torch.float32) * 0.02).to(
        torch.float8_e4m3fn
    )
    a_s = torch.randint(124, 128, (M, K // GROUP), device="cuda", dtype=torch.uint8)
    b_s = torch.randint(124, 128, (N, K // GROUP), device="cuda", dtype=torch.uint8)

    cfg = helion.Config(
        block_sizes=[32, 32],
        num_warps=4,
        num_stages=2,
    )

    @helion.kernel(
        config=cfg,
        settings=helion.Settings(autotune_effort="none", static_shapes=True),
    )
    def bs_gemm(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty((M, N), dtype=torch.float16, device=a.device)
        for m in hl.tile(M):
            for n in hl.tile(N):
                acc = hl.dot_scaled(
                    a[m, :],
                    a_s[m, :],
                    "e4m3",
                    b[:, n],
                    b_s[n, :],
                    "e4m3",
                    out_dtype=torch.float32,
                )
                out[m, n] = acc.to(torch.float16)
        return out

    out = bs_gemm(a, a_s, b, b_s)
    ref = ref_triton_style(a, a_s, b, b_s, k_dim=K, g=GROUP)
    diff = (out.float() - ref).abs()
    print("max err", float(diff.max().item()))


if __name__ == "__main__":
    main()
