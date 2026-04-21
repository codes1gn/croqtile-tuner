#!/usr/bin/env python3
"""
Autonomous STORE rounds for Helion blockscale GEMM (8192^3).
Generates iterNNN_<tag>.py from a template, VERIFY+bench, then calls store_round.sh.

K inner tile BK must be a multiple of GROUP=32 (FP8 blockscale scale granularity).
Scale columns per dot = BK // 32.

Usage (repo root):
  HELION_AUTOTUNE_EFFORT=none PYTHONPATH=$PWD python3 tuning/.../blockscale_auto_rounds.py --from 351 --to 500

Round routing: 5–100 legacy GRID; 101–200 EXPANDED; 201–250 PHASE2;
251–350 PHASE3; 351+ PHASE4 (structural: loop order, K-outer, deeper stages, odd tiles).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _transient_cuda_failure(log: str) -> bool:
    t = log.lower()
    return (
        "busy or unavailable" in t
        or "cudaerrordevicesunavailable" in t
        or "cuda error: cuda-capable device" in t
        or ("acceleratorerror" in t and "cuda" in t)
    )

KERNEL_TEMPLATE = '''#!/usr/bin/env python3
"""{doc}"""
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
        block_sizes=[{bm}, {bn}],
        num_warps={nw},
        num_stages={st},
        indexing="{ix}",
{lo_line}    )

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
            for t_k in hl.tile(kd, block_size={bk}):
                kb0 = t_k.begin // 32
                acc = hl.dot_scaled(
                    a[t_m, t_k],
                    a_s[t_m, kb0 : kb0 + {ncol}],
                    "e4m3",
                    b[t_k, t_n],
                    b_s[t_n, kb0 : kb0 + {ncol}],
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
    print(f"VERIFY: {{'PASS' if ok else 'FAIL'}} max_abs_err={{max_err:.6f}}")
    return ok


def bench(warmup: int = 3, iters: int = 12) -> float:
    cfg = helion.Config(
        block_sizes=[{bm}, {bn}],
        num_warps={nw},
        num_stages={st},
        indexing="{ix}",
{lo_line}    )

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
            for t_k in hl.tile(kd, block_size={bk}):
                kb0 = t_k.begin // 32
                acc = hl.dot_scaled(
                    a[t_m, t_k],
                    a_s[t_m, kb0 : kb0 + {ncol}],
                    "e4m3",
                    b[t_k, t_n],
                    b_s[t_n, kb0 : kb0 + {ncol}],
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
    print(f"TFLOPS: {{tflops:.2f}}   time_ms: {{elapsed_ms:.3f}}")
    return tflops


if __name__ == "__main__":
    if verify():
        bench()
'''

# K-dimension outer: partial sums into fp32 workspace, then cast to fp16 (split-style traversal).
KERNEL_TEMPLATE_K_OUTER = '''#!/usr/bin/env python3
"""{doc}"""
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
        block_sizes=[{bm}, {bn}],
        num_warps={nw},
        num_stages={st},
        indexing="{ix}",
{lo_line}    )

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
        acc_buf = torch.zeros((m, n), dtype=torch.float32, device=a.device)
        out = torch.empty((m, n), dtype=torch.float16, device=a.device)
        for t_k in hl.tile(kd, block_size={bk}):
            kb0 = t_k.begin // 32
            for t_m, t_n in hl.tile([m, n]):
                p = hl.dot_scaled(
                    a[t_m, t_k],
                    a_s[t_m, kb0 : kb0 + {ncol}],
                    "e4m3",
                    b[t_k, t_n],
                    b_s[t_n, kb0 : kb0 + {ncol}],
                    "e4m3",
                    acc=None,
                    out_dtype=torch.float32,
                )
                acc_buf[t_m, t_n] = acc_buf[t_m, t_n] + p
        for t_m, t_n in hl.tile([m, n]):
            out[t_m, t_n] = acc_buf[t_m, t_n].to(torch.float16)
        return out

    out = k_small(a, a_s, b, b_s)
    ref = _ref_blockscale(a, a_s, b, b_s)
    max_err = float((out.float() - ref).abs().max().item())
    ok = max_err < 0.25
    print(f"VERIFY: {{'PASS' if ok else 'FAIL'}} max_abs_err={{max_err:.6f}}")
    return ok


def bench(warmup: int = 3, iters: int = 12) -> float:
    cfg = helion.Config(
        block_sizes=[{bm}, {bn}],
        num_warps={nw},
        num_stages={st},
        indexing="{ix}",
{lo_line}    )

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
        acc_buf = torch.zeros((m, n), dtype=torch.float32, device=a.device)
        out = torch.empty((m, n), dtype=torch.float16, device=a.device)
        for t_k in hl.tile(kd, block_size={bk}):
            kb0 = t_k.begin // 32
            for t_m, t_n in hl.tile([m, n]):
                p = hl.dot_scaled(
                    a[t_m, t_k],
                    a_s[t_m, kb0 : kb0 + {ncol}],
                    "e4m3",
                    b[t_k, t_n],
                    b_s[t_n, kb0 : kb0 + {ncol}],
                    "e4m3",
                    acc=None,
                    out_dtype=torch.float32,
                )
                acc_buf[t_m, t_n] = acc_buf[t_m, t_n] + p
        for t_m, t_n in hl.tile([m, n]):
            out[t_m, t_n] = acc_buf[t_m, t_n].to(torch.float16)
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
    print(f"TFLOPS: {{tflops:.2f}}   time_ms: {{elapsed_ms:.3f}}")
    return tflops


if __name__ == "__main__":
    if verify():
        bench()
'''

# Legacy grid (rounds 5–100): BK always 32 → ncol=1
GRID = [
    (64, 64, 4, 1, "pointer", 32),
    (64, 128, 4, 1, "pointer", 32),
    (128, 64, 4, 1, "pointer", 32),
    (128, 128, 4, 1, "pointer", 32),
    (128, 256, 4, 1, "pointer", 32),
    (256, 128, 4, 1, "pointer", 32),
    (256, 256, 4, 1, "pointer", 32),
    (64, 256, 4, 1, "pointer", 32),
    (256, 64, 4, 1, "pointer", 32),
    (64, 64, 8, 1, "pointer", 32),
    (64, 128, 8, 1, "pointer", 32),
    (128, 64, 8, 1, "pointer", 32),
    (128, 128, 8, 1, "pointer", 32),
    (128, 256, 8, 1, "pointer", 32),
    (256, 128, 8, 1, "pointer", 32),
    (256, 256, 8, 1, "pointer", 32),
    (64, 64, 4, 1, "block_ptr", 32),
    (64, 128, 4, 1, "block_ptr", 32),
    (128, 64, 4, 1, "block_ptr", 32),
    (128, 128, 4, 1, "block_ptr", 32),
    (128, 256, 4, 1, "block_ptr", 32),
    (256, 128, 4, 1, "block_ptr", 32),
    (256, 256, 4, 1, "block_ptr", 32),
    (64, 64, 8, 1, "block_ptr", 32),
    (128, 128, 8, 1, "block_ptr", 32),
    (256, 256, 8, 1, "block_ptr", 32),
    (64, 64, 4, 2, "block_ptr", 32),
    (128, 128, 4, 2, "block_ptr", 32),
    (256, 256, 4, 2, "block_ptr", 32),
    (64, 128, 4, 2, "pointer", 32),
    (128, 64, 4, 2, "pointer", 32),
    (128, 256, 4, 2, "pointer", 32),
    (256, 128, 4, 2, "pointer", 32),
]


def build_expanded_grid() -> list[tuple[int, int, int, int, str, int]]:
    """Rich search for rounds >= 101: asymmetric MN, warps through 16, stages 1–4, BK 32|64|128."""
    blocks = [
        (128, 128),  # near current best
        (64, 256),
        (256, 64),
        (128, 256),
        (256, 128),
        (64, 128),
        (128, 64),
        (256, 256),
    ]
    warps = [4, 8, 16]
    stages = [1, 2, 3, 4]
    idxs = ["block_ptr", "pointer"]
    bks = [32, 64, 128]
    grid: list[tuple[int, int, int, int, str, int]] = []
    for bm, bn in blocks:
        for nw in warps:
            for st in stages:
                for ix in idxs:
                    for bk in bks:
                        grid.append((bm, bn, nw, st, ix, bk))
    return grid


EXPANDED_GRID = build_expanded_grid()


def build_phase2_grid() -> list[tuple[int, int, int, int, str, int]]:
    """Narrow search near current best (128², BK 64, stages 3): fine-tune stages/warps/indexing/BK."""
    out: list[tuple[int, int, int, int, str, int]] = []
    base = (128, 128)
    # Priority: BK 64/128, stages 3–4, warps 4/8/16, both indexings
    for bk in (32, 64, 128):
        for st in (3, 4):
            for nw in (4, 8, 16):
                for ix in ("pointer", "block_ptr"):
                    out.append((*base, nw, st, ix, bk))
    # Asymmetric + stages3 BK64 (user request)
    for bm, bn in ((64, 256), (256, 64), (96, 192), (192, 96)):
        for nw in (4, 8, 16):
            for ix in ("pointer", "block_ptr"):
                out.append((bm, bn, nw, 3, ix, 64))
                out.append((bm, bn, nw, 2, ix, 64))
    return out


PHASE2_GRID = build_phase2_grid()


def build_phase3_grid() -> list[tuple[int, int, int, int, str, int]]:
    """
    Fine-tune around iter117 (~162T, 128², w4, s3, pointer, BK64).
    Includes: s4+BK64, s3+BK128, w8+s3+BK64, 64×128 / 128×64 asym, BK=96, block_ptr pairs.
    """
    seen: set[tuple[int, int, int, int, str, int]] = set()
    out: list[tuple[int, int, int, int, str, int]] = []

    def add(
        t: tuple[int, int, int, int, str, int],
    ) -> None:
        if t not in seen:
            seen.add(t)
            out.append(t)

    # User-priority probes first
    priority: list[tuple[int, int, int, int, str, int]] = [
        # Deeper pipeline: stages=4, BK=64 (128²)
        (128, 128, 4, 4, "pointer", 64),
        (128, 128, 4, 4, "block_ptr", 64),
        (128, 128, 8, 4, "pointer", 64),
        (128, 128, 8, 4, "block_ptr", 64),
        # Larger K-tile: stages=3, BK=128
        (128, 128, 4, 3, "pointer", 128),
        (128, 128, 4, 3, "block_ptr", 128),
        (128, 128, 8, 3, "pointer", 128),
        (128, 128, 8, 3, "block_ptr", 128),
        # warps=8, s=3, BK=64
        (128, 128, 8, 3, "pointer", 64),
        (128, 128, 8, 3, "block_ptr", 64),
        # Asymmetric 64×128 / 128×64, s=3, BK=64
        (64, 128, 4, 3, "pointer", 64),
        (64, 128, 4, 3, "block_ptr", 64),
        (128, 64, 4, 3, "pointer", 64),
        (128, 64, 4, 3, "block_ptr", 64),
        # BK=96 (ncol=3), 128² anchor configs
        (128, 128, 4, 3, "pointer", 96),
        (128, 128, 4, 3, "block_ptr", 96),
        (128, 128, 4, 4, "pointer", 96),
        (128, 128, 4, 4, "block_ptr", 96),
        (128, 128, 8, 3, "pointer", 96),
        (128, 128, 8, 3, "block_ptr", 96),
    ]
    for t in priority:
        add(t)

    blocks = [
        (128, 128),
        (64, 128),
        (128, 64),
        (96, 128),
        (128, 96),
    ]
    for bm, bn in blocks:
        for nw in (4, 8, 16):
            for st in (3, 4):
                for ix in ("pointer", "block_ptr"):
                    for bk in (64, 96, 128):
                        add((bm, bn, nw, st, ix, bk))
    return out


PHASE3_GRID = build_phase3_grid()


def build_phase4_grid() -> list[tuple[int, int, int, int, str, int, str]]:
    """
    Structural search past iter117 plateau: deeper stages, warps=2,
    fused MN loop permutation (lo1).

    Helion requires each block_sizes entry to be a **power of two** (non-Po2 e.g. 96
    fails InvalidConfig). K-outer ``kmn`` layout was removed from autogrid: it hits
    LoopDependencyError on fp32 acc_buf between tile loops (needs a different design).
    """
    seen: set[tuple[int, int, int, int, str, int, str]] = set()
    out: list[tuple[int, int, int, int, str, int, str]] = []

    def add(t: tuple[int, int, int, int, str, int, str]) -> None:
        if t not in seen:
            seen.add(t)
            out.append(t)

    # Only mnk / lo1 — kmn template not safe for hl.tile lowering without host refactor
    layouts = ("mnk", "lo1")
    # Powers of two only (8192 divisible by each)
    blocks = [
        (128, 128),
        (64, 128),
        (128, 64),
        (256, 128),
        (128, 256),
        (256, 256),
        (64, 256),
        (256, 64),
        (512, 128),
        (128, 512),
    ]
    warps = (2, 4, 8)
    stages = (3, 4, 5, 6)
    bks = (32, 64, 128)
    idxs = ("pointer", "block_ptr")

    priority = [
        # iter117 family: stages 4–6, BK=64
        (128, 128, 4, 4, "pointer", 64, "mnk"),
        (128, 128, 4, 5, "pointer", 64, "mnk"),
        (128, 128, 4, 6, "pointer", 64, "mnk"),
        (128, 128, 4, 4, "pointer", 64, "lo1"),
        (128, 128, 4, 4, "block_ptr", 64, "lo1"),
        # warps=2, BK=64 (more registers / thread)
        (128, 128, 2, 3, "pointer", 64, "mnk"),
        (128, 128, 2, 3, "block_ptr", 64, "mnk"),
        (128, 128, 2, 4, "pointer", 64, "mnk"),
        # larger Po2 tiles
        (256, 256, 4, 3, "pointer", 64, "mnk"),
        (256, 256, 4, 3, "block_ptr", 64, "mnk"),
    ]
    for t in priority:
        add(t)

    for bm, bn in blocks:
        for nw in warps:
            for st in stages:
                for ix in idxs:
                    for bk in bks:
                        for lay in layouts:
                            add((bm, bn, nw, st, ix, bk, lay))
    return out


PHASE4_GRID = build_phase4_grid()


def format_kernel(
    doc: str,
    bm: int,
    bn: int,
    nw: int,
    st: int,
    ix: str,
    bk: int,
    ncol: int,
    layout: str,
) -> str:
    lo_line = "        loop_orders=[[1, 0]],\n" if layout == "lo1" else ""
    common = dict(
        doc=doc,
        bm=bm,
        bn=bn,
        nw=nw,
        st=st,
        ix=ix,
        bk=bk,
        ncol=ncol,
        lo_line=lo_line,
    )
    # layout "kmn" (K-outer) is not emitted by build_phase4_grid; legacy iter files may still say kmn.
    return KERNEL_TEMPLATE.format(**common)


def spec_for_round(
    r: int,
) -> tuple[int, int, int, int, str, int, int, str, str]:
    layout = "mnk"
    if r < 101:
        g = GRID[(r - 5) % len(GRID)]
        bm, bn, nw, st, ix, bk = g
    elif r < 201:
        g = EXPANDED_GRID[(r - 101) % len(EXPANDED_GRID)]
        bm, bn, nw, st, ix, bk = g
    elif r < 251:
        g = PHASE2_GRID[(r - 201) % len(PHASE2_GRID)]
        bm, bn, nw, st, ix, bk = g
    elif r < 351:
        g = PHASE3_GRID[(r - 251) % len(PHASE3_GRID)]
        bm, bn, nw, st, ix, bk = g
    else:
        bm, bn, nw, st, ix, bk, layout = PHASE4_GRID[(r - 351) % len(PHASE4_GRID)]
    assert bk % 32 == 0 and bk in (32, 64, 96, 128), f"invalid BK={bk}"
    ncol = bk // 32
    if r >= 351:
        tag = f"m{bm}n{bn}_w{nw}_s{st}_bk{bk}_{ix[:5]}_{layout}"
    else:
        tag = f"m{bm}n{bn}_w{nw}_s{st}_bk{bk}_{ix[:5]}"
    return bm, bn, nw, st, ix, bk, ncol, tag, layout


def run_one(
    rnd: int,
    out_dir: Path,
    gpu: str,
    shape_key: str,
    model: str,
) -> None:
    bm, bn, nw, st, ix, bk, ncol, tag, layout = spec_for_round(rnd)
    doc = (
        f"iter{rnd:03d}_autogrid: BM={bm} BN={bn} warps={nw} stages={st} "
        f"indexing={ix} Ktile={bk} ncol={ncol} layout={layout}"
    )
    src = format_kernel(doc, bm, bn, nw, st, ix, bk, ncol, layout)
    fname = f"iter{rnd:03d}_{tag}.py"
    path = out_dir / fname
    path.write_text(src)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    env["HELION_AUTOTUNE_EFFORT"] = "none"
    env["PYTHONUNBUFFERED"] = "1"
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    proc: subprocess.CompletedProcess[str] | None = None
    out = ""
    for attempt in range(8):
        proc = subprocess.run(
            [sys.executable, "-u", str(path)],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        out = proc.stdout + proc.stderr
        ok = proc.returncode == 0 and not _transient_cuda_failure(out)
        if ok:
            break
        if _transient_cuda_failure(out) and attempt < 7:
            wait_s = 5 + attempt * 5
            print(
                f"[blockscale_auto_rounds] transient CUDA failure, "
                f"retry {attempt + 1}/8 in {wait_s}s...",
                flush=True,
            )
            time.sleep(wait_s)
            continue
        break

    assert proc is not None
    print(out[-2000:] if len(out) > 2000 else out)
    if proc.returncode != 0 or _transient_cuda_failure(out):
        subprocess.run(
            [
                str(REPO / ".cursor/skills/cursor-croq-tune/tools/store_round.sh"),
                "--gpu",
                gpu,
                "--dsl",
                "helion",
                "--shape-key",
                shape_key,
                "--model",
                model,
                "--iter",
                f"iter{rnd:03d}",
                "--kernel",
                f"iter{rnd:03d}_{tag}",
                "--tflops",
                "0",
                "--decision",
                "COMPILE_FAIL",
                "--bottleneck",
                "compile",
                "--idea",
                doc[:200],
                "--round",
                str(rnd),
                "--category",
                "grid",
            ],
            cwd=str(REPO),
            check=False,
        )
        return
    vf = re.search(r"VERIFY:\s*(\w+)", out)
    tf = re.search(r"TFLOPS:\s*([0-9.]+)", out)
    if not vf or vf.group(1) != "PASS" or not tf:
        subprocess.run(
            [
                str(REPO / ".cursor/skills/cursor-croq-tune/tools/store_round.sh"),
                "--gpu",
                gpu,
                "--dsl",
                "helion",
                "--shape-key",
                shape_key,
                "--model",
                model,
                "--iter",
                f"iter{rnd:03d}",
                "--kernel",
                f"iter{rnd:03d}_{tag}",
                "--tflops",
                "0",
                "--decision",
                "DISCARD",
                "--bottleneck",
                "verify",
                "--idea",
                doc[:200],
                "--round",
                str(rnd),
                "--category",
                "grid",
            ],
            cwd=str(REPO),
            check=False,
        )
        return
    tflops = float(tf.group(1))
    tsv = REPO / f"tuning/{gpu}/helion/logs/{shape_key}/{model}/results.tsv"
    best = 0.0
    if tsv.exists():
        for line in tsv.read_text().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 3:
                if parts[0].startswith("iter000") or "cublas" in parts[1]:
                    continue
                try:
                    best = max(best, float(parts[2]))
                except ValueError:
                    pass
    decision = "KEEP" if tflops >= best * 0.998 else "DISCARD"
    bt = "compute_bound"
    subprocess.run(
        [
            str(REPO / ".cursor/skills/cursor-croq-tune/tools/store_round.sh"),
            "--gpu",
            gpu,
            "--dsl",
            "helion",
            "--shape-key",
            shape_key,
            "--model",
            model,
            "--iter",
            f"iter{rnd:03d}",
            "--kernel",
            f"iter{rnd:03d}_{tag}",
            "--tflops",
            f"{tflops:.2f}",
            "--decision",
            decision,
            "--bottleneck",
            bt,
            "--idea",
            doc[:220],
            "--round",
            str(rnd),
            "--category",
            "grid",
        ],
        cwd=str(REPO),
        check=True,
    )
    subprocess.run(
        [
            str(REPO / ".cursor/skills/cursor-croq-tune/tools/reinforce.sh"),
            "--dsl",
            "helion",
            "--shape-key",
            shape_key,
            "--model",
            model,
        ],
        cwd=str(REPO),
        check=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_r", type=int, default=5)
    ap.add_argument("--to", dest="to_r", type=int, default=100)
    ap.add_argument(
        "--shape-key",
        default="blockscale_gemm_e4m3fp16fp32_8192x8192x8192",
    )
    ap.add_argument("--model", default="opus-4")
    args = ap.parse_args()
    out_dir = REPO / (
        "tuning/sm90_NVIDIA_H800_PCIe/helion/srcs/"
        f"blockscale_gemm_e4m3fp16fp32_8192x8192x8192/opus-4"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.check_output(
        [str(REPO / ".cursor/skills/cursor-croq-tune/tools/detect_gpu.sh")],
        text=True,
    ).strip()
    print(
        f"[blockscale_auto_rounds] grids: legacy={len(GRID)} "
        f"expanded(101-200)={len(EXPANDED_GRID)} "
        f"phase2(201-250)={len(PHASE2_GRID)} "
        f"phase3(251-350)={len(PHASE3_GRID)} "
        f"phase4(351+)={len(PHASE4_GRID)}",
        flush=True,
    )
    for rnd in range(args.from_r, args.to_r + 1):
        print(f"\n=== ROUND {rnd} ===", flush=True)
        run_one(rnd, out_dir, gpu, args.shape_key, args.model)


if __name__ == "__main__":
    main()
