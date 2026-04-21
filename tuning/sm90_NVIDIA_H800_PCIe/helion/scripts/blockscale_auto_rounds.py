#!/usr/bin/env python3
"""
Autonomous STORE rounds for Helion blockscale GEMM (8192^3).
Generates iterNNN_<tag>.py from a template, VERIFY+bench, then calls store_round.sh.

Usage (repo root):
  HELION_AUTOTUNE_EFFORT=none PYTHONPATH=$PWD python3 tuning/.../blockscale_auto_rounds.py --from 5 --to 100

Requires: helion, CUDA; store_round.sh + detect_gpu.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

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
    print(f"VERIFY: {{'PASS' if ok else 'FAIL'}} max_abs_err={{max_err:.6f}}")
    return ok


def bench(warmup: int = 3, iters: int = 12) -> float:
    cfg = helion.Config(
        block_sizes=[{bm}, {bn}],
        num_warps={nw},
        num_stages={st},
        indexing="{ix}",
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
    print(f"TFLOPS: {{tflops:.2f}}   time_ms: {{elapsed_ms:.3f}}")
    return tflops


if __name__ == "__main__":
    if verify():
        bench()
'''

# Deterministic search grid (BM, BN, warps, stages, indexing, short_tag)
GRID = [
    (64, 64, 4, 1, "pointer"),
    (64, 128, 4, 1, "pointer"),
    (128, 64, 4, 1, "pointer"),
    (128, 128, 4, 1, "pointer"),
    (128, 256, 4, 1, "pointer"),
    (256, 128, 4, 1, "pointer"),
    (256, 256, 4, 1, "pointer"),
    (64, 256, 4, 1, "pointer"),
    (256, 64, 4, 1, "pointer"),
    (64, 64, 8, 1, "pointer"),
    (64, 128, 8, 1, "pointer"),
    (128, 64, 8, 1, "pointer"),
    (128, 128, 8, 1, "pointer"),
    (128, 256, 8, 1, "pointer"),
    (256, 128, 8, 1, "pointer"),
    (256, 256, 8, 1, "pointer"),
    (64, 64, 4, 1, "block_ptr"),
    (64, 128, 4, 1, "block_ptr"),
    (128, 64, 4, 1, "block_ptr"),
    (128, 128, 4, 1, "block_ptr"),
    (128, 256, 4, 1, "block_ptr"),
    (256, 128, 4, 1, "block_ptr"),
    (256, 256, 4, 1, "block_ptr"),
    (64, 64, 8, 1, "block_ptr"),
    (128, 128, 8, 1, "block_ptr"),
    (256, 256, 8, 1, "block_ptr"),
    (64, 64, 4, 2, "block_ptr"),
    (128, 128, 4, 2, "block_ptr"),
    (256, 256, 4, 2, "block_ptr"),
    (64, 128, 4, 2, "pointer"),
    (128, 64, 4, 2, "pointer"),
    (128, 256, 4, 2, "pointer"),
    (256, 128, 4, 2, "pointer"),
]


def spec_for_round(r: int) -> tuple[int, int, int, int, str, str]:
    g = GRID[(r - 5) % len(GRID)]
    bm, bn, nw, st, ix = g
    tag = f"m{bm}n{bn}_w{nw}_s{st}_{ix[:5]}"
    return bm, bn, nw, st, ix, tag


def run_one(
    rnd: int,
    out_dir: Path,
    gpu: str,
    shape_key: str,
    model: str,
) -> None:
    bm, bn, nw, st, ix, tag = spec_for_round(rnd)
    doc = f"iter{rnd:03d}_autogrid: BM={bm} BN={bn} warps={nw} stages={st} indexing={ix}"
    src = KERNEL_TEMPLATE.format(
        doc=doc,
        bm=bm,
        bn=bn,
        nw=nw,
        st=st,
        ix=ix,
    )
    fname = f"iter{rnd:03d}_{tag}.py"
    path = out_dir / fname
    path.write_text(src)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    env["HELION_AUTOTUNE_EFFORT"] = "none"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        [sys.executable, "-u", str(path)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = proc.stdout + proc.stderr
    print(out[-2000:] if len(out) > 2000 else out)
    if proc.returncode != 0:
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
    # Compare to current best from results.tsv (simple read)
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
    for rnd in range(args.from_r, args.to_r + 1):
        print(f"\n=== ROUND {rnd} ===", flush=True)
        run_one(rnd, out_dir, gpu, args.shape_key, args.model)


if __name__ == "__main__":
    main()
