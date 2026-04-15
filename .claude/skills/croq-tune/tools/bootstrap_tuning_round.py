#!/usr/bin/env python3
"""Bootstrap first croq-tune round when no kernel source exists yet."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_shape(shape: str) -> tuple[int, int, int]:
    parts = shape.lower().split("x")
    if len(parts) != 3:
        raise SystemExit(f"Invalid shape '{shape}', expected MxNxK")
    return int(parts[0]), int(parts[1]), int(parts[2])


def shape_key(operator: str, dtype: str, m: int, n: int, k: int) -> str:
    return f"{operator}_{dtype}_{m}x{n}x{k}"


def get_torch_dtype(dtype_str: str) -> "torch.dtype":
    import torch
    dtype_map = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
        "bf16fp32": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise SystemExit(f"Unsupported dtype '{dtype_str}', supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def bench_torch_mm(m: int, n: int, k: int, dtype: str, warmup: int, iters: int, samples: int) -> dict:
    import torch  # type: ignore

    input_dtype = get_torch_dtype(dtype)
    use_fp32_output = dtype == "bf16fp32"

    sample_rows = []
    for sample_id in range(1, samples + 1):
        a = torch.randn(m, k, device="cuda", dtype=input_dtype)
        b = torch.randn(k, n, device="cuda", dtype=input_dtype)

        if use_fp32_output:
            for _ in range(warmup):
                torch.mm(a.float(), b.float())
        else:
            for _ in range(warmup):
                torch.mm(a, b)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            if use_fp32_output:
                torch.mm(a.float(), b.float())
            else:
                torch.mm(a, b)
        end.record()
        torch.cuda.synchronize()

        time_ms = start.elapsed_time(end) / iters
        tflops = (2.0 * m * n * k) / (time_ms * 1e-3) / 1e12
        sample_rows.append({"sample": sample_id, "time_ms": time_ms, "tflops": tflops})

    return {
        "samples": sample_rows,
        "median_time_ms": statistics.median([row["time_ms"] for row in sample_rows]),
        "median_tflops": statistics.median([row["tflops"] for row in sample_rows]),
        "input_dtype": str(input_dtype),
        "output_dtype": "torch.float32" if use_fp32_output else str(input_dtype),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap first tuning round artifacts for croq-tune.")
    parser.add_argument("--dsl", default="cuda")
    parser.add_argument("--operator", default="matmul", help="Operator type (matmul, spmm, conv2d, attention)")
    parser.add_argument("--dtype", default="f16")
    parser.add_argument("--shape", required=True, help="MxNxK")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    if args.dsl != "cuda":
        raise SystemExit("bootstrap_tuning_round currently supports --dsl cuda only")

    m, n, k = parse_shape(args.shape)
    key = shape_key(args.operator, args.dtype, m, n, k)
    root = repo_root()
    now = datetime.now(timezone.utc).isoformat()

    tuning_root = root / "tuning" / "aitune" / args.dsl
    log_dir = tuning_root / "logs" / key
    perf_dir = tuning_root / "perf" / key
    mem_dir = tuning_root / "memory" / key
    ckpt_dir = tuning_root / "checkpoints"
    for directory in [log_dir, perf_dir, mem_dir, ckpt_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    bench = bench_torch_mm(m, n, k, args.dtype, args.warmup, args.iters, args.samples)

    results_path = log_dir / "results.tsv"
    results_lines = [
        "iter\tkernel\ttflops\thw_eff_pct\tdecision\tbottleneck\tidea_summary\trun_command",
        (
            f"iter000\tframework/torch_mm_{args.dtype}\t{bench['median_tflops']:.2f}\t0.00\tKEEP\tbaseline_profile\t"
            f"Bootstrap baseline profile with torch.mm {args.dtype.upper()}.\t"
            f"python3 .claude/skills/croq-tune/tools/bootstrap_tuning_round.py --dsl {args.dsl} --dtype {args.dtype} --shape {args.shape}"
        ),
    ]
    results_path.write_text("\n".join(results_lines) + "\n", encoding="utf-8")

    idea = {
        "timestamp": now,
        "round": 1,
        "category": "structural",
        "bottleneck": "unknown_pre_ncu",
        "idea": "No kernel source found in repo; next action is to create first compilable custom matmul kernel scaffold for this shape.",
        "expected_gain": "Enable IMPLEMENT path and measured iter001 candidate.",
    }
    (log_dir / "idea-log.jsonl").write_text(json.dumps(idea, sort_keys=True) + "\n", encoding="utf-8")

    attempt = {
        "timestamp": now,
        "attempt_id": "attempt0001",
        "status": "failed",
        "reason": "no_kernel_source_in_repo",
        "next_state": "IDEA",
    }
    (log_dir / "attempt-log.jsonl").write_text(json.dumps(attempt, sort_keys=True) + "\n", encoding="utf-8")

    perf_payload = {
        "timestamp": now,
        "shape_key": key,
        "kernel": f"framework/torch_mm_{args.dtype}",
        "bench": bench,
    }
    (perf_dir / "timing_iter000_baseline.json").write_text(
        json.dumps(perf_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    round_payload = {
        "timestamp": now,
        "shape_key": key,
        "state": "IDEA",
        "summary": "Baseline profiled, first implementation attempt blocked by missing kernel source.",
    }
    (mem_dir / "rounds.raw.jsonl").write_text(json.dumps(round_payload, sort_keys=True) + "\n", encoding="utf-8")
    (mem_dir / "rounds.md").write_text(
        "\n".join(
            [
                f"## Round bootstrap - {now}",
                f"- shape: `{key}`",
                f"- baseline_tflops: `{bench['median_tflops']:.2f}`",
                "- result: baseline stored; next state returns to IDEA due to missing custom kernel source.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    checkpoint = {
        "schema": "croq-tune-checkpoint-v1",
        "timestamp": now,
        "dsl": args.dsl,
        "dtype": args.dtype,
        "shape_key": key,
        "shape": {"m": m, "n": n, "k": k},
        "current_iter": 0,
        "best_tflops": round(bench["median_tflops"], 2),
        "best_kernel": f"framework/torch_mm_{args.dtype}",
        "next_state": "IDEA",
        "last_attempt": "attempt0001",
    }
    (ckpt_dir / f"{key}.json").write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[bootstrap-round] created bootstrap artifacts for {key}")
    print(f"[bootstrap-round] baseline_tflops={bench['median_tflops']:.2f}")
    print(f"[bootstrap-round] next_state=IDEA reason=no_kernel_source_in_repo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
