#!/usr/bin/env python3
"""Mock opencode simulator for testing CroqTuner without a GPU.

Usage: python3 mock_opencode.py <shape_key> <max_iterations>

Simulates an opencode tuning session by printing iteration lines to stdout
and writing fake checkpoint/results files to ../croktile_paper/tuning/.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: mock_opencode.py <shape_key> <max_iterations>", file=sys.stderr)
        sys.exit(1)

    shape_key = sys.argv[1]
    max_iter = int(sys.argv[2])

    tuning_dir = Path(os.environ.get(
        "CROQTUNER_TUNING_DIR",
        str(Path(__file__).parent.parent.parent / "croktile_paper" / "tuning"),
    ))

    logs_dir = tuning_dir / "logs" / shape_key
    srcs_dir = tuning_dir / "srcs" / shape_key
    perf_dir = tuning_dir / "perf" / shape_key
    checkpoints_dir = tuning_dir / "checkpoints"

    for d in [logs_dir, srcs_dir, perf_dir, checkpoints_dir]:
        d.mkdir(parents=True, exist_ok=True)

    baseline_tflops = random.uniform(30.0, 100.0)
    print(f"[mock] Baseline measurement: {baseline_tflops:.1f} TFLOPS")
    sys.stdout.flush()

    results_tsv = logs_dir / "results.tsv"
    with open(results_tsv, "w") as f:
        f.write(f"# Mock tuning: {shape_key}\n")
        f.write("iter\tkernel\ttflops\thw_eff_pct\tdecision\tbottleneck\tidea_summary\trun_command\n")
        f.write(f"0\tseed.cu\t{baseline_tflops:.1f}\t0.0\tBASELINE\t-\tbaseline\t-\n")

    best_tflops = baseline_tflops
    best_iter = 0

    sim_iters = min(max_iter, random.randint(5, 15))

    for i in range(1, sim_iters + 1):
        time.sleep(random.uniform(0.3, 1.0))

        delta = random.uniform(-10.0, 15.0)
        tflops = max(10.0, best_tflops + delta)
        decision = "KEEP" if tflops > best_tflops else "DISCARD"

        if decision == "KEEP":
            best_tflops = tflops
            best_iter = i

        bottleneck = random.choice(["compute_bound", "smem_bw", "l2_thrash", "warp_stall", "dram_bw"])
        idea = random.choice([
            "pipeline depth 3",
            "swizzle factor 128",
            "unroll factor 4",
            "barrier depth 2",
            "launch_bounds 128",
            "warp_spec 1p2c",
            "tile_k 128",
            "prefetch hints",
        ])

        print(f"iter{i:03d}: {idea} — {tflops:.1f} TFLOPS ({decision})")
        sys.stdout.flush()

        with open(results_tsv, "a") as f:
            f.write(f"{i}\titer{i:03d}_{idea.replace(' ','_')}.cu\t{tflops:.1f}\t0.0\t{decision}\t{bottleneck}\t{idea}\t-\n")

        checkpoint = {
            "key": shape_key,
            "current_iter": i,
            "best_iter": best_iter,
            "best_tflops": round(best_tflops, 1),
            "baseline_tflops": round(baseline_tflops, 1),
            "best_kernel": f"tuning/srcs/{shape_key}/iter{best_iter:03d}.cu",
            "status": "active" if i < sim_iters else "done",
            "max_iter": max_iter,
        }
        (checkpoints_dir / f"{shape_key}.json").write_text(json.dumps(checkpoint, indent=2))

    print(f"[mock] Tuning complete. Best: {best_tflops:.1f} TFLOPS at iter {best_iter}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
