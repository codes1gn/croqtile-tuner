#!/usr/bin/env bash
BIN="tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/bin/matmul_bf16fp32_512x512x512/opus-4/iter001_baseline"
LOG="tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/perf/matmul_bf16fp32_512x512x512/opus-4/timing_iter001.txt"
"$BIN" --execute 2>&1 | tee "$LOG"
