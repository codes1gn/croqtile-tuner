#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_fp16fp32_1024x1024x1024/opus-4/iter073_p1c2_lb_wait1"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_fp16fp32_1024x1024x1024/opus-4/timing_iter073.txt"
mkdir -p "$(dirname "$LOG")"
"$BIN" "$@" 2>&1 | tee "$LOG"
