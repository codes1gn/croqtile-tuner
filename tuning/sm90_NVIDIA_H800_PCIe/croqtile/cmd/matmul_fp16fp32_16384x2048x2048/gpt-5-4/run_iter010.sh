#!/usr/bin/env bash
set -e
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_f16fp32_16384x2048x2048/gpt-5-4/iter010_wait1"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_f16fp32_16384x2048x2048/gpt-5-4/timing_iter010.txt"
LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}" "$BIN" "$@" 2>&1 | tee "$LOG"