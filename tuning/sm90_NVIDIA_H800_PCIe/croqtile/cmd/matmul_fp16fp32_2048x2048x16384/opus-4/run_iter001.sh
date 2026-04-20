#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_fp16fp32_2048x2048x16384/opus-4/iter001_1p2c_base"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_fp16fp32_2048x2048x16384/opus-4/timing_iter001.txt"
mkdir -p "$(dirname "$LOG")"
"$BIN" "$@" 2>&1 | tee "$LOG"
