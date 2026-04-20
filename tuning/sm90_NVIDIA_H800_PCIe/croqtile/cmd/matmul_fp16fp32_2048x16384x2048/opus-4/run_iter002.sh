#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_fp16fp32_2048x16384x2048/opus-4/iter002_1p2c_tm128"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_fp16fp32_2048x16384x2048/opus-4/timing_iter002.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"
