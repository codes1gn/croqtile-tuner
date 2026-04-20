#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_e4m3fp16fp32_8192x8192x8192/opus-4/iter001_warpspec_1p1c"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_e4m3fp16fp32_8192x8192x8192/opus-4/timing_iter001.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"
