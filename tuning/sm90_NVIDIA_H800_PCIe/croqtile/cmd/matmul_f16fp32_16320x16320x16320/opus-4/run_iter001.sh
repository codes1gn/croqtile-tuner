#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_f16fp32_16320x16320x16320/opus-4/iter001_wn192_s4_1p1c"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_f16fp32_16320x16320x16320/opus-4/timing_iter001.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"
