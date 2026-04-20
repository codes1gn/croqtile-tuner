#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_f16fp32_16416x16416x16416/opus-4/iter001_1p2c_w1_ku64"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_f16fp32_16416x16416x16416/opus-4/timing_iter001.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"
