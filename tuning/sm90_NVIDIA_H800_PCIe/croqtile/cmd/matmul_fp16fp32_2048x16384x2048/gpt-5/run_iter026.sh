#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_f16fp32_2048x16384x2048/gpt-5/iter026_padded_repack"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_f16fp32_2048x16384x2048/gpt-5/timing_iter026.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"