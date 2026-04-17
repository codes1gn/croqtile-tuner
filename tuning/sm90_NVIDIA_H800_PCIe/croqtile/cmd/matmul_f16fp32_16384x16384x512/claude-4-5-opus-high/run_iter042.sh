#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter042_persis_dlcmca"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/timing_iter042.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"
