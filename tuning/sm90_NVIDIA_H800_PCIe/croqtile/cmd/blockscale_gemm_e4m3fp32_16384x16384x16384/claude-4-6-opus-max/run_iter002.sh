#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/blockscale_gemm_e4m3fp32_16384x16384x16384/claude-4-6-opus-max/iter002_warpspec_1p1c_2stage"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/blockscale_gemm_e4m3fp32_16384x16384x16384/claude-4-6-opus-max/timing_iter002.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"
