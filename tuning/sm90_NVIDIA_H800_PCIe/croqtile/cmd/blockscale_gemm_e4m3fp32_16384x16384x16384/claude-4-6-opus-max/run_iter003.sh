#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/blockscale_gemm_e4m3fp32_16384x16384x16384/claude-4-6-opus-max/iter003_warpspec_1p1c_lbounds"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/blockscale_gemm_e4m3fp32_16384x16384x16384/claude-4-6-opus-max/timing_iter003.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"
