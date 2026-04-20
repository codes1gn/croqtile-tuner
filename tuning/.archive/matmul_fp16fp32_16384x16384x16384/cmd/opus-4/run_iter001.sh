#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_fp16fp32_16384x16384x16384/opus-4/iter001_tierb_sonnet069"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_fp16fp32_16384x16384x16384/opus-4/timing_iter001.txt"
export CHOREO_TIMING_WARMUP=10
export CHOREO_TIMING_REPEAT=50
"$BIN" "$@" 2>&1 | tee "$LOG"
