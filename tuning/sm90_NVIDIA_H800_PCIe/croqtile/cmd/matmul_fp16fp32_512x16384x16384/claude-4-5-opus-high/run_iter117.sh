#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_fp16fp32_512x16384x16384/claude-4-5-opus-high/iter117_tma_outside"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_fp16fp32_512x16384x16384/claude-4-5-opus-high/timing_iter117.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"
