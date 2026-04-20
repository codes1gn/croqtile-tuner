#!/usr/bin/env bash
set -e
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter004_stage3"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/timing_iter004.txt"
LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}" "$BIN" "$@" 2>&1 | tee "$LOG"
