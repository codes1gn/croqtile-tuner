#!/usr/bin/env bash
set -e
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_e4m3fp16_8192x8192x4096/opus-4/iter139_both_mbar"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_e4m3fp16_8192x8192x4096/opus-4/timing_iter139.txt"
cd /home/albert/workspace/croqtile-tuner-ctune
"$BIN" "$@" 2>&1 | tee "$LOG"
