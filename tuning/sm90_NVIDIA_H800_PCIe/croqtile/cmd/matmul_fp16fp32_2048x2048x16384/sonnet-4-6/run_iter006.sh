#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter006_pingpong_wn128"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/timing_iter006.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"
