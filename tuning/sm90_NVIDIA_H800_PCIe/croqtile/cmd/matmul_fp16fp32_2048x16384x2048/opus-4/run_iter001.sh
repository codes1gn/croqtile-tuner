#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_fp16fp32_2048x16384x2048/opus-4/iter001_base_wait2"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_fp16fp32_2048x16384x2048/opus-4/timing_iter001.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"
