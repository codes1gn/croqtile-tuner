#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_fp16fp32_16384x16384x512/claude-4-5-opus-high/iter020_tm256_wn200_128sms"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_fp16fp32_16384x16384x512/claude-4-5-opus-high/timing_iter020.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"
