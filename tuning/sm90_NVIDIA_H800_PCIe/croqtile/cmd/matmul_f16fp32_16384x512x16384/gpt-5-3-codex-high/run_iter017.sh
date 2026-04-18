#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter017_wn96_s2_2cta"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/timing_iter017.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"
