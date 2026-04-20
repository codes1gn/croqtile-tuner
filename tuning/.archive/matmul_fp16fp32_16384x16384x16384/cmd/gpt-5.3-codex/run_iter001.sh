#!/usr/bin/env bash
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_fp16fp32_16384x16384x16384/gpt-5.3-codex/iter001_draft"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_fp16fp32_16384x16384x16384/gpt-5.3-codex/timing_iter001.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"
