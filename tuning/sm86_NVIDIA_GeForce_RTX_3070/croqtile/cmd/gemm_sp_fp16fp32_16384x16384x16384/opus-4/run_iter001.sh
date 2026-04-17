#!/usr/bin/env bash
BIN="tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/bin/gemm_sp_fp16fp32_16384x16384x16384/opus-4/iter001_sm86_mma_sp_base"
LOG="tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/perf/gemm_sp_fp16fp32_16384x16384x16384/opus-4/timing_iter001.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"
