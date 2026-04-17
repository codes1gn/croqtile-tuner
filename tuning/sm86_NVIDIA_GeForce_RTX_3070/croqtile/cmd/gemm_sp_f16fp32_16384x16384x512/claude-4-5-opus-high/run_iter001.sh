#!/usr/bin/env bash
# Run script for iter001_baseline (gemm_sp_f16fp32_16384x16384x512)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
GPU="sm86_NVIDIA_GeForce_RTX_3070"
DSL="croqtile"
KEY="gemm_sp_f16fp32_16384x16384x512"
MODEL="claude-4-5-opus-high"

BIN="$REPO_ROOT/tuning/$GPU/$DSL/bin/$KEY/$MODEL/iter001_baseline"
LOG="$REPO_ROOT/tuning/$GPU/$DSL/perf/$KEY/$MODEL/timing_iter001.txt"

mkdir -p "$(dirname "$LOG")"

"$BIN" "$@" 2>&1 | tee "$LOG"
