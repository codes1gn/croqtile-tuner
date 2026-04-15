#!/usr/bin/env bash
set -e

GPU_KEY="sm86_NVIDIA_GeForce_RTX_3070"
DSL="croqtile"
SHAPE_KEY="matmul_bf16fp32_16384x16384x16384"
ITER="iter002_tile32x64"

SCRIPT="tuning/$GPU_KEY/$DSL/cmd/$SHAPE_KEY/${ITER}.cute.result"
LOG="tuning/$GPU_KEY/$DSL/perf/$SHAPE_KEY/timing_iter002.txt"

echo "Running $ITER..."
CHOREO_TIMING_WARMUP=10 CHOREO_TIMING_REPEAT=50 bash "$SCRIPT" --execute 2>&1 | tee "$LOG"
