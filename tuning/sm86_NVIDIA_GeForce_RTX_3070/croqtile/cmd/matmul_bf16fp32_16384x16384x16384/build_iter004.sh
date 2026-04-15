#!/usr/bin/env bash
set -e

export CHOREO_HOME=/home/albert/workspace/croqtile
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$CHOREO_HOME/build:$PATH

GPU_KEY="sm86_NVIDIA_GeForce_RTX_3070"
DSL="croqtile"
SHAPE_KEY="matmul_bf16fp32_16384x16384x16384"
ITER="iter004_tilek64"

SRC="tuning/$GPU_KEY/$DSL/srcs/$SHAPE_KEY/${ITER}.co"
OUT="tuning/$GPU_KEY/$DSL/cmd/$SHAPE_KEY/${ITER}.cute.result"

echo "Building ${ITER}..."
$CHOREO_HOME/build/choreo -gs -t cute -arch=sm_86 "$SRC" -o "$OUT"
echo "Build complete: $OUT"
