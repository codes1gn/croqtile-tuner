#!/usr/bin/env bash
set -e

GPU="sm90_NVIDIA_H800_PCIe"
DSL="cuda"
SHAPE_KEY="gemm_sp_e4m3fp16_4096x8192x8192"
MODEL="opus-4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/home/albert/workspace/croqtile-tuner"

SRC_DIR="$ROOT_DIR/tuning/$GPU/$DSL/srcs/$SHAPE_KEY/$MODEL"
BIN_DIR="$ROOT_DIR/tuning/$GPU/$DSL/bin/$SHAPE_KEY/$MODEL"
PERF_DIR="$ROOT_DIR/tuning/$GPU/$DSL/perf/$SHAPE_KEY/$MODEL"

mkdir -p "$BIN_DIR" "$PERF_DIR"

CUSPARSELT_ROOT="/home/albert/.local/lib/python3.10/site-packages/nvidia/cusparselt"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

nvcc -O3 -arch=sm_90a -std=c++17 \
    -I"$CUDA_HOME/include" \
    -I"$CUSPARSELT_ROOT/include" \
    -L"$CUDA_HOME/lib64" \
    -L"$CUSPARSELT_ROOT/lib" \
    -lcuda -lcusparseLt \
    --expt-relaxed-constexpr \
    -Xcompiler "-Wno-psabi" \
    -o "$BIN_DIR/iter001_baseline_wgmma_1p2c" \
    "$SRC_DIR/iter001_baseline_wgmma_1p2c.cu" \
    2>&1 | tee "$PERF_DIR/build_iter001.txt"

echo "Build complete: $BIN_DIR/iter001_baseline_wgmma_1p2c"
