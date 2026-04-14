#!/bin/bash
# Build script for iter001_draft - matmul f16 512x16384x16384
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
SHAPE_KEY="matmul_f16_512x16384x16384"

SRC_DIR="$REPO_ROOT/tuning/aitune/cuda/srcs/$SHAPE_KEY"
PERF_DIR="$REPO_ROOT/tuning/aitune/cuda/perf/$SHAPE_KEY"

mkdir -p "$PERF_DIR"

SRC_FILE="$SRC_DIR/iter001_draft.cu"
OUT_FILE="$PERF_DIR/iter001_draft"

# CUDA toolkit path
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# Compile with optimizations for Ampere (RTX 3070 = sm_86)
# Override with GPU_ARCH env var if needed
GPU_ARCH="${GPU_ARCH:-sm_86}"

echo "[build] Compiling $SRC_FILE -> $OUT_FILE"
echo "[build] Target architecture: $GPU_ARCH"

nvcc -O3 \
    -arch="$GPU_ARCH" \
    -use_fast_math \
    --expt-relaxed-constexpr \
    -Xcompiler -fPIC \
    -o "$OUT_FILE" \
    "$SRC_FILE" \
    2>&1 | tee "$PERF_DIR/build_iter001.txt"

if [ -f "$OUT_FILE" ]; then
    echo "[build] Success: $OUT_FILE"
else
    echo "[build] FAILED - see $PERF_DIR/build_iter001.txt"
    exit 1
fi
