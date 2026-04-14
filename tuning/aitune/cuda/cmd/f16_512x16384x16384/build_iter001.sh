#!/bin/bash
# Build script for iter001_draft - f16 matmul 512x16384x16384
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

SRC_DIR="$REPO_ROOT/tuning/aitune/cuda/srcs/f16_512x16384x16384"
PERF_DIR="$REPO_ROOT/tuning/aitune/cuda/perf/f16_512x16384x16384"

mkdir -p "$PERF_DIR"

SRC_FILE="$SRC_DIR/iter001_draft.cu"
OUT_FILE="$PERF_DIR/iter001_draft"

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

echo "[build] Success: $OUT_FILE"
