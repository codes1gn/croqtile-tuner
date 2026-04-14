#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

SRC_DIR="$REPO_ROOT/tuning/aitune/cuda/srcs/matmul_bf16fp32_512x16384x16384"
PERF_DIR="$REPO_ROOT/tuning/aitune/cuda/perf/matmul_bf16fp32_512x16384x16384"
mkdir -p "$PERF_DIR"

nvcc -O3 -arch=sm_86 \
    -I/usr/local/cuda/include \
    "$SRC_DIR/iter001_draft.cu" \
    -o "$PERF_DIR/iter001_draft" \
    2>&1 | tee "$PERF_DIR/build_iter001.txt"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "[build] SUCCESS: iter001_draft"
else
    echo "[build] FAILED: iter001_draft"
    exit 1
fi
