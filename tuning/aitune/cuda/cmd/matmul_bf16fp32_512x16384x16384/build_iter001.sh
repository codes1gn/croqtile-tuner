#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
SRC_DIR="$REPO_ROOT/tuning/aitune/cuda/srcs/matmul_bf16fp32_512x16384x16384"
BIN_DIR="$REPO_ROOT/tuning/aitune/cuda/bin/matmul_bf16fp32_512x16384x16384"

mkdir -p "$BIN_DIR"

export PATH=/usr/local/cuda/bin:$PATH

nvcc -O3 -arch=sm_86 \
    "$SRC_DIR/iter001_draft.cu" \
    -o "$BIN_DIR/iter001_draft"

echo "Built: $BIN_DIR/iter001_draft"
