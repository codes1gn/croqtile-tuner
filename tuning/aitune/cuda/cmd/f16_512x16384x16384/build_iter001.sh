#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../../../.." && pwd)"
SRC="$ROOT_DIR/tuning/aitune/cuda/srcs/f16_512x16384x16384/iter001_cublas_scaffold.cu"
BIN="$ROOT_DIR/tuning/aitune/cuda/perf/f16_512x16384x16384/iter001_cublas_scaffold"

NVCC_BIN="${NVCC_BIN:-}"
if [[ -z "$NVCC_BIN" ]]; then
  for candidate in \
    "/usr/local/cuda/bin/nvcc" \
    "/usr/local/cuda-13.0/bin/nvcc" \
    "/usr/local/cuda-13/bin/nvcc" \
    "/usr/local/cuda-12.8/bin/nvcc" \
    "/usr/local/cuda-12/bin/nvcc"; do
    if [[ -x "$candidate" ]]; then
      NVCC_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$NVCC_BIN" ]]; then
  echo "nvcc not found. Set NVCC_BIN or install CUDA toolkit." >&2
  exit 127
fi

mkdir -p "$(dirname "$BIN")"
"$NVCC_BIN" -O3 -std=c++17 "$SRC" -lcublas -o "$BIN"
echo "$BIN"
