#!/usr/bin/env bash
# Build script for iter001_baseline (gemm_sp_f16fp32_16384x16384x512)
# Phase 1: .co -> .cu via choreo, Phase 2: nvcc compile
set -e

export CHOREO_HOME=/home/albert/workspace/croqtile
export CUTE_HOME=$CHOREO_HOME/extern/cutlass
export CUDA_HOME=/usr/local/cuda

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
GPU="sm86_NVIDIA_GeForce_RTX_3070"
DSL="croqtile"
KEY="gemm_sp_f16fp32_16384x16384x512"
MODEL="claude-4-5-opus-high"

SRC_DIR="$REPO_ROOT/tuning/$GPU/$DSL/srcs/$KEY/$MODEL"
CMD_DIR="$REPO_ROOT/tuning/$GPU/$DSL/cmd/$KEY/$MODEL"
BIN_DIR="$REPO_ROOT/tuning/$GPU/$DSL/bin/$KEY/$MODEL"
PERF_DIR="$REPO_ROOT/tuning/$GPU/$DSL/perf/$KEY/$MODEL"
HEADERS="$REPO_ROOT/.claude/skills/croq-tune/tools/choreo_headers"

mkdir -p "$BIN_DIR" "$PERF_DIR"

CO_FILE="$SRC_DIR/iter001_baseline.co"
CU_FILE="$SRC_DIR/iter001_baseline.cu"
RESULT_FILE="$CMD_DIR/iter001_baseline.cute.result"
BIN_FILE="$BIN_DIR/iter001_baseline"
LOG_FILE="$PERF_DIR/build_iter001.txt"

echo "[build_iter001] Phase 1: choreo .co -> .cute.result"
$CHOREO_HOME/build/choreo -gs -t cute -arch=sm_86 "$CO_FILE" -o "$RESULT_FILE" 2>&1 | tee "$LOG_FILE"

echo "[build_iter001] Extracting .cu from .cute.result"
# Extract the CUDA source from the heredoc in the .cute.result script
sed -n '/^cat << .CHOREO_CUTE_EOF./,/^.CHOREO_CUTE_EOF$/p' "$RESULT_FILE" | sed '1d;$d' > "$CU_FILE"

echo "[build_iter001] Phase 2: nvcc compile .cu -> binary"
nvcc -O3 -arch=sm_86 -std=c++17 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
    -D__USE_CUDA_TYPE__ \
    -I"$HEADERS" -I${CUTE_HOME}/include \
    -Xcompiler -static-libstdc++ \
    -L${CUDA_HOME}/lib64 -lcuda \
    -o "$BIN_FILE" "$CU_FILE" \
    2>&1 | tee -a "$LOG_FILE"

echo "[build_iter001] Build complete: $BIN_FILE"
