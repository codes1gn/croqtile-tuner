#!/usr/bin/env bash
set -e
export CHOREO_HOME=/home/albert/workspace/croqtile
export CUTE_HOME=$CHOREO_HOME/extern/cutlass
export CUDA_HOME=/usr/local/cuda
HEADERS=".claude/skills/croq-tune/tools/choreo_headers"

SRC="tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter001_draft.cu"
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter001_draft"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/build_iter001.txt"

mkdir -p "$(dirname "$BIN")" "$(dirname "$LOG")"

${CUDA_HOME}/bin/nvcc \
    -gencode arch=compute_90a,code=sm_90a \
    -std=c++17 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
    -D__CHOREO_TARGET_CUTE__ \
    -D__USE_CUDA_TYPE__ \
    -D__CHOREO_DMA_DIAGNOSIS__ \
    --expt-relaxed-constexpr \
    -Xcompiler -static-libstdc++ \
    -I"${HEADERS}" -I"${CUTE_HOME}/include" \
    -L${CUDA_HOME}/lib64 -lcuda \
    -O3 \
    -o "$BIN" "$SRC" \
    2>&1 | tee "$LOG"
