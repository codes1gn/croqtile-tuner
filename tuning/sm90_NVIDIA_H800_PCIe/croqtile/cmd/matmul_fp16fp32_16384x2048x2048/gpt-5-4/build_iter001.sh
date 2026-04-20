#!/usr/bin/env bash
set -e
export CHOREO_HOME=/home/albert/workspace/croqtile
export CUTE_HOME=$CHOREO_HOME/extern/cutlass
export CUDA_HOME=/usr/local/cuda-12.8
HEADERS="/home/albert/workspace/croqtile-tuner/.claude/skills/croq-tune/tools/choreo_headers"

SRC="tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x2048x2048/gpt-5-4/iter001_draft.cu"
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_f16fp32_16384x2048x2048/gpt-5-4/iter001_draft"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_f16fp32_16384x2048x2048/gpt-5-4/build_iter001.txt"

mkdir -p "$(dirname "$BIN")" "$(dirname "$LOG")"
/usr/local/cuda-12.8/bin/nvcc -O3 -gencode arch=compute_90a,code=sm_90a -std=c++17 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
    -D__USE_CUDA_TYPE__ -D__CHOREO_DMA_DIAGNOSIS__ \
    --expt-relaxed-constexpr \
    -Xcompiler -static-libstdc++ \
    -I"$HEADERS" -I${CUTE_HOME}/include \
    "$SRC" -o "$BIN" \
    -L/usr/local/cuda-12.8/lib64 -lcudart -lcuda \
    2>&1 | tee "$LOG"