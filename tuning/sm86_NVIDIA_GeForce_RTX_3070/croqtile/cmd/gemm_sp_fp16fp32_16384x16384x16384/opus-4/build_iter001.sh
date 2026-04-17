#!/usr/bin/env bash
set -e
export CHOREO_HOME=/home/albert/workspace/croqtile
export CUTE_HOME=$CHOREO_HOME/extern/cutlass
export CUDA_HOME=/usr/local/cuda-12.2
HEADERS="/home/albert/workspace/croqtile-tuner/.claude/skills/croq-tune/tools/choreo_headers"

SRC="tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/gemm_sp_fp16fp32_16384x16384x16384/opus-4/iter001_sm86_mma_sp_base.cu"
BIN="tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/bin/gemm_sp_fp16fp32_16384x16384x16384/opus-4/iter001_sm86_mma_sp_base"
LOG="tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/perf/gemm_sp_fp16fp32_16384x16384x16384/opus-4/build_iter001.txt"

mkdir -p "$(dirname "$BIN")"
$CUDA_HOME/bin/nvcc -O3 -arch=sm_86 -std=c++17 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
    -D__USE_CUDA_TYPE__ -D__CHOREO_DMA_DIAGNOSIS__ \
    -I"$HEADERS" -I${CUTE_HOME}/include \
    -Xcompiler -static-libstdc++ \
    -L${CUDA_HOME}/lib64 -lcuda \
    -o "$BIN" "$SRC" \
    2>&1 | tee "$LOG"
