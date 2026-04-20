#!/usr/bin/env bash
set -e
export CHOREO_HOME=/home/albert/workspace/croqtile
export CUTE_HOME=$CHOREO_HOME/extern/cutlass
export CUDA_HOME=/usr/local/cuda-13.0
HEADERS="/home/albert/workspace/croqtile-tuner/.claude/skills/croq-tune/tools/choreo_headers"
mkdir -p tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_fp16fp32_2048x16384x2048/opus-4
${CUDA_HOME}/bin/nvcc -O3 -gencode arch=compute_90a,code=sm_90a -std=c++17 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
    -D__USE_CUDA_TYPE__ -D__CHOREO_DMA_DIAGNOSIS__ \
    --expt-relaxed-constexpr --use_fast_math -maxrregcount 114 \
    -I"$HEADERS" -I${CUTE_HOME}/include -Xcompiler -static-libstdc++ \
    -L${CUDA_HOME}/lib64 -lcuda \
    -o tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_fp16fp32_2048x16384x2048/opus-4/iter028_maxreg114 \
    tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x16384x2048/opus-4/iter028_maxreg114.cu 2>&1 | tail -3
