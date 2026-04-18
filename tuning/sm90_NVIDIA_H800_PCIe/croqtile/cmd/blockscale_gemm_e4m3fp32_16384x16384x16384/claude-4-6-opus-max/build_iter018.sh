#!/usr/bin/env bash
set -e
export CHOREO_HOME=/home/albert/workspace/croqtile
export CUTE_HOME=$CHOREO_HOME/extern/cutlass
export CUDA_HOME=/usr/local/cuda
HEADERS=".claude/skills/croq-tune/tools/choreo_headers"

SRC="tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/blockscale_gemm_e4m3fp32_16384x16384x16384/claude-4-6-opus-max/iter018_scale_dma_smem.cu"
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/blockscale_gemm_e4m3fp32_16384x16384x16384/claude-4-6-opus-max/iter018_scale_dma_smem"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/blockscale_gemm_e4m3fp32_16384x16384x16384/claude-4-6-opus-max/build_iter018.txt"

mkdir -p "$(dirname "$BIN")" "$(dirname "$LOG")"
nvcc -O3 -gencode arch=compute_90a,code=sm_90a -std=c++17 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
    -D__USE_CUDA_TYPE__ -D__CHOREO_DMA_DIAGNOSIS__ \
    --expt-relaxed-constexpr \
    -I"$HEADERS" -I${CUTE_HOME}/include \
    -Xcompiler -static-libstdc++ \
    -L${CUDA_HOME}/lib64 -lcuda \
    -o "$BIN" "$SRC" \
    2>&1 | tee "$LOG"
