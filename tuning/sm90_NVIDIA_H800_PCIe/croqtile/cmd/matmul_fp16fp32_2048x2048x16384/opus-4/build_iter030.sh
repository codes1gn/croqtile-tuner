#!/usr/bin/env bash
set -euo pipefail
export CUTE_HOME=/home/albert/workspace/croqtile/extern/cutlass
export CUDA_HOME=/usr/local/cuda
SRC="tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter030_np_wn160_s4_O2.cu"
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_fp16fp32_2048x2048x16384/opus-4/iter030_np_wn160_s4_O2"
mkdir -p "$(dirname "$BIN")"
${CUDA_HOME}/bin/nvcc -O2 -gencode arch=compute_90a,code=sm_90a -std=c++17 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
    -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
    -D__USE_CUDA_TYPE__ -D__CHOREO_DMA_DIAGNOSIS__ \
    --expt-relaxed-constexpr \
    -I".claude/skills/croq-tune/tools/choreo_headers" -I${CUTE_HOME}/include \
    -Xcompiler -static-libstdc++ -L${CUDA_HOME}/lib64 -lcuda -o "$BIN" "$SRC" 2>&1
echo "Built: $BIN"
