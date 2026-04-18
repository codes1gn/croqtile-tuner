#!/usr/bin/env bash
set -e
export CHOREO_HOME=/home/albert/workspace/croqtile
export CUTE_HOME=$CHOREO_HOME/extern/cutlass
export CUDA_HOME=/usr/local/cuda
HEADERS="/home/albert/workspace/croqtile-tuner/.claude/skills/croq-tune/tools/choreo_headers"
SRC="tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_1024x1024x1024/opus-4/iter010_p1c2_stages5.cu"
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_fp16fp32_1024x1024x1024/opus-4/iter010_p1c2_stages5"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_fp16fp32_1024x1024x1024/opus-4/build_iter010.txt"
mkdir -p "$(dirname "$BIN")" "$(dirname "$LOG")"
nvcc -O3 -gencode arch=compute_90a,code=sm_90a -std=c++17 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
    -D__USE_CUDA_TYPE__ -D__CHOREO_DMA_DIAGNOSIS__ --expt-relaxed-constexpr \
    -I"$HEADERS" -I${CUTE_HOME}/include -Xcompiler -static-libstdc++ \
    -L${CUDA_HOME}/lib64 -lcuda -o "$BIN" "$SRC" 2>&1 | tee "$LOG"
echo "Built: $BIN"
