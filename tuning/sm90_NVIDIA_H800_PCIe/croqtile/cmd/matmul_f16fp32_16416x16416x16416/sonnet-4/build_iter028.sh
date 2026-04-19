#!/usr/bin/env bash
set -e
export CHOREO_HOME=/home/albert/workspace/croqtile
export CUTE_HOME=$CHOREO_HOME/extern/cutlass
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda-12.8/bin:$PATH"
HEADERS="/home/albert/workspace/croqtile-tuner/.claude/skills/croq-tune/tools/choreo_headers"

SRC="tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16416x16416x16416/sonnet-4/iter028_maxreg160.cu"
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_f16fp32_16416x16416x16416/sonnet-4/iter028_maxreg160"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_f16fp32_16416x16416x16416/sonnet-4/build_iter028.txt"

mkdir -p "$(dirname "$BIN")" "$(dirname "$LOG")"
nvcc -O3 -gencode arch=compute_90a,code=sm_90a -std=c++17 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
    -D__USE_CUDA_TYPE__ \
    -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
    --expt-relaxed-constexpr \
    --maxrregcount=160 \
    -I"$HEADERS" -I${CUTE_HOME}/include \
    -Xcompiler -static-libstdc++ \
    -L/usr/local/cuda-12.8/lib64 -lcuda -lcudart \
    -o "$BIN" "$SRC" \
    2>&1 | tee "$LOG"
