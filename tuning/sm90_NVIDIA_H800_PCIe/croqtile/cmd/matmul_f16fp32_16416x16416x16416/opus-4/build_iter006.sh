#!/usr/bin/env bash
set -e
export CHOREO_HOME=/home/albert/workspace/croqtile
export CUTE_HOME=$CHOREO_HOME/extern/cutlass
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
HEADERS="/home/albert/workspace/croqtile-tuner-btune/.claude/skills/croq-tune/tools/choreo_headers"
SRC="tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16416x16416x16416/opus-4/iter006_wn240_s4_1p1c.cu"
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_f16fp32_16416x16416x16416/opus-4/iter006_wn240_s4_1p1c"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_f16fp32_16416x16416x16416/opus-4/build_iter006.txt"
mkdir -p "$(dirname "$BIN")" "$(dirname "$LOG")"
nvcc -O3 -gencode arch=compute_90a,code=sm_90a -std=c++17 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
    -D__USE_CUDA_TYPE__ -D__CHOREO_DMA_DIAGNOSIS__ \
    -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
    --expt-relaxed-constexpr \
    -I"$HEADERS" -I${CUTE_HOME}/include \
    -Xcompiler -static-libstdc++ \
    -L${CUDA_HOME}/lib64 -lcuda \
    -o "$BIN" "$SRC" \
    2>&1 | tee "$LOG"
echo "Build complete: $BIN"
