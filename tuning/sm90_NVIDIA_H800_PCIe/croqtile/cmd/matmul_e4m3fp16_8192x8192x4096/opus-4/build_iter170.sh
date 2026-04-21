#!/usr/bin/env bash
set -e
export CHOREO_HOME=/home/albert/workspace/croqtile
export CUTE_HOME=$CHOREO_HOME/extern/cutlass
export CUDA_HOME=/usr/local/cuda
HEADERS="/home/albert/workspace/croqtile-tuner-ctune/.cursor/skills/cursor-croq-tune/tools/choreo_headers"

SRC="tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter170_mbar_inval.cu"
BIN="tuning/sm90_NVIDIA_H800_PCIe/croqtile/bin/matmul_e4m3fp16_8192x8192x4096/opus-4/iter170_mbar_inval"
LOG="tuning/sm90_NVIDIA_H800_PCIe/croqtile/perf/matmul_e4m3fp16_8192x8192x4096/opus-4/build_iter170.txt"

mkdir -p "$(dirname "$BIN")" "$(dirname "$LOG")"
nvcc -O3 -gencode arch=compute_90a,code=sm_90a -std=c++17 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
    -D__USE_CUDA_TYPE__ -D__CHOREO_DMA_DIAGNOSIS__ \
    --expt-relaxed-constexpr --use_fast_math -Xptxas -O3 \
    -I"$HEADERS" -I${CUTE_HOME}/include \
    -Xcompiler -static-libstdc++ \
    -L${CUDA_HOME}/lib64 -lcuda \
    -o "$BIN" "$SRC" \
    2>&1 | tee "$LOG"

echo "[build_iter170] Built: $BIN"
