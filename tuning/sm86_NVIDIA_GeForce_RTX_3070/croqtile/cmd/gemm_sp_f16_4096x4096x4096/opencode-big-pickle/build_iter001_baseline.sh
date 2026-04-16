#!/usr/bin/env bash
set -e

export CHOREO_HOME=/home/albert/workspace/croqtile
export CUTE_HOME=$CHOREO_HOME/extern/cutlass
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

HEADERS="/home/albert/workspace/croqtile-tuner/.claude/skills/croq-tune/tools/choreo_headers"

SRC="tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/gemm_sp_f16_4096x4096x4096/opencode-big-pickle/iter001_baseline.cu"
BIN="tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/bin/gemm_sp_f16_4096x4096x4096/opencode-big-pickle/iter001_baseline"
LOG="tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/perf/gemm_sp_f16_4096x4096x4096/opencode-big-pickle/build_iter001_baseline.txt"

mkdir -p "$(dirname "$BIN")"

# Compile for sm_86 (RTX 3070)
nvcc -O3 \
    -arch=sm_86 \
    -std=c++17 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
    -D__CHOREO_TARGET_CUTE__ \
    -D__USE_CUDA_TYPE__ \
    -I"$HEADERS" \
    -I${CUTE_HOME}/include \
    -Xcompiler -static-libstdc++ \
    -L${CUDA_HOME}/lib64 -lcuda \
    -o "$BIN" "$SRC" \
    2>&1 | tee "$LOG"

echo "Build complete: $BIN"
