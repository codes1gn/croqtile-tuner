#!/usr/bin/env bash
set -e

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

SRC="tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/gemm_sp_f16_4096x4096x4096/opencode-big-pickle/iter001_baseline_cublas.cu"
BIN="tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/bin/gemm_sp_f16_4096x4096x4096/opencode-big-pickle/iter001_baseline_cublas"
LOG="tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/perf/gemm_sp_f16_4096x4096x4096/opencode-big-pickle/build_iter001_baseline_cublas.txt"

mkdir -p "$(dirname "$BIN")"

nvcc -O3 \
    -arch=sm_86 \
    -std=c++17 \
    -I${CUDA_HOME}/include \
    -L${CUDA_HOME}/lib64 -lcublas -lcudart \
    -o "$BIN" "$SRC" \
    2>&1 | tee "$LOG"

echo "Build complete: $BIN"
