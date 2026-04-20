#!/usr/bin/env bash
set -e
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

GPU="sm90_NVIDIA_H800_PCIe"
SHAPE="matmul_f16fp32_16416x16416x16416"
MODEL="opus-4"
BASE="tuning/${GPU}/cuda"

mkdir -p "${BASE}/bin/${SHAPE}/${MODEL}"
mkdir -p "${BASE}/perf/${SHAPE}/${MODEL}"

nvcc -O3 -arch=sm_90 -std=c++17 --maxrregcount=128 \
     -I${CUDA_HOME}/include \
     -lcublas \
     -o "${BASE}/bin/${SHAPE}/${MODEL}/iter002_reduced_regs" \
     "${BASE}/srcs/${SHAPE}/${MODEL}/iter002_reduced_regs.cu" \
     2>&1 | tee "${BASE}/perf/${SHAPE}/${MODEL}/build_iter002.txt"

echo "Build complete: ${BASE}/bin/${SHAPE}/${MODEL}/iter002_reduced_regs"
