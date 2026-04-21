#!/usr/bin/env bash
set -e
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
GPU="sm90_NVIDIA_H800_PCIe"
SHAPE="matmul_f16fp32_16384x16384x16384"
MODEL="opus-4"
ITER="iter018_wmma_unroll"

nvcc -O3 -arch=sm_90 -std=c++17 --maxrregcount=128 \
     -I$CUDA_HOME/include \
     -o tuning/$GPU/cuda/bin/$SHAPE/$MODEL/$ITER \
     tuning/$GPU/cuda/srcs/$SHAPE/$MODEL/${ITER}.cu \
     -lcublas \
     2>&1 | tee tuning/$GPU/cuda/perf/$SHAPE/$MODEL/build_iter018.txt
