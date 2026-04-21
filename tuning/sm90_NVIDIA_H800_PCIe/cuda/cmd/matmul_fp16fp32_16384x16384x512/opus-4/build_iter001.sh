#!/usr/bin/env bash
set -e
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
GPU="sm90_NVIDIA_H800_PCIe"
SHAPE_KEY="matmul_fp16fp32_16384x16384x512"
MODEL="opus-4"
nvcc -O3 -arch=sm_90 -std=c++17 -I$CUDA_HOME/include \
     -o tuning/$GPU/cuda/bin/$SHAPE_KEY/$MODEL/iter001_adapted \
     tuning/$GPU/cuda/srcs/$SHAPE_KEY/$MODEL/iter001_adapted.cu \
     -lcublas \
     2>&1 | tee tuning/$GPU/cuda/perf/$SHAPE_KEY/$MODEL/build_iter001.txt
