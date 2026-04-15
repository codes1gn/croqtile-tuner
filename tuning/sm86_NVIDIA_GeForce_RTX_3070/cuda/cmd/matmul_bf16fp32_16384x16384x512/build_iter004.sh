#!/usr/bin/env bash
set -e
export PATH=/usr/local/cuda/bin:$PATH
GPU=sm86_NVIDIA_GeForce_RTX_3070
DSL=cuda
KEY=matmul_bf16fp32_16384x16384x512
nvcc -O3 -arch=sm_86 -std=c++17 -Xptxas -v --maxrregcount=56 \
     -o tuning/${GPU}/${DSL}/bin/${KEY}/iter004_maxr56 \
     tuning/${GPU}/${DSL}/srcs/${KEY}/iter004_maxr56.cu \
     2>&1 | tee tuning/${GPU}/${DSL}/perf/${KEY}/build_iter004.txt
