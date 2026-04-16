#!/usr/bin/env bash
set -e
export PATH="/usr/local/cuda/bin:$PATH"
nvcc -O3 -arch=sm_90 -std=c++17 -I/usr/local/cuda/include \
     -lcublas \
     -o tuning/sm90_NVIDIA_H800_PCIe/cuda/bin/matmul_bf16fp32_16384x16384x16384/iter006_coalesced \
     tuning/sm90_NVIDIA_H800_PCIe/cuda/srcs/matmul_bf16fp32_16384x16384x16384/iter006_coalesced.cu \
     2>&1 | tee tuning/sm90_NVIDIA_H800_PCIe/cuda/perf/matmul_bf16fp32_16384x16384x16384/build_iter006.txt
