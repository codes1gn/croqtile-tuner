#!/usr/bin/env bash
GPU="sm90_NVIDIA_H800_PCIe"
SHAPE="matmul_f16fp32_16384x16384x16384"
MODEL="opus-4"
ITER="iter002_large_tiles"

tuning/$GPU/cuda/bin/$SHAPE/$MODEL/$ITER \
    2>&1 | tee tuning/$GPU/cuda/perf/$SHAPE/$MODEL/timing_iter002.txt
