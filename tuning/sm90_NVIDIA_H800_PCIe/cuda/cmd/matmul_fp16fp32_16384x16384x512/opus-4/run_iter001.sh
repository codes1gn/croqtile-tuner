#!/usr/bin/env bash
GPU="sm90_NVIDIA_H800_PCIe"
SHAPE_KEY="matmul_fp16fp32_16384x16384x512"
MODEL="opus-4"
tuning/$GPU/cuda/bin/$SHAPE_KEY/$MODEL/iter001_adapted "$@" \
    2>&1 | tee tuning/$GPU/cuda/perf/$SHAPE_KEY/$MODEL/timing_iter001.txt
