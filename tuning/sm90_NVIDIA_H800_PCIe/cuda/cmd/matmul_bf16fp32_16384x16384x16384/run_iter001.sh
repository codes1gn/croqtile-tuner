#!/usr/bin/env bash
tuning/sm90_NVIDIA_H800_PCIe/cuda/bin/matmul_bf16fp32_16384x16384x16384/iter001_draft "$@" \
    2>&1 | tee tuning/sm90_NVIDIA_H800_PCIe/cuda/perf/matmul_bf16fp32_16384x16384x16384/timing_iter001.txt
