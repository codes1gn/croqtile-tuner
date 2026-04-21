#!/usr/bin/env bash
set -e
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 tuning/sm90_NVIDIA_H800_PCIe/tilelang/srcs/matmul_f16fp32_16416x16416x16416/claude-opus-4/iter001_base_pipelined.py \
    2>&1 | tee tuning/sm90_NVIDIA_H800_PCIe/tilelang/perf/matmul_f16fp32_16416x16416x16416/claude-opus-4/timing_iter001.txt
