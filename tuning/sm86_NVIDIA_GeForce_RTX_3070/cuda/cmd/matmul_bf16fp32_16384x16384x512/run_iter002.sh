#!/usr/bin/env bash
GPU=sm86_NVIDIA_GeForce_RTX_3070
DSL=cuda
KEY=matmul_bf16fp32_16384x16384x512
tuning/${GPU}/${DSL}/bin/${KEY}/iter002_warp2x4 \
    2>&1 | tee tuning/${GPU}/${DSL}/perf/${KEY}/timing_iter002.txt
