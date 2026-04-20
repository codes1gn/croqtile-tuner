#!/usr/bin/env bash
GPU="sm90_NVIDIA_H800_PCIe"
SHAPE="matmul_f16fp32_16416x16416x16416"
MODEL="opus-4"
BASE="tuning/${GPU}/cuda"

"${BASE}/bin/${SHAPE}/${MODEL}/iter001_mma_tiled" "$@" \
    2>&1 | tee "${BASE}/perf/${SHAPE}/${MODEL}/timing_iter001.txt"
