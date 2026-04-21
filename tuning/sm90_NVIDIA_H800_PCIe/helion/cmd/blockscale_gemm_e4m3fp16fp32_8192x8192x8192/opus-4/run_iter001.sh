#!/usr/bin/env bash
set -euo pipefail
REPO="/home/albert/workspace/croqtile-tuner"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export HELION_AUTOTUNE_EFFORT="${HELION_AUTOTUNE_EFFORT:-none}"
cd "$REPO"
python3 tuning/sm90_NVIDIA_H800_PCIe/helion/srcs/blockscale_gemm_e4m3fp16fp32_8192x8192x8192/opus-4/iter001_blockscale_dotscaled.py \
  2>&1 | tee tuning/sm90_NVIDIA_H800_PCIe/helion/perf/blockscale_gemm_e4m3fp16fp32_8192x8192x8192/opus-4/timing_iter001.txt
