#!/usr/bin/env bash
# Single-kernel bench for Nsight Compute (warmup=1, iters=1).
set -euo pipefail
REPO="/home/albert/workspace/croqtile-tuner"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export HELION_AUTOTUNE_EFFORT="${HELION_AUTOTUNE_EFFORT:-none}"
export PYTHONUNBUFFERED=1
cd "$REPO"
python3 -u -c "
import runpy
ns = runpy.run_path(
    'tuning/sm90_NVIDIA_H800_PCIe/helion/srcs/blockscale_gemm_e4m3fp16fp32_8192x8192x8192/opus-4/iter001_blockscale_dotscaled.py',
    run_name='ncu_bench',
)
ns['bench'](warmup=1, iters=1)
"
