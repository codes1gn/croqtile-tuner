#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
SRC="$ROOT/tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_16384x16384x512/claude-4-5-opus-high/iter026_tm128_wn128_s4.cu"
BIN="$ROOT/tuning/sm90_NVIDIA_H800_PCIe/croqtile/bins/matmul_fp16fp32_16384x16384x512/claude-4-5-opus-high/iter026_tm128_wn128_s4"
mkdir -p "$(dirname "$BIN")"
nvcc -arch=sm_90a -O3 --use_fast_math \
  -I"$ROOT/.claude/skills/croq-tune/tools/choreo_headers" \
  "$SRC" -o "$BIN" -lcuda
echo "Built: $BIN"
