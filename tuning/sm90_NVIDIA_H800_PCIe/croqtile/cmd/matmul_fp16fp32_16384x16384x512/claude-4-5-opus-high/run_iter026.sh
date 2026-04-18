#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
BIN="$ROOT/tuning/sm90_NVIDIA_H800_PCIe/croqtile/bins/matmul_fp16fp32_16384x16384x512/claude-4-5-opus-high/iter026_tm128_wn128_s4"
"$BIN" "$@"
