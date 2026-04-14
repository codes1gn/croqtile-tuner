#!/bin/bash
# Run script for iter001_draft - f16 matmul 512x16384x16384
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

PERF_DIR="$REPO_ROOT/tuning/aitune/cuda/perf/f16_512x16384x16384"
BIN_FILE="$PERF_DIR/iter001_draft"

WARMUP="${1:-5}"
ITERS="${2:-10}"

if [ ! -f "$BIN_FILE" ]; then
    echo "[run] Binary not found: $BIN_FILE"
    echo "[run] Run build_iter001.sh first"
    exit 1
fi

echo "[run] Executing: $BIN_FILE $WARMUP $ITERS"
"$BIN_FILE" "$WARMUP" "$ITERS" 2>&1 | tee "$PERF_DIR/timing_iter001.txt"
