#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

PERF_DIR="$REPO_ROOT/tuning/aitune/cuda/perf/matmul_bf16fp32_512x16384x16384"

"$PERF_DIR/iter001_draft" ${1:-5} ${2:-10}
