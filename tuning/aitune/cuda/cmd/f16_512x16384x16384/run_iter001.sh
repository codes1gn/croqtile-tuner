#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../../../.." && pwd)"
BIN="$ROOT_DIR/tuning/aitune/cuda/perf/f16_512x16384x16384/iter001_cublas_scaffold"

"$BIN" 512 16384 16384 3 10
