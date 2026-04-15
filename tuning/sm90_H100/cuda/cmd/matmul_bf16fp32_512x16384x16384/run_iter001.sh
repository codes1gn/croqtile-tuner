#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
BIN_DIR="$REPO_ROOT/tuning/sm90_H100/cuda/bin/matmul_bf16fp32_512x16384x16384"

"$BIN_DIR/iter001_draft" "$@"
