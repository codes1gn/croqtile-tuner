#!/usr/bin/env bash
# Usage: SRC=path/to/iterXXX.py bash ncu_bench_any.sh
set -euo pipefail
REPO="/home/albert/workspace/croqtile-tuner"
: "${SRC:?Set SRC= to helion iterXXX.py path}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export HELION_AUTOTUNE_EFFORT="${HELION_AUTOTUNE_EFFORT:-none}"
export PYTHONUNBUFFERED=1
cd "$REPO"
python3 -u -c "
import runpy
ns = runpy.run_path('${SRC}', run_name='ncu_bench')
ns['bench'](warmup=1, iters=1)
"
