#!/usr/bin/env bash
set -euo pipefail
REPO="/home/albert/workspace/croqtile-tuner"
SRC="$REPO/tuning/sm90_NVIDIA_H800_PCIe/helion/srcs/blockscale_gemm_e4m3fp16fp32_8192x8192x8192/opus-4/iter001_blockscale_dotscaled.py"
python3 -c "import ast; ast.parse(open('$SRC').read()); print('parse OK')"
python3 -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('k', '$SRC')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('import OK')
"
