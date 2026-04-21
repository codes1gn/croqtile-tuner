#!/usr/bin/env bash
set -e
SRC="tuning/sm90_NVIDIA_H800_PCIe/tilelang/srcs/matmul_f16fp32_16416x16416x16416/claude-opus-4/iter001_base_pipelined.py"
python3 -c "import ast; ast.parse(open('$SRC').read()); print('parse OK')"
python3 -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('kernel', '$SRC')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
print('import OK')
"
