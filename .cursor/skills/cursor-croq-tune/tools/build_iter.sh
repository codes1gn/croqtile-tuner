#!/usr/bin/env bash
# build_iter.sh: Build a .cu file using the correct direct nvcc flags that achieved 408.8T
# Usage: bash build_iter.sh <input.cu> <output_binary>
set -e

INPUT_CU="$1"
OUTPUT_BIN="$2"

if [[ -z "$INPUT_CU" || -z "$OUTPUT_BIN" ]]; then
  echo "Usage: bash build_iter.sh <input.cu> <output_binary>" >&2
  exit 1
fi

HEADERS="$(dirname "$(realpath "$0")")/choreo_headers"
NVCC_FLAGS="-gencode arch=compute_90a,code=sm_90a -std=c++17 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
    -Xcompiler -static-libstdc++ -lcuda -O3 \
    -D__USE_CUDA_TYPE__ -D__CHOREO_DMA_DIAGNOSIS__ \
    --expt-relaxed-constexpr \
    -Xptxas --allow-expensive-optimizations=true,-O3 \
    --maxrregcount=224"

${CUDA_HOME}/bin/nvcc $NVCC_FLAGS \
  -I${CUTE_HOME}/include -L${CUDA_HOME}/lib64 -I"${HEADERS}" \
  "${INPUT_CU}" -o "${OUTPUT_BIN}"
