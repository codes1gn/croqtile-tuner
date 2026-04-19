#!/bin/bash
# Profile triton kernel with ncu

set -e

PYTHON=/home/baldlee/workspace/choreo-attn/.venv/bin/python
SCRIPT=export_inputs.py
OUTPUT_PREFIX=triton_profile

cd /home/baldlee/workspace/choreo-attn/gdn/reference

echo "Profiling Triton kernel with ncu..."
echo "Command: ncu --target-processes all --set full --import-source on -f -o ${OUTPUT_PREFIX} ${PYTHON} ${SCRIPT}"

ncu --target-processes all --set full --import-source on -f -o ${OUTPUT_PREFIX} ${PYTHON} ${SCRIPT}

echo ""
echo "Profiling complete!"
echo "Output file: ${OUTPUT_PREFIX}.ncu-rep"
echo ""
echo "To view results, run: ncu-ui ${OUTPUT_PREFIX}.ncu-rep"
