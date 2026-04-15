#!/usr/bin/env bash
set -e

export CHOREO_HOME=/home/albert/workspace/croqtile
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$CHOREO_HOME/build:$PATH

GPU_KEY="sm86_NVIDIA_GeForce_RTX_3070"
DSL="croqtile"
SHAPE_KEY="matmul_bf16fp32_16384x16384x16384"
ITER="iter001_draft"
ROUND="round1"

SCRIPT="tuning/$GPU_KEY/$DSL/cmd/$SHAPE_KEY/${ITER}.cute.result"
NCU_REP="tuning/$GPU_KEY/$DSL/perf/$SHAPE_KEY/ncu_${ITER}_${ROUND}.ncu-rep"
NCU_CSV="tuning/$GPU_KEY/$DSL/perf/$SHAPE_KEY/ncu_${ITER}_${ROUND}.csv"
NCU_METRICS="tuning/$GPU_KEY/$DSL/perf/$SHAPE_KEY/ncu_${ITER}_${ROUND}.metrics.csv"
PROFILE_TXT="tuning/$GPU_KEY/$DSL/perf/$SHAPE_KEY/profile_${ITER}_${ROUND}.txt"

echo "Profiling $ITER with ncu..."

# Run ncu with full metrics - need --target-processes all for bash wrapper
CHOREO_TIMING_WARMUP=0 CHOREO_TIMING_REPEAT=1 CHOREO_SKIP_VERIFY=1 \
ncu --target-processes all \
    --set full \
    --export "$NCU_REP" \
    --force-overwrite \
    bash "$SCRIPT" --execute --skip-verify 2>&1 | tee "$PROFILE_TXT"

echo ""
echo "Extracting metrics to CSV..."
ncu --import "$NCU_REP" --csv --page raw > "$NCU_CSV" 2>/dev/null || true

# Extract key metrics
ncu --import "$NCU_REP" --csv --page details > "$NCU_METRICS" 2>/dev/null || true

echo "Profile complete:"
echo "  Report: $NCU_REP"
echo "  CSV: $NCU_CSV"
