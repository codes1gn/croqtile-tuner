#!/usr/bin/env bash
# setup_mock_workspace.sh — Build a minimal pre-seeded mock tuning workspace for e2e tests.
#
# Creates a self-contained fixture workspace under a given TMP directory,
# pre-populated with:
#   - iter000 baseline artifacts (cuBLAS ref results)
#   - iter001_draft source (from mock_kernel)
#   - build/run scripts that call mock_kernel instead of nvcc output
#   - rounds.raw.jsonl, results.tsv, idea-log.jsonl with iter000 baseline row
#   - checkpoint.json at state PROFILE (ready for round 1)
#
# USAGE:
#   source testing/e2e/setup_mock_workspace.sh
#   setup_mock_workspace <tmpdir> <dsl> <shape_key>
#
# The function sets MOCK_WS_ROOT and MOCK_WS_TUNING as side effects.
# GPU key used in tests: sm90_testgpu (static stub — GPU-agnostic).
#
# After calling setup_mock_workspace, a subagent started with /croq-tune <dsl>
# will skip baseline and start immediately at round 1 (PROFILE step).

REPO=$(git rev-parse --show-toplevel)
MOCK_KERNEL="$REPO/testing/mocks/mock_kernel"
MOCK_NCU="$REPO/testing/mocks/mock_ncu"

setup_mock_workspace() {
    local TMP="$1"
    local DSL="$2"
    local SHAPE_KEY="$3"
    local TFLOPS="${4:-28.5}"
    local GPU="${5:-sm90_testgpu}"

    export MOCK_WS_ROOT="$TMP"
    export MOCK_WS_GPU="$GPU"
    export MOCK_WS_TUNING="$TMP/tuning/$GPU/$DSL"
    # Legacy alias for any callers still using the old name
    export MOCK_WS_AITUNE="$MOCK_WS_TUNING"

    local SRCS="$MOCK_WS_TUNING/srcs/$SHAPE_KEY"
    local CMDS="$MOCK_WS_TUNING/cmd/$SHAPE_KEY"
    local PERF="$MOCK_WS_TUNING/perf/$SHAPE_KEY"
    local LOGS="$MOCK_WS_TUNING/logs/$SHAPE_KEY"
    local MEM="$MOCK_WS_TUNING/memory/$SHAPE_KEY"
    local BIN="$MOCK_WS_TUNING/bin/$SHAPE_KEY"
    local CHKDIR="$MOCK_WS_TUNING/checkpoints"

    mkdir -p "$SRCS" "$CMDS" "$PERF" "$LOGS" "$MEM" "$BIN" "$CHKDIR"

    local TS
    TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # ── iter000 baseline source (stub) ────────────────────────────────────────
    cat > "$SRCS/iter000_baseline.cu" <<'CUDA'
// Mock cuBLAS baseline — not compiled; replaced by mock_kernel at runtime.
#include <stdio.h>
int main() { printf("TFLOPS: 20.00   time_ms: 150.000\n"); }
CUDA

    # ── iter001_draft source (stub) ───────────────────────────────────────────
    cat > "$SRCS/iter001_draft.cu" <<'CUDA'
// Mock iter001 draft kernel — not compiled; replaced by mock_kernel at runtime.
#include <stdio.h>
int main() { printf("VERIFY: PASS\nTFLOPS: 28.50   time_ms: 120.300\n"); }
CUDA

    # ── build script for iter001_draft ────────────────────────────────────────
    cat > "$CMDS/build_iter001.sh" <<SCRIPT
#!/usr/bin/env bash
# Mock build: copy mock_kernel as the output binary instead of running nvcc
set -e
cp "$MOCK_KERNEL" "$BIN/iter001_draft"
chmod +x "$BIN/iter001_draft"
echo "[mock_build] iter001_draft built successfully (mock_kernel shim)"
SCRIPT
    chmod +x "$CMDS/build_iter001.sh"

    # ── run script for iter001_draft ──────────────────────────────────────────
    cat > "$CMDS/run_iter001.sh" <<SCRIPT
#!/usr/bin/env bash
set -e
MOCK_KERNEL_TFLOPS=28.5 MOCK_KERNEL_TIME_MS=120.3 \\
    python3 "$MOCK_KERNEL" \\
    2>&1 | tee "$PERF/timing_iter001_draft.txt"
SCRIPT
    chmod +x "$CMDS/run_iter001.sh"

    # ── profile script for iter001_draft ──────────────────────────────────────
    cat > "$CMDS/profile_iter001.sh" <<SCRIPT
#!/usr/bin/env bash
# Mock profile: use mock_ncu to generate synthetic CSV
set -e
export PATH="$REPO/testing/mocks:\$PATH"
export MOCK_NCU_SCENARIO="\${MOCK_NCU_SCENARIO:-memory_bound}"

# Step 1: write .ncu-rep stub
ncu --set full \\
    --export "$PERF/ncu_iter001_draft.ncu-rep" \\
    --force-overwrite \\
    python3 "$MOCK_KERNEL" 2>&1

# Step 2: export CSV
ncu --import "$PERF/ncu_iter001_draft.ncu-rep" \\
    --csv --page raw \\
    > "$PERF/ncu_iter001_draft.csv" 2>&1
echo "[mock_profile] ncu CSV written: $PERF/ncu_iter001_draft.csv"
SCRIPT
    chmod +x "$CMDS/profile_iter001.sh"

    # ── baseline env log ──────────────────────────────────────────────────────
    cat > "$LOGS/env_iter000.txt" <<ENV
[mock_env] GPU: Mock H100 80GB HBM3
[mock_env] Driver: 550.54.15
[mock_env] CUDA: 12.4
[mock_env] ncu: NVIDIA Nsight Compute 2024.1 (mock)
[mock_env] nvcc: Cuda compilation tools, release 12.4 (mock)
ENV

    # ── baseline timing output ────────────────────────────────────────────────
    printf "VERIFY: PASS\nTFLOPS: 20.00   time_ms: 150.000\n" \
        > "$PERF/timing_iter000_baseline.txt"

    # ── results.tsv (with iter000 baseline row) ───────────────────────────────
    printf "iter\tkernel\ttflops\ttime_ms\tdecision\tbottleneck\tidea\ttimestamp\n" \
        > "$LOGS/results.tsv"
    printf "iter000\titer000_baseline\t20.0\t150.0\tKEEP\tmemory_bound\tcuBLAS baseline\t%s\n" \
        "$TS" >> "$LOGS/results.tsv"

    # ── idea-log.jsonl (with iter000 row) ─────────────────────────────────────
    printf '{"round": 0, "iter": "iter000_baseline", "bottleneck": "memory_bound", "idea": "cuBLAS baseline", "category": "baseline", "expected_gain": "N/A", "timestamp": "%s"}\n' \
        "$TS" > "$LOGS/idea-log.jsonl"

    # ── attempt-log.jsonl (empty) ─────────────────────────────────────────────
    touch "$LOGS/attempt-log.jsonl"

    # ── rounds.raw.jsonl (with iter000 row) ──────────────────────────────────
    printf '{"iter": "iter000_baseline", "kernel": "iter000_baseline", "tflops": 20.0, "decision": "KEEP", "bottleneck": "memory_bound", "idea": "cuBLAS baseline", "timestamp": "%s"}\n' \
        "$TS" > "$MEM/rounds.raw.jsonl"

    # ── rounds.md ─────────────────────────────────────────────────────────────
    cat > "$MEM/rounds.md" <<MD
## iter000_baseline - $TS
- kernel: \`iter000_baseline\`
- tflops: \`20.0\`
- decision: **KEEP**
- bottleneck: \`memory_bound\`
- idea: cuBLAS baseline
MD

    # ── checkpoint.json (PROFILE state, ready for round 1) ───────────────────
    cat > "$CHKDIR/$SHAPE_KEY.json" <<JSON
{
  "dsl": "$DSL",
  "shape_key": "$SHAPE_KEY",
  "status": "PROFILE",
  "last_iter": "iter000_baseline",
  "next_state": "PROFILE",
  "planned_iter": "iter001_draft",
  "timestamp": "$TS"
}
JSON

    # ── state.json ────────────────────────────────────────────────────────────
    cat > "$MOCK_WS_TUNING/state.json" <<JSON
{
  "dsl": "$DSL",
  "shape_key": "$SHAPE_KEY",
  "current_best": "iter000_baseline",
  "current_best_tflops": 20.0,
  "round": 0,
  "updated_at": "$TS"
}
JSON

    echo "[setup_mock_workspace] Workspace ready at: $MOCK_WS_TUNING"
    echo "[setup_mock_workspace]   GPU=$GPU  DSL=$DSL  SHAPE_KEY=$SHAPE_KEY  baseline_tflops=20.0"
    echo "[setup_mock_workspace]   Preloaded: iter000 baseline, iter001_draft stub"
    echo "[setup_mock_workspace]   Ready for: PROFILE step of round 1"
}
