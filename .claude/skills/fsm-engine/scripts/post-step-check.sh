#!/bin/bash
set -euo pipefail

# post-step-check.sh — Validate postconditions after each FSM step
#
# Usage: post-step-check.sh <STEP_NAME>
# Returns 0 if all postconditions met, non-zero with diagnostic message if not.
#
# This is the "PostToolUse hook" equivalent for the CroqTuner skill.
# The LLM MUST run this after each step. If it exits non-zero, the step is INCOMPLETE.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CROQTUNER_DIR="$(dirname "$SCRIPT_DIR")"
DEFAULT_DSL="${CROQTUNER_DSL:-croqtile}"
STATE_FILE="${CROQTUNER_STATE_FILE:-$CROQTUNER_DIR/state/$DEFAULT_DSL/loop-state.json}"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: post-step-check.sh <STEP_NAME>"
    echo "Steps: INIT BASELINE PROFILE IDEATE IMPLEMENT MEASURE DECIDE STORE SHAPE_COMPLETE SELECT_SHAPE CHOREO_PHASE1 NCU_PROFILE CHOREO_PHASE2 GENERATE_CU SWEEPER_PHASE COMPARE UPDATE_BANK"
    exit 1
fi

STEP="$1"
ERRORS=0

err() { echo "INCOMPLETE: $1"; ERRORS=$((ERRORS + 1)); }

path_within_repo() {
    local target="$1"
    local resolved
    [ -n "$target" ] && [ "$target" != "null" ] || return 1
    resolved="$(realpath -m "$target" 2>/dev/null || true)"
    [ -n "$resolved" ] && [[ "$resolved" == "$REPO_ROOT"/* ]]
}

validate_kernel_includes() {
    local file="$1"
    local include_path
    [ -f "$file" ] || return 0
    while IFS= read -r include_path; do
        case "$include_path" in
            /*)
                if ! path_within_repo "$include_path"; then
                    err "Kernel file $file includes external path outside repo: $include_path"
                    return 1
                fi
                ;;
        esac
    done < <(sed -n 's/^[[:space:]]*#include[[:space:]]*"\([^"]*\)".*$/\1/p' "$file")
    return 0
}

if [ ! -f "$STATE_FILE" ]; then
    err "Active state file not found."
    exit 1
fi

SHAPE_KEY=$(jq -r '.fsm.shape_key' "$STATE_FILE")
ITERATION=$(jq -r '.fsm.iteration' "$STATE_FILE")
MAX_ITER=$(jq -r '.fsm.max_iteration' "$STATE_FILE")
DSL=$(jq -r '.fsm.dsl // "croqtile"' "$STATE_FILE")

case "$STEP" in
    INIT)
        # Verify directories exist
        LOGS_DIR=$(jq -r '.paths.shape_dir_logs' "$STATE_FILE")
        SRCS_DIR=$(jq -r '.paths.shape_dir_srcs' "$STATE_FILE")
        PERF_DIR=$(jq -r '.paths.shape_dir_perf' "$STATE_FILE")

        [ -d "$LOGS_DIR" ] || err "Logs directory not created: $LOGS_DIR"
        [ -d "$SRCS_DIR" ] || err "Srcs directory not created: $SRCS_DIR"
        [ -d "$PERF_DIR" ] || err "Perf directory not created: $PERF_DIR"
        CMD_DIR=$(jq -r '.paths.shape_dir_cmd' "$STATE_FILE")
        [ -d "$CMD_DIR" ] || err "Cmd directory not created: $CMD_DIR"

        ITER000_COUNT=$(find "$SRCS_DIR" -maxdepth 1 -type f -name 'iter000_*' 2>/dev/null | wc -l)
        [ "$ITER000_COUNT" -ge 1 ] || err "iter000 reference kernel not found in $SRCS_DIR"
        [ -f "$CMD_DIR/build_iter000.sh" ] || err "iter000 build script not found"
        [ -f "$CMD_DIR/run_iter000.sh" ] || err "iter000 run script not found"
        while IFS= read -r kernel_file; do
            validate_kernel_includes "$kernel_file"
        done < <(find "$SRCS_DIR" -maxdepth 1 -type f \( -name '*.cu' -o -name '*.co' \) 2>/dev/null | sort)

        # Verify results.tsv
        [ -f "$LOGS_DIR/results.tsv" ] || err "results.tsv not created: $LOGS_DIR/results.tsv"
        ;;

    BASELINE)
        PERF_DIR=$(jq -r '.paths.shape_dir_perf' "$STATE_FILE")
        BASELINE_TFLOPS=$(jq -r '.metrics.baseline_tflops' "$STATE_FILE")

        [ -f "$PERF_DIR/timing_iter000_baseline.txt" ] || err "Baseline timing file not found"
        [ "$BASELINE_TFLOPS" != "null" ] || err "baseline_tflops not set in active state file"

        GPU_CHECKED=$(jq -r '.guard_flags.gpu_health_checked' "$STATE_FILE")
        [ "$GPU_CHECKED" = "true" ] || err "GPU health not checked before baseline"
        ;;

    PROFILE)
        NCU_RAN=$(jq -r '.guard_flags.ncu_ran_this_iter' "$STATE_FILE")
        BOTTLENECK=$(jq -r '.guard_flags.bottleneck_identified' "$STATE_FILE")
        LAST_BN=$(jq -r '.metrics.last_bottleneck' "$STATE_FILE")

        [ "$NCU_RAN" = "true" ] || err "ncu_ran_this_iter not set to true"
        [ "$BOTTLENECK" = "true" ] || err "bottleneck_identified not set to true"
        [ "$LAST_BN" != "null" ] || err "last_bottleneck not set in metrics"
        ;;

    IDEATE)
        NOVEL=$(jq -r '.guard_flags.idea_is_novel' "$STATE_FILE")
        LOGGED=$(jq -r '.guard_flags.idea_logged' "$STATE_FILE")

        [ "$NOVEL" = "true" ] || err "idea_is_novel not set to true"
        [ "$LOGGED" = "true" ] || err "idea_logged not set to true"

        # Verify idea was actually appended to log
        IDEA_LOG=$(jq -r '.paths.idea_log' "$STATE_FILE")
        if [ -f "$IDEA_LOG" ]; then
            if ! tail -1 "$IDEA_LOG" | jq -e ".iter == $ITERATION" > /dev/null 2>&1; then
                err "Last entry in idea-log.jsonl does not match current iteration ($ITERATION)"
            fi
        else
            err "idea-log.jsonl not found at $IDEA_LOG"
        fi
        ;;

    IMPLEMENT)
        COMPILE_OK=$(jq -r '.guard_flags.compile_succeeded' "$STATE_FILE")
        [ "$COMPILE_OK" = "true" ] || err "compile_succeeded not set to true"

        # Check kernel file exists (iter number = fsm.iteration)
        SRCS_DIR=$(jq -r '.paths.shape_dir_srcs' "$STATE_FILE")
        ITER_TAG=$(printf "iter%03d" "$ITERATION")
        KERNEL_COUNT=$(find "$SRCS_DIR" -name "${ITER_TAG}_*" 2>/dev/null | wc -l)
        [ "$KERNEL_COUNT" -ge 1 ] || err "No kernel file found matching ${ITER_TAG}_* in $SRCS_DIR"
        while IFS= read -r kernel_file; do
            validate_kernel_includes "$kernel_file"
        done < <(find "$SRCS_DIR" -maxdepth 1 -type f \( -name '*.cu' -o -name '*.co' \) 2>/dev/null | sort)
        ;;

    MEASURE)
        TIMING_OK=$(jq -r '.guard_flags.timing_captured' "$STATE_FILE")
        THIS_TFLOPS=$(jq -r '.metrics.this_iter_tflops' "$STATE_FILE")

        [ "$TIMING_OK" = "true" ] || err "timing_captured not set to true"
        [ "$THIS_TFLOPS" != "null" ] || err "this_iter_tflops not set in metrics"

        # Check timing file exists (iter number = fsm.iteration)
        PERF_DIR=$(jq -r '.paths.shape_dir_perf' "$STATE_FILE")
        ITER_TAG=$(printf "timing_iter%03d" "$ITERATION")
        TIMING_COUNT=$(find "$PERF_DIR" -name "${ITER_TAG}*" 2>/dev/null | wc -l)
        [ "$TIMING_COUNT" -ge 1 ] || err "No timing file found matching ${ITER_TAG}* in $PERF_DIR"
        ;;

    DECIDE)
        DECISION=$(jq -r '.guard_flags.decision_made' "$STATE_FILE")
        THIS_DEC=$(jq -r '.metrics.this_iter_decision' "$STATE_FILE")

        [ "$DECISION" = "true" ] || err "decision_made not set to true"
        [ "$THIS_DEC" != "null" ] || err "this_iter_decision not set (must be KEEP or DISCARD)"
        ;;

    STORE)
        RESULTS_OK=$(jq -r '.guard_flags.results_appended' "$STATE_FILE")
        CKPT_OK=$(jq -r '.guard_flags.checkpoint_written' "$STATE_FILE")
        GIT_OK=$(jq -r '.guard_flags.git_committed' "$STATE_FILE")
        RAW_MEM_OK=$(jq -r '.guard_flags.round_memory_raw_saved // false' "$STATE_FILE")
        MD_MEM_OK=$(jq -r '.guard_flags.round_memory_md_saved // false' "$STATE_FILE")

        [ "$RESULTS_OK" = "true" ] || err "results_appended not set to true"
        [ "$CKPT_OK" = "true" ] || err "checkpoint_written not set to true"
        [ "$GIT_OK" = "true" ] || err "git_committed not set to true"
        [ "$RAW_MEM_OK" = "true" ] || err "round_memory_raw_saved not set to true"
        [ "$MD_MEM_OK" = "true" ] || err "round_memory_md_saved not set to true"

        # Verify measured rows or attempt logs depending on whether the candidate reached MEASURE
        LOGS_DIR=$(jq -r '.paths.shape_dir_logs' "$STATE_FILE")
        THIS_TFLOPS=$(jq -r '.metrics.this_iter_tflops' "$STATE_FILE")
        if [ "$THIS_TFLOPS" != "null" ]; then
            if [ -f "$LOGS_DIR/results.tsv" ]; then
                if ! grep -q "^$ITERATION[[:space:]]" "$LOGS_DIR/results.tsv" 2>/dev/null; then
                    err "results.tsv missing row for measured iteration $ITERATION"
                fi
            else
                err "results.tsv missing for measured iteration storage"
            fi
        else
            ATTEMPT_LOG=$(jq -r '.paths.attempt_log // ""' "$STATE_FILE")
            [ -f "$ATTEMPT_LOG" ] || err "attempt-log.jsonl missing for failed attempt storage"
        fi

        # Verify checkpoint file exists
        CKPT_FILE=$(jq -r '.paths.checkpoint_file' "$STATE_FILE")
        [ -f "$CKPT_FILE" ] || err "Checkpoint file not found: $CKPT_FILE"
        if [ -f "$CKPT_FILE" ]; then
            BEST_KERNEL_PATH=$(jq -r '.best_kernel // empty' "$CKPT_FILE")
            if [ -n "$BEST_KERNEL_PATH" ]; then
                path_within_repo "$REPO_ROOT/$BEST_KERNEL_PATH" || err "Checkpoint best_kernel escapes repo root: $BEST_KERNEL_PATH"
            fi
        fi

        # Verify per-round agent history persistence exists for this iteration
        RAW_MEM_FILE=$(jq -r '.paths.round_memory_raw_log // ""' "$STATE_FILE")
        MD_MEM_FILE=$(jq -r '.paths.round_memory_md_log // ""' "$STATE_FILE")
        [ -f "$RAW_MEM_FILE" ] || err "Raw round-memory log not found: $RAW_MEM_FILE"
        [ -f "$MD_MEM_FILE" ] || err "Markdown round-memory log not found: $MD_MEM_FILE"
        if [ -f "$RAW_MEM_FILE" ] && ! grep -q "\"iter\"[[:space:]]*:[[:space:]]*$ITERATION" "$RAW_MEM_FILE"; then
            err "Raw round-memory log missing iter=$ITERATION entry"
        fi
        if [ -f "$MD_MEM_FILE" ] && ! grep -q "Iter $ITERATION" "$MD_MEM_FILE"; then
            err "Markdown round-memory log missing Iter $ITERATION section"
        fi
        ;;

    SHAPE_COMPLETE)
        if [ "$ITERATION" -lt "$MAX_ITER" ]; then
            err "COMPLETION PROMISE FAILED: iteration=$ITERATION < max_iteration=$MAX_ITER. Shape is NOT complete."
        fi

        DTYPE=$(jq -r '.fsm.dtype' "$STATE_FILE")
        BEST_KERNEL="kernels/gemm_sp_${DTYPE}/${SHAPE_KEY}_best.cu"
        BEST_CO="kernels/gemm_sp_${DTYPE}/${SHAPE_KEY}_best.co"
        if [ ! -f "$BEST_KERNEL" ] && [ ! -f "$BEST_CO" ]; then
            err "Best kernel not registered at $BEST_KERNEL (or .co)"
        fi

        DSL_STATE="tuning/aitune/${DSL}/state.json"
        LEGACY_STATE="tuning/state.json"
        if [ -f "$DSL_STATE" ]; then
            STATUS=$(jq -r ".[\"$SHAPE_KEY\"].status // \"missing\"" "$DSL_STATE")
            if [ "$STATUS" != "done" ]; then
                err "$DSL_STATE does not show status=done for $SHAPE_KEY (got: $STATUS)"
            fi
        elif [ -f "$LEGACY_STATE" ]; then
            STATUS=$(jq -r ".[\"$SHAPE_KEY\"].status // \"missing\"" "$LEGACY_STATE")
            if [ "$STATUS" != "done" ]; then
                err "$LEGACY_STATE does not show status=done for $SHAPE_KEY (got: $STATUS)"
            fi
        else
            err "No state registry found at $DSL_STATE (or legacy $LEGACY_STATE)"
        fi

        UNCOMMITTED=$(git status --porcelain -- "tuning/aitune/$DSL/" "kernels/" 2>/dev/null | wc -l)
        if [ "$UNCOMMITTED" -gt 0 ]; then
            err "Uncommitted changes in tuning/aitune/$DSL or kernels/ — final commit required"
        fi
        ;;

    SELECT_SHAPE)
        [ -n "$SHAPE_KEY" ] && [ "$SHAPE_KEY" != "null" ] && [ "$SHAPE_KEY" != "" ] || err "shape_key not set after SELECT_SHAPE"
        ;;

    CHOREO_PHASE1|CHOREO_PHASE2)
        TIMING_OK=$(jq -r '.guard_flags.timing_captured' "$STATE_FILE")
        BEST_TFLOPS=$(jq -r '.metrics.best_tflops // "null"' "$STATE_FILE")
        [ "$TIMING_OK" = "true" ] || err "No timing captured from choreo phase"
        ;;

    NCU_PROFILE)
        NCU_RAN=$(jq -r '.guard_flags.ncu_ran_this_iter' "$STATE_FILE")
        BOTTLENECK=$(jq -r '.guard_flags.bottleneck_identified' "$STATE_FILE")
        [ "$NCU_RAN" = "true" ] || err "ncu not run"
        [ "$BOTTLENECK" = "true" ] || err "bottleneck not identified"
        ;;

    GENERATE_CU)
        COMPILE_OK=$(jq -r '.guard_flags.compile_succeeded' "$STATE_FILE")
        [ "$COMPILE_OK" = "true" ] || err "Seed .cu not generated"
        ;;

    SWEEPER_PHASE)
        TIMING_OK=$(jq -r '.guard_flags.timing_captured' "$STATE_FILE")
        [ "$TIMING_OK" = "true" ] || err "No sweeper results captured"
        ;;

    COMPARE)
        DECISION_OK=$(jq -r '.guard_flags.decision_made' "$STATE_FILE")
        [ "$DECISION_OK" = "true" ] || err "No comparison decision made"
        ;;

    UPDATE_BANK)
        RESULTS_OK=$(jq -r '.guard_flags.results_appended' "$STATE_FILE")
        [ "$RESULTS_OK" = "true" ] || err "Bank not updated"
        ;;
esac

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "=== POST-CHECK FAILED: $ERRORS error(s) for step $STEP ==="
    echo "Step is INCOMPLETE. Fix the errors above."
    exit 1
fi

echo "=== POST-CHECK PASSED for step $STEP ==="
exit 0
