#!/bin/bash
set -euo pipefail

# pre-step-check.sh — Validate preconditions before each FSM step
#
# Usage: pre-step-check.sh <STEP_NAME>
# Returns 0 if all preconditions met, non-zero with diagnostic message if not.
#
# This is the "PreToolUse hook" equivalent for the CroqTuner skill.
# The LLM MUST run this before each step. If it exits non-zero, the step is BLOCKED.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CROQTUNER_DIR="$(dirname "$SCRIPT_DIR")"
DEFAULT_DSL="${CROQTUNER_DSL:-croqtile}"
STATE_FILE="${CROQTUNER_STATE_FILE:-$CROQTUNER_DIR/state/$DEFAULT_DSL/loop-state.json}"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: pre-step-check.sh <STEP_NAME>"
    echo "Steps: BASELINE PROFILE IDEATE IMPLEMENT MEASURE DECIDE STORE SHAPE_COMPLETE SELECT_SHAPE CHOREO_PHASE1 NCU_PROFILE CHOREO_PHASE2 GENERATE_CU SWEEPER_PHASE COMPARE UPDATE_BANK"
    exit 1
fi

STEP="$1"
ERRORS=0
WARNINGS=0

err() { echo "BLOCKED: $1"; ERRORS=$((ERRORS + 1)); }
warn() { echo "WARNING: $1"; WARNINGS=$((WARNINGS + 1)); }

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

validate_resume_locality() {
    local tuning_state="$REPO_ROOT/tuning/aitune/$DEFAULT_DSL/state.json"
    local checkpoint
    local best_kernel

    if [ -f "$tuning_state" ]; then
        best_kernel="$(jq -r 'to_entries[]? | select(.value.status == "in_progress") | .value.best_kernel // empty' "$tuning_state" | head -1)"
        if [ -n "$best_kernel" ]; then
            path_within_repo "$REPO_ROOT/$best_kernel" || err "In-progress best_kernel escapes repo root: $best_kernel"
        fi
    fi

    checkpoint="$(jq -r '.paths.checkpoint_file // empty' "$STATE_FILE")"
    if [ -n "$checkpoint" ] && [ -f "$REPO_ROOT/$checkpoint" ]; then
        best_kernel="$(jq -r '.best_kernel // empty' "$REPO_ROOT/$checkpoint")"
        if [ -n "$best_kernel" ]; then
            path_within_repo "$REPO_ROOT/$best_kernel" || err "Checkpoint best_kernel escapes repo root: $best_kernel"
        fi
    fi
}

if [ ! -f "$STATE_FILE" ]; then
    err "Active state file not found. Run state-transition.sh INIT first."
    exit 1
fi

validate_resume_locality

CURRENT_STATE=$(jq -r '.fsm.current_state' "$STATE_FILE")
ITERATION=$(jq -r '.fsm.iteration' "$STATE_FILE")
MAX_ITER=$(jq -r '.fsm.max_iteration' "$STATE_FILE")
SHAPE_KEY=$(jq -r '.fsm.shape_key' "$STATE_FILE")
CONSEC_DISCARDS=$(jq -r '.metrics.consecutive_discards' "$STATE_FILE")

# Check state matches requested step
if [ "$CURRENT_STATE" != "$STEP" ]; then
    err "Current state is '$CURRENT_STATE', not '$STEP'. Cannot run pre-check for wrong state."
    exit 1
fi

case "$STEP" in
    BASELINE)
        # Check iter000 reference source exists
        SRCS_DIR=$(jq -r '.paths.shape_dir_srcs' "$STATE_FILE")
        ITER000_COUNT=$(find "$SRCS_DIR" -maxdepth 1 -type f -name 'iter000_*' 2>/dev/null | wc -l)
        if [ "$ITER000_COUNT" -lt 1 ]; then
            err "iter000 reference kernel not found in $SRCS_DIR"
        fi
        while IFS= read -r kernel_file; do
            validate_kernel_includes "$kernel_file"
        done < <(find "$SRCS_DIR" -maxdepth 1 -type f \( -name '*.cu' -o -name '*.co' \) 2>/dev/null | sort)
        ;;

    PROFILE)
        # The transition to PROFILE only happens from BASELINE or STORE.
        # STORE post-check already verified git_committed=true.
        # Flags are reset on entry, so nothing to check here.
        # Just verify we're in a valid state.
        if [ -z "$SHAPE_KEY" ] || [ "$SHAPE_KEY" = "null" ] || [ "$SHAPE_KEY" = "" ]; then
            err "shape_key not set. Run INIT first."
        fi
        ;;

    IDEATE)
        # Check profiling ran for this iteration or was explicitly satisfied by the active PROFILE step
        NCU_RAN=$(jq -r '.guard_flags.ncu_ran_this_iter' "$STATE_FILE")
        if [ "$NCU_RAN" != "true" ]; then
            err "ncu has not run this iteration (ncu_ran_this_iter=false). Complete PROFILE step first."
        fi

        BOTTLENECK=$(jq -r '.guard_flags.bottleneck_identified' "$STATE_FILE")
        if [ "$BOTTLENECK" != "true" ]; then
            err "Bottleneck not identified (bottleneck_identified=false). Complete PROFILE step first."
        fi

        # Check idea diversity
        IDEA_LOG=$(jq -r '.paths.idea_log' "$STATE_FILE")
        if [ -f "$IDEA_LOG" ] && [ -s "$IDEA_LOG" ]; then
            LINE_COUNT=$(wc -l < "$IDEA_LOG")
            if [ "$LINE_COUNT" -ge 2 ]; then
                LAST_2_CATS=$(tail -2 "$IDEA_LOG" | jq -r '.category' 2>/dev/null || echo "")
                if [ -n "$LAST_2_CATS" ]; then
                    UNIQUE_CATS=$(echo "$LAST_2_CATS" | sort -u | wc -l)
                    if [ "$UNIQUE_CATS" -eq 1 ]; then
                        REPEATED_CAT=$(echo "$LAST_2_CATS" | head -1)
                        warn "Last 2 ideas both in category '$REPEATED_CAT'. Rule D3 requires different category next."
                    fi
                fi
            fi

            STRUCTURAL_COUNT=$(jq -r 'select(.category == "structural")' "$IDEA_LOG" 2>/dev/null | grep -c "structural" || echo 0)
            CHOREO_COUNT=$(jq -r 'select(.category == "choreo")' "$IDEA_LOG" 2>/dev/null | grep -c "choreo" || echo 0)
            NCU_MICRO_COUNT=$(jq -r 'select(.category == "ncu_micro")' "$IDEA_LOG" 2>/dev/null | grep -c "ncu_micro" || echo 0)

            if [ "$ITERATION" -ge 10 ] && [ "$STRUCTURAL_COUNT" -lt 2 ]; then
                warn "Phase progression: at iter $ITERATION but only $STRUCTURAL_COUNT structural ideas (need >=2 by iter 10)."
            fi
            if [ "$ITERATION" -ge 25 ] && [ "$CHOREO_COUNT" -lt 1 ]; then
                warn "Phase progression: at iter $ITERATION but only $CHOREO_COUNT choreo ideas (need >=1 by iter 25)."
            fi
            if [ "$ITERATION" -ge 40 ] && [ "$NCU_MICRO_COUNT" -lt 1 ]; then
                warn "Phase progression: at iter $ITERATION but only $NCU_MICRO_COUNT ncu_micro ideas (need >=1 by iter 40)."
            fi
        fi

        # Discard-triggered escalation
        if [ "$CONSEC_DISCARDS" -ge 10 ]; then
            warn "consecutive_discards=$CONSEC_DISCARDS (>=10). MUST try radical changes (choreo rewrite, split-K, different WGMMA tile)."
        elif [ "$CONSEC_DISCARDS" -ge 5 ]; then
            warn "consecutive_discards=$CONSEC_DISCARDS (>=5). MUST try completely different approach category."
        elif [ "$CONSEC_DISCARDS" -ge 3 ]; then
            warn "consecutive_discards=$CONSEC_DISCARDS (>=3). SHOULD run full ncu, switch category, and widen research."
        fi
        ;;

    IMPLEMENT)
        IDEA_LOGGED=$(jq -r '.guard_flags.idea_logged' "$STATE_FILE")
        if [ "$IDEA_LOGGED" != "true" ]; then
            err "Idea not logged (idea_logged=false). Complete IDEATE step first."
        fi
        SRCS_DIR=$(jq -r '.paths.shape_dir_srcs' "$STATE_FILE")
        while IFS= read -r kernel_file; do
            validate_kernel_includes "$kernel_file"
        done < <(find "$SRCS_DIR" -maxdepth 1 -type f \( -name '*.cu' -o -name '*.co' \) 2>/dev/null | sort)
        ;;

    MEASURE)
        COMPILE_OK=$(jq -r '.guard_flags.compile_succeeded' "$STATE_FILE")
        if [ "$COMPILE_OK" != "true" ]; then
            err "Kernel not compiled successfully (compile_succeeded=false). Complete IMPLEMENT step first."
        fi
        ;;

    DECIDE)
        TIMING_OK=$(jq -r '.guard_flags.timing_captured' "$STATE_FILE")
        if [ "$TIMING_OK" != "true" ]; then
            err "Timing not captured (timing_captured=false). Complete MEASURE step first."
        fi
        ;;

    STORE)
        DECISION_OK=$(jq -r '.guard_flags.decision_made' "$STATE_FILE")
        if [ "$DECISION_OK" != "true" ]; then
            err "Decision not made (decision_made=false). Complete DECIDE step first."
        fi
        ;;

    SHAPE_COMPLETE)
        if [ "$ITERATION" -lt "$MAX_ITER" ]; then
            err "iteration=$ITERATION < max_iteration=$MAX_ITER. Shape is NOT complete. Continue the loop."
        fi
        ;;

    SELECT_SHAPE)
        :
        ;;

    CHOREO_PHASE1|CHOREO_PHASE2)
        if [ -z "$SHAPE_KEY" ] || [ "$SHAPE_KEY" = "null" ] || [ "$SHAPE_KEY" = "" ]; then
            err "shape_key not set. Run SELECT_SHAPE first."
        fi
        ;;

    NCU_PROFILE)
        COMPILE_OK=$(jq -r '.guard_flags.compile_succeeded' "$STATE_FILE")
        if [ "$COMPILE_OK" != "true" ]; then
            warn "No compiled kernel for ncu profiling (compile_succeeded=false). At least one choreo iteration should have succeeded."
        fi
        ;;

    GENERATE_CU)
        TIMING_OK=$(jq -r '.guard_flags.timing_captured' "$STATE_FILE")
        if [ "$TIMING_OK" != "true" ]; then
            warn "No timing from choreo phases (timing_captured=false)."
        fi
        ;;

    SWEEPER_PHASE)
        COMPILE_OK=$(jq -r '.guard_flags.compile_succeeded' "$STATE_FILE")
        if [ "$COMPILE_OK" != "true" ]; then
            err "Seed .cu not generated (compile_succeeded=false). Complete GENERATE_CU first."
        fi
        ;;

    COMPARE)
        TIMING_OK=$(jq -r '.guard_flags.timing_captured' "$STATE_FILE")
        if [ "$TIMING_OK" != "true" ]; then
            err "No timing results (timing_captured=false). Complete SWEEPER_PHASE first."
        fi
        ;;

    UPDATE_BANK)
        DECISION_OK=$(jq -r '.guard_flags.decision_made' "$STATE_FILE")
        if [ "$DECISION_OK" != "true" ]; then
            err "Decision not made (decision_made=false). Complete COMPARE first."
        fi
        ;;
esac

# Summary
if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "=== PRE-CHECK FAILED: $ERRORS error(s), $WARNINGS warning(s) for step $STEP ==="
    echo "Fix the errors above before proceeding."
    exit 1
fi

if [ "$WARNINGS" -gt 0 ]; then
    echo ""
    echo "=== PRE-CHECK PASSED with $WARNINGS warning(s) for step $STEP ==="
    echo "Proceed, but address the warnings above."
    exit 0
fi

echo "=== PRE-CHECK PASSED for step $STEP ==="
exit 0
