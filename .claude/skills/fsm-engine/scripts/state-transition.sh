#!/bin/bash
set -euo pipefail

# state-transition.sh -- Atomically transition the CroqTuner FSM state.
# Supports per-DSL state/memory isolation and strict non-stop looping.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CROQTUNER_DIR="$(dirname "$SCRIPT_DIR")"

if [ $# -lt 1 ]; then
    echo "Usage: state-transition.sh <NEXT_STATE|SET> [key=value ...]"
    exit 1
fi

ACTION="$1"
shift

DEFAULT_DSL="${CROQTUNER_DSL:-croqtile}"
STATE_FILE="${CROQTUNER_STATE_FILE:-$CROQTUNER_DIR/state/$DEFAULT_DSL/loop-state.json}"

VALID_STATES="INIT BASELINE PROFILE IDEATE IMPLEMENT MEASURE DECIDE STORE SHAPE_COMPLETE NEXT_SHAPE SELECT_SHAPE CHOREO_PHASE1 NCU_PROFILE CHOREO_PHASE2 GENERATE_CU SWEEPER_PHASE COMPARE UPDATE_BANK"
if [ "$ACTION" != "SET" ] && ! echo "$VALID_STATES" | grep -qw "$ACTION"; then
    echo "ERROR: Invalid state '$ACTION'. Valid states: $VALID_STATES or SET"
    exit 1
fi

declare -A LEGAL_TRANSITIONS
LEGAL_TRANSITIONS=(
    ["INIT"]="BASELINE SELECT_SHAPE"
    ["BASELINE"]="PROFILE"
    ["PROFILE"]="IDEATE"
    ["IDEATE"]="IMPLEMENT PROFILE"
    ["IMPLEMENT"]="MEASURE STORE"
    ["MEASURE"]="DECIDE STORE"
    ["DECIDE"]="STORE"
    ["STORE"]="PROFILE SHAPE_COMPLETE"
    ["SHAPE_COMPLETE"]="NEXT_SHAPE"
    ["NEXT_SHAPE"]="INIT"
    ["_NEW_"]="INIT"
    ["SELECT_SHAPE"]="CHOREO_PHASE1"
    ["CHOREO_PHASE1"]="NCU_PROFILE"
    ["NCU_PROFILE"]="CHOREO_PHASE2"
    ["CHOREO_PHASE2"]="GENERATE_CU"
    ["GENERATE_CU"]="SWEEPER_PHASE"
    ["SWEEPER_PHASE"]="COMPARE"
    ["COMPARE"]="UPDATE_BANK SELECT_SHAPE"
    ["UPDATE_BANK"]="SELECT_SHAPE"
)

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

write_json() {
    local expr="$1"
    jq "$expr" "$STATE_FILE" > "${STATE_FILE}.tmp"
    mv "${STATE_FILE}.tmp" "$STATE_FILE"
}

write_json_args() {
    jq "$@" "$STATE_FILE" > "${STATE_FILE}.tmp"
    mv "${STATE_FILE}.tmp" "$STATE_FILE"
}

apply_paths() {
    local dsl key base logs srcs perf cmd ckpt idea attempts memdir raw md

    dsl="$(jq -r '.fsm.dsl // "croqtile"' "$STATE_FILE")"
    key="$(jq -r '.fsm.shape_key // ""' "$STATE_FILE")"
    base="tuning/aitune/${dsl}"

    if [ -z "$key" ] || [ "$key" = "null" ]; then
        logs=""
        srcs=""
        perf=""
        cmd=""
        ckpt=""
        idea=""
        attempts=""
        memdir=""
        raw=""
        md=""
    else
        logs="${base}/logs/${key}"
        srcs="${base}/srcs/${key}"
        perf="${base}/perf/${key}"
        cmd="${base}/cmd/${key}"
        ckpt="${base}/checkpoints/${key}.json"
        idea="${base}/logs/${key}/idea-log.jsonl"
        attempts="${base}/logs/${key}/attempt-log.jsonl"
        memdir="${base}/memory/${key}"
        raw="${memdir}/rounds.raw.jsonl"
        md="${memdir}/rounds.md"
    fi

    write_json_args \
        --arg logs "$logs" \
        --arg srcs "$srcs" \
        --arg perf "$perf" \
        --arg cmd "$cmd" \
        --arg ckpt "$ckpt" \
        --arg idea "$idea" \
        --arg attempts "$attempts" \
        --arg memdir "$memdir" \
        --arg raw "$raw" \
        --arg md "$md" \
        '.paths.shape_dir_logs = $logs
         | .paths.shape_dir_srcs = $srcs
         | .paths.shape_dir_perf = $perf
         | .paths.shape_dir_cmd = $cmd
         | .paths.checkpoint_file = $ckpt
         | .paths.idea_log = $idea
         | .paths.attempt_log = $attempts
         | .paths.memory_dir = $memdir
         | .paths.round_memory_raw_log = $raw
         | .paths.round_memory_md_log = $md'
}

apply_kv_overrides() {
    local path_dirty=0
    local kv key val
    for kv in "$@"; do
        key="${kv%%=*}"
        val="${kv#*=}"
        case "$key" in
            gpu_health_checked|ncu_ran_this_iter|bottleneck_identified|idea_is_novel|idea_logged|compile_succeeded|correctness_verified|timing_captured|decision_made|results_appended|checkpoint_written|git_committed|round_memory_raw_saved|round_memory_md_saved)
                write_json ".guard_flags.$key = $val | .last_updated = \"$(timestamp)\""
                ;;
            baseline_tflops|current_best_tflops|this_iter_tflops)
                write_json ".metrics.$key = $val | .last_updated = \"$(timestamp)\""
                ;;
            current_best_iter|consecutive_discards)
                write_json ".metrics.$key = $val | .last_updated = \"$(timestamp)\""
                ;;
            current_best_kernel|this_iter_decision|last_bottleneck|last_idea_category|loop_status)
                write_json ".metrics.$key = \"$val\" | .last_updated = \"$(timestamp)\""
                ;;
            iteration|max_iteration)
                write_json ".fsm.$key = $val | .last_updated = \"$(timestamp)\""
                ;;
            shape_key|dtype|dsl|branch_name)
                write_json ".fsm.$key = \"$val\" | .last_updated = \"$(timestamp)\""
                path_dirty=1
                ;;
            shape)
                write_json ".fsm.shape = $val | .last_updated = \"$(timestamp)\""
                ;;
            *)
                ;;
        esac
    done

    if [ "$path_dirty" -eq 1 ]; then
        apply_paths
    fi
}

if [ "$ACTION" = "SET" ]; then
    if [ ! -f "$STATE_FILE" ]; then
        echo "ERROR: No state file at $STATE_FILE"
        exit 1
    fi
    apply_kv_overrides "$@"
    echo "OK: Updated state via SET"
    exit 0
fi

NEXT_STATE="$ACTION"

if [ ! -f "$STATE_FILE" ]; then
    if [ "$NEXT_STATE" != "INIT" ]; then
        echo "ERROR: No state file found. First transition must be INIT."
        exit 1
    fi

    mkdir -p "$(dirname "$STATE_FILE")"
    cat > "$STATE_FILE" <<INITJSON
{
  "schema_version": 2,
  "fsm": {
    "current_state": "INIT",
    "iteration": 0,
    "max_iteration": 30,
    "shape_key": "",
    "dtype": "",
    "shape": [],
    "dsl": "$DEFAULT_DSL",
    "branch_name": "aitune/$DEFAULT_DSL"
  },
  "guard_flags": {
    "gpu_health_checked": false,
    "ncu_ran_this_iter": false,
    "bottleneck_identified": false,
    "idea_is_novel": false,
    "idea_logged": false,
    "compile_succeeded": false,
    "correctness_verified": false,
    "timing_captured": false,
    "decision_made": false,
    "results_appended": false,
    "checkpoint_written": false,
    "git_committed": false,
    "round_memory_raw_saved": false,
    "round_memory_md_saved": false
  },
  "metrics": {
    "baseline_tflops": null,
    "current_best_tflops": null,
    "current_best_iter": null,
    "current_best_kernel": null,
    "this_iter_tflops": null,
    "this_iter_decision": null,
    "consecutive_discards": 0,
    "last_bottleneck": null,
    "last_idea_category": null,
    "loop_status": "running_non_stop"
  },
  "paths": {
    "shape_dir_logs": "",
    "shape_dir_srcs": "",
    "shape_dir_perf": "",
        "shape_dir_cmd": "",
    "checkpoint_file": "",
    "idea_log": "",
        "attempt_log": "",
    "memory_dir": "",
    "round_memory_raw_log": "",
    "round_memory_md_log": ""
  },
  "completion_promise": "never stop at shape boundary; continue NEXT_SHAPE until user interrupts",
  "non_stop_keyphrase": "NON_STOP_CONTINUE_WITHOUT_WAIT",
  "last_updated": "$(timestamp)"
}
INITJSON

    apply_kv_overrides "$@"
    apply_paths
    echo "OK: Created $STATE_FILE in INIT"
    exit 0
fi

CURRENT_STATE="$(jq -r '.fsm.current_state' "$STATE_FILE")"
ALLOWED="${LEGAL_TRANSITIONS[$CURRENT_STATE]:-}"
if [ -z "$ALLOWED" ]; then
    echo "ERROR: Unknown current state '$CURRENT_STATE'"
    exit 1
fi
if ! echo "$ALLOWED" | grep -qw "$NEXT_STATE"; then
    echo "ERROR: Illegal transition $CURRENT_STATE -> $NEXT_STATE"
    echo "Allowed from $CURRENT_STATE: $ALLOWED"
    exit 1
fi

# Hard enforcement: no early shape completion for regular sweep modes.
if [ "$NEXT_STATE" = "SHAPE_COMPLETE" ]; then
    CUR_ITER="$(jq -r '.fsm.iteration // 0' "$STATE_FILE")"
    CUR_MAX="$(jq -r '.fsm.max_iteration // 0' "$STATE_FILE")"
    if [ "$CUR_ITER" -lt "$CUR_MAX" ]; then
        echo "ERROR: Refusing SHAPE_COMPLETE early (iteration=$CUR_ITER < max_iteration=$CUR_MAX)"
        exit 1
    fi
fi

# NEXT_SHAPE -> INIT means re-seed a fresh state for the next shape, never terminal.
if [ "$NEXT_STATE" = "INIT" ] && [ "$CURRENT_STATE" = "NEXT_SHAPE" ]; then
    rm -f "$STATE_FILE"
    exec "$0" INIT "$@"
fi

JQ_EXPR=".fsm.current_state = \"$NEXT_STATE\" | .last_updated = \"$(timestamp)\""

case "$NEXT_STATE" in
    PROFILE)
        if [ "$CURRENT_STATE" = "BASELINE" ]; then
            JQ_EXPR="$JQ_EXPR | .fsm.iteration = 1"
        elif [ "$CURRENT_STATE" = "STORE" ]; then
            CURRENT_ITER="$(jq -r '.fsm.iteration' "$STATE_FILE")"
            THIS_TFLOPS="$(jq -r '.metrics.this_iter_tflops' "$STATE_FILE")"
            if [ "$THIS_TFLOPS" != "null" ]; then
                NEXT_ITER=$((CURRENT_ITER + 1))
                JQ_EXPR="$JQ_EXPR | .fsm.iteration = $NEXT_ITER"
            fi
        fi
        JQ_EXPR="$JQ_EXPR
            | .guard_flags.gpu_health_checked = false
            | .guard_flags.ncu_ran_this_iter = false
            | .guard_flags.bottleneck_identified = false
            | .guard_flags.idea_is_novel = false
            | .guard_flags.idea_logged = false
            | .guard_flags.compile_succeeded = false
            | .guard_flags.correctness_verified = false
            | .guard_flags.timing_captured = false
            | .guard_flags.decision_made = false
            | .guard_flags.results_appended = false
            | .guard_flags.checkpoint_written = false
            | .guard_flags.git_committed = false
            | .guard_flags.round_memory_raw_saved = false
            | .guard_flags.round_memory_md_saved = false
            | .metrics.this_iter_tflops = null
            | .metrics.this_iter_decision = null"
        ;;
    IDEATE)
        JQ_EXPR="$JQ_EXPR | .guard_flags.idea_is_novel = false | .guard_flags.idea_logged = false"
        ;;
    IMPLEMENT)
        JQ_EXPR="$JQ_EXPR | .guard_flags.compile_succeeded = false | .guard_flags.correctness_verified = false"
        ;;
    MEASURE)
        JQ_EXPR="$JQ_EXPR | .guard_flags.timing_captured = false"
        ;;
    DECIDE)
        JQ_EXPR="$JQ_EXPR | .guard_flags.decision_made = false"
        ;;
    STORE)
        JQ_EXPR="$JQ_EXPR
            | .guard_flags.results_appended = false
            | .guard_flags.checkpoint_written = false
            | .guard_flags.git_committed = false
            | .guard_flags.round_memory_raw_saved = false
            | .guard_flags.round_memory_md_saved = false"
        ;;
    SELECT_SHAPE)
        JQ_EXPR="$JQ_EXPR
            | .guard_flags.gpu_health_checked = false
            | .guard_flags.ncu_ran_this_iter = false
            | .guard_flags.bottleneck_identified = false
            | .guard_flags.compile_succeeded = false
            | .guard_flags.correctness_verified = false
            | .guard_flags.timing_captured = false
            | .guard_flags.decision_made = false
            | .guard_flags.results_appended = false
            | .guard_flags.checkpoint_written = false
            | .guard_flags.git_committed = false
            | .guard_flags.round_memory_raw_saved = false
            | .guard_flags.round_memory_md_saved = false"
        ;;
    CHOREO_PHASE1|CHOREO_PHASE2)
        JQ_EXPR="$JQ_EXPR | .guard_flags.compile_succeeded = false | .guard_flags.correctness_verified = false | .guard_flags.timing_captured = false"
        ;;
    NCU_PROFILE)
        JQ_EXPR="$JQ_EXPR | .guard_flags.ncu_ran_this_iter = false | .guard_flags.bottleneck_identified = false"
        ;;
    GENERATE_CU)
        JQ_EXPR="$JQ_EXPR | .guard_flags.compile_succeeded = false"
        ;;
    SWEEPER_PHASE)
        JQ_EXPR="$JQ_EXPR | .guard_flags.compile_succeeded = false | .guard_flags.correctness_verified = false | .guard_flags.timing_captured = false"
        ;;
    COMPARE)
        JQ_EXPR="$JQ_EXPR | .guard_flags.decision_made = false"
        ;;
    UPDATE_BANK)
        JQ_EXPR="$JQ_EXPR | .guard_flags.results_appended = false | .guard_flags.checkpoint_written = false"
        ;;
esac

write_json "$JQ_EXPR"
apply_kv_overrides "$@"
apply_paths

echo "OK: $CURRENT_STATE -> $NEXT_STATE"
