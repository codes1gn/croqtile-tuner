#!/usr/bin/env bash
# checkpoint_write.sh — Structured IDEA→IMPLEMENT gate for croq-tune.
#
# PURPOSE:
#   Eliminate the gap between what the agent planned in IDEA and what it
#   actually implements. The agent writes a checkpoint at the END of IDEA,
#   reads it back at the START of IMPLEMENT, and verifies it at VERIFY.
#
# USAGE:
#   # At end of IDEA step — write the plan:
#   bash .claude/skills/croq-store/checkpoint_write.sh write \
#       --dsl cuda \
#       --shape-key matmul_bf16fp32_512x16384x16384 \
#       --iter iter201_swizzle \
#       --bottleneck memory_bound \
#       --idea "Increase L2 swizzle stride from 4 to 8 blocks to reduce L2 thrashing" \
#       --expected-gain "+2 TFLOPS (5-8%)" \
#       --levers "SWIZZLE_STRIDE,BM,BN"
#
#   # At start of IMPLEMENT step — read the plan back:
#   bash .claude/skills/croq-store/checkpoint_write.sh read \
#       --dsl cuda \
#       --shape-key matmul_bf16fp32_512x16384x16384
#   # Emits JSON to stdout; agent reads this to confirm what to build.
#
#   # At VERIFY step — check what was actually changed vs. the plan:
#   bash .claude/skills/croq-store/checkpoint_write.sh verify \
#       --dsl cuda \
#       --shape-key matmul_bf16fp32_512x16384x16384 \
#       --iter iter201_swizzle
#   # Prints a pass/warn report; exits 0=ok, 1=drift detected
#
# EXIT:
#   0 — success
#   1 — validation error (bad args, required fields missing)
#   2 — checkpoint file not found (for read/verify)
#   3 — verify detected significant drift from plan

set -euo pipefail

MODE="${1:-}"
if [[ -z "$MODE" || ! "$MODE" =~ ^(write|read|verify)$ ]]; then
    echo "[checkpoint_write] ERROR: first arg must be 'write', 'read', or 'verify'" >&2
    echo "Usage: $0 write|read|verify [--dsl DSL] [--shape-key KEY] ..." >&2
    exit 1
fi
shift

# ── argument parsing ──────────────────────────────────────────────────────────
DSL=""
SHAPE_KEY=""
ITER=""
BOTTLENECK=""
IDEA=""
EXPECTED_GAIN=""
LEVERS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dsl)           DSL="$2";           shift 2 ;;
        --shape-key)     SHAPE_KEY="$2";     shift 2 ;;
        --iter)          ITER="$2";          shift 2 ;;
        --bottleneck)    BOTTLENECK="$2";    shift 2 ;;
        --idea)          IDEA="$2";          shift 2 ;;
        --expected-gain) EXPECTED_GAIN="$2"; shift 2 ;;
        --levers)        LEVERS="$2";        shift 2 ;;
        *) echo "[checkpoint_write] ERROR: unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── validate required args ────────────────────────────────────────────────────
if [[ -z "$DSL" || -z "$SHAPE_KEY" ]]; then
    echo "[checkpoint_write] ERROR: --dsl and --shape-key are required" >&2
    exit 1
fi

# ── paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR="tuning/aitune/${DSL}/checkpoints/${SHAPE_KEY}"
CHECKPOINT_FILE="$CHECKPOINT_DIR/current_idea.json"
mkdir -p "$CHECKPOINT_DIR"

# ════════════════════════════════════════════════════════════════════════════════
# MODE: write
# ════════════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "write" ]]; then
    # Validate required write args
    if [[ -z "$ITER" ]]; then
        echo "[checkpoint_write] ERROR: write mode requires --iter" >&2; exit 1
    fi
    if [[ -z "$IDEA" ]]; then
        echo "[checkpoint_write] ERROR: write mode requires --idea" >&2; exit 1
    fi
    if [[ -z "$BOTTLENECK" ]]; then
        echo "[checkpoint_write] ERROR: write mode requires --bottleneck" >&2; exit 1
    fi

    # Validate iter tag format
    if ! echo "$ITER" | grep -qE '^iter[0-9]{3}_[a-z][a-z0-9_]{1,15}$'; then
        echo "[checkpoint_write] ERROR: --iter must match iter<NNN>_<tag>, got: $ITER" >&2
        exit 1
    fi

    TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    python3 -c "
import json, sys
checkpoint = {
    'schema': 'croq-checkpoint-v1',
    'timestamp': '$TIMESTAMP',
    'dsl': '$DSL',
    'shape_key': '$SHAPE_KEY',
    'iter': '$ITER',
    'bottleneck': '$BOTTLENECK',
    'idea': '$IDEA',
    'expected_gain': '$EXPECTED_GAIN',
    'levers': [l.strip() for l in '$LEVERS'.split(',') if l.strip()],
    'status': 'PLANNED'
}
print(json.dumps(checkpoint, indent=2))
" > "$CHECKPOINT_FILE"

    echo "[checkpoint_write] PLANNED: $ITER"
    echo "[checkpoint_write] Bottleneck: $BOTTLENECK"
    echo "[checkpoint_write] Idea: $IDEA"
    echo "[checkpoint_write] Expected gain: $EXPECTED_GAIN"
    echo "[checkpoint_write] Checkpoint: $CHECKPOINT_FILE"
    exit 0
fi

# ════════════════════════════════════════════════════════════════════════════════
# MODE: read
# ════════════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "read" ]]; then
    if [[ ! -f "$CHECKPOINT_FILE" ]]; then
        echo "[checkpoint_write] ERROR: no checkpoint found at $CHECKPOINT_FILE" >&2
        echo "[checkpoint_write] Run 'write' at end of IDEA step first." >&2
        exit 2
    fi

    # Print the checkpoint JSON to stdout (agent reads this)
    cat "$CHECKPOINT_FILE"

    # Also print a human-readable reminder to stderr
    PLAN_ITER=$(python3 -c "import json; d=json.load(open('$CHECKPOINT_FILE')); print(d.get('iter','?'))" 2>/dev/null || echo "?")
    PLAN_IDEA=$(python3 -c "import json; d=json.load(open('$CHECKPOINT_FILE')); print(d.get('idea','?'))" 2>/dev/null || echo "?")
    echo "" >&2
    echo "[checkpoint_write] IMPLEMENT CONTRACT:" >&2
    echo "  Build exactly: $PLAN_ITER" >&2
    echo "  Idea:          $PLAN_IDEA" >&2
    echo "  Do NOT deviate without rewriting the checkpoint." >&2
    exit 0
fi

# ════════════════════════════════════════════════════════════════════════════════
# MODE: verify
# ════════════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "verify" ]]; then
    if [[ ! -f "$CHECKPOINT_FILE" ]]; then
        echo "[checkpoint_write] ERROR: no checkpoint to verify at $CHECKPOINT_FILE" >&2
        exit 2
    fi

    if [[ -z "$ITER" ]]; then
        echo "[checkpoint_write] ERROR: verify mode requires --iter (the actual iter built)" >&2
        exit 1
    fi

    PLAN=$(cat "$CHECKPOINT_FILE")
    PLAN_ITER=$(echo "$PLAN" | python3 -c "import sys,json; print(json.load(sys.stdin).get('iter',''))")
    PLAN_BOTTLENECK=$(echo "$PLAN" | python3 -c "import sys,json; print(json.load(sys.stdin).get('bottleneck',''))")
    PLAN_IDEA=$(echo "$PLAN" | python3 -c "import sys,json; print(json.load(sys.stdin).get('idea',''))")

    DRIFT=0
    echo "[checkpoint_write] VERIFY against checkpoint:"
    echo ""

    # Check 1: iter name matches
    if [[ "$ITER" == "$PLAN_ITER" ]]; then
        echo "  PASS  Iter name matches plan: $ITER"
    else
        echo "  WARN  Iter name drift: planned '$PLAN_ITER', built '$ITER'"
        DRIFT=$((DRIFT + 1))
    fi

    # Check 2: src file exists
    SRC_DIR="tuning/aitune/${DSL}/srcs/${SHAPE_KEY}"
    SRC_FILE=$(ls "${SRC_DIR}/${ITER}".* 2>/dev/null | head -1 || echo "")
    if [[ -n "$SRC_FILE" ]]; then
        echo "  PASS  Source file exists: $(basename "$SRC_FILE")"
    else
        echo "  FAIL  Source file missing: $SRC_DIR/$ITER.<ext>"
        DRIFT=$((DRIFT + 2))
    fi

    # Check 3: bin exists (compiled)
    BIN_DIR="tuning/aitune/${DSL}/bin/${SHAPE_KEY}"
    if [[ -d "${BIN_DIR}/${ITER}" ]]; then
        echo "  PASS  Binary directory exists: bin/$ITER"
    else
        echo "  WARN  Binary directory missing (not yet compiled?): $BIN_DIR/$ITER"
        DRIFT=$((DRIFT + 1))
    fi

    # Mark checkpoint as verified
    python3 -c "
import json
with open('$CHECKPOINT_FILE') as f:
    d = json.load(f)
d['status'] = 'VERIFIED'
d['verified_iter'] = '$ITER'
d['drift_score'] = $DRIFT
with open('$CHECKPOINT_FILE', 'w') as f:
    json.dump(d, f, indent=2)
"

    echo ""
    echo "  Planned idea: $PLAN_IDEA"
    echo "  Drift score:  $DRIFT  (0=clean, 1-2=minor, 3+=significant)"
    echo ""

    if [[ "$DRIFT" -ge 3 ]]; then
        echo "[checkpoint_write] RESULT: DRIFT DETECTED — recheck what was built vs. planned"
        exit 3
    elif [[ "$DRIFT" -ge 1 ]]; then
        echo "[checkpoint_write] RESULT: MINOR DRIFT — acceptable, but note the differences"
    else
        echo "[checkpoint_write] RESULT: CLEAN — implementation matches plan"
    fi

    exit 0
fi
