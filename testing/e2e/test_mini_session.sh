#!/usr/bin/env bash
# test_mini_session.sh — E2E validation for one completed croq-tune round.
#
# This script has two modes:
#
# MODE 1 — VALIDATE-ONLY (default):
#   Verify a completed session transcript + filesystem artifacts.
#   Requires: a session JSONL file produced by a real agent run.
#
#   USAGE:
#     bash testing/e2e/test_mini_session.sh \
#         --session ~/.cursor/projects/.../agent-transcripts/<uuid>.jsonl \
#         --dsl cuda --shape-key matmul_bf16fp32_512x16384x16384
#
# MODE 2 — MOCK-ONLY (no agent, no GPU):
#   Set up a mock workspace and run a filesystem-only validation.
#   Simulates one round of filesystem output, then checks artifact correctness.
#   Does NOT test agent behaviour (no agent is launched).
#
#   USAGE:
#     bash testing/e2e/test_mini_session.sh --mock-only
#
# HOW TO RUN A REAL E2E SESSION (for Mode 1):
#   1. In Cursor IDE or CLI, run:
#        /croq-tune cuda bf16fp32 matmul_bf16fp32_512x16384x16384
#   2. If testing with mocks (no GPU), prepend PATH first:
#        export PATH="$(git rev-parse --show-toplevel)/testing/mocks:$PATH"
#        export MOCK_NCU_SCENARIO=memory_bound
#   3. After the session completes (≥1 round), note the session JSONL path
#      from ~/.cursor/projects/<project>/agent-transcripts/<uuid>.jsonl
#   4. Run this script with --session <path>
#
# EXIT CODES:
#   0  — all checks passed (or --mock-only: mock artifacts validated)
#   1  — one or more required checks failed
#   2  — bad arguments or missing file

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

source testing/e2e/setup_mock_workspace.sh
source testing/harness/tap_helpers.sh

# ── defaults ──────────────────────────────────────────────────────────────────
SESSION=""
DSL="cuda"
SHAPE_KEY="matmul_bf16fp32_512x16384x16384"
MOCK_ONLY=0

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --session)   SESSION="$2";    shift 2 ;;
        --dsl)       DSL="$2";        shift 2 ;;
        --shape-key) SHAPE_KEY="$2";  shift 2 ;;
        --mock-only) MOCK_ONLY=1;     shift   ;;
        *) echo "ERROR: unknown arg $1" >&2; exit 2 ;;
    esac
done

# ── MODE 2: MOCK-ONLY — filesystem-only validation ────────────────────────────
if [[ "$MOCK_ONLY" -eq 1 ]]; then
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  E2E Mock-Only: Filesystem Artifact Validation       ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""

    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT

    # Set up mock workspace (baseline only)
    setup_mock_workspace "$TMP" "$DSL" "$SHAPE_KEY"

    # Simulate one completed round by writing iter001 artifacts
    AITUNE="$MOCK_WS_TUNING"
    PERF="$AITUNE/perf/$SHAPE_KEY"
    MEM="$AITUNE/memory/$SHAPE_KEY"
    LOGS="$AITUNE/logs/$SHAPE_KEY"
    SRCS="$AITUNE/srcs/$SHAPE_KEY"
    CHKDIR="$AITUNE/checkpoints"
    TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    REPO=$(git rev-parse --show-toplevel)
    MOCK_NCU_BIN="$REPO/testing/mocks/mock_ncu"

    # Write mock ncu CSV (as if mock_ncu ran) — call mock_ncu directly by path
    MOCK_NCU_SCENARIO=memory_bound bash "$MOCK_NCU_BIN" \
        --set full --export "$PERF/ncu_iter001_draft.ncu-rep" \
        --force-overwrite python3 "$REPO/testing/mocks/mock_kernel" >/dev/null 2>&1
    MOCK_NCU_SCENARIO=memory_bound bash "$MOCK_NCU_BIN" \
        --import "$PERF/ncu_iter001_draft.ncu-rep" --csv --page raw \
        > "$PERF/ncu_iter001_draft.csv" 2>/dev/null

    # Write iter001 timing (as if mock_kernel ran)
    printf "VERIFY: PASS\nTFLOPS: 28.50   time_ms: 120.300\n" \
        > "$PERF/timing_iter001_draft.txt"

    # Append iter001 to log files (as if store_round ran)
    printf '{"round": 1, "iter": "iter001_draft", "bottleneck": "memory_bound", "idea": "vectorized loads", "category": "memory", "expected_gain": "+8 TFLOPS", "decision": "KEEP", "tflops": 28.5, "timestamp": "%s"}\n' \
        "$TS" >> "$LOGS/idea-log.jsonl"

    printf 'iter001\titer001_draft\t28.5\t120.3\tKEEP\tmemory_bound\tvectorized loads\t%s\n' \
        "$TS" >> "$LOGS/results.tsv"

    # Update checkpoint
    cat > "$CHKDIR/$SHAPE_KEY.json" <<JSON
{
  "dsl": "$DSL",
  "shape_key": "$SHAPE_KEY",
  "status": "VERIFIED",
  "last_iter": "iter001_draft",
  "next_state": "PROFILE",
  "timestamp": "$TS"
}
JSON

    echo ""
    echo "--- Simulated round 1 artifacts written ---"
    echo ""

    plan 14

    # Source file naming
    like "iter001_draft.cu exists in srcs/" \
        "$(ls "$SRCS")" "iter001_draft.cu"

    like "iter000_baseline.cu exists in srcs/" \
        "$(ls "$SRCS")" "iter000_baseline.cu"

    # ncu artifacts
    [ -f "$PERF/ncu_iter001_draft.ncu-rep" ] && ok "ncu .ncu-rep sentinel exists" 0 \
        || ok "ncu .ncu-rep sentinel exists" 1

    [ -f "$PERF/ncu_iter001_draft.csv" ] && ok "ncu CSV exists" 0 \
        || ok "ncu CSV exists" 1

    # Validate CSV has the required metric columns
    like "ncu CSV has dram metric" \
        "$(cat "$PERF/ncu_iter001_draft.csv")" "dram__throughput"
    like "ncu CSV has sm metric" \
        "$(cat "$PERF/ncu_iter001_draft.csv")" "sm__throughput"

    # profile_extract.sh classifies correctly
    PROFILE_JSON=$(bash .claude/skills/croq-tune/tools/profile_extract.sh \
        --csv "$PERF/ncu_iter001_draft.csv" --iter "iter001_draft" 2>/dev/null)
    like "profile_extract classifies memory_bound" "$PROFILE_JSON" '"bottleneck": "memory_bound"'
    like "profile_extract produces high confidence" "$PROFILE_JSON" '"confidence"'

    # Timing output
    like "timing file has TFLOPS:" \
        "$(cat "$PERF/timing_iter001_draft.txt")" "TFLOPS:"
    like "timing file has VERIFY: PASS" \
        "$(cat "$PERF/timing_iter001_draft.txt")" "VERIFY: PASS"

    # idea-log.jsonl
    IDEA_LINES=$(wc -l < "$LOGS/idea-log.jsonl")
    [ "$IDEA_LINES" -ge 1 ] && ok "idea-log.jsonl has ≥1 entry (round 1)" 0 \
        || ok "idea-log.jsonl has ≥1 entry (round 1)" 1

    # results.tsv
    DATA_ROWS=$(tail -n +2 "$LOGS/results.tsv" | grep -c . || true)
    [ "$DATA_ROWS" -ge 2 ] && ok "results.tsv has ≥2 data rows" 0 \
        || ok "results.tsv has ≥2 data rows" 1

    # Kernel tag naming — no bare iter files
    BAD_BARE=$(ls "$SRCS"/iter[0-9][0-9][0-9].cu 2>/dev/null | wc -l || true)
    is "No bare iter*.cu source files (all tagged)" "$BAD_BARE" "0"

    # Checkpoint updated
    STATUS=$(python3 -c "import json; print(json.load(open('$CHKDIR/$SHAPE_KEY.json'))['status'])" 2>/dev/null || echo "MISSING")
    like "Checkpoint status is VERIFIED or PROFILE" "$STATUS" "VERIFIED\|PROFILE"

    done_testing

    echo ""
    echo "(Mock-only mode: skipping grep_session_markers.sh filesystem check —"
    echo " that check runs against the real repo and requires a completed real session.)"
    exit 0
fi

# ── MODE 1: VALIDATE session transcript ──────────────────────────────────────
if [[ -z "$SESSION" ]]; then
    echo "ERROR: --session <path.jsonl> required in validate mode (or use --mock-only)" >&2
    exit 2
fi

if [[ ! -f "$SESSION" ]]; then
    echo "ERROR: session file not found: $SESSION" >&2
    exit 2
fi

echo "╔══════════════════════════════════════════════════════╗"
echo "║  E2E Session Validation                              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Session:   $SESSION"
echo "  Lines:     $(wc -l < "$SESSION")"
echo "  DSL:       $DSL"
echo "  Shape-key: $SHAPE_KEY"
echo ""

# Extract all text from session JSONL
TRANSCRIPT=$(python3 -c "
import sys, json
for line in sys.stdin:
    try:
        print(json.dumps(json.loads(line.strip())))
    except Exception:
        pass
" < "$SESSION" 2>/dev/null || echo "")

plan 20

check_marker() {
    local desc="$1"
    local pattern="$2"
    echo "$TRANSCRIPT" | grep -q "$pattern" && RC=0 || RC=1
    ok "$desc" "$RC"
}

echo "--- PROFILE step ---"
check_marker "ncu invoked in PROFILE"           "ncu "
check_marker "profile_extract.sh called"        "profile_extract\\.sh"
check_marker "bottleneck classified"            '"bottleneck"'

echo ""
echo "--- IDEA step ---"
check_marker "WebSearch called in IDEA"         "WebSearch\|web_search"
check_marker "checkpoint_write.sh called"       "checkpoint_write\\.sh"
check_marker "PLANNED checkpoint written"       "PLANNED\|planned_iter"

echo ""
echo "--- IMPLEMENT step ---"
check_marker "checkpoint read back"             "checkpoint_write.*read\|read.*checkpoint"
check_marker "next_iter.sh called"              "next_iter\\.sh"
check_marker "build script executed"            "build_iter\|build_attempt"

echo ""
echo "--- VERIFY step ---"
check_marker "VERIFY: PASS appeared"            "VERIFY: PASS"
check_marker "checkpoint verify step ran"       "verify.*checkpoint_write\|checkpoint_write.*verify"

echo ""
echo "--- MEASURE step ---"
check_marker "TFLOPS measurement present"       "TFLOPS:"
check_marker "KEEP or DISCARD decision"         "KEEP\|DISCARD"

echo ""
echo "--- STORE step ---"
check_marker "store_round.sh called"            "store_round\\.sh"
check_marker "STORE complete message"           "\[store_round\] STORE complete"
check_marker "idea-log.jsonl updated"           "idea-log\.jsonl\|idea.log"

echo ""
echo "--- Anti-patterns (absent = PASS) ---"
BAD_BARE=$(echo "$TRANSCRIPT" | grep -cE 'iter[0-9]{3}\.(cu|co|py)\b' || true)
is "No bare iter source files" "$BAD_BARE" "0"

BAD_NCU=$(echo "$TRANSCRIPT" | grep -cE 'ncu_iter[0-9]{3}\.csv\b' || true)
is "No bare ncu CSV filenames" "$BAD_NCU" "0"

echo ""
echo "--- Filesystem artifacts ---"
# Detect GPU key from the session filesystem (try sm90_H100 as default for real sessions)
GPU_KEY="${CROQTUNER_GPU:-sm90_H100}"
FS_RESULTS="tuning/${GPU_KEY}/${DSL}/logs/${SHAPE_KEY}"
# Find results.tsv under any model subdir
FS_TSV=$(find "$FS_RESULTS" -name results.tsv 2>/dev/null | head -1)
if [[ -n "$FS_TSV" && -f "$FS_TSV" ]]; then
    DATA_ROWS=$(tail -n +2 "$FS_TSV" | grep -c . || true)
    [ "$DATA_ROWS" -ge 2 ] && ok "results.tsv ≥2 data rows" 0 || ok "results.tsv ≥2 data rows" 1
else
    ok "results.tsv ≥2 data rows" 1
fi

FS_PERF="tuning/${GPU_KEY}/${DSL}/perf/${SHAPE_KEY}"
NCU_CSVs=$(ls "$FS_PERF"/ncu_iter*.csv 2>/dev/null | wc -l || echo "0")
[ "$NCU_CSVs" -ge 1 ] && ok "≥1 ncu CSV in perf/" 0 || ok "≥1 ncu CSV in perf/" 1

done_testing

echo ""
echo "--- Running grep_session_markers.sh ---"
bash testing/e2e/grep_session_markers.sh \
    --session "$SESSION" \
    --fs --dsl "$DSL" --shape-key "$SHAPE_KEY"
