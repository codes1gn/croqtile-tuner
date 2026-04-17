#!/usr/bin/env bash
# store_round.sh — Atomic STORE harness for croq-tune.
#
# USAGE:
#   bash .claude/skills/croq-tune/tools/store_round.sh \
#       --gpu       sm90_H100 \
#       --dsl       cuda \
#       --shape-key matmul_bf16fp32_512x16384x16384 \
#       --model     opus-4 \
#       --iter      iter045 \
#       --kernel    iter045_myidea \
#       --tflops    36.12 \
#       --decision  KEEP \
#       --bottleneck memory_bound \
#       --idea      "Increased tile BK from 32 to 64 to improve L2 reuse" \
#       --round     45 \
#       --category  "tiling"
#
# The --gpu value is emitted by:
#   bash .claude/skills/croq-tune/tools/detect_gpu.sh (auto-detected if --gpu omitted)
#
# EXIT CODES:
#   0  — all files written and verified
#   1  — missing required argument
#   2  — write failed for one or more files
#   3  — post-write verification failed
#
# The script writes exactly these files:
#   tuning/<gpu>/<dsl>/logs/<key>/<model>/idea-log.jsonl        (append JSON line)
#   tuning/<gpu>/<dsl>/logs/<key>/<model>/results.tsv           (append TSV row)
#
# results.tsv format:
#   iter  kernel  tflops  decision  bottleneck  idea_summary
#
# The script does NOT write:
#   - source files (.cu / .py / .co)
#   - build/run/profile scripts
#   - checkpoint JSON
#   - git commits
# Those remain the agent's responsibility.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/activity_trace.sh"

# ── argument parsing ───────────────────────────────────────────────────────────
GPU=""
DSL=""
SHAPE_KEY=""
MODEL=""
ITER=""
KERNEL=""
TFLOPS=""
DECISION=""
BOTTLENECK=""
IDEA=""
ROUND=""
CATEGORY="general"
EXPECTED_GAIN="unknown"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)           GPU="$2";            shift 2 ;;
    --dsl)           DSL="$2";            shift 2 ;;
    --shape-key)     SHAPE_KEY="$2";      shift 2 ;;
    --model)         MODEL="$2";          shift 2 ;;
    --iter)          ITER="$2";           shift 2 ;;
    --kernel)        KERNEL="$2";         shift 2 ;;
    --tflops)        TFLOPS="$2";         shift 2 ;;
    --decision)      DECISION="$2";       shift 2 ;;
    --bottleneck)    BOTTLENECK="$2";     shift 2 ;;
    --idea)          IDEA="$2";           shift 2 ;;
    --round)         ROUND="$2";          shift 2 ;;
    --category)      CATEGORY="$2";       shift 2 ;;
    --expected-gain) EXPECTED_GAIN="$2";  shift 2 ;;
    *) echo "ERROR: unknown argument $1" >&2; echo "[SUGGESTION] Use your judgement to decide autonomously. Remove the unknown argument and retry. Valid args: --gpu --dsl --shape-key --model --iter --kernel --tflops --decision --bottleneck --idea --round --category --expected-gain" >&2; exit 1 ;;
  esac
done

# ── auto-detect GPU if not provided ───────────────────────────────────────────
# --gpu is optional; if omitted the script calls detect_gpu.sh automatically.
if [[ -z "$GPU" ]]; then
  DETECT_SCRIPT="$(dirname "$0")/detect_gpu.sh"
  GPU=$(bash "$DETECT_SCRIPT" 2>/dev/null || echo "sm00_unknown")
fi

# ── validate required args ─────────────────────────────────────────────────────
MISSING=()
[ -z "$DSL" ]        && MISSING+=("--dsl")
[ -z "$SHAPE_KEY" ]  && MISSING+=("--shape-key")
[ -z "$MODEL" ]      && MISSING+=("--model")
[ -z "$ITER" ]       && MISSING+=("--iter")
[ -z "$KERNEL" ]     && MISSING+=("--kernel")
[ -z "$TFLOPS" ]     && MISSING+=("--tflops")
[ -z "$DECISION" ]   && MISSING+=("--decision")
[ -z "$BOTTLENECK" ] && MISSING+=("--bottleneck")
[ -z "$IDEA" ]       && MISSING+=("--idea")
[ -z "$ROUND" ]      && MISSING+=("--round")

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "ERROR: missing required arguments: ${MISSING[*]}" >&2
  echo "[SUGGESTION] Use your judgement to decide autonomously. Add the missing arguments and retry. All listed args are mandatory. Use resume_state.sh to get current state if unsure about values." >&2
  exit 1
fi

# ── validate decision value ────────────────────────────────────────────────────
case "$DECISION" in
  KEEP|DISCARD|SEGFAULT|HANG|COMPILE_FAIL) ;;
  *) echo "ERROR: --decision must be one of KEEP|DISCARD|SEGFAULT|HANG|COMPILE_FAIL" >&2; echo "[SUGGESTION] Use your judgement to decide autonomously. Fix the --decision value. Use KEEP if TFLOPS improved, DISCARD if not, COMPILE_FAIL if build failed, SEGFAULT/HANG for runtime errors." >&2; exit 1 ;;
esac

# ── validate iter tag ──────────────────────────────────────────────────────────
if [[ ! "$ITER" =~ ^iter[0-9]+$ ]]; then
  echo "ERROR: --iter must match iter<N+> (bare number, no tag), got: $ITER" >&2
  echo "[SUGGESTION] Use your judgement to decide autonomously. Use the bare iteration number like iter005. Get the correct number from next_iter.sh output." >&2
  exit 1
fi

if [[ ! "$KERNEL" =~ ^iter[0-9]+_[a-z][a-z0-9_]{1,30}$ ]]; then
  echo "ERROR: --kernel must match iter<N+>_<tag> where tag is 2-31 lowercase chars, got: $KERNEL" >&2
  echo "[SUGGESTION] Use your judgement to decide autonomously. Fix the --kernel name. Format: iter<NNN>_<descriptive_tag>. Example: iter045_myidea. Tag must be 2-31 lowercase alphanumeric+underscore chars starting with a letter." >&2
  exit 1
fi

# ── paths ──────────────────────────────────────────────────────────────────────
BASE="tuning/${GPU}/${DSL}"
LOG_DIR="${BASE}/logs/${SHAPE_KEY}/${MODEL}"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

mkdir -p "$LOG_DIR"

trace_init --gpu "$GPU" --dsl "$DSL" --shape-key "$SHAPE_KEY" --model "$MODEL"
trace_event "store_round" "Storing $KERNEL ($DECISION $TFLOPS TFLOPS)"

# ── 1. idea-log.jsonl ─────────────────────────────────────────────────────────
IDEA_LOG="${LOG_DIR}/idea-log.jsonl"
IDEA_LINE=$(python3 -c "
import json
print(json.dumps({
  'round':         int('$ROUND'),
  'iter':          '$ITER',
  'bottleneck':    '$BOTTLENECK',
  'idea':          '$IDEA',
  'category':      '$CATEGORY',
  'expected_gain': '$EXPECTED_GAIN',
  'decision':      '$DECISION',
  'tflops':        float('$TFLOPS'),
  'timestamp':     '$TIMESTAMP'
}))
")
echo "$IDEA_LINE" >> "$IDEA_LOG" || { echo "ERROR: failed to write $IDEA_LOG" >&2; echo "[SUGGESTION] Use your judgement to decide autonomously. Check filesystem permissions and disk space. Ensure the parent directory exists. Create it with: mkdir -p $(dirname $IDEA_LOG)" >&2; exit 2; }

# ── 4. results.tsv ────────────────────────────────────────────────────────────
TSV="${LOG_DIR}/results.tsv"
if [ ! -f "$TSV" ]; then
  echo -e "iter\tkernel\ttflops\tdecision\tbottleneck\tidea_summary" > "$TSV"
fi
IDEA_SHORT="${IDEA:0:80}"
echo -e "${ITER}\t${KERNEL}\t${TFLOPS}\t${DECISION}\t${BOTTLENECK}\t${IDEA_SHORT}" >> "$TSV" \
  || { echo "ERROR: failed to write $TSV" >&2; echo "[SUGGESTION] Use your judgement to decide autonomously. Check filesystem permissions and disk space for $TSV. Create directory with: mkdir -p $(dirname $TSV)" >&2; exit 2; }

# ── post-write verification ────────────────────────────────────────────────────
FAIL=0

[ -f "$IDEA_LOG" ] || { echo "VERIFY FAIL: $IDEA_LOG missing"; FAIL=1; }
[ -f "$TSV" ] && grep -q "$ITER" "$TSV" \
              || { echo "VERIFY FAIL: $TSV missing row for $ITER"; FAIL=1; }

if [ $FAIL -ne 0 ]; then
  echo "ERROR: post-write verification failed" >&2
  echo "[SUGGESTION] Use your judgement to decide autonomously. Files were not written correctly. Retry the store_round.sh command. If it persists, check disk space and file permissions, then continue with the next iteration." >&2
  exit 3
fi

# ── success ────────────────────────────────────────────────────────────────────
trace_event "store_round" "Stored $KERNEL: $DECISION $TFLOPS TFLOPS (round $ROUND)"
echo "[store_round] STORE complete for ${ITER} (${DECISION} ${TFLOPS} TFLOPS)"
echo "[store_round] Written:"
echo "  ${IDEA_LOG} ($(wc -l < "$IDEA_LOG") lines)"
echo "  ${TSV}    ($(wc -l < "$TSV") rows)"
echo "[NEXT] CONTINUE to next round — advance immediately to PROFILE"
