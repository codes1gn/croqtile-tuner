#!/usr/bin/env bash
# store_round.sh — Atomic STORE harness for croq-tune.
#
# USAGE:
#   bash .claude/skills/croq-store/store_round.sh \
#       --gpu       sm90_H100 \
#       --dsl       cuda \
#       --shape-key matmul_bf16fp32_512x16384x16384 \
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
#   bash .claude/skills/croq-tune/tools/detect_gpu.sh
#
# EXIT CODES:
#   0  — all files written and verified
#   1  — missing required argument
#   2  — write failed for one or more files
#   3  — post-write verification failed
#
# The script writes exactly these files:
#   tuning/<gpu>/<dsl>/memory/<key>/rounds.raw.jsonl   (append JSON line)
#   tuning/<gpu>/<dsl>/memory/<key>/rounds.md           (append markdown section)
#   tuning/<gpu>/<dsl>/logs/<key>/idea-log.jsonl        (append JSON line)
#   tuning/<gpu>/<dsl>/logs/<key>/results.tsv           (append TSV row)
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

# ── argument parsing ───────────────────────────────────────────────────────────
GPU=""
DSL=""
SHAPE_KEY=""
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
    --gpu)         GPU="$2";          shift 2 ;;
    --dsl)         DSL="$2";          shift 2 ;;
    --shape-key)   SHAPE_KEY="$2";   shift 2 ;;
    --iter)        ITER="$2";         shift 2 ;;
    --kernel)      KERNEL="$2";       shift 2 ;;
    --tflops)      TFLOPS="$2";       shift 2 ;;
    --decision)    DECISION="$2";     shift 2 ;;
    --bottleneck)  BOTTLENECK="$2";   shift 2 ;;
    --idea)        IDEA="$2";         shift 2 ;;
    --round)       ROUND="$2";        shift 2 ;;
    --category)    CATEGORY="$2";     shift 2 ;;
    --expected-gain) EXPECTED_GAIN="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument $1" >&2; exit 1 ;;
  esac
done

# ── validate required args ─────────────────────────────────────────────────────
MISSING=()
[ -z "$GPU" ]        && MISSING+=("--gpu")
[ -z "$DSL" ]        && MISSING+=("--dsl")
[ -z "$SHAPE_KEY" ]  && MISSING+=("--shape-key")
[ -z "$ITER" ]       && MISSING+=("--iter")
[ -z "$KERNEL" ]     && MISSING+=("--kernel")
[ -z "$TFLOPS" ]     && MISSING+=("--tflops")
[ -z "$DECISION" ]   && MISSING+=("--decision")
[ -z "$BOTTLENECK" ] && MISSING+=("--bottleneck")
[ -z "$IDEA" ]       && MISSING+=("--idea")
[ -z "$ROUND" ]      && MISSING+=("--round")

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "ERROR: missing required arguments: ${MISSING[*]}" >&2
  exit 1
fi

# ── validate decision value ────────────────────────────────────────────────────
case "$DECISION" in
  KEEP|DISCARD|SEGFAULT|HANG|COMPILE_FAIL) ;;
  *) echo "ERROR: --decision must be one of KEEP|DISCARD|SEGFAULT|HANG|COMPILE_FAIL" >&2; exit 1 ;;
esac

# ── validate iter tag ──────────────────────────────────────────────────────────
if [[ ! "$ITER" =~ ^iter[0-9]{3}$ ]]; then
  echo "ERROR: --iter must match iter<NNN> (3-digit), got: $ITER" >&2
  exit 1
fi

if [[ ! "$KERNEL" =~ ^iter[0-9]{3}_[a-z0-9_]{2,16}$ ]]; then
  echo "ERROR: --kernel must match iter<NNN>_<tag> where tag is 2-16 lowercase chars, got: $KERNEL" >&2
  echo "       Example: iter045_myidea" >&2
  exit 1
fi

# ── paths ──────────────────────────────────────────────────────────────────────
BASE="tuning/${GPU}/${DSL}"
MEM_DIR="${BASE}/memory/${SHAPE_KEY}"
LOG_DIR="${BASE}/logs/${SHAPE_KEY}"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

mkdir -p "$MEM_DIR" "$LOG_DIR"

# ── 1. rounds.raw.jsonl ────────────────────────────────────────────────────────
JSONL="${MEM_DIR}/rounds.raw.jsonl"
JSON_LINE=$(python3 -c "
import json, sys
print(json.dumps({
  'iter':       '$ITER',
  'kernel':     '$KERNEL',
  'tflops':     float('$TFLOPS'),
  'decision':   '$DECISION',
  'bottleneck': '$BOTTLENECK',
  'idea':       '$IDEA',
  'round':      int('$ROUND'),
  'timestamp':  '$TIMESTAMP'
}))
")
echo "$JSON_LINE" >> "$JSONL" || { echo "ERROR: failed to write $JSONL" >&2; exit 2; }

# ── 2. rounds.md ──────────────────────────────────────────────────────────────
MD="${MEM_DIR}/rounds.md"
cat >> "$MD" << MDEOF

## ${ITER} — ${TIMESTAMP}
- kernel: \`${KERNEL}\`
- tflops: \`${TFLOPS}\`
- decision: **${DECISION}**
- bottleneck: \`${BOTTLENECK}\`
- idea: ${IDEA}
MDEOF
[ $? -eq 0 ] || { echo "ERROR: failed to write $MD" >&2; exit 2; }

# ── 3. idea-log.jsonl ─────────────────────────────────────────────────────────
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
echo "$IDEA_LINE" >> "$IDEA_LOG" || { echo "ERROR: failed to write $IDEA_LOG" >&2; exit 2; }

# ── 4. results.tsv ────────────────────────────────────────────────────────────
TSV="${LOG_DIR}/results.tsv"
if [ ! -f "$TSV" ]; then
  echo -e "iter\tkernel\ttflops\tdecision\tbottleneck\tidea_summary" > "$TSV"
fi
IDEA_SHORT="${IDEA:0:80}"
echo -e "${ITER}\t${KERNEL}\t${TFLOPS}\t${DECISION}\t${BOTTLENECK}\t${IDEA_SHORT}" >> "$TSV" \
  || { echo "ERROR: failed to write $TSV" >&2; exit 2; }

# ── post-write verification ────────────────────────────────────────────────────
FAIL=0

[ -f "$JSONL" ] || { echo "VERIFY FAIL: $JSONL missing"; FAIL=1; }
[ -f "$MD" ]    || { echo "VERIFY FAIL: $MD missing"; FAIL=1; }
[ -f "$IDEA_LOG" ] || { echo "VERIFY FAIL: $IDEA_LOG missing"; FAIL=1; }
[ -f "$TSV" ] && grep -q "$ITER" "$TSV" \
              || { echo "VERIFY FAIL: $TSV missing row for $ITER"; FAIL=1; }

if [ $FAIL -ne 0 ]; then
  echo "ERROR: post-write verification failed" >&2
  exit 3
fi

# ── success ────────────────────────────────────────────────────────────────────
echo "[store_round] STORE complete for ${ITER} (${DECISION} ${TFLOPS} TFLOPS)"
echo "[store_round] Written:"
echo "  ${JSONL}  ($(wc -l < "$JSONL") lines)"
echo "  ${MD}     ($(wc -l < "$MD") lines)"
echo "  ${IDEA_LOG} ($(wc -l < "$IDEA_LOG") lines)"
echo "  ${TSV}    ($(wc -l < "$TSV") rows)"
