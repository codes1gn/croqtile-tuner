#!/usr/bin/env bash
# store_baseline.sh — Atomic PREPARATION baseline: measure + store cuBLAS reference.
#
# Wraps cublas_baseline.sh + store_round.sh into a single atomic step that
# the agent cannot skip. Saves baseline artifacts to a dedicated subfolder
# for easy parsing by the monitor.
#
# USAGE:
#   bash .claude/skills/croq-tune/tools/store_baseline.sh \
#       --dsl <dsl> --shape-key <shape_key> --model <model> \
#       --dtype <dtype> --m <M> --n <N> --k <K> \
#       [--warmup 10] [--iters 50]
#
# OUTPUT (stdout): JSON summary + store confirmation
#
# ARTIFACTS written:
#   tuning/<gpu>/<dsl>/baseline/<shape_key>/<model>/cublas_result.json
#   tuning/<gpu>/<dsl>/logs/<shape_key>/<model>/results.tsv  (iter000 row)
#   tuning/<gpu>/<dsl>/logs/<shape_key>/<model>/idea-log.jsonl (round 0)
#
# EXIT CODES:
#   0 — baseline measured and stored
#   1 — missing arguments
#   2 — cublas_baseline.sh failed
#   3 — store_round.sh failed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/activity_trace.sh" 2>/dev/null || true

DSL=""
SHAPE_KEY=""
MODEL=""
DTYPE=""
M_DIM=""
N_DIM=""
K_DIM=""
WARMUP=10
ITERS=50
TASK_UID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dsl)       DSL="$2";       shift 2 ;;
    --shape-key) SHAPE_KEY="$2"; shift 2 ;;
    --model)     MODEL="$2";     shift 2 ;;
    --dtype)     DTYPE="$2";     shift 2 ;;
    --m)         M_DIM="$2";     shift 2 ;;
    --n)         N_DIM="$2";     shift 2 ;;
    --k)         K_DIM="$2";     shift 2 ;;
    --warmup)    WARMUP="$2";    shift 2 ;;
    --iters)     ITERS="$2";     shift 2 ;;
    --task-uid)  TASK_UID="$2";  shift 2 ;;
    *) echo "[store_baseline] ERROR: unknown arg: $1" >&2; exit 1 ;;
  esac
done

MISSING=()
[ -z "$DSL" ]       && MISSING+=("--dsl")
[ -z "$SHAPE_KEY" ] && MISSING+=("--shape-key")
[ -z "$MODEL" ]     && MISSING+=("--model")
[ -z "$DTYPE" ]     && MISSING+=("--dtype")
[ -z "$M_DIM" ]     && MISSING+=("--m")
[ -z "$N_DIM" ]     && MISSING+=("--n")
[ -z "$K_DIM" ]     && MISSING+=("--k")

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "[store_baseline] ERROR: missing required: ${MISSING[*]}" >&2
  exit 1
fi

# Auto-detect GPU
DETECT_SCRIPT="$SCRIPT_DIR/detect_gpu.sh"
GPU=$(bash "$DETECT_SCRIPT" 2>/dev/null || echo "sm00_unknown")

trace_init --gpu "$GPU" --dsl "$DSL" --shape-key "$SHAPE_KEY" --model "$MODEL"
trace_event "store_baseline" "Measuring cuBLAS baseline for ${DTYPE} ${M_DIM}x${N_DIM}x${K_DIM}"

# ── 0. Write task_config.json early (prevents phantom tasks) ───────────
BASE="tuning/${GPU}/${DSL}"
LOG_DIR="${BASE}/logs/${SHAPE_KEY}/${MODEL}"
CONFIG_FILE="${LOG_DIR}/task_config.json"
mkdir -p "$LOG_DIR"

if [[ -n "$TASK_UID" ]]; then
  if [[ -f "$CONFIG_FILE" ]]; then
    python3 -c "
import json, sys
cfg = json.load(open('$CONFIG_FILE'))
if cfg.get('task_uid') != '$TASK_UID':
    cfg['task_uid'] = '$TASK_UID'
    json.dump(cfg, open('$CONFIG_FILE', 'w'), indent=2, default=str)
    print('[store_baseline] Updated task_config.json with task_uid=$TASK_UID')
" 2>/dev/null || true
  else
    python3 -c "
import json
cfg = {'task_uid': '$TASK_UID', 'dsl': '$DSL', 'model': '$MODEL', 'device': '$GPU', 'status': 'running'}
json.dump(cfg, open('$CONFIG_FILE', 'w'), indent=2, default=str)
print('[store_baseline] Created task_config.json with task_uid=$TASK_UID')
"
  fi
fi

# ── 1. Check if baseline already exists ────────────────────────────────
BASELINE_DIR="${BASE}/baseline/${SHAPE_KEY}/${MODEL}"
RESULT_FILE="${BASELINE_DIR}/cublas_result.json"
TSV="${BASE}/logs/${SHAPE_KEY}/${MODEL}/results.tsv"

if [[ -f "$RESULT_FILE" ]]; then
  EXISTING_TFLOPS=$(python3 -c "import json; print(json.load(open('$RESULT_FILE')).get('tflops', 0))" 2>/dev/null || echo "0")
  if (( $(echo "$EXISTING_TFLOPS > 0" | bc -l 2>/dev/null || echo 0) )); then
    echo "[store_baseline] Baseline already exists: ${EXISTING_TFLOPS} TFLOPS"
    echo "[store_baseline] Artifact: ${RESULT_FILE}"
    # Verify it's also in results.tsv
    if [[ -f "$TSV" ]] && grep -q "iter000" "$TSV"; then
      echo "[store_baseline] Already in results.tsv — skipping"
      cat "$RESULT_FILE"
      exit 0
    fi
    echo "[store_baseline] Missing from results.tsv — re-storing..."
  fi
fi

# ── 2. Measure cuBLAS baseline ─────────────────────────────────────────
echo "[store_baseline] Running cuBLAS baseline: ${DTYPE} ${M_DIM}x${N_DIM}x${K_DIM}..."

BASELINE_JSON=$(bash "$SCRIPT_DIR/cublas_baseline.sh" \
    --dtype "$DTYPE" --m "$M_DIM" --n "$N_DIM" --k "$K_DIM" \
    --warmup "$WARMUP" --iters "$ITERS") || {
  echo "[store_baseline] ERROR: cublas_baseline.sh failed" >&2
  trace_event "store_baseline" "cuBLAS baseline measurement failed" "error"
  exit 2
}

STATUS=$(echo "$BASELINE_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status','error'))")
if [[ "$STATUS" != "ok" ]]; then
  echo "[store_baseline] ERROR: baseline returned status=$STATUS" >&2
  echo "$BASELINE_JSON" >&2
  trace_event "store_baseline" "cuBLAS baseline returned error: $STATUS" "error"
  exit 2
fi

TFLOPS=$(echo "$BASELINE_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['tflops'])")

echo "[store_baseline] cuBLAS baseline: ${TFLOPS} TFLOPS"

# ── 3. Save baseline artifacts ─────────────────────────────────────────
mkdir -p "$BASELINE_DIR"
echo "$BASELINE_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
data['gpu'] = '$GPU'
data['dsl'] = '$DSL'
data['shape_key'] = '$SHAPE_KEY'
data['model'] = '$MODEL'
print(json.dumps(data, indent=2))
" > "$RESULT_FILE"

echo "[store_baseline] Saved baseline artifact: ${RESULT_FILE}"
trace_event "store_baseline" "cuBLAS baseline: ${TFLOPS} TFLOPS (saved to ${RESULT_FILE})"

# ── 4. Remove old iter000 rows (dedup before re-store) ─────────────────
if [[ -f "$TSV" ]]; then
  TEMP_TSV=$(mktemp)
  grep -v "^iter000" "$TSV" > "$TEMP_TSV" || true
  mv "$TEMP_TSV" "$TSV"
fi

IDEA_LOG="${BASE}/logs/${SHAPE_KEY}/${MODEL}/idea-log.jsonl"
if [[ -f "$IDEA_LOG" ]]; then
  TEMP_JSONL=$(mktemp)
  grep -v '"iter": "iter000"' "$IDEA_LOG" > "$TEMP_JSONL" || true
  mv "$TEMP_JSONL" "$IDEA_LOG"
fi

# ── 5. Store as iter000 in results.tsv + idea-log.jsonl ────────────────
bash "$SCRIPT_DIR/store_round.sh" \
  --gpu "$GPU" \
  --dsl "$DSL" \
  --shape-key "$SHAPE_KEY" \
  --model "$MODEL" \
  --iter iter000 \
  --kernel iter000_cublas \
  --tflops "$TFLOPS" \
  --decision KEEP \
  --bottleneck baseline \
  --idea "cuBLAS reference baseline" \
  --round 0 \
  --category baseline \
  --expected-gain "baseline reference" || {
  echo "[store_baseline] ERROR: store_round.sh failed" >&2
  trace_event "store_baseline" "Failed to store baseline as iter000" "error"
  exit 3
}

trace_event "store_baseline" "Stored iter000_cublas: ${TFLOPS} TFLOPS (baseline)"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  BASELINE STORED SUCCESSFULLY                               ║"
echo "║  TFLOPS:   ${TFLOPS}                                       ║"
echo "║  Artifact: ${RESULT_FILE}                                   ║"
echo "║  iter000_cublas in results.tsv ✓                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "$BASELINE_JSON"
