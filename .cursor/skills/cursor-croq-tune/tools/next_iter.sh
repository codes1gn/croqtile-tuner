#!/usr/bin/env bash
# next_iter.sh — Canonical next-iteration name resolver for croq-tune.
#
# USAGE:
#   bash .cursor/skills/cursor-croq-tune/tools/next_iter.sh \
#       --gpu       sm90_H100 \
#       --dsl       cuda \
#       --shape-key matmul_bf16fp32_512x16384x16384 \
#       --model     opus-4 \
#       --tag       myoptimization
#
#   Prints one line to stdout: iter<NNN>_<tag>
#   (e.g. iter069_myoptimization)
#   Prints nothing else on stdout so the output can be captured:
#       ITER=$(bash .cursor/skills/cursor-croq-tune/tools/next_iter.sh --gpu sm90_H100 --dsl cuda --shape-key <key> --tag <tag>)
#
# The --gpu value is emitted by:
#   bash .cursor/skills/cursor-croq-tune/tools/detect_gpu.sh
#
# For attempt naming (compile-fail), pass --attempt:
#   bash .cursor/skills/cursor-croq-tune/tools/next_iter.sh ... --attempt
#   → attempt0042_myoptimization
#
# EXIT CODES:
#   0  — success, canonical name printed to stdout
#   1  — missing required argument
#   2  — tag validation failed
#
# This script is READ-ONLY. It writes nothing. It only reads existing artifact
# filenames to determine the next sequence number.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/activity_trace.sh"

# ── argument parsing ───────────────────────────────────────────────────────────
GPU=""
DSL=""
SHAPE_KEY=""
MODEL=""
TAG=""
ATTEMPT_MODE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)       GPU="$2";       shift 2 ;;
    --dsl)       DSL="$2";       shift 2 ;;
    --shape-key) SHAPE_KEY="$2"; shift 2 ;;
    --model)     MODEL="$2";     shift 2 ;;
    --tag)       TAG="$2";       shift 2 ;;
    --attempt)   ATTEMPT_MODE=true; shift ;;
    *) echo "ERROR: unknown argument $1" >&2; echo "[SUGGESTION] Use your judgement to decide autonomously. Remove the unknown argument '$1' and retry. Valid args: --gpu --dsl --shape-key --model --tag --attempt" >&2; exit 1 ;;
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
[ -z "$DSL" ]       && MISSING+=("--dsl")
[ -z "$SHAPE_KEY" ] && MISSING+=("--shape-key")
[ -z "$MODEL" ]     && MISSING+=("--model")
[ -z "$TAG" ]       && MISSING+=("--tag")

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "ERROR: missing required arguments: ${MISSING[*]}" >&2
  echo "[SUGGESTION] Use your judgement to decide autonomously. Provide all missing arguments and retry. --dsl is the DSL name (cuda/croqtile/triton/etc), --shape-key is the full shape key, --model is the model name, --tag is a short lowercase descriptor for your optimization idea." >&2
  exit 1
fi

# ── validate tag ───────────────────────────────────────────────────────────────
if [[ ! "$TAG" =~ ^[a-z][a-z0-9_]{1,30}$ ]]; then
  echo "ERROR: --tag must be 2-31 chars, start with a letter, lowercase alphanumeric+underscore only." >&2
  echo "       Got: $TAG" >&2
  echo "[SUGGESTION] Use your judgement to decide autonomously. Fix the --tag value. Use a short lowercase descriptor like 'swizzle', 'tilek32', 'doublebuf'. Must start with a letter, only a-z 0-9 and underscore, 2-31 chars total." >&2
  exit 2
fi

# ── paths ──────────────────────────────────────────────────────────────────────
SRC_DIR="tuning/${GPU}/${DSL}/srcs/${SHAPE_KEY}/${MODEL}"
trace_init --gpu "$GPU" --dsl "$DSL" --shape-key "$SHAPE_KEY" --model "$MODEL"

# ── find highest existing sequence number ──────────────────────────────────────
if $ATTEMPT_MODE; then
  # Look for attempt<AAAA>* files (4-digit)
  HIGHEST=0
  if [ -d "$SRC_DIR" ]; then
    while IFS= read -r f; do
      NUM=$(echo "$f" | grep -oP 'attempt\K[0-9]+' || true | head -1)
      if [ -n "$NUM" ] && [ "$((10#$NUM))" -gt "$HIGHEST" ]; then
        HIGHEST=$((10#$NUM))
      fi
    done < <({ ls "$SRC_DIR" 2>/dev/null | grep -P '^attempt[0-9]{4}'; } 2>/dev/null || true)
  fi
  NEXT=$((HIGHEST + 1))
  trace_event "next_iter" "Resolved attempt$(printf '%04d' "$NEXT")_$TAG"
  printf "attempt%04d_%s\n" "$NEXT" "$TAG"
else
  # Look for iter<NNN>* files (3-digit)
  # Also check logs/results.tsv to catch iterations whose sources may be missing
  HIGHEST=0

  if [ -d "$SRC_DIR" ]; then
    while IFS= read -r f; do
      NUM=$({ echo "$f" | grep -oP '^iter\K[0-9]+'; } 2>/dev/null || true)
      NUM=$(echo "$NUM" | head -1)
      if [ -n "$NUM" ] && [ "$((10#$NUM))" -gt "$HIGHEST" ]; then
        HIGHEST=$((10#$NUM))
      fi
    done < <({ ls "$SRC_DIR" 2>/dev/null | grep -P '^iter[0-9]{3}'; } 2>/dev/null || true)
  fi

  # Also check results.tsv
  TSV="tuning/${GPU}/${DSL}/logs/${SHAPE_KEY}/${MODEL}/results.tsv"
  if [ -f "$TSV" ]; then
    while IFS=$'\t' read -r iter _rest; do
      NUM=$({ echo "$iter" | grep -oP '^iter\K[0-9]+'; } 2>/dev/null || true)
      NUM=$(echo "$NUM" | head -1)
      if [ -n "$NUM" ] && [ "$((10#$NUM))" -gt "$HIGHEST" ]; then
        HIGHEST=$((10#$NUM))
      fi
    done < <(tail -n +2 "$TSV" 2>/dev/null || true)
  fi

  # Also check idea-log.jsonl for iter numbers
  IDEA_LOG="tuning/${GPU}/${DSL}/logs/${SHAPE_KEY}/${MODEL}/idea-log.jsonl"
  if [ -f "$IDEA_LOG" ]; then
    while IFS= read -r line; do
      iter=$(echo "$line" | python3 -c "
import sys,json
try:
    d=json.loads(sys.stdin.read())
    print(d.get('iter',''))
except Exception:
    pass
" 2>/dev/null || true)
      NUM=$({ echo "$iter" | grep -oP '^iter\K[0-9]+'; } 2>/dev/null || true)
      NUM=$(echo "$NUM" | head -1)
      if [ -n "$NUM" ] && [ "$((10#$NUM))" -gt "$HIGHEST" ]; then
        HIGHEST=$((10#$NUM))
      fi
    done < "$IDEA_LOG"
  fi

  NEXT=$((HIGHEST + 1))
  trace_event "next_iter" "Resolved iter$(printf '%03d' "$NEXT")_$TAG"
  printf "iter%03d_%s\n" "$NEXT" "$TAG"
fi
