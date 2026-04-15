#!/usr/bin/env bash
# next_iter.sh — Canonical next-iteration name resolver for croq-tune.
#
# USAGE:
#   bash .claude/skills/croq-store/next_iter.sh \
#       --gpu       sm90_H100 \
#       --dsl       cuda \
#       --shape-key matmul_bf16fp32_512x16384x16384 \
#       --tag       myoptimization
#
#   Prints one line to stdout: iter<NNN>_<tag>
#   (e.g. iter069_myoptimization)
#   Prints nothing else on stdout so the output can be captured:
#       ITER=$(bash .claude/skills/croq-store/next_iter.sh --gpu sm90_H100 --dsl cuda --shape-key <key> --tag <tag>)
#
# The --gpu value is emitted by:
#   bash .claude/skills/croq-tune/tools/detect_gpu.sh
#
# For attempt naming (compile-fail), pass --attempt:
#   bash .claude/skills/croq-store/next_iter.sh ... --attempt
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

# ── argument parsing ───────────────────────────────────────────────────────────
GPU=""
DSL=""
SHAPE_KEY=""
TAG=""
ATTEMPT_MODE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)       GPU="$2";       shift 2 ;;
    --dsl)       DSL="$2";       shift 2 ;;
    --shape-key) SHAPE_KEY="$2"; shift 2 ;;
    --tag)       TAG="$2";       shift 2 ;;
    --attempt)   ATTEMPT_MODE=true; shift ;;
    *) echo "ERROR: unknown argument $1" >&2; exit 1 ;;
  esac
done

# ── validate required args ─────────────────────────────────────────────────────
MISSING=()
[ -z "$GPU" ]       && MISSING+=("--gpu")
[ -z "$DSL" ]       && MISSING+=("--dsl")
[ -z "$SHAPE_KEY" ] && MISSING+=("--shape-key")
[ -z "$TAG" ]       && MISSING+=("--tag")

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "ERROR: missing required arguments: ${MISSING[*]}" >&2
  exit 1
fi

# ── validate tag ───────────────────────────────────────────────────────────────
if [[ ! "$TAG" =~ ^[a-z][a-z0-9_]{1,15}$ ]]; then
  echo "ERROR: --tag must be 2-16 chars, start with a letter, lowercase alphanumeric+underscore only." >&2
  echo "       Got: $TAG" >&2
  exit 2
fi

# ── paths ──────────────────────────────────────────────────────────────────────
SRC_DIR="tuning/${GPU}/${DSL}/srcs/${SHAPE_KEY}"

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
  TSV="tuning/${GPU}/${DSL}/logs/${SHAPE_KEY}/results.tsv"
  if [ -f "$TSV" ]; then
    while IFS=$'\t' read -r iter _rest; do
      NUM=$({ echo "$iter" | grep -oP '^iter\K[0-9]+'; } 2>/dev/null || true)
      NUM=$(echo "$NUM" | head -1)
      if [ -n "$NUM" ] && [ "$((10#$NUM))" -gt "$HIGHEST" ]; then
        HIGHEST=$((10#$NUM))
      fi
    done < <(tail -n +2 "$TSV" 2>/dev/null || true)
  fi

  # Also check rounds.raw.jsonl — only parse valid JSON lines, skip malformed
  JSONL="tuning/${GPU}/${DSL}/memory/${SHAPE_KEY}/rounds.raw.jsonl"
  if [ -f "$JSONL" ]; then
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
    done < "$JSONL"
  fi

  NEXT=$((HIGHEST + 1))
  printf "iter%03d_%s\n" "$NEXT" "$TAG"
fi
