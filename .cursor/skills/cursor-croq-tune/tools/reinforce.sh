#!/usr/bin/env bash
# reinforce.sh — Mandatory continuation gate for croq-tune loop.
#
# Called after every STORE step to reinforce the tuning loop contract.
# Reads current progress from results.tsv and emits structured instructions
# that force the agent to continue iterating.
#
# USAGE:
#   bash .cursor/skills/cursor-croq-tune/tools/reinforce.sh \
#       --dsl <dsl> --shape-key <shape_key> --model <model>
#
# EXIT CODES:
#   0 — reinforcement complete, agent MUST continue
#   1 — missing arguments
#
# This script is MANDATORY after every store_round.sh call.
# The agent MUST NOT skip this step.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/activity_trace.sh" 2>/dev/null || true

# ── argument parsing ─────────────────────────────────────────────────────
GPU=""
DSL=""
SHAPE_KEY=""
MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)       GPU="$2";       shift 2 ;;
    --dsl)       DSL="$2";       shift 2 ;;
    --shape-key) SHAPE_KEY="$2"; shift 2 ;;
    --model)     MODEL="$2";     shift 2 ;;
    *) shift ;;
  esac
done

# Auto-detect GPU if not provided
if [[ -z "$GPU" ]]; then
  DETECT_SCRIPT="$(dirname "$0")/detect_gpu.sh"
  GPU=$(bash "$DETECT_SCRIPT" 2>/dev/null || echo "sm00_unknown")
fi

if [[ -z "$DSL" || -z "$SHAPE_KEY" || -z "$MODEL" ]]; then
  echo "ERROR: --dsl, --shape-key, and --model are required" >&2
  exit 1
fi

# ── read current state ───────────────────────────────────────────────────
BASE="tuning/${GPU}/${DSL}"
LOG_DIR="${BASE}/logs/${SHAPE_KEY}/${MODEL}"
TSV="${LOG_DIR}/results.tsv"
CHECKPOINT="${BASE}/checkpoints/${SHAPE_KEY}/${MODEL}/current_idea.json"

TOTAL_ITERS=0
BEST_TFLOPS="0"
BEST_KERNEL="none"
BASELINE_TFLOPS="0"
LAST_DECISION=""
CONSECUTIVE_FAILS=0

if [[ -f "$TSV" ]]; then
  while IFS=$'\t' read -r iter kernel tflops decision rest; do
    [[ "$iter" =~ ^iter ]] || continue
    [[ "$iter" == "iter" ]] && continue  # header
    
    if [[ "$iter" == "iter000" ]]; then
      BASELINE_TFLOPS="$tflops"
      continue
    fi
    
    TOTAL_ITERS=$((TOTAL_ITERS + 1))
    LAST_DECISION="$decision"
    
    if [[ "$decision" == "KEEP" ]] && (( $(echo "$tflops > $BEST_TFLOPS" | bc -l 2>/dev/null || echo 0) )); then
      BEST_TFLOPS="$tflops"
      BEST_KERNEL="$kernel"
    fi
    
    if [[ "$decision" == "COMPILE_FAIL" || "$decision" == "SEGFAULT" || "$decision" == "HANG" ]]; then
      CONSECUTIVE_FAILS=$((CONSECUTIVE_FAILS + 1))
    else
      CONSECUTIVE_FAILS=0
    fi
  done < "$TSV"
fi

# Calculate percentage of baseline
PCT="?"
if (( $(echo "$BASELINE_TFLOPS > 0" | bc -l 2>/dev/null || echo 0) )) && \
   (( $(echo "$BEST_TFLOPS > 0" | bc -l 2>/dev/null || echo 0) )); then
  PCT=$(echo "scale=1; $BEST_TFLOPS * 100 / $BASELINE_TFLOPS" | bc -l 2>/dev/null || echo "?")
fi

# ── emit reinforcement ──────────────────────────────────────────────────
echo "[reinforce] iters=${TOTAL_ITERS} best=${BEST_TFLOPS}T (${PCT}% of ${BASELINE_TFLOPS}T baseline) kernel=${BEST_KERNEL} consecutive_fails=${CONSECUTIVE_FAILS}"
echo ""
echo "MANDATORY: Read the full tuning protocol NOW before doing anything else:"
echo "  Read file: .cursor/skills/cursor-croq-tune/SKILL.md"
echo "Then re-read: .cursor/skills/cursor-croq-dsl-${DSL}/SKILL.md"
echo "Then continue the loop — PROFILE → IDEA → IMPLEMENT → VERIFY → MEASURE → DECIDE → STORE → reinforce."
