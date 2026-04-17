#!/usr/bin/env bash
# activity_trace.sh — Shared harness activity logger
#
# Source this from any croq-* tool script to log structured activity entries:
#   source "$(dirname "$0")/activity_trace.sh"
#   trace_init --gpu "$GPU_KEY" --dsl "$DSL" --shape-key "$SHAPE" --model "$MODEL"
#   trace_event "store_round" "Stored iter005_tiled with 12.3 TFLOPS (KEEP)"
#   trace_event "ncu_profile" "Profiling iter005_tiled (ncu full set)" "info"
#   trace_event "co2cu" "Choreo compile failed: missing DMA form" "error"
#
# Writes JSONL to: tuning/<gpu>/<dsl>/memory/<shape_key>/<model>/activity.jsonl
# Each line: {"ts":"...","tool":"...","msg":"...","level":"info|warn|error"}

set -euo pipefail

_TRACE_DIR=""
_TRACE_FILE=""

trace_init() {
  local gpu="" dsl="" shape_key="" model=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu)        gpu="$2"; shift 2 ;;
      --dsl)        dsl="$2"; shift 2 ;;
      --shape-key)  shape_key="$2"; shift 2 ;;
      --model)      model="$2"; shift 2 ;;
      *)            shift ;;
    esac
  done

  if [[ -z "$gpu" || -z "$dsl" || -z "$shape_key" ]]; then
    return 0
  fi

  local project_root
  project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
  local mem_base="$project_root/tuning/$gpu/$dsl/memory/$shape_key"

  if [[ -n "$model" ]]; then
    _TRACE_DIR="$mem_base/$model"
  else
    _TRACE_DIR="$mem_base"
  fi
  mkdir -p "$_TRACE_DIR"
  _TRACE_FILE="$_TRACE_DIR/activity.jsonl"
}

trace_event() {
  local tool="${1:-unknown}" msg="${2:-}" level="${3:-info}"

  if [[ -z "$_TRACE_FILE" ]]; then
    return 0
  fi

  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ)"

  # Escape JSON strings (minimal: backslash, double-quote, newlines)
  msg="${msg//\\/\\\\}"
  msg="${msg//\"/\\\"}"
  msg="${msg//$'\n'/\\n}"
  tool="${tool//\"/\\\"}"

  printf '{"ts":"%s","tool":"%s","msg":"%s","level":"%s"}\n' \
    "$ts" "$tool" "$msg" "$level" >> "$_TRACE_FILE"
}
