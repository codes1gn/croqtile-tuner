#!/usr/bin/env bash
# detect_gpu.sh — Emit GPU key in sm<CC>_<MODEL> format using nvidia-smi.
#
# USAGE (from repo root or any directory):
#   bash .claude/skills/croq-tune/tools/detect_gpu.sh
#
# OUTPUT (stdout, one line):
#   sm90_H100            (Hopper H100)
#   sm80_A100_80GB_PCIe  (Ampere A100 80GB PCIe)
#   sm89_L40S            (Ada L40S)
#
# The key is assembled as:
#   sm<CC>_<MODEL>
# where:
#   <CC>    = CUDA compute capability with the dot removed (e.g. 9.0 → 90)
#   <MODEL> = GPU name with spaces replaced by underscores
#
# EXIT CODES:
#   0 — success, key printed to stdout
#   1 — nvidia-smi not found or failed (prints a stub key to stdout and warns to stderr)

set -euo pipefail

if ! command -v nvidia-smi &>/dev/null; then
  echo "WARN: nvidia-smi not found; using stub key 'sm00_unknown'" >&2
  echo "sm00_unknown"
  exit 0
fi

MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/ /_/g')
CC_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
CC=$(echo "$CC_RAW" | sed 's/\.//')

if [[ -z "$MODEL" || -z "$CC" ]]; then
  echo "WARN: nvidia-smi returned empty output; using stub key 'sm00_unknown'" >&2
  echo "sm00_unknown"
  exit 0
fi

echo "sm${CC}_${MODEL}"
