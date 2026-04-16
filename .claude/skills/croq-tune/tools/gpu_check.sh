#!/usr/bin/env bash
# gpu_check.sh — Check GPU idle state and handle contention
#
# Usage:
#   bash gpu_check.sh                     # check GPU utilization, exit 0 if idle
#   bash gpu_check.sh --wait              # wait for GPU to become idle (poll)
#   bash gpu_check.sh --wait --timeout 60 # wait up to 60s
#   bash gpu_check.sh --kill-others       # kill other GPU processes (requires sudo)
#   bash gpu_check.sh --reset             # reset GPU (requires sudo, dangerous)
#
# Exit codes:
#   0 — GPU is idle (utilization < threshold)
#   1 — GPU is busy (contention detected)
#   2 — nvidia-smi not available
#   3 — timeout waiting for idle GPU
#
# Output (JSON to stdout):
#   { "idle": true/false, "gpu_util": N, "mem_util": N, "processes": [...], "action": "..." }

set -euo pipefail

UTIL_THRESHOLD=15
WAIT=false
TIMEOUT=120
KILL_OTHERS=false
RESET=false
POLL_INTERVAL=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait) WAIT=true; shift ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --threshold) UTIL_THRESHOLD="$2"; shift 2 ;;
    --kill-others) KILL_OTHERS=true; shift ;;
    --reset) RESET=true; shift ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
  esac
done

if ! command -v nvidia-smi &>/dev/null; then
  echo '{"idle": false, "error": "nvidia-smi not found"}' 
  exit 2
fi

check_gpu() {
  local gpu_util mem_util
  gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ') || gpu_util="0"
  mem_util=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ') || mem_util="0"

  local procs
  procs=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null | head -20) || procs=""

  local proc_json="[]"
  if [[ -n "$procs" ]]; then
    proc_json=$(echo "$procs" | python3 -c "
import sys, json
entries = []
for line in sys.stdin:
    parts = [p.strip() for p in line.strip().split(',')]
    if len(parts) >= 3:
        entries.append({'pid': int(parts[0]), 'name': parts[1], 'mem': parts[2]})
print(json.dumps(entries))
" 2>/dev/null || echo "[]")
  fi

  local idle=false
  if [[ "$gpu_util" -lt "$UTIL_THRESHOLD" ]]; then
    idle=true
  fi

  echo "{\"idle\": $idle, \"gpu_util\": $gpu_util, \"mem_util\": $mem_util, \"processes\": $proc_json}"
  
  if [[ "$idle" == "true" ]]; then
    return 0
  else
    return 1
  fi
}

kill_other_gpu_processes() {
  local my_pid=$$
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ') || return

  local killed=0
  for pid in $pids; do
    if [[ "$pid" != "$my_pid" ]]; then
      echo "[gpu_check] Killing GPU process PID=$pid" >&2
      if sudo kill -9 "$pid" 2>/dev/null; then
        killed=$((killed + 1))
      fi
    fi
  done
  echo "[gpu_check] Killed $killed GPU process(es)" >&2
  sleep 2
}

reset_gpu() {
  echo "[gpu_check] Attempting GPU reset via nvidia-smi..." >&2
  if sudo nvidia-smi --gpu-reset 2>&1; then
    echo "[gpu_check] GPU reset complete" >&2
    sleep 3
  else
    echo "[gpu_check] GPU reset failed (may need reboot)" >&2
    return 1
  fi
}

if [[ "$KILL_OTHERS" == "true" ]]; then
  kill_other_gpu_processes
fi

if [[ "$RESET" == "true" ]]; then
  reset_gpu
fi

if [[ "$WAIT" == "true" ]]; then
  start_time=$(date +%s)
  attempt=0
  while true; do
    result=$(check_gpu) && { echo "$result"; exit 0; }

    elapsed=$(( $(date +%s) - start_time ))
    if [[ "$elapsed" -ge "$TIMEOUT" ]]; then
      echo "[gpu_check] Timeout after ${elapsed}s waiting for idle GPU" >&2
      echo "$result" | python3 -c "
import sys, json
d = json.load(sys.stdin)
d['action'] = 'timeout'
d['waited_seconds'] = $elapsed
print(json.dumps(d))
"
      exit 3
    fi

    attempt=$((attempt + 1))
    echo "[gpu_check] GPU busy (util=${result##*gpu_util\": }), waiting ${POLL_INTERVAL}s... (attempt $attempt, ${elapsed}s elapsed)" >&2
    sleep "$POLL_INTERVAL"
  done
else
  result=$(check_gpu)
  rc=$?
  echo "$result"
  exit $rc
fi
