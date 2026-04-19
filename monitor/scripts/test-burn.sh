#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAFE_SCRIPT="$SCRIPT_DIR/test-safe.sh"
RUNS="${1:-5}"

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [ "$RUNS" -lt 1 ]; then
  echo "Usage: $0 [runs>=1]"
  exit 1
fi

echo "=== CroqTuner Burn Test ==="
echo "Runs: $RUNS"
echo ""

pass=0
for i in $(seq 1 "$RUNS"); do
  echo "--- Run $i/$RUNS ---"
  if "$SAFE_SCRIPT" >/tmp/croqtuner-burn-last.log 2>&1; then
    pass=$((pass + 1))
    echo "PASS"
  else
    echo "FAIL (see /tmp/croqtuner-burn-last.log)"
    cat /tmp/croqtuner-burn-last.log
    exit 1
  fi
done

echo ""
echo "Burn test complete: $pass/$RUNS passed"
