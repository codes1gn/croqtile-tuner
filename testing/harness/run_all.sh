#!/usr/bin/env bash
# run_all.sh — Run all harness unit tests and report pass/fail summary.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PASS=0
FAIL=0
ERRORS=()

run_test() {
  local test_script="$1"
  local name
  name=$(basename "$test_script" .sh)
  echo ""
  echo "=== $name ==="
  if bash "$test_script"; then
    PASS=$((PASS + 1))
    echo "  PASSED: $name"
  else
    FAIL=$((FAIL + 1))
    ERRORS+=("$name")
    echo "  FAILED: $name"
  fi
}

for test in testing/harness/test_*.sh; do
  run_test "$test"
done

echo ""
echo "==============================================="
echo " Results: $PASS passed, $FAIL failed"
echo "==============================================="

if [ "${#ERRORS[@]}" -gt 0 ]; then
  echo "FAILED tests:"
  for e in "${ERRORS[@]}"; do
    echo "  - $e"
  done
  exit 1
fi

echo "All tests passed."
