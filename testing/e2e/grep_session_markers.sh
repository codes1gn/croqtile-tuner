#!/usr/bin/env bash
# grep_session_markers.sh — Check a croq-tune session transcript for
# key behavioural markers that indicate the agent followed the protocol.
#
# USAGE:
#   bash testing/e2e/grep_session_markers.sh <session.jsonl>
#
# Where <session.jsonl> is an agent session transcript file from:
#   ~/.cursor/projects/<project>/agent-transcripts/<uuid>/<uuid>.jsonl
#
# EXIT CODES:
#   0  — all required markers found
#   1  — one or more required markers missing
#   2  — session file not found or not readable

set -euo pipefail

SESSION="${1:-}"

if [ -z "$SESSION" ]; then
  echo "USAGE: $0 <session.jsonl>" >&2
  exit 2
fi

if [ ! -f "$SESSION" ]; then
  echo "ERROR: session file not found: $SESSION" >&2
  exit 2
fi

echo "=== Behavioural Marker Check ==="
echo "Session: $SESSION"
echo "Size:    $(wc -l < "$SESSION") lines"
echo ""

# Extract all text content from the JSONL for grep
TEXT=$(python3 -c "
import sys, json
for line in sys.stdin:
    try:
        obj = json.loads(line.strip())
        print(json.dumps(obj))
    except Exception:
        pass
" < "$SESSION")

PASS=0
FAIL=0

check() {
  local desc="$1"
  local pattern="$2"
  local required="${3:-yes}"
  if echo "$TEXT" | grep -q "$pattern"; then
    echo "  PRESENT  $desc"
    PASS=$((PASS + 1))
  else
    if [ "$required" = "yes" ]; then
      echo "  MISSING  $desc   [REQUIRED]"
      FAIL=$((FAIL + 1))
    else
      echo "  MISSING  $desc   [optional]"
    fi
  fi
}

echo "--- Storage Harness ---"
check "store_round.sh was called"          "store_round\.sh"
check "STORE complete message"             "\[store_round\] STORE complete"
check "next_iter.sh was called"           "next_iter\.sh"

echo ""
echo "--- Protocol Steps ---"
check "ncu profiling was run"              "ncu "
check "VERIFY: PASS appeared"             "VERIFY: PASS"
check "TFLOPS measurement appeared"       "TFLOPS:"
check "KEEP or DISCARD decision made"     "KEEP\|DISCARD"

echo ""
echo "--- Anti-Patterns (should be MISSING) ---"
BAD_BARE=$(echo "$TEXT" | grep -c 'iter[0-9][0-9][0-9]\.cu\b' || true)
if [ "$BAD_BARE" -eq 0 ]; then
  echo "  GOOD     No bare-number iter files (iter<NNN>.cu without tag)"
  PASS=$((PASS + 1))
else
  echo "  BAD      Found $BAD_BARE bare-number iter files (iter<NNN>.cu without tag)"
  FAIL=$((FAIL + 1))
fi

echo ""
echo "--- Summary ---"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"

if [ "$FAIL" -gt 0 ]; then
  echo ""
  echo "RESULT: FAILED — $FAIL required markers missing"
  exit 1
fi

echo ""
echo "RESULT: PASSED — all required markers present"
