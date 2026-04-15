#!/usr/bin/env bash
# test_store_round.sh — Unit tests for .claude/skills/croq-store/store_round.sh
#
# Uses a temporary fixture workspace; no GPU required.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source testing/harness/tap_helpers.sh

SCRIPT=".claude/skills/croq-store/store_round.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

DSL="cuda"
KEY="matmul_test_4x8x8"
BASE="$TMP/tuning/aitune/${DSL}"
mkdir -p "$BASE/memory/$KEY" "$BASE/logs/$KEY"

plan 18

# ── helper: call store_round from TMP dir ─────────────────────────────────────
store() {
  (cd "$TMP" && bash "$OLDPWD/$SCRIPT" "$@" 2>&1)
}

# Test 1: successful store for iter005_swizzle
result=$(store \
  --dsl "$DSL" --shape-key "$KEY" \
  --iter iter005 --kernel iter005_swizzle \
  --tflops 12.34 --decision KEEP \
  --bottleneck memory_bound \
  --idea "swizzle tile ordering for L2 reuse" \
  --round 5 2>&1)
ok "store exit code 0" $?
like "store success message" "$result" "\[store_round\] STORE complete"

# Test 2: rounds.raw.jsonl created
[ -f "$BASE/memory/$KEY/rounds.raw.jsonl" ]
ok "rounds.raw.jsonl exists" $?

# Test 3: rounds.raw.jsonl contains the iter
like "rounds.raw has iter005" "$(cat "$BASE/memory/$KEY/rounds.raw.jsonl")" "iter005"

# Test 4: rounds.md created
[ -f "$BASE/memory/$KEY/rounds.md" ]
ok "rounds.md exists" $?

# Test 5: rounds.md contains iter005
like "rounds.md has iter005" "$(cat "$BASE/memory/$KEY/rounds.md")" "iter005"

# Test 6: idea-log.jsonl created
[ -f "$BASE/logs/$KEY/idea-log.jsonl" ]
ok "idea-log.jsonl exists" $?

# Test 7: idea-log has the round info
like "idea-log has round 5" "$(cat "$BASE/logs/$KEY/idea-log.jsonl")" '"round": 5'

# Test 8: results.tsv created with header
[ -f "$BASE/logs/$KEY/results.tsv" ]
ok "results.tsv exists" $?

# Test 9: results.tsv has header line
like "results.tsv has header" "$(head -1 "$BASE/logs/$KEY/results.tsv")" "iter"

# Test 10: results.tsv has iter005 row
like "results.tsv has iter005 row" "$(cat "$BASE/logs/$KEY/results.tsv")" "iter005"

# Test 11: second store call appends (does not overwrite)
store \
  --dsl "$DSL" --shape-key "$KEY" \
  --iter iter006 --kernel iter006_pipeline \
  --tflops 13.00 --decision DISCARD \
  --bottleneck compute_bound \
  --idea "pipeline stages" \
  --round 6 > /dev/null 2>&1
LINES=$(wc -l < "$BASE/memory/$KEY/rounds.raw.jsonl")
[ "$LINES" -eq 2 ]
ok "rounds.raw has 2 lines after 2 stores" $?

# Test 12: results.tsv has 3 lines (header + 2 data rows)
TSVCNT=$(wc -l < "$BASE/logs/$KEY/results.tsv")
[ "$TSVCNT" -eq 3 ]
ok "results.tsv has 3 lines (header+2)" $?

# ── rejection tests ───────────────────────────────────────────────────────────

# Test 13: missing --kernel
result13=$(store --dsl "$DSL" --shape-key "$KEY" --iter iter007 \
  --tflops 1.0 --decision KEEP --bottleneck mem --idea "x" --round 7 2>&1) || true
like "missing kernel rejected" "$result13" "ERROR"

# Test 14: bare-number kernel (no tag)
result14=$(store --dsl "$DSL" --shape-key "$KEY" \
  --iter iter008 --kernel iter008 \
  --tflops 1.0 --decision KEEP --bottleneck mem --idea "x" --round 8 2>&1) || true
like "bare-number kernel rejected" "$result14" "ERROR.*iter008"

# Test 15: invalid decision value
result15=$(store --dsl "$DSL" --shape-key "$KEY" \
  --iter iter009 --kernel iter009_test \
  --tflops 1.0 --decision BADVALUE --bottleneck mem --idea "x" --round 9 2>&1) || true
like "invalid decision rejected" "$result15" "ERROR"

# Test 16: iter format wrong (only 2 digits)
result16=$(store --dsl "$DSL" --shape-key "$KEY" \
  --iter iter09 --kernel iter09_test \
  --tflops 1.0 --decision KEEP --bottleneck mem --idea "x" --round 9 2>&1) || true
like "2-digit iter rejected" "$result16" "ERROR"

# Test 17: rounds.raw is valid JSON (each line parseable)
python3 -c "
import json
with open('$BASE/memory/$KEY/rounds.raw.jsonl') as f:
    for line in f:
        json.loads(line.strip())
print('OK')
" > /dev/null 2>&1
ok "rounds.raw.jsonl is valid JSON per line" $?

# Test 18: idea-log is valid JSON per line
python3 -c "
import json
with open('$BASE/logs/$KEY/idea-log.jsonl') as f:
    for line in f:
        json.loads(line.strip())
print('OK')
" > /dev/null 2>&1
ok "idea-log.jsonl is valid JSON per line" $?

done_testing
