#!/usr/bin/env bash
# test_store_round.sh — Unit tests for .claude/skills/croq-tune/tools/store_round.sh
#
# Uses a temporary fixture workspace; no GPU required.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source testing/harness/tap_helpers.sh

SCRIPT=".claude/skills/croq-tune/tools/store_round.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

DSL="cuda"
GPU="sm90_testgpu"
KEY="matmul_test_4x8x8"
MODEL="opus-4"
BASE="$TMP/tuning/${GPU}/${DSL}"
mkdir -p "$BASE/logs/$KEY/$MODEL"

plan 14

# ── helper: call store_round from TMP dir ─────────────────────────────────────
store() {
  (cd "$TMP" && bash "$OLDPWD/$SCRIPT" "$@" 2>&1)
}

# Test 1: successful store for iter005_swizzle
result=$(store \
  --gpu "$GPU" --dsl "$DSL" --shape-key "$KEY" --model "$MODEL" \
  --iter iter005 --kernel iter005_swizzle \
  --tflops 12.34 --decision KEEP \
  --bottleneck memory_bound \
  --idea "swizzle tile ordering for L2 reuse" \
  --round 5 2>&1)
ok "store exit code 0" $?
like "store success message" "$result" "\[store_round\] STORE complete"

# Test 2: idea-log.jsonl created
[ -f "$BASE/logs/$KEY/$MODEL/idea-log.jsonl" ]
ok "idea-log.jsonl exists" $?

# Test 3: idea-log has the round info
like "idea-log has round 5" "$(cat "$BASE/logs/$KEY/$MODEL/idea-log.jsonl")" '"round": 5'

# Test 4: results.tsv created with header
[ -f "$BASE/logs/$KEY/$MODEL/results.tsv" ]
ok "results.tsv exists" $?

# Test 5: results.tsv has header line
like "results.tsv has header" "$(head -1 "$BASE/logs/$KEY/$MODEL/results.tsv")" "iter"

# Test 6: results.tsv has iter005 row
like "results.tsv has iter005 row" "$(cat "$BASE/logs/$KEY/$MODEL/results.tsv")" "iter005"

# Test 7: second store call appends (does not overwrite)
store \
  --gpu "$GPU" --dsl "$DSL" --shape-key "$KEY" --model "$MODEL" \
  --iter iter006 --kernel iter006_pipeline \
  --tflops 13.00 --decision DISCARD \
  --bottleneck compute_bound \
  --idea "pipeline stages" \
  --round 6 > /dev/null 2>&1
TSVCNT=$(wc -l < "$BASE/logs/$KEY/$MODEL/results.tsv")
[ "$TSVCNT" -eq 3 ]
ok "results.tsv has 3 lines (header+2)" $?

# ── rejection tests ───────────────────────────────────────────────────────────

# Test 8: missing --kernel
result8=$(store --gpu "$GPU" --dsl "$DSL" --shape-key "$KEY" --model "$MODEL" --iter iter007 \
  --tflops 1.0 --decision KEEP --bottleneck mem --idea "x" --round 7 2>&1) || true
like "missing kernel rejected" "$result8" "ERROR"

# Test 9: bare-number kernel (no tag)
result9=$(store --gpu "$GPU" --dsl "$DSL" --shape-key "$KEY" --model "$MODEL" \
  --iter iter008 --kernel iter008 \
  --tflops 1.0 --decision KEEP --bottleneck mem --idea "x" --round 8 2>&1) || true
like "bare-number kernel rejected" "$result9" "ERROR.*iter008"

# Test 10: invalid decision value
result10=$(store --gpu "$GPU" --dsl "$DSL" --shape-key "$KEY" --model "$MODEL" \
  --iter iter009 --kernel iter009_test \
  --tflops 1.0 --decision BADVALUE --bottleneck mem --idea "x" --round 9 2>&1) || true
like "invalid decision rejected" "$result10" "ERROR"

# Test 11: iter format wrong (letters instead of digits)
result11=$(store --gpu "$GPU" --dsl "$DSL" --shape-key "$KEY" --model "$MODEL" \
  --iter iterabc --kernel iterabc_test \
  --tflops 1.0 --decision KEEP --bottleneck mem --idea "x" --round 9 2>&1) || true
like "non-numeric iter rejected" "$result11" "ERROR"

# Test 12: idea-log is valid JSON per line
python3 -c "
import json
with open('$BASE/logs/$KEY/$MODEL/idea-log.jsonl') as f:
    for line in f:
        json.loads(line.strip())
print('OK')
" > /dev/null 2>&1
ok "idea-log.jsonl is valid JSON per line" $?

# Test 13: idea-log has 2 lines after 2 stores
IDEA_LINES=$(wc -l < "$BASE/logs/$KEY/$MODEL/idea-log.jsonl")
[ "$IDEA_LINES" -eq 2 ]
ok "idea-log has 2 lines after 2 stores" $?

# Test 14: no rounds.raw.jsonl or rounds.md created (removed from store)
[ ! -f "$BASE/memory/$KEY/$MODEL/rounds.raw.jsonl" ] && [ ! -f "$BASE/memory/$KEY/$MODEL/rounds.md" ]
ok "no legacy rounds.raw.jsonl or rounds.md created" $?

done_testing
