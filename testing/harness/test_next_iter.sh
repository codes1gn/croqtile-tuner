#!/usr/bin/env bash
# test_next_iter.sh — Unit tests for .claude/skills/croq-store/next_iter.sh
#
# Uses a temporary fixture workspace; no GPU required.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source testing/harness/tap_helpers.sh

SCRIPT=".claude/skills/croq-store/next_iter.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# ── fixture setup ──────────────────────────────────────────────────────────────
# Create a fake tuning workspace with some iter files
DSL="cuda"
KEY="matmul_test_4x8x8"
SRC_DIR="$TMP/tuning/aitune/${DSL}/srcs/${KEY}"
LOG_DIR="$TMP/tuning/aitune/${DSL}/logs/${KEY}"
MEM_DIR="$TMP/tuning/aitune/${DSL}/memory/${KEY}"
mkdir -p "$SRC_DIR" "$LOG_DIR" "$MEM_DIR"

# Simulate: iter000, iter001, iter005 exist in srcs
touch "$SRC_DIR/iter000_baseline.cu"
touch "$SRC_DIR/iter001_draft.cu"
touch "$SRC_DIR/iter005_pipeline.cu"

# results.tsv has iter007 (higher than srcs)
echo -e "iter\tkernel\ttflops\tdecision\tbottleneck\tidea_summary" > "$LOG_DIR/results.tsv"
echo -e "iter007\titer007_swizzle\t10.0\tKEEP\tmemory_bound\ttest" >> "$LOG_DIR/results.tsv"

# rounds.raw.jsonl has iter009 (highest)
echo '{"iter": "iter009", "kernel": "iter009_pipeline", "tflops": 11.0, "decision": "KEEP"}' > "$MEM_DIR/rounds.raw.jsonl"
echo 'not valid json line' >> "$MEM_DIR/rounds.raw.jsonl"

plan 10

# Test 1: next public iter is 010 (from rounds.raw having iter009 as highest)
cd "$TMP"
result=$(bash "$OLDPWD/$SCRIPT" --dsl "$DSL" --shape-key "$KEY" --tag mytest)
cd "$OLDPWD"
is "next iter after 009 is iter010_mytest" "$result" "iter010_mytest"

# Test 2: empty workspace → next iter is iter001
TMP2=$(mktemp -d)
trap 'rm -rf "$TMP2"' EXIT
SRC2="$TMP2/tuning/aitune/cuda/srcs/matmul_empty_1x1x1"
mkdir -p "$SRC2"
cd "$TMP2"
result2=$(bash "$OLDPWD/$SCRIPT" --dsl cuda --shape-key matmul_empty_1x1x1 --tag first)
cd "$OLDPWD"
is "empty workspace → iter001_first" "$result2" "iter001_first"

# Test 3: attempt mode in empty workspace → attempt0001
cd "$TMP2"
attempt1=$(bash "$OLDPWD/$SCRIPT" --dsl cuda --shape-key matmul_empty_1x1x1 --tag trythis --attempt)
cd "$OLDPWD"
is "empty workspace attempt → attempt0001_trythis" "$attempt1" "attempt0001_trythis"

# Test 4: invalid tag — uppercase letter
cd "$TMP"
result4=$(bash "$OLDPWD/$SCRIPT" --dsl "$DSL" --shape-key "$KEY" --tag BadTag 2>&1) || true
cd "$OLDPWD"
like "invalid tag rejected" "$result4" "ERROR"

# Test 5: invalid tag — too short (1 char)
cd "$TMP"
result5=$(bash "$OLDPWD/$SCRIPT" --dsl "$DSL" --shape-key "$KEY" --tag a 2>&1) || true
cd "$OLDPWD"
like "too-short tag rejected" "$result5" "ERROR"

# Test 6: invalid tag — contains spaces
cd "$TMP"
result6=$(bash "$OLDPWD/$SCRIPT" --dsl "$DSL" --shape-key "$KEY" --tag "my tag" 2>&1) || true
cd "$OLDPWD"
like "tag with space rejected" "$result6" "ERROR"

# Test 7: missing --tag
cd "$TMP"
result7=$(bash "$OLDPWD/$SCRIPT" --dsl "$DSL" --shape-key "$KEY" 2>&1) || true
cd "$OLDPWD"
like "missing tag rejected" "$result7" "ERROR"

# Test 8: missing --dsl
cd "$TMP"
result8=$(bash "$OLDPWD/$SCRIPT" --shape-key "$KEY" --tag test 2>&1) || true
cd "$OLDPWD"
like "missing dsl rejected" "$result8" "ERROR"

# Test 9: output has no trailing whitespace or newline issues
cd "$TMP"
result9=$(bash "$OLDPWD/$SCRIPT" --dsl "$DSL" --shape-key "$KEY" --tag cleanout)
cd "$OLDPWD"
unlike "no ERROR in clean output" "$result9" "ERROR"
like "output matches iter pattern" "$result9" "^iter[0-9][0-9][0-9]_cleanout$"

# Test 10: malformed JSON in rounds.raw does not crash
# (already present in fixture — iter009 is the last valid line, non-JSON is after)
cd "$TMP"
result10=$(bash "$OLDPWD/$SCRIPT" --dsl "$DSL" --shape-key "$KEY" --tag afterjson)
cd "$OLDPWD"
unlike "malformed JSON line does not crash script" "$result10" "ERROR"

done_testing
