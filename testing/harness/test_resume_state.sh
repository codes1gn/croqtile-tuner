#!/usr/bin/env bash
# test_resume_state.sh — Unit tests for .claude/skills/croq-tune/tools/resume_state.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source testing/harness/tap_helpers.sh

SCRIPT=".claude/skills/croq-tune/tools/resume_state.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Create a fake git root so the script finds git
mkdir -p "$TMP/.git"
cd "$TMP"
git init -q

# Copy the script into TMP workspace (use absolute path from ORIG_DIR which was set before cd)
ORIG_REPO="/home/albert/workspace/croqtile-tuner"
mkdir -p .claude/skills/croq-tune/tools
cp "$ORIG_REPO/.claude/skills/croq-tune/tools/resume_state.sh" .claude/skills/croq-tune/tools/

DSL="cuda"
GPU="sm90_testgpu"
KEY="test_matmul_4x8x8"
MODEL="opus-4"
SRC="tuning/$GPU/$DSL/srcs/$KEY/$MODEL"
LOG="tuning/$GPU/$DSL/logs/$KEY/$MODEL"
CP="tuning/$GPU/$DSL/checkpoints/$KEY/$MODEL"

mkdir -p "$SRC" "$LOG" "$CP"

# ── fixture: some iter source files ──────────────────────────────────────────
touch "$SRC/iter000_baseline.cu"
touch "$SRC/iter001_draft.cu"
touch "$SRC/iter005_pipeline.cu"
touch "$SRC/iter009_swizzle.cu"

# ── fixture: results.tsv (primary state source) ─────────────────────────────
printf "iter\tkernel\ttflops\tdecision\tbottleneck\tidea_summary\n" > "$LOG/results.tsv"
printf "iter001\titer001_draft\t10.0\tDISCARD\tmemory_bound\ttest idea 1\n" >> "$LOG/results.tsv"
printf "iter012\titer012_pipe\t25.0\tKEEP\tcompute_bound\ttest idea 12\n" >> "$LOG/results.tsv"

# ── fixture: idea-log.jsonl ──────────────────────────────────────────────────
printf '{"round":1,"iter":"iter001","idea":"test idea 1","decision":"DISCARD","tflops":10.0}\n' > "$LOG/idea-log.jsonl"
printf '{"round":12,"iter":"iter012","idea":"test idea 12","decision":"KEEP","tflops":25.0}\n' >> "$LOG/idea-log.jsonl"

plan 17

# Run script from TMP (acts as git root)
OUT=$(bash .claude/skills/croq-tune/tools/resume_state.sh --gpu "$GPU" --dsl "$DSL" --shape-key "$KEY" --model "$MODEL" 2>&1)
ok "exit code 0" $?

# Validate JSON
VALID=$(echo "$OUT" | python3 -c "import sys,json; json.load(sys.stdin); print('ok')" 2>/dev/null || echo "fail")
is "output is valid JSON" "$VALID" "ok"

# next_iter_number = max(9+1 from src, 12+1 from tsv) = 13
NEXT=$(echo "$OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['next_iter_number'])" 2>/dev/null || echo "0")
is "next_iter_number is 13 (from tsv max=12)" "$NEXT" "13"

# src_count = 4 (iter000, iter001, iter005, iter009)
SRC_CNT=$(echo "$OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['src_count'])" 2>/dev/null || echo "0")
is "src_count is 4" "$SRC_CNT" "4"

# current_best = iter012_pipe at 25.0 (KEEP, non-iter000)
BEST_T=$(echo "$OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['current_best_tflops'])" 2>/dev/null || echo "0")
is "current_best_tflops is 25.0" "$BEST_T" "25.0"

BEST_K=$(echo "$OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['current_best_kernel'])" 2>/dev/null || echo "")
is "current_best_kernel is iter012_pipe" "$BEST_K" "iter012_pipe"

# last round from idea-log = 12
LAST_R=$(echo "$OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['last_round'])" 2>/dev/null || echo "0")
is "last_round is 12" "$LAST_R" "12"

# last_decision = KEEP
LAST_D=$(echo "$OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['last_decision'])" 2>/dev/null || echo "")
is "last_decision is KEEP" "$LAST_D" "KEEP"

# memory_files_ok = true (results.tsv + idea-log.jsonl present)
MEM_OK=$(echo "$OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['memory_files_ok'])" 2>/dev/null || echo "False")
is "memory_files_ok is True" "$MEM_OK" "True"

# no warnings
WARN_COUNT=$(echo "$OUT" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['warnings']))" 2>/dev/null || echo "99")
is "no warnings" "$WARN_COUNT" "0"

# open_checkpoint is null
OPEN_CP=$(echo "$OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['open_checkpoint'])" 2>/dev/null || echo "err")
is "open_checkpoint is None" "$OPEN_CP" "None"

# ── test: open checkpoint detected ───────────────────────────────────────────
cat > "$CP/current_idea.json" <<'EOF'
{"schema":"croq-checkpoint-v1","iter":"iter013_myplan","status":"PLANNED","idea":"test","bottleneck":"memory_bound"}
EOF

OUT2=$(bash .claude/skills/croq-tune/tools/resume_state.sh --gpu "$GPU" --dsl "$DSL" --shape-key "$KEY" --model "$MODEL" 2>&1)
OPEN2=$(echo "$OUT2" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['open_checkpoint']['iter'] if d['open_checkpoint'] else 'None')" 2>/dev/null || echo "err")
is "open_checkpoint detected" "$OPEN2" "iter013_myplan"

WARN2=$(echo "$OUT2" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['warnings']))" 2>/dev/null || echo "0")
like "warning about open checkpoint" "$(echo "$OUT2" | python3 -c "import sys,json; print(json.load(sys.stdin)['warnings'])" 2>/dev/null || echo "")" "Open checkpoint"

# ── test: missing log files flagged ──────────────────────────────────────────
rm -f "$LOG/idea-log.jsonl"
OUT3=$(bash .claude/skills/croq-tune/tools/resume_state.sh --gpu "$GPU" --dsl "$DSL" --shape-key "$KEY" --model "$MODEL" 2>&1)
MEM_OK3=$(echo "$OUT3" | python3 -c "import sys,json; print(json.load(sys.stdin)['memory_files_ok'])" 2>/dev/null || echo "True")
is "memory_files_ok False when idea-log.jsonl missing" "$MEM_OK3" "False"

# ── test: missing args exits 1 ───────────────────────────────────────────────
bash .claude/skills/croq-tune/tools/resume_state.sh --gpu "$GPU" --dsl "$DSL" >/dev/null 2>&1 && RC=0 || RC=$?
is "missing --shape-key exits 1" "$RC" "1"

# ── test: nonexistent dsl/shape exits 2 ──────────────────────────────────────
bash .claude/skills/croq-tune/tools/resume_state.sh --gpu "$GPU" --dsl "nonexistent" --shape-key "nope" --model "opus-4" >/dev/null 2>&1 && RC=0 || RC=$?
is "nonexistent dsl/shape exits 2" "$RC" "2"

# ── test: empty workspace returns next_iter_number=1 ─────────────────────────
mkdir -p "tuning/$GPU/cuda/srcs/empty_key/opus-4" "tuning/$GPU/cuda/logs/empty_key/opus-4"
printf "iter\tkernel\ttflops\tdecision\n" > "tuning/$GPU/cuda/logs/empty_key/opus-4/results.tsv"
printf '{"round":0}\n' > "tuning/$GPU/cuda/logs/empty_key/opus-4/idea-log.jsonl"
OUT4=$(bash .claude/skills/croq-tune/tools/resume_state.sh --gpu "$GPU" --dsl cuda --shape-key empty_key --model opus-4 2>&1)
NEXT4=$(echo "$OUT4" | python3 -c "import sys,json; print(json.load(sys.stdin)['next_iter_number'])" 2>/dev/null || echo "0")
is "empty workspace: next_iter_number is 1" "$NEXT4" "1"

cd "$ORIG_REPO"
done_testing
