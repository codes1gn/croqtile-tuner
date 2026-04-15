#!/usr/bin/env bash
# test_checkpoint_write.sh — Unit tests for .claude/skills/croq-store/checkpoint_write.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source testing/harness/tap_helpers.sh

SCRIPT=".claude/skills/croq-store/checkpoint_write.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Override tuning dir to temp
export HOME="$TMP"  # not needed; script uses git root relative paths

# We need to run in a temp workspace that has the expected directory structure
ORIG_DIR="$(pwd)"
cp -r tuning "$TMP/tuning" 2>/dev/null || mkdir -p "$TMP/tuning"
cd "$TMP"

# Make the script runnable from TMP by creating a symlink to the original script
mkdir -p .claude/skills/croq-store
cp "$ORIG_DIR/.claude/skills/croq-store/checkpoint_write.sh" .claude/skills/croq-store/

# Also create a fake src/bin dir for verify tests
mkdir -p "tuning/aitune/cuda/srcs/test_shape_4x8x8"
touch "tuning/aitune/cuda/srcs/test_shape_4x8x8/iter007_mytag.cu"
mkdir -p "tuning/aitune/cuda/bin/test_shape_4x8x8/iter007_mytag"

plan 18

DSL="cuda"
KEY="test_shape_4x8x8"

# ── write mode ────────────────────────────────────────────────────────────────
OUT=$(bash .claude/skills/croq-store/checkpoint_write.sh write \
    --dsl "$DSL" --shape-key "$KEY" \
    --iter "iter007_mytag" \
    --bottleneck "memory_bound" \
    --idea "Increase swizzle stride to reduce L2 thrashing" \
    --expected-gain "+3 TFLOPS" \
    --levers "SWIZZLE_STRIDE,BM" 2>&1)

ok "write exits 0" $?
like "write prints PLANNED" "$OUT" "PLANNED"
like "write prints iter name" "$OUT" "iter007_mytag"

# Checkpoint file created
CP="tuning/aitune/cuda/checkpoints/test_shape_4x8x8/current_idea.json"
ok "checkpoint file exists" "$([ -f "$CP" ] && echo 0 || echo 1)"

# Checkpoint is valid JSON
VALID=$(python3 -c "import json; json.load(open('$CP')); print('ok')" 2>/dev/null || echo "fail")
is "checkpoint is valid JSON" "$VALID" "ok"

# Checkpoint has required fields
ITER_FIELD=$(python3 -c "import json; d=json.load(open('$CP')); print(d.get('iter',''))" 2>/dev/null || echo "")
is "checkpoint.iter correct" "$ITER_FIELD" "iter007_mytag"

BN_FIELD=$(python3 -c "import json; d=json.load(open('$CP')); print(d.get('bottleneck',''))" 2>/dev/null || echo "")
is "checkpoint.bottleneck correct" "$BN_FIELD" "memory_bound"

STATUS=$(python3 -c "import json; d=json.load(open('$CP')); print(d.get('status',''))" 2>/dev/null || echo "")
is "checkpoint.status is PLANNED" "$STATUS" "PLANNED"

LEVERS=$(python3 -c "import json; d=json.load(open('$CP')); print(','.join(d.get('levers',[])))" 2>/dev/null || echo "")
is "checkpoint.levers correct" "$LEVERS" "SWIZZLE_STRIDE,BM"

# ── read mode ─────────────────────────────────────────────────────────────────
READ_OUT=$(bash .claude/skills/croq-store/checkpoint_write.sh read \
    --dsl "$DSL" --shape-key "$KEY" 2>/dev/null)
ok "read exits 0" $?
like "read outputs JSON bottleneck" "$READ_OUT" '"bottleneck"'
like "read outputs correct iter" "$READ_OUT" '"iter007_mytag"'

# ── verify mode — clean (iter matches, src exists, bin exists) ────────────────
VERIFY_OUT=$(bash .claude/skills/croq-store/checkpoint_write.sh verify \
    --dsl "$DSL" --shape-key "$KEY" --iter "iter007_mytag" 2>&1)
VERIFY_RC=$?
ok "verify exits 0 for clean match" "$VERIFY_RC"
like "verify reports CLEAN" "$VERIFY_OUT" "CLEAN"
like "verify PASS iter name" "$VERIFY_OUT" "PASS.*Iter name"
like "verify PASS src file" "$VERIFY_OUT" "PASS.*Source file"

# Checkpoint status updated to VERIFIED
STATUS2=$(python3 -c "import json; d=json.load(open('$CP')); print(d.get('status',''))" 2>/dev/null || echo "")
is "checkpoint.status updated to VERIFIED" "$STATUS2" "VERIFIED"

# ── verify mode — drift (different iter built) ────────────────────────────────
# Rewrite checkpoint as PLANNED to re-verify
python3 -c "
import json
with open('$CP') as f: d = json.load(f)
d['status'] = 'PLANNED'
d['iter'] = 'iter007_mytag'
with open('$CP', 'w') as f: json.dump(d, f)
"
bash .claude/skills/croq-store/checkpoint_write.sh verify \
    --dsl "$DSL" --shape-key "$KEY" --iter "iter099_different" >/dev/null 2>&1 && DRIFT_RC=0 || DRIFT_RC=$?
# Drift: iter name mismatch (1 point) + missing src (2 points) + missing bin (1 point) = 4 → exit 3
is "verify exits 3 on significant drift" "$DRIFT_RC" "3"

# ── write: missing --iter exits 1 ────────────────────────────────────────────
bash .claude/skills/croq-store/checkpoint_write.sh write \
    --dsl "$DSL" --shape-key "$KEY" \
    --bottleneck "memory_bound" --idea "test" >/dev/null 2>&1 && RC=0 || RC=$?
is "write without --iter exits 1" "$RC" "1"

# ── write: bare iter (no _tag) exits 1 ───────────────────────────────────────
bash .claude/skills/croq-store/checkpoint_write.sh write \
    --dsl "$DSL" --shape-key "$KEY" \
    --iter "iter007" --bottleneck "memory_bound" --idea "test" >/dev/null 2>&1 && RC=0 || RC=$?
is "write with bare iter (no _tag) exits 1" "$RC" "1"

# ── read: no checkpoint exits 2 ──────────────────────────────────────────────
rm -f "$CP"
bash .claude/skills/croq-store/checkpoint_write.sh read \
    --dsl "$DSL" --shape-key "nonexistent_key" >/dev/null 2>&1 && RC=0 || RC=$?
is "read with no checkpoint exits 2" "$RC" "2"

cd "$ORIG_DIR"
done_testing
