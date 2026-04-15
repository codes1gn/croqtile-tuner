#!/usr/bin/env bash
# test_profile_extract.sh — Unit tests for .claude/skills/croq-profile/profile_extract.sh
#
# Builds synthetic ncu-style CSV fixtures; no GPU required.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source testing/harness/tap_helpers.sh

SCRIPT=".claude/skills/croq-profile/profile_extract.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# ── fixture: minimal ncu CSV columns ─────────────────────────────────────────
# Real ncu --csv --page raw columns (subset we use):
#   "Metric Name","Metric Unit","Maximum","Minimum","Average"
make_csv() {
    local path="$1"; shift
    mkdir -p "$(dirname "$path")"
    printf '"Metric Name","Metric Unit","Maximum","Minimum","Average"\n' > "$path"
    while [[ $# -ge 2 ]]; do
        local metric="$1"; local val="$2"; shift 2
        printf '"%s","%%","%.2f","%.2f","%.2f"\n' "$metric" "$val" "$val" "$val" >> "$path"
    done
}

ITER="iter012_testcase"

# ── CSV A: clearly memory-bound (DRAM=92, compute=32) ─────────────────────────
CSV_A="$TMP/ncu_a.csv"
make_csv "$CSV_A" \
    "dram__throughput.avg.pct_of_peak_sustained_elapsed" 92.0 \
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"   32.0 \
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed" 65.0 \
    "smsp__warp_issue_stalled_long_scoreboard_pct"        15.0

# ── CSV B: clearly compute-bound (DRAM=30, compute=88) ───────────────────────
CSV_B="$TMP/ncu_b.csv"
make_csv "$CSV_B" \
    "dram__throughput.avg.pct_of_peak_sustained_elapsed" 30.0 \
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"   88.0 \
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed" 72.0 \
    "smsp__warp_issue_stalled_long_scoreboard_pct"        12.0

# ── CSV C: launch-bound (occupancy=8, DRAM=20, compute=15) ───────────────────
CSV_C="$TMP/ncu_c.csv"
make_csv "$CSV_C" \
    "dram__throughput.avg.pct_of_peak_sustained_elapsed" 20.0 \
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"   15.0 \
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed"  8.0 \
    "smsp__warp_issue_stalled_long_scoreboard_pct"         5.0

# ── CSV D: latency-bound (stall=55, DRAM=45, compute=35) ─────────────────────
CSV_D="$TMP/ncu_d.csv"
make_csv "$CSV_D" \
    "dram__throughput.avg.pct_of_peak_sustained_elapsed" 45.0 \
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"   35.0 \
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed" 50.0 \
    "smsp__warp_issue_stalled_long_scoreboard_pct"        55.0

plan 14

# ── 1-4: memory-bound CSV A ───────────────────────────────────────────────────
OUT_A=$(bash "$SCRIPT" --csv "$CSV_A" --iter "$ITER")
ok "exit code 0 for memory-bound CSV" $?
like "output is valid JSON (has bottleneck key)" "$OUT_A" '"bottleneck"'
like "memory-bound classification" "$OUT_A" '"bottleneck": "memory_bound"'
like "confidence present" "$OUT_A" '"confidence"'

# ── 5-7: compute-bound CSV B ──────────────────────────────────────────────────
OUT_B=$(bash "$SCRIPT" --csv "$CSV_B" --iter "$ITER")
like "compute-bound classification" "$OUT_B" '"bottleneck": "compute_bound"'
like "iter field present" "$OUT_B" "\"iter\": \"$ITER\""
like "evidence.key_metrics present" "$OUT_B" '"key_metrics"'

# ── 8: launch-bound CSV C ─────────────────────────────────────────────────────
OUT_C=$(bash "$SCRIPT" --csv "$CSV_C" --iter "$ITER")
like "launch-bound classification" "$OUT_C" '"bottleneck": "launch_bound"'

# ── 9: latency-bound CSV D ────────────────────────────────────────────────────
OUT_D=$(bash "$SCRIPT" --csv "$CSV_D" --iter "$ITER")
like "latency-bound classification" "$OUT_D" '"bottleneck": "latency_bound"'

# ── 10: output is valid JSON (python can parse it) ────────────────────────────
VALID=$(echo "$OUT_A" | python3 -c "import sys,json; json.load(sys.stdin); print('ok')" 2>/dev/null || echo "fail")
is "output is valid JSON (python parse)" "$VALID" "ok"

# ── 11: missing CSV exits 2 ───────────────────────────────────────────────────
bash "$SCRIPT" --csv "$TMP/nonexistent.csv" --iter "$ITER" >/dev/null 2>&1 && RC=0 || RC=$?
is "missing CSV exits 2" "$RC" "2"

# ── 12: missing --csv arg exits 1 ────────────────────────────────────────────
bash "$SCRIPT" --iter "$ITER" >/dev/null 2>&1 && RC=0 || RC=$?
is "missing --csv exits 1" "$RC" "1"

# ── 13: missing --iter arg exits 1 ───────────────────────────────────────────
bash "$SCRIPT" --csv "$CSV_A" >/dev/null 2>&1 && RC=0 || RC=$?
is "missing --iter exits 1" "$RC" "1"

# ── 14: bare iter tag (no _tag) rejected with exit 1 ─────────────────────────
bash "$SCRIPT" --csv "$CSV_A" --iter "iter012" >/dev/null 2>&1 && RC=0 || RC=$?
is "bare iter without _tag exits 1" "$RC" "1"

done_testing
