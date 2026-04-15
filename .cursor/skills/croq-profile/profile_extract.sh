#!/usr/bin/env bash
# profile_extract.sh — Deterministic bottleneck classifier from an ncu CSV.
#
# USAGE (from repo root):
#   bash .claude/skills/croq-profile/profile_extract.sh \
#       --csv   tuning/<gpu>/cuda/perf/512x16384x16384/ncu_iter012_myname.csv \
#       --iter  iter012_myname
#
# OUTPUT (stdout): one-line JSON matching croq-profile handoff schema
#   {"bottleneck":"memory_bound","confidence":"high","evidence":{...}}
#
# EXIT:
#   0  — classification succeeded, JSON on stdout
#   1  — usage error (bad args)
#   2  — CSV not found or unreadable
#   3  — required metrics not found in CSV
#
# RULES:
#   • Never print anything to stdout except the final JSON.
#   • All diagnostics go to stderr.
#   • Classification thresholds are fixed — do NOT change them without a test.

set -euo pipefail

# ── argument parsing ──────────────────────────────────────────────────────────
CSV=""
ITER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv)   CSV="$2";  shift 2 ;;
        --iter)  ITER="$2"; shift 2 ;;
        *) echo "[profile_extract] ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CSV" || -z "$ITER" ]]; then
    echo "[profile_extract] ERROR: --csv and --iter are required" >&2
    echo "Usage: bash profile_extract.sh --csv <path.csv> --iter <iter_tag>" >&2
    exit 1
fi

# ── validate iter tag ─────────────────────────────────────────────────────────
if ! echo "$ITER" | grep -qE '^iter[0-9]{3}_[a-z][a-z0-9_]{1,15}$'; then
    echo "[profile_extract] ERROR: ITER must match iter<NNN>_<tag>, got: $ITER" >&2
    exit 1
fi

# ── validate CSV exists ───────────────────────────────────────────────────────
if [[ ! -f "$CSV" ]]; then
    echo "[profile_extract] ERROR: CSV not found: $CSV" >&2
    exit 2
fi

# ── parse metrics from ncu CSV ────────────────────────────────────────────────
# ncu --csv --page raw produces a CSV with one header row and metric rows.
# Key metric IDs we look for (ncu uses exact metric names):
#   dram__throughput.avg.pct_of_peak_sustained_elapsed  → DRAM throughput %
#   sm__throughput.avg.pct_of_peak_sustained_elapsed    → SM (compute) throughput %
#   ipc_occupancy.avg.pct                               → SM occupancy % (Hopper)
#   sm__warps_active.avg.pct_of_peak_sustained_elapsed  → occupancy (older)
#   derived__memory_l2_theoretical_bandwidth_optimal    → L2 BW (fallback for DRAM)

extract_metric() {
    local name="$1"
    # ncu CSV: columns are "Metric Name","Metric Unit","Maximum","Minimum","Average"
    # We want the Average column (5th field).  Value may be float or "N/A".
    python3 - "$CSV" "$name" <<'PYEOF' 2>/dev/null || echo ""
import sys, csv, re
csv_path, target = sys.argv[1], sys.argv[2]
with open(csv_path, newline='', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        metric_name = row[0].strip().strip('"')
        if metric_name == target and len(row) >= 5:
            val = row[4].strip().strip('"')
            try:
                print(float(val))
            except ValueError:
                pass
            break
PYEOF
}

# Primary metrics
DRAM_PCT=$(extract_metric "dram__throughput.avg.pct_of_peak_sustained_elapsed")
COMPUTE_PCT=$(extract_metric "sm__throughput.avg.pct_of_peak_sustained_elapsed")
OCCUPANCY_PCT=$(extract_metric "sm__warps_active.avg.pct_of_peak_sustained_elapsed")
# Hopper / newer arch fallback for occupancy
if [[ -z "$OCCUPANCY_PCT" ]]; then
    OCCUPANCY_PCT=$(extract_metric "ipc_occupancy.avg.pct")
fi
# BW in GB/s (reported metric name varies by ncu version)
BW_GBS=$(extract_metric "dram__bytes.sum.per_second" 2>/dev/null || echo "")
if [[ -z "$BW_GBS" ]]; then
    BW_GBS="0"
fi

# Stall cycles (latency proxy)
STALL_PCT=$(extract_metric "smsp__warp_issue_stalled_long_scoreboard_pct")
if [[ -z "$STALL_PCT" ]]; then
    STALL_PCT="0"
fi

# Check we got at least DRAM and compute
if [[ -z "$DRAM_PCT" || -z "$COMPUTE_PCT" ]]; then
    echo "[profile_extract] ERROR: could not extract dram__throughput or sm__throughput from: $CSV" >&2
    echo "[profile_extract] Available metrics:" >&2
    python3 -c "
import csv, sys
with open('$CSV', newline='', encoding='utf-8-sig') as f:
    for row in csv.reader(f):
        if row: print(' ', row[0][:80])
" >&2 2>/dev/null || true
    exit 3
fi

if [[ -z "$OCCUPANCY_PCT" ]]; then
    OCCUPANCY_PCT="0"
fi

# ── classify bottleneck ───────────────────────────────────────────────────────
# Thresholds (empirically calibrated for H100/A100; change only with test evidence):
#   memory_bound : DRAM% > 80 AND compute% < 50
#   compute_bound: compute% > 70 AND DRAM% < 60
#   latency_bound: stall% > 40 AND DRAM% < 60 AND compute% < 60
#   launch_bound : occupancy% < 15 AND DRAM% < 40 AND compute% < 40
#   mixed        : anything else

BOTTLENECK=$(python3 - "$DRAM_PCT" "$COMPUTE_PCT" "$OCCUPANCY_PCT" "$STALL_PCT" <<'PYEOF'
import sys
dram, compute, occ, stall = [float(x) for x in sys.argv[1:]]

if dram > 80 and compute < 50:
    print("memory_bound")
elif compute > 70 and dram < 60:
    print("compute_bound")
elif occ < 15 and dram < 40 and compute < 40:
    print("launch_bound")
elif stall > 40 and dram < 60 and compute < 60:
    print("latency_bound")
else:
    # Mixed: whichever is higher wins
    if dram >= compute:
        print("memory_bound")
    else:
        print("compute_bound")
PYEOF
)

# Confidence: high if leading dimension is unambiguous, medium otherwise
CONFIDENCE=$(python3 - "$DRAM_PCT" "$COMPUTE_PCT" "$OCCUPANCY_PCT" "$STALL_PCT" "$BOTTLENECK" <<'PYEOF'
import sys
dram, compute, occ, stall = [float(x) for x in sys.argv[1:5]]
bottleneck = sys.argv[5]

if bottleneck == "memory_bound" and dram > 90:
    print("high")
elif bottleneck == "compute_bound" and compute > 85:
    print("high")
elif bottleneck == "launch_bound" and occ < 10:
    print("high")
elif bottleneck == "latency_bound" and stall > 55:
    print("high")
else:
    print("medium")
PYEOF
)

# ── emit handoff JSON ─────────────────────────────────────────────────────────
python3 - "$ITER" "$CSV" "$BOTTLENECK" "$CONFIDENCE" \
          "$DRAM_PCT" "$COMPUTE_PCT" "$OCCUPANCY_PCT" "$BW_GBS" <<'PYEOF'
import sys, json
iter_tag, csv_path, bottleneck, confidence = sys.argv[1:5]
dram_pct, compute_pct, occ_pct, bw_gbs = [float(x) for x in sys.argv[5:]]

result = {
    "iter": iter_tag,
    "bottleneck": bottleneck,
    "confidence": confidence,
    "evidence": {
        "ncu_csv": csv_path,
        "key_metrics": {
            "dram_throughput_pct":     round(dram_pct, 1),
            "compute_throughput_pct":  round(compute_pct, 1),
            "sm_occupancy_pct":        round(occ_pct, 1),
            "achieved_bandwidth_gb_s": round(bw_gbs / 1e9, 1) if bw_gbs > 1e6 else bw_gbs
        }
    }
}
print(json.dumps(result))
PYEOF
