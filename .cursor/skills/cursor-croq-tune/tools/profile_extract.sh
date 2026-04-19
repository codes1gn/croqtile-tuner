#!/usr/bin/env bash
# profile_extract.sh — Deterministic bottleneck classifier from an ncu CSV.
#
# USAGE (from repo root):
#   bash .cursor/skills/cursor-croq-tune/tools/profile_extract.sh \
#       --csv   tuning/<gpu>/croqtile/perf/<shape_key>/<model>/ncu_iter021_warp2x4_round18.csv \
#       --iter  iter021_warp2x4
#
# SUPPORTED CSV FORMATS:
#   1. WIDE (raw)   — from: ncu --csv --page raw
#      Row 1 = column headers (metric names), Row 2 = units, Row 3+ = per-launch values.
#      This is what `ncu --csv` produces by default on most versions.
#
#   2. TALL (details) — from: ncu --import --csv --page details
#      Each row is one metric: col[12]=Metric Name, col[14]=Metric Value.
#      Produced by: ncu --import file.ncu-rep --csv --page details
#
#   The script auto-detects the format from the header row.
#
# OUTPUT (stdout): one-line JSON matching croq-profile handoff schema
#   {"iter":"...","bottleneck":"...","confidence":"...","evidence":{...}}
#
# EXIT CODES:
#   0  — success, JSON on stdout
#   1  — usage error (bad args)
#   2  — CSV not found or unreadable
#   3  — required metrics not found in CSV
#
# RULES:
#   • Nothing to stdout except the final JSON.
#   • All diagnostics go to stderr.
#   • Thresholds are fixed — change only with documented evidence.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/activity_trace.sh"

# ── argument parsing ──────────────────────────────────────────────────────────
CSV=""
ITER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv)   CSV="$2";  shift 2 ;;
        --iter)  ITER="$2"; shift 2 ;;
        *) echo "[profile_extract] ERROR: unknown argument: $1" >&2; echo "[SUGGESTION] Use your judgement to decide autonomously. Remove '$1' and retry. Valid args: --csv <path_to_ncu_csv> --iter <iter_tag>" >&2; exit 1 ;;
    esac
done

if [[ -z "$CSV" || -z "$ITER" ]]; then
    echo "[profile_extract] ERROR: --csv and --iter are required" >&2
    echo "Usage: bash profile_extract.sh --csv <path.csv> --iter <iter_tag>" >&2
    echo "[SUGGESTION] Use your judgement to decide autonomously. Provide --csv pointing to the ncu CSV output file, and --iter with the iteration tag (e.g. iter045_swizzle). The CSV is produced by ncu_profile.sh." >&2
    exit 1
fi

# ── validate iter tag ─────────────────────────────────────────────────────────
if ! echo "$ITER" | grep -qE '^iter[0-9]+_[a-z][a-z0-9_]{1,30}$'; then
    echo "[profile_extract] ERROR: ITER must match iter<N+>_<tag>, got: $ITER" >&2
    echo "[SUGGESTION] Use your judgement to decide autonomously. Fix --iter format. Must be iter<NNN>_<tag> like iter045_swizzle. This should match the kernel you profiled." >&2
    exit 1
fi

# ── validate CSV exists ───────────────────────────────────────────────────────
if [[ ! -f "$CSV" ]]; then
    echo "[profile_extract] ERROR: CSV not found: $CSV" >&2
    echo "[SUGGESTION] Use your judgement to decide autonomously. The CSV file does not exist at the given path. Run ncu_profile.sh first to produce the CSV, then retry profile_extract.sh with the correct path." >&2
    exit 2
fi

# ── init trace from CSV path (tuning/<gpu>/<dsl>/perf/<shape_key>/<model>/...)
_csv_parts=(${CSV//\// })
if [[ ${#_csv_parts[@]} -ge 6 && "${_csv_parts[0]}" == "tuning" ]]; then
    trace_init --gpu "${_csv_parts[1]}" --dsl "${_csv_parts[2]}" \
               --shape-key "${_csv_parts[4]}" --model "${_csv_parts[5]}"
fi
trace_event "profile_extract" "Analyzing $ITER from $CSV"

# ── parse all key metrics in one Python pass ──────────────────────────────────
# Single Python script reads the CSV once, auto-detects format, returns JSON.
METRICS_JSON=$(python3 - "$CSV" <<'PYEOF'
import sys, csv, json

csv_path = sys.argv[1]

# Metric IDs we want (in priority order per quantity)
DRAM_IDS = [
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
]
COMPUTE_IDS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
]
OCCUPANCY_IDS = [
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
    "ipc_occupancy.avg.pct",
]
STALL_IDS = [
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__warp_issue_stalled_long_scoreboard_pct",
]
BW_IDS = [
    "dram__bytes.sum.per_second",
    "dram__bytes_read.sum.per_second",
]

# Human-readable names used in TALL format (col 12 = Metric Name)
HUMAN_MAP = {
    "DRAM Throughput": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "Memory Throughput": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "Compute (SM) Throughput": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "SM Throughput": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "Achieved Occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
}
# Build reverse map: internal id -> set of human names
HUMAN_REVERSE = {}
for human, internal in HUMAN_MAP.items():
    HUMAN_REVERSE.setdefault(internal, set()).add(human)

def first_float(d, id_list):
    """Return the first non-None float from dict d for id_list."""
    for k in id_list:
        v = d.get(k)
        if v is not None:
            return v
    return None

def parse_float(s):
    try:
        v = float(str(s).strip().strip('"').replace(',', ''))
        return v if abs(v) < 1e30 else None
    except (ValueError, TypeError):
        return None

with open(csv_path, newline='', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    rows = list(reader)

if not rows:
    print(json.dumps({"error": "empty CSV"}))
    sys.exit(0)

header = [h.strip().strip('"') for h in rows[0]]

# ── Format detection ──────────────────────────────────────────────────────────
# TALL format: col[11] = "Section Name", col[12] = "Metric Name", col[14] = "Metric Value"
is_tall = (len(header) > 14 and header[11] == "Section Name"
                              and header[12] == "Metric Name"
                              and header[14] == "Metric Value")

# COMPACT-TALL: "Metric Name","Metric Unit","Maximum","Minimum","Average" (5 cols)
is_compact_tall = (not is_tall and len(header) >= 5
                   and header[0] == "Metric Name")

metrics = {}  # internal_id -> float value
all_ids = set(DRAM_IDS + COMPUTE_IDS + OCCUPANCY_IDS + STALL_IDS + BW_IDS)

if is_tall:
    for row in rows[1:]:
        if len(row) <= 14:
            continue
        name = row[12].strip().strip('"')
        val  = parse_float(row[14])
        if val is None:
            continue
        if name in all_ids and name not in metrics:
            metrics[name] = val
            continue
        internal = HUMAN_MAP.get(name)
        if internal and internal not in metrics:
            metrics[internal] = val
elif is_compact_tall:
    # Each row: metric_name, unit, max, min, avg — use avg (col 4) or max (col 2)
    avg_idx = next((i for i, h in enumerate(header) if h.lower() == "average"), -1)
    max_idx = next((i for i, h in enumerate(header) if h.lower() == "maximum"), -1)
    val_idx = avg_idx if avg_idx >= 0 else max_idx
    for row in rows[1:]:
        if len(row) <= val_idx or val_idx < 0:
            continue
        name = row[0].strip().strip('"')
        val = parse_float(row[val_idx])
        if val is None:
            continue
        if name in all_ids and name not in metrics:
            metrics[name] = val
            continue
        internal = HUMAN_MAP.get(name)
        if internal and internal not in metrics:
            metrics[internal] = val
else:
    # WIDE format: row[0] = headers, row[1] = units, row[2+] = data per launch
    if len(rows) < 3:
        print(json.dumps({"error": "wide CSV has fewer than 3 rows"}))
        sys.exit(0)
    data = rows[2]
    for h, v in zip(header, data):
        val = parse_float(v)
        if val is not None:
            metrics[h] = val

# ── Extract values ────────────────────────────────────────────────────────────
dram_pct    = first_float(metrics, DRAM_IDS)
compute_pct = first_float(metrics, COMPUTE_IDS)
occ_pct     = first_float(metrics, OCCUPANCY_IDS)
stall_ratio = first_float(metrics, STALL_IDS)
bw_gbs      = first_float(metrics, BW_IDS)

missing = []
if dram_pct    is None: missing.append("DRAM throughput %")
if compute_pct is None: missing.append("SM compute throughput %")
if missing:
    print(json.dumps({"error": f"metrics not found: {missing}",
                      "format": "tall" if is_tall else "wide",
                      "available": sorted(list(metrics.keys()))[:30]}))
    sys.exit(0)

occ_pct     = occ_pct    if occ_pct    is not None else 0.0
stall_ratio = stall_ratio if stall_ratio is not None else 0.0
bw_gbs      = bw_gbs     if bw_gbs     is not None else 0.0

result = {
    "format": "tall" if is_tall else "wide",
    "dram_pct": round(dram_pct, 2),
    "compute_pct": round(compute_pct, 2),
    "occ_pct": round(occ_pct, 2),
    "stall_ratio": round(stall_ratio, 2),
    "bw_gbs": round(bw_gbs, 2),
}
print(json.dumps(result))
PYEOF
)

# Check for parse errors
PARSE_ERROR=$(echo "$METRICS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',''))")
if [[ -n "$PARSE_ERROR" ]]; then
    echo "[profile_extract] ERROR: metric extraction failed: $PARSE_ERROR" >&2
    echo "[profile_extract] Raw parse output: $METRICS_JSON" >&2
    echo "[SUGGESTION] Use your judgement to decide autonomously. The CSV does not contain required metrics (DRAM throughput %, SM compute throughput %). This may mean the ncu profiling used insufficient metric sets. Re-run ncu_profile.sh with --set full and try again. If this persists, classify bottleneck manually based on available ncu output and skip profile_extract." >&2
    exit 3
fi

# ── Extract individual values ─────────────────────────────────────────────────
DRAM_PCT=$(echo "$METRICS_JSON"    | python3 -c "import sys,json; print(json.load(sys.stdin)['dram_pct'])")
COMPUTE_PCT=$(echo "$METRICS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['compute_pct'])")
OCCUPANCY_PCT=$(echo "$METRICS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['occ_pct'])")
STALL_RATIO=$(echo "$METRICS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['stall_ratio'])")
BW_GBS=$(echo "$METRICS_JSON"      | python3 -c "import sys,json; print(json.load(sys.stdin)['bw_gbs'])")

# ── Classify bottleneck ───────────────────────────────────────────────────────
# Thresholds calibrated for Ampere/Turing (SM86/SM80). Valid for H100 too.
# For stall: ratio > 4.0 (warps stalled per active warp) indicates latency issue.
#   memory_bound : DRAM% > 70 AND compute% < 60
#   compute_bound: compute% > 70 AND DRAM% < 60
#   latency_bound: stall_ratio > 4.0 AND DRAM% < 60 AND compute% < 60
#   launch_bound : occupancy% < 15 AND DRAM% < 40 AND compute% < 40
#   mixed        : highest% wins
BOTTLENECK=$(python3 - "$DRAM_PCT" "$COMPUTE_PCT" "$OCCUPANCY_PCT" "$STALL_RATIO" <<'PYEOF'
import sys
dram, compute, occ, stall = [float(x) for x in sys.argv[1:]]
if dram > 70 and compute < 60:
    print("memory_bound")
elif compute > 70 and dram < 60:
    print("compute_bound")
elif occ < 15 and dram < 40 and compute < 40:
    print("launch_bound")
elif stall > 4.0 and dram < 60 and compute < 60:
    print("latency_bound")
else:
    print("memory_bound" if dram >= compute else "compute_bound")
PYEOF
)

# Confidence: high if the leading dimension is unambiguous
CONFIDENCE=$(python3 - "$DRAM_PCT" "$COMPUTE_PCT" "$OCCUPANCY_PCT" "$STALL_RATIO" "$BOTTLENECK" <<'PYEOF'
import sys
dram, compute, occ, stall = [float(x) for x in sys.argv[1:5]]
bottleneck = sys.argv[5]
if bottleneck == "memory_bound" and dram > 85:
    print("high")
elif bottleneck == "compute_bound" and compute > 85:
    print("high")
elif bottleneck == "launch_bound" and occ < 10:
    print("high")
elif bottleneck == "latency_bound" and stall > 6.0:
    print("high")
else:
    print("medium")
PYEOF
)

# ── Derive .ncu-rep path from CSV path ───────────────────────────────────────
NCU_REP="${CSV%.csv}.ncu-rep"
if [[ ! -f "$NCU_REP" ]]; then
    NCU_REP=""
fi

# ── Emit handoff JSON ─────────────────────────────────────────────────────────
python3 - "$ITER" "$CSV" "$NCU_REP" "$BOTTLENECK" "$CONFIDENCE" \
          "$DRAM_PCT" "$COMPUTE_PCT" "$OCCUPANCY_PCT" "$BW_GBS" <<'PYEOF'
import sys, json
iter_tag, csv_path, ncu_rep, bottleneck, confidence = sys.argv[1:6]
dram_pct, compute_pct, occ_pct, bw_gbs = [float(x) for x in sys.argv[6:]]

result = {
    "iter": iter_tag,
    "bottleneck": bottleneck,
    "confidence": confidence,
    "evidence": {
        "ncu_csv": csv_path,
        "ncu_rep": ncu_rep if ncu_rep else None,
        "key_metrics": {
            "dram_throughput_pct":    round(dram_pct, 1),
            "compute_throughput_pct": round(compute_pct, 1),
            "sm_occupancy_pct":       round(occ_pct, 1),
            "achieved_bandwidth_gb_s": round(bw_gbs, 1),
        }
    }
}

if confidence == "medium":
    result["hint"] = (
        "Confidence is medium. For a better IDEA, consider: "
        "(1) ncu --import <ncu_rep> --page details for warp stall breakdown, L1/L2 hit rates; "
        "(2) bash .cursor/skills/cursor-croq-tune/tools/sass_compare.sh dump-baseline + dump-custom + compare "
        "for baseline vs custom SASS comparison (instruction mix, register pressure, tensor core types); "
        "(3) load perf-nsight-compute-analysis skill for systematic analysis."
    )

print(json.dumps(result))
PYEOF
