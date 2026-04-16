#!/usr/bin/env bash
# ncu_profile.sh — One-shot ncu profile: capture .ncu-rep then export .csv
#
# USAGE (from repo root):
#   bash .claude/skills/croq-tune/tools/ncu_profile.sh \
#       --out  tuning/<gpu>/<dsl>/perf/<shape_key>/<model>/ncu_iter021_warp2x4 \
#       --cmd  "./build/iter021_warp2x4.exe 16384 16384 16384"
#
# OUTPUT files:
#   <out>.ncu-rep   — full ncu report
#   <out>.csv       — wide raw CSV, ready for profile_extract.sh
#
# EXIT CODES:
#   0  — both files produced, metrics parseable
#   1  — usage error
#   2  — ncu not found
#   3  — ncu capture failed
#   4  — CSV export failed
#   5  — preflight failed (perf_event_paranoid)
#
# STDOUT: nothing (all progress to stderr)
# STDERR: progress messages + error details
#
# DESIGN NOTES:
#   • --set full: all metric groups collected in one pass
#   • --target-processes all: needed when binary forks a child process
#   • --launch-count 1: profile only the first kernel launch (saves ~10x time)
#   • --page raw: wide CSV format; profile_extract.sh handles this format natively
#   • Atomic: if either step fails, exits immediately with non-zero code

set -euo pipefail

# ── Find ncu ──────────────────────────────────────────────────────────────────
NCU=""
for candidate in ncu /usr/local/cuda/bin/ncu /usr/local/cuda-12.2/bin/ncu /usr/local/cuda-11.8/bin/ncu; do
    if command -v "$candidate" &>/dev/null 2>&1; then
        NCU="$candidate"
        break
    fi
done

if [[ -z "$NCU" ]]; then
    # Try standard CUDA paths
    for p in /usr/local/cuda*/bin/ncu; do
        if [[ -x "$p" ]]; then NCU="$p"; break; fi
    done
fi

if [[ -z "$NCU" ]]; then
    echo "[ncu_profile] ERROR: ncu not found. Add /usr/local/cuda/bin to PATH." >&2
    exit 2
fi

# ── Argument parsing ──────────────────────────────────────────────────────────
OUT=""
CMD_ARGS=()
LAUNCH_COUNT=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)           OUT="$2"; shift 2 ;;
        --launch-count)  LAUNCH_COUNT="$2"; shift 2 ;;
        --cmd)           shift; CMD_ARGS=("$@"); break ;;
        *) echo "[ncu_profile] ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$OUT" || ${#CMD_ARGS[@]} -eq 0 ]]; then
    echo "[ncu_profile] ERROR: --out and --cmd are required" >&2
    echo "Usage: bash ncu_profile.sh --out <path_without_ext> --cmd <binary> [args...]" >&2
    exit 1
fi

# ── Preflight: perf_event_paranoid ───────────────────────────────────────────
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "99")
if (( PARANOID > 2 )); then
    echo "[ncu_profile] ERROR: kernel.perf_event_paranoid=$PARANOID — ncu cannot profile." >&2
    echo "[ncu_profile] Fix:   sudo sysctl -w kernel.perf_event_paranoid=2" >&2
    exit 5
fi

REP="${OUT}.ncu-rep"
CSV="${OUT}.csv"

# ── Step 1: ncu capture ───────────────────────────────────────────────────────
echo "[ncu_profile] Capturing: $NCU --set full --target-processes all -c $LAUNCH_COUNT --export $REP" >&2
echo "[ncu_profile] Running:   ${CMD_ARGS[*]}" >&2

"$NCU" \
    --set full \
    --target-processes all \
    --launch-count "$LAUNCH_COUNT" \
    --force-overwrite \
    --export "$REP" \
    -- "${CMD_ARGS[@]}" >&2

if [[ ! -f "${REP}" && ! -f "${REP}" ]]; then
    # ncu appends .ncu-rep automatically; check both forms
    echo "[ncu_profile] ERROR: ncu did not produce report at $REP" >&2
    exit 3
fi

# ncu may write either <out>.ncu-rep or <out>.ncu-rep (sometimes adds ext twice on older)
# Normalise
ACTUAL_REP="$REP"
if [[ ! -f "$REP" && -f "${REP}.ncu-rep" ]]; then
    ACTUAL_REP="${REP}.ncu-rep"
fi

echo "[ncu_profile] Report:    $ACTUAL_REP" >&2

# ── Step 2: Export raw CSV ────────────────────────────────────────────────────
echo "[ncu_profile] Exporting: $NCU --import $ACTUAL_REP --csv --page raw" >&2

"$NCU" \
    --import "$ACTUAL_REP" \
    --csv \
    --page raw \
    > "$CSV"

if [[ ! -s "$CSV" ]]; then
    echo "[ncu_profile] ERROR: CSV export produced empty file at $CSV" >&2
    exit 4
fi

echo "[ncu_profile] CSV:       $CSV" >&2
echo "[ncu_profile] Done. Run profile_extract.sh --csv $CSV --iter <tag>" >&2
