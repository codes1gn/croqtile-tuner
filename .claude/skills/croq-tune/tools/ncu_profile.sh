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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/activity_trace.sh"

# ── Find ncu ──────────────────────────────────────────────────────────────────
NCU=""
# Prefer the real binary over the wrapper scripts (wrappers can fail to resolve)
for candidate in \
    /usr/local/cuda-12.2/nsight-compute-*/target/linux-desktop-glibc_2_11_3-x64/ncu \
    /usr/local/cuda-12.*/nsight-compute-*/target/linux-desktop-glibc_2_11_3-x64/ncu \
    /usr/local/cuda/nsight-compute-*/target/linux-desktop-glibc_2_11_3-x64/ncu \
    /usr/local/cuda-11.*/nsight-compute-*/target/linux-desktop-glibc_2_11_3-x64/ncu; do
    if [[ -x "$candidate" ]]; then
        NCU="$candidate"
        break
    fi
done

if [[ -z "$NCU" ]]; then
    for p in /usr/local/cuda*/bin/ncu; do
        if [[ -x "$p" ]]; then NCU="$p"; break; fi
    done
fi

if [[ -z "$NCU" ]]; then
    # Last-resort search: walk common CUDA install roots for the ncu binary
    for ncu_root in /usr/local/cuda /opt/cuda /usr/cuda; do
        _found=$(find "$ncu_root" -maxdepth 6 -name ncu -type f -executable 2>/dev/null | head -1)
        if [[ -n "$_found" ]]; then NCU="$_found"; break; fi
    done
fi

if [[ -z "$NCU" ]] && command -v ncu &>/dev/null; then
    NCU=$(command -v ncu)
fi

if [[ -z "$NCU" ]]; then
    _search_report=$(find /usr/local/cuda /opt/cuda /usr/cuda -maxdepth 8 -name "ncu" 2>/dev/null | head -5 | tr '\n' ' ')
    echo "[ncu_profile] ERROR: ncu not found in any standard CUDA path." >&2
    echo "[ncu_profile] Searched: /usr/local/cuda /opt/cuda /usr/cuda" >&2
    [[ -n "$_search_report" ]] && echo "[ncu_profile] Partial matches found: $_search_report" >&2
    echo "[SUGGESTION] Use your judgement to decide autonomously. Try: (1) locate ncu with 'find / -name ncu -type f -executable 2>/dev/null | head -5' and set NCU= to that path; (2) add the containing directory to PATH and retry; (3) if ncu truly absent, skip profiling and classify bottleneck manually as memory_bound to start." >&2
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
        *) echo "[ncu_profile] ERROR: unknown argument: $1" >&2; echo "[SUGGESTION] Use your judgement to decide autonomously. Remove '$1' and retry. Valid args: --out <path_without_ext> --launch-count <N> --cmd <binary> [args...]" >&2; exit 1 ;;
    esac
done

if [[ -z "$OUT" || ${#CMD_ARGS[@]} -eq 0 ]]; then
    echo "[ncu_profile] ERROR: --out and --cmd are required" >&2
    echo "Usage: bash ncu_profile.sh --out <path_without_ext> --cmd <binary> [args...]" >&2
    echo "[SUGGESTION] Use your judgement to decide autonomously. Provide --out with the output path prefix (e.g. tuning/<gpu>/<dsl>/perf/<key>/<model>/ncu_iter045_swizzle) and --cmd with the executable followed by its arguments." >&2
    exit 1
fi

# ── Preflight: GPU contention check ──────────────────────────────────────────
_CONTENTION_TOOL="$(dirname "$0")/gpu_contention.sh"
if [[ -x "$_CONTENTION_TOOL" ]]; then
    echo "[ncu_profile] Checking GPU contention before profiling..." >&2
    _GPU_STATUS=$(bash "$_CONTENTION_TOOL" --json 2>/dev/null || echo '{"idle":true}')
    _IDLE=$(echo "$_GPU_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('idle', True))" 2>/dev/null || echo "True")
    _FOREIGN=$(echo "$_GPU_STATUS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
foreign = [c for c in d.get('contenders', []) if not c.get('is_ours', False)]
print(len(foreign))
" 2>/dev/null || echo "0")
    if [[ "$_IDLE" != "True" && "$_FOREIGN" -gt 0 ]]; then
        echo "[ncu_profile] WARNING: ${_FOREIGN} foreign GPU process(es) detected. Killing them for clean profile..." >&2
        bash "$_CONTENTION_TOOL" --kill 2>&1 | grep -E '^\s+\[(kill|FAIL|spare|skip)\]' >&2 || true
        echo "[ncu_profile] GPU contention cleared. Proceeding." >&2
    elif [[ "$_IDLE" != "True" ]]; then
        echo "[ncu_profile] INFO: GPU busy but only croq-tune harness processes found — proceeding." >&2
    else
        echo "[ncu_profile] GPU idle. Proceeding." >&2
    fi
fi

# ── Preflight: perf_event_paranoid ───────────────────────────────────────────
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "99")
if (( PARANOID > 2 )); then
    echo "[ncu_profile] ERROR: kernel.perf_event_paranoid=$PARANOID — ncu cannot profile." >&2
    echo "[ncu_profile] Fix:   sudo sysctl -w kernel.perf_event_paranoid=2" >&2
    echo "[SUGGESTION] Use your judgement to decide autonomously. Run 'sudo sysctl -w kernel.perf_event_paranoid=2' to allow ncu profiling, then retry. If sudo is unavailable, skip profiling and classify bottleneck manually based on timing data." >&2
    exit 5
fi

REP="${OUT}.ncu-rep"
CSV="${OUT}.csv"

# ── init trace from --out path (tuning/<gpu>/<dsl>/perf/<shape_key>/<model>/...)
_out_parts=(${OUT//\// })
if [[ ${#_out_parts[@]} -ge 6 && "${_out_parts[0]}" == "tuning" ]]; then
    trace_init --gpu "${_out_parts[1]}" --dsl "${_out_parts[2]}" \
               --shape-key "${_out_parts[4]}" --model "${_out_parts[5]}"
fi
trace_event "ncu_profile" "Capturing profile: ${CMD_ARGS[*]}"

# ── Step 1: ncu capture ───────────────────────────────────────────────────────
echo "[ncu_profile] Capturing: $NCU --set full --target-processes all -c $LAUNCH_COUNT --export $REP" >&2
echo "[ncu_profile] Running:   ${CMD_ARGS[*]}" >&2

"$NCU" \
    --set full \
    --target-processes all \
    --launch-count "$LAUNCH_COUNT" \
    --force-overwrite \
    --export "$REP" \
    "${CMD_ARGS[@]}" >&2

if [[ ! -f "${REP}" && ! -f "${REP}" ]]; then
    echo "[ncu_profile] ERROR: ncu did not produce report at $REP" >&2
    echo "[SUGGESTION] Use your judgement to decide autonomously. ncu profiling failed to produce a report. Common causes: (1) the kernel binary crashed or segfaulted, (2) ncu permissions issue, (3) the binary didn't launch any GPU kernels. Check that the binary runs correctly without ncu first, then retry profiling." >&2
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
    echo "[SUGGESTION] Use your judgement to decide autonomously. ncu captured a report but CSV export was empty. Try exporting manually: ncu --import $ACTUAL_REP --csv --page raw > $CSV. If that fails, try --page details instead. Then pass the CSV to profile_extract.sh." >&2
    exit 4
fi

trace_event "ncu_profile" "Profile complete: $REP + $CSV"
echo "[ncu_profile] CSV:       $CSV" >&2
echo "[ncu_profile] Done. Run profile_extract.sh --csv $CSV --iter <tag>" >&2
