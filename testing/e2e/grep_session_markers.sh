#!/usr/bin/env bash
# grep_session_markers.sh — Dual-mode e2e checker for croq-tune sessions.
#
# MODE 1 — session transcript check (pass a JSONL file):
#   Greps for behavioural markers proving the agent followed the protocol.
#
# MODE 2 — filesystem artifact check (pass a DSL + shape-key):
#   Validates that actual artifacts on disk conform to naming and
#   co-location rules (ncu CSVs present, filenames tagged, etc.)
#
# USAGE:
#   # Mode 1: session transcript
#   bash testing/e2e/grep_session_markers.sh --session <uuid>.jsonl
#
#   # Mode 2: filesystem artifacts
#   bash testing/e2e/grep_session_markers.sh --fs --dsl cuda --shape-key matmul_bf16fp32_512x16384x16384
#
#   # Both modes together:
#   bash testing/e2e/grep_session_markers.sh \
#       --session <uuid>.jsonl \
#       --fs --dsl cuda --shape-key matmul_bf16fp32_512x16384x16384
#
# EXIT CODES:
#   0  — all checks passed
#   1  — one or more required checks failed
#   2  — bad arguments or file not found

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ── argument parsing ──────────────────────────────────────────────────────────
SESSION=""
DO_FS=0
DSL=""
SHAPE_KEY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session)  SESSION="$2";   shift 2 ;;
        --fs)       DO_FS=1;        shift   ;;
        --dsl)      DSL="$2";       shift 2 ;;
        --shape-key) SHAPE_KEY="$2"; shift 2 ;;
        -*)
            echo "ERROR: unknown argument: $1" >&2
            echo "USAGE: $0 [--session <file.jsonl>] [--fs --dsl <dsl> --shape-key <key>]" >&2
            exit 2
            ;;
        *)
            # Positional compat: first positional arg = session file (legacy usage)
            SESSION="$1"; shift ;;
    esac
done

if [[ -z "$SESSION" && "$DO_FS" -eq 0 ]]; then
    echo "ERROR: at least --session <file> or --fs must be specified" >&2
    exit 2
fi

if [[ "$DO_FS" -eq 1 && (-z "$DSL" || -z "$SHAPE_KEY") ]]; then
    echo "ERROR: --fs requires both --dsl and --shape-key" >&2
    exit 2
fi

TOTAL_PASS=0
TOTAL_FAIL=0

# ── shared helpers ─────────────────────────────────────────────────────────────
result_pass() { echo "  PASS  $1"; TOTAL_PASS=$((TOTAL_PASS + 1)); }
result_fail() { echo "  FAIL  $1"; TOTAL_FAIL=$((TOTAL_FAIL + 1)); }
result_warn() { echo "  WARN  $1 [optional]"; }

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: SESSION TRANSCRIPT CHECKS
# ══════════════════════════════════════════════════════════════════════════════
if [[ -n "$SESSION" ]]; then
    if [[ ! -f "$SESSION" ]]; then
        echo "ERROR: session file not found: $SESSION" >&2
        exit 2
    fi

    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  MODE 1: Session Transcript Markers                     ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo "  File: $SESSION"
    echo "  Size: $(wc -l < "$SESSION") lines"
    echo ""

    # Extract all text content from the JSONL
    TEXT=$(python3 -c "
import sys, json
for line in sys.stdin:
    try:
        obj = json.loads(line.strip())
        print(json.dumps(obj))
    except Exception:
        pass
" < "$SESSION")

    marker_check() {
        local desc="$1"
        local pattern="$2"
        local required="${3:-yes}"
        if echo "$TEXT" | grep -q "$pattern"; then
            result_pass "$desc"
        else
            if [[ "$required" == "yes" ]]; then
                result_fail "$desc  [REQUIRED marker missing]"
            else
                result_warn "$desc"
            fi
        fi
    }

    echo "--- Storage Harness ---"
    marker_check "store_round.sh was called"       "store_round\.sh"
    marker_check "STORE complete message"          "\[store_round\] STORE complete"
    marker_check "next_iter.sh was called"         "next_iter\.sh"

    echo ""
    echo "--- Profiling ---"
    marker_check "ncu was invoked"                 "ncu "
    marker_check "profile_extract.sh was called"  "profile_extract\.sh"  "no"
    marker_check "bottleneck classification present" '"bottleneck"'        "no"

    echo ""
    echo "--- Protocol Steps ---"
    marker_check "VERIFY: PASS appeared"           "VERIFY: PASS"
    marker_check "TFLOPS measurement appeared"     "TFLOPS:"
    marker_check "KEEP or DISCARD decision made"   "KEEP\|DISCARD"

    echo ""
    echo "--- Anti-Patterns (presence = FAIL) ---"

    # Bare iter files without tag: iter012.cu (3 digits, no underscore + word)
    BAD_BARE=$(echo "$TEXT" | grep -cE 'iter[0-9]{3}\.(cu|co|py)\b' || true)
    if [[ "$BAD_BARE" -eq 0 ]]; then
        result_pass "No bare-number iter source files (iter<NNN>.cu without _tag)"
    else
        result_fail "Found $BAD_BARE bare-number iter files — missing _tag suffix"
    fi

    # ncu CSV without iter tag: ncu_iter012.csv (missing _tag)
    BAD_NCU=$(echo "$TEXT" | grep -cE 'ncu_iter[0-9]{3}\.csv\b' || true)
    if [[ "$BAD_NCU" -eq 0 ]]; then
        result_pass "No bare ncu CSV filenames (ncu_iter<NNN>.csv without _tag)"
    else
        result_fail "Found $BAD_NCU ncu CSVs without _tag — name them ncu_iter<NNN>_<tag>.csv"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: FILESYSTEM ARTIFACT CHECKS
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$DO_FS" -eq 1 ]]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  MODE 2: Filesystem Artifact Checks                     ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo "  DSL:       $DSL"
    echo "  Shape-key: $SHAPE_KEY"
    echo ""

    SRC_DIR="tuning/aitune/${DSL}/srcs/${SHAPE_KEY}"
    LOG_DIR="tuning/aitune/${DSL}/logs/${SHAPE_KEY}"
    MEM_DIR="tuning/aitune/${DSL}/memory/${SHAPE_KEY}"
    PERF_DIR="tuning/aitune/${DSL}/perf/${SHAPE_KEY}"

    # ── 1. Source file naming ──────────────────────────────────────────────────
    echo "--- Source File Names ($SRC_DIR) ---"
    if [[ ! -d "$SRC_DIR" ]]; then
        result_fail "Source directory does not exist: $SRC_DIR"
    else
        SRC_COUNT=0
        BARE_COUNT=0
        TAGGED_COUNT=0

        # Iterate source files matching iter pattern
        for f in "$SRC_DIR"/iter[0-9][0-9][0-9]*; do
            [[ -f "$f" ]] || continue
            name=$(basename "$f")
            SRC_COUNT=$((SRC_COUNT + 1))
            if echo "$name" | grep -qE '^iter[0-9]{3}_[a-z][a-z0-9_]{1,15}\.[a-z]+$'; then
                TAGGED_COUNT=$((TAGGED_COUNT + 1))
            else
                BARE_COUNT=$((BARE_COUNT + 1))
                echo "    BAD NAME: $name  (expected iter<NNN>_<tag>.<ext>)"
            fi
        done

        if [[ "$SRC_COUNT" -eq 0 ]]; then
            result_warn "No iter source files found in $SRC_DIR"
        elif [[ "$BARE_COUNT" -eq 0 ]]; then
            result_pass "All $TAGGED_COUNT iter source files have correct _tag suffix"
        else
            result_fail "$BARE_COUNT of $SRC_COUNT iter source files missing _tag suffix"
        fi
    fi

    echo ""
    echo "--- Memory Files ($MEM_DIR) ---"

    RAW="$MEM_DIR/rounds.raw.jsonl"
    MD="$MEM_DIR/rounds.md"

    if [[ -f "$RAW" ]]; then
        LINES=$(wc -l < "$RAW")
        result_pass "rounds.raw.jsonl exists ($LINES lines)"

        # Validate every line is parseable JSON with required fields
        BAD_JSON=$(python3 -c "
import sys, json
bad = 0
required = ['iter','kernel','tflops','decision','bottleneck']
for i, line in enumerate(sys.stdin, 1):
    line = line.strip()
    if not line: continue
    try:
        obj = json.loads(line)
        missing = [k for k in required if k not in obj]
        if missing:
            print(f'  line {i}: missing fields {missing}')
            bad += 1
    except Exception as e:
        print(f'  line {i}: invalid JSON — {e}')
        bad += 1
sys.exit(bad)
" < "$RAW" 2>&1 || true)
        BAD_COUNT=$(echo "$BAD_JSON" | grep -c '  line' || true)
        if [[ "$BAD_COUNT" -eq 0 ]]; then
            result_pass "rounds.raw.jsonl: all lines valid JSON with required fields"
        else
            result_fail "rounds.raw.jsonl: $BAD_COUNT malformed lines"
            echo "$BAD_JSON" | head -10 | sed 's/^/    /'
        fi

        # Check iters in rounds match tagged src files
        BARE_ITERS=$(python3 -c "
import sys, json
bad = []
for line in sys.stdin:
    try:
        obj = json.loads(line.strip())
        k = obj.get('kernel','')
        if k and not __import__('re').match(r'^iter[0-9]{3}_[a-z][a-z0-9_]{1,15}$', k):
            bad.append(k)
    except: pass
for b in bad: print(b)
" < "$RAW" 2>/dev/null || true)
        if [[ -z "$BARE_ITERS" ]]; then
            result_pass "rounds.raw.jsonl: all kernel names are tagged (iter<NNN>_<tag>)"
        else
            COUNT=$(echo "$BARE_ITERS" | wc -l)
            result_fail "rounds.raw.jsonl: $COUNT kernel entries missing _tag: $(echo "$BARE_ITERS" | tr '\n' ' ')"
        fi
    else
        result_fail "rounds.raw.jsonl does not exist at $RAW"
    fi

    if [[ -f "$MD" ]]; then
        result_pass "rounds.md exists ($(wc -l < "$MD") lines)"
    else
        result_fail "rounds.md does not exist at $MD"
    fi

    echo ""
    echo "--- Log Files ($LOG_DIR) ---"

    IDEA_LOG="$LOG_DIR/idea-log.jsonl"
    TSV="$LOG_DIR/results.tsv"

    if [[ -f "$IDEA_LOG" ]]; then
        result_pass "idea-log.jsonl exists ($(wc -l < "$IDEA_LOG") lines)"
    else
        result_fail "idea-log.jsonl does not exist at $IDEA_LOG"
    fi

    if [[ -f "$TSV" ]]; then
        DATA_ROWS=$(tail -n +2 "$TSV" | grep -c . || true)
        result_pass "results.tsv exists ($DATA_ROWS data rows)"

        # Check that iter kernel column entries are all tagged (skip non-iter baseline names)
        BARE_TSV=$(tail -n +2 "$TSV" | awk -F'\t' '{print $2}' | \
            grep -E '^iter[0-9]{3}' | \
            grep -vE '^iter[0-9]{3}_[a-z][a-z0-9_]{1,15}$' | grep -v '^$' || true)
        if [[ -z "$BARE_TSV" ]]; then
            result_pass "results.tsv: all iter kernel entries are tagged"
        else
            COUNT=$(echo "$BARE_TSV" | wc -l)
            result_fail "results.tsv: $COUNT untagged iter kernel entries: $(echo "$BARE_TSV" | tr '\n' ' ')"
        fi
    else
        result_fail "results.tsv does not exist at $TSV"
    fi

    echo ""
    echo "--- ncu Artifacts ($PERF_DIR) ---"

    if [[ ! -d "$PERF_DIR" ]]; then
        result_warn "perf directory does not exist yet: $PERF_DIR"
    else
        NCU_CSV_TOTAL=0
        NCU_CSV_BAD=0
        NCU_REP_TOTAL=0
        ORPHAN_CSV=0

        for csv in "$PERF_DIR"/ncu_iter*.csv; do
            [[ -f "$csv" ]] || continue
            name=$(basename "$csv")
            NCU_CSV_TOTAL=$((NCU_CSV_TOTAL + 1))

            # Check filename format: ncu_iter<NNN>_<tag>.csv
            if ! echo "$name" | grep -qE '^ncu_iter[0-9]{3}_[a-z][a-z0-9_]{1,15}\.csv$'; then
                NCU_CSV_BAD=$((NCU_CSV_BAD + 1))
                echo "    BAD CSV NAME: $name  (expected ncu_iter<NNN>_<tag>.csv)"
            fi

            # Check that a corresponding .ncu-rep exists
            rep="${csv%.csv}.ncu-rep"
            if [[ ! -f "$rep" ]]; then
                ORPHAN_CSV=$((ORPHAN_CSV + 1))
                echo "    ORPHAN CSV (no .ncu-rep): $name"
            fi
        done

        for rep in "$PERF_DIR"/ncu_iter*.ncu-rep; do
            [[ -f "$rep" ]] || continue
            NCU_REP_TOTAL=$((NCU_REP_TOTAL + 1))
        done

        if [[ "$NCU_CSV_TOTAL" -eq 0 ]]; then
            result_warn "No ncu CSVs found in $PERF_DIR (expected after profiling)"
        elif [[ "$NCU_CSV_BAD" -eq 0 ]]; then
            result_pass "All $NCU_CSV_TOTAL ncu CSVs have correct naming (ncu_iter<NNN>_<tag>.csv)"
        else
            result_fail "$NCU_CSV_BAD of $NCU_CSV_TOTAL ncu CSVs have bad names"
        fi

        if [[ "$NCU_REP_TOTAL" -gt 0 ]]; then
            result_pass "$NCU_REP_TOTAL .ncu-rep report files present"
        else
            result_warn "No .ncu-rep files found (expected after ncu --export)"
        fi

        if [[ "$ORPHAN_CSV" -eq 0 && "$NCU_CSV_TOTAL" -gt 0 ]]; then
            result_pass "All ncu CSVs have matching .ncu-rep counterparts"
        elif [[ "$ORPHAN_CSV" -gt 0 ]]; then
            result_fail "$ORPHAN_CSV ncu CSVs have no matching .ncu-rep file"
        fi

        # Cross-check: every iter kernel in results.tsv should have a ncu CSV
        # (skip non-iter baseline entries like cuBLAS_GemmEx_bf16fp32)
        if [[ -f "$TSV" ]]; then
            MISSING_CSV=0
            while IFS=$'\t' read -r iter kernel rest; do
                [[ "$iter" == "iter" ]] && continue  # skip header
                [[ -z "$kernel" ]] && continue
                # Only enforce ncu CSV presence for iter-prefixed kernels
                [[ "$kernel" =~ ^iter[0-9]{3} ]] || continue
                expected_csv="$PERF_DIR/ncu_${kernel}.csv"
                if [[ ! -f "$expected_csv" ]]; then
                    MISSING_CSV=$((MISSING_CSV + 1))
                    echo "    NO ncu CSV for: $kernel  (expected $expected_csv)"
                fi
            done < "$TSV"

            if [[ "$MISSING_CSV" -eq 0 && "$NCU_CSV_TOTAL" -gt 0 ]]; then
                result_pass "Every iter results.tsv entry has a corresponding ncu CSV"
            elif [[ "$MISSING_CSV" -gt 0 ]]; then
                result_fail "$MISSING_CSV iter kernels in results.tsv are missing ncu CSVs"
            fi
        fi
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PASS: $TOTAL_PASS   FAIL: $TOTAL_FAIL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$TOTAL_FAIL" -gt 0 ]]; then
    echo "RESULT: FAILED — $TOTAL_FAIL checks did not pass"
    exit 1
fi
echo "RESULT: PASSED"
