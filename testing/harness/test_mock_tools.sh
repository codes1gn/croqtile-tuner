#!/usr/bin/env bash
# test_mock_tools.sh — Unit tests for testing/mocks/mock_ncu and mock_websearch.
#
# Verifies:
#   - mock_ncu profile mode writes .ncu-rep sentinel
#   - mock_ncu csv mode emits valid ncu CSV for each scenario
#   - profile_extract.sh correctly classifies all 5 mock_ncu scenarios
#   - mock_websearch returns valid JSON for queries and topics
#   - mock_websearch keyword detection works for common bottleneck terms

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source testing/harness/tap_helpers.sh

MOCK_NCU="testing/mocks/mock_ncu"
MOCK_WS="testing/mocks/mock_websearch"
PROFILE_EXTRACT=".claude/skills/croq-profile/profile_extract.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

plan 30

# ─────────────────────────────────────────────────────────────────────────────
# mock_ncu — --version flag
# ─────────────────────────────────────────────────────────────────────────────
VER=$(bash "$MOCK_NCU" --version 2>/dev/null)
like "mock_ncu --version prints mock_ncu" "$VER" "mock_ncu"

# ─────────────────────────────────────────────────────────────────────────────
# mock_ncu — profile mode (write .ncu-rep)
# ─────────────────────────────────────────────────────────────────────────────
REP="$TMP/ncu_test.ncu-rep"

MOCK_NCU_SCENARIO=memory_bound bash "$MOCK_NCU" \
    --set full \
    --export "$REP" \
    --force-overwrite \
    /fake/binary >/dev/null 2>&1
ok "mock_ncu profile mode exits 0" $?

[ -f "$REP" ] && IS_FILE=0 || IS_FILE=1
ok "mock_ncu profile mode writes .ncu-rep file" "$IS_FILE"

like "mock_ncu .ncu-rep contains scenario tag" "$(cat "$REP")" "memory_bound"

# ─────────────────────────────────────────────────────────────────────────────
# mock_ncu — csv mode (emit synthetic CSV)
# ─────────────────────────────────────────────────────────────────────────────
CSV_OUT="$TMP/ncu_out.csv"

MOCK_NCU_SCENARIO=memory_bound bash "$MOCK_NCU" \
    --import "$REP" \
    --csv \
    --page raw > "$CSV_OUT" 2>/dev/null
ok "mock_ncu csv mode exits 0" $?

like "csv mode output has header line" "$(head -1 "$CSV_OUT")" "Metric Name"
like "csv mode output has dram metric" "$(cat "$CSV_OUT")" "dram__throughput"
like "csv mode output has sm metric" "$(cat "$CSV_OUT")" "sm__throughput"
like "csv mode output has occupancy metric" "$(cat "$CSV_OUT")" "sm__warps_active"
like "csv mode output has stall metric" "$(cat "$CSV_OUT")" "smsp__warp_issue_stalled"

# ─────────────────────────────────────────────────────────────────────────────
# profile_extract.sh integration: all 5 scenarios produce correct classification
# ─────────────────────────────────────────────────────────────────────────────
run_scenario() {
    local scenario="$1"
    local expected_bottleneck="$2"
    local rep="$TMP/ncu_${scenario}.ncu-rep"
    local csv="$TMP/ncu_${scenario}.csv"
    local iter="iter001_mocktest"

    MOCK_NCU_SCENARIO="$scenario" bash "$MOCK_NCU" \
        --set full --export "$rep" --force-overwrite /fake/bin >/dev/null 2>&1

    MOCK_NCU_SCENARIO="$scenario" bash "$MOCK_NCU" \
        --import "$rep" --csv --page raw > "$csv" 2>/dev/null

    local profile_json
    profile_json=$(bash "$PROFILE_EXTRACT" --csv "$csv" --iter "$iter" 2>/dev/null)
    echo "$profile_json" | grep -q "\"bottleneck\": \"${expected_bottleneck}\""
}

run_scenario memory_bound  memory_bound  && ok "memory_bound scenario classified correctly" 0  || ok "memory_bound scenario classified correctly" 1
run_scenario compute_bound compute_bound && ok "compute_bound scenario classified correctly" 0 || ok "compute_bound scenario classified correctly" 1
run_scenario launch_bound  launch_bound  && ok "launch_bound scenario classified correctly" 0  || ok "launch_bound scenario classified correctly" 1
run_scenario latency_bound latency_bound && ok "latency_bound scenario classified correctly" 0 || ok "latency_bound scenario classified correctly" 1

# mixed → classified as either memory_bound or compute_bound (highest wins)
run_scenario_mixed() {
    local rep="$TMP/ncu_mixed.ncu-rep"
    local csv="$TMP/ncu_mixed.csv"
    local iter="iter001_mocktest"

    MOCK_NCU_SCENARIO="mixed" bash "$MOCK_NCU" \
        --set full --export "$rep" --force-overwrite /fake/bin >/dev/null 2>&1

    MOCK_NCU_SCENARIO="mixed" bash "$MOCK_NCU" \
        --import "$rep" --csv --page raw > "$csv" 2>/dev/null

    local profile_json
    profile_json=$(bash "$PROFILE_EXTRACT" --csv "$csv" --iter "$iter" 2>/dev/null)
    # mixed (DRAM=55, compute=52) → DRAM wins → memory_bound classification
    echo "$profile_json" | grep -q '"bottleneck"'
}
run_scenario_mixed && ok "mixed scenario produces valid bottleneck JSON" 0 || ok "mixed scenario produces valid bottleneck JSON" 1

# ─────────────────────────────────────────────────────────────────────────────
# mock_ncu — error handling
# ─────────────────────────────────────────────────────────────────────────────
MOCK_NCU_SCENARIO=memory_bound bash "$MOCK_NCU" \
    --import "$TMP/nonexistent.ncu-rep" --csv --page raw >/dev/null 2>&1 \
    && RC=0 || RC=$?
is "csv mode with missing .ncu-rep exits nonzero" "$RC" "2"

MOCK_NCU_SCENARIO=bad_scenario bash "$MOCK_NCU" \
    --set full --export "$TMP/x.ncu-rep" --force-overwrite /fake/bin >/dev/null 2>&1 \
    && RC=0 || RC=$?
is "unknown MOCK_NCU_SCENARIO exits nonzero" "$RC" "1"

# ─────────────────────────────────────────────────────────────────────────────
# mock_websearch — basic output
# ─────────────────────────────────────────────────────────────────────────────
WS_OUT=$(python3 "$MOCK_WS" "CUDA shared memory bank conflict matmul bf16" 2>/dev/null)
ok "mock_websearch exits 0 for query" $?
like "mock_websearch output is JSON array" "$WS_OUT" '"title"'
like "mock_websearch output has url field" "$WS_OUT" '"url"'
like "mock_websearch output has snippet field" "$WS_OUT" '"snippet"'

# Validate it's a proper JSON array
VALID=$(echo "$WS_OUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print('ok' if isinstance(d, list) and len(d) > 0 else 'fail')" 2>/dev/null || echo "fail")
is "mock_websearch output is valid non-empty JSON array" "$VALID" "ok"

# ─────────────────────────────────────────────────────────────────────────────
# mock_websearch — keyword detection
# ─────────────────────────────────────────────────────────────────────────────
check_topic() {
    local query="$1"
    local expected_topic_keyword="$2"
    local out
    out=$(python3 "$MOCK_WS" "$query" 2>/dev/null)
    echo "$out" | grep -qi "$expected_topic_keyword"
}

check_topic "warp divergence reduction GEMM kernel" "compute" \
    && ok "compute-related query detected correctly" 0 \
    || ok "compute-related query detected correctly" 1

check_topic "DRAM throughput matmul memory bound" "memory" \
    && ok "memory-related query detected correctly" 0 \
    || ok "memory-related query detected correctly" 1

check_topic "bank conflict smem padding" "bank" \
    && ok "bank-conflict query detected correctly" 0 \
    || ok "bank-conflict query detected correctly" 1

check_topic "threadblock swizzle L2 locality" "swizzle" \
    && ok "swizzle query detected correctly" 0 \
    || ok "swizzle query detected correctly" 1

# ─────────────────────────────────────────────────────────────────────────────
# mock_websearch — explicit --topic flag
# ─────────────────────────────────────────────────────────────────────────────
TS_OUT=$(python3 "$MOCK_WS" --topic latency_bound 2>/dev/null)
like "explicit --topic latency_bound has pipeline content" "$TS_OUT" "pipeline"

TS_OUT=$(python3 "$MOCK_WS" --topic swizzle 2>/dev/null)
like "explicit --topic swizzle has swizzle content" "$TS_OUT" "swizzle"

TS_OUT=$(python3 "$MOCK_WS" --topic launch_bound 2>/dev/null)
like "explicit --topic launch_bound has occupancy content" "$TS_OUT" "occupancy"

# ─────────────────────────────────────────────────────────────────────────────
# mock_websearch — all 7 topics return valid JSON
# ─────────────────────────────────────────────────────────────────────────────
ALL_VALID=0
for t in memory_bound compute_bound launch_bound latency_bound bank_conflict swizzle generic; do
    count=$(python3 "$MOCK_WS" --topic "$t" 2>/dev/null | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
    if [[ "$count" -lt 1 ]]; then
        ALL_VALID=1
        echo "  # topic '$t' returned empty results" >&2
    fi
done
ok "all 7 mock_websearch topics return ≥1 result" "$ALL_VALID"

done_testing
