#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MONITOR_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$MONITOR_DIR/backend"
FRONTEND_DIR="$MONITOR_DIR/frontend"

TMP_ROOT="$(mktemp -d -t croqtuner-test-XXXXXX)"
trap 'rm -rf "$TMP_ROOT"' EXIT

# Hard-isolate tests from real tuning artifacts and DB.
export CROQTUNER_TUNING_DIR="$TMP_ROOT/tuning"
export CROQTUNER_DB_PATH="$TMP_ROOT/monitor.db"
export CROQTUNER_MOCK_MODE=true
mkdir -p "$CROQTUNER_TUNING_DIR"

echo "=== CroqTuner Safe Test Suite ==="
echo "Isolated tuning dir: $CROQTUNER_TUNING_DIR"
echo "Isolated DB path:    $CROQTUNER_DB_PATH"
echo ""

echo "[1/2] Backend tests (pytest)..."
cd "$BACKEND_DIR"
source .venv/bin/activate
python -m pytest tests -q
echo ""

echo "[2/2] Frontend tests (vitest)..."
cd "$FRONTEND_DIR"
./node_modules/.bin/vitest run
echo ""

echo "=== Safe tests passed ==="
