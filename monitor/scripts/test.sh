#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== CroqTuner Test Suite ==="
echo ""

echo "[1/2] Backend tests (pytest)..."
cd "$PROJECT_DIR/backend"
source .venv/bin/activate
python -m pytest tests/ -v
echo ""

echo "[2/2] Frontend tests (vitest)..."
cd "$PROJECT_DIR/frontend"
npm test
echo ""

echo "=== All tests passed ==="
