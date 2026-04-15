#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$MONITOR_DIR")"

cd "$MONITOR_DIR"

# Load environment if exists
if [[ -f .env ]]; then
    source .env
fi

# Set defaults
export CROQTUNER_TUNING_DIR="${CROQTUNER_TUNING_DIR:-$PROJECT_DIR/tuning}"
export CROQTUNER_SKILLS_DIR="${CROQTUNER_SKILLS_DIR:-$PROJECT_DIR/.claude/skills}"
export CROQTUNER_PROJECT_DIR="${CROQTUNER_PROJECT_DIR:-$PROJECT_DIR}"

# Ensure data directory exists
mkdir -p backend/data

echo "=== CroqTuner Monitor ==="
echo "Project dir: $PROJECT_DIR"
echo "Tuning dir:  $CROQTUNER_TUNING_DIR"
echo "Monitor dir: $MONITOR_DIR"
echo ""

# Start backend
echo "Starting backend on port ${CROQTUNER_PORT:-8642}..."
cd backend
if [[ ! -d .venv ]]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

uvicorn app.main:app \
    --host "${CROQTUNER_HOST:-0.0.0.0}" \
    --port "${CROQTUNER_PORT:-8642}" \
    --reload &
BACKEND_PID=$!

cd "$MONITOR_DIR"

# Start frontend
echo "Starting frontend on port 5173..."
cd frontend
if [[ ! -d node_modules ]]; then
    echo "Installing npm dependencies..."
    npm install
fi
npm run dev &
FRONTEND_PID=$!

cd "$MONITOR_DIR"

echo ""
echo "=== Services Started ==="
echo "Backend:  http://localhost:${CROQTUNER_PORT:-8642}"
echo "Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop"

# Handle cleanup
cleanup() {
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

wait
