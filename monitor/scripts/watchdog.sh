#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
LOG_DIR="$ROOT_DIR/tuning/logs/system-watchdog"
WATCHDOG_LOG="$LOG_DIR/watchdog.log"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
NODE_BIN_DIR="/home/albert/local/node-v20.11.1-linux-x64/bin"
BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8642"
FRONTEND_HOST="127.0.0.1"
FRONTEND_PORT="5173"
HEALTH_URL="http://${BACKEND_HOST}:${BACKEND_PORT}/api/health"
HEALTH_FAILURE_THRESHOLD=2

mkdir -p "$LOG_DIR"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$WATCHDOG_LOG"
}

repo_pids() {
  local pattern="$1"
  pgrep -af "$pattern" | grep "$ROOT_DIR" || true
}

repo_pid_count() {
  local pattern="$1"
  local lines
  lines="$(repo_pids "$pattern")"
  if [[ -z "$lines" ]]; then
    echo 0
  else
    printf '%s\n' "$lines" | wc -l
  fi
}

stop_backend() {
  local pids
  pids="$(repo_pids 'uvicorn app.main:app')"
  if [[ -n "$pids" ]]; then
    printf '%s\n' "$pids" | awk '{print $1}' | xargs -r kill
    sleep 1
  fi
}

start_backend() {
  log "starting backend"
  (
    cd "$BACKEND_DIR"
    source .venv/bin/activate
    exec uvicorn app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT"
  ) >>"$BACKEND_LOG" 2>&1 &
}

stop_frontend() {
  local pids
  pids="$(repo_pids 'vite --host 127.0.0.1 --port 5173')"
  if [[ -z "$pids" ]]; then
    pids="$(repo_pids '/frontend/node_modules/.bin/vite')"
  fi
  if [[ -n "$pids" ]]; then
    printf '%s\n' "$pids" | awk '{print $1}' | xargs -r kill
    sleep 1
  fi
}

start_frontend() {
  log "starting frontend"
  (
    export PATH="$NODE_BIN_DIR:$PATH"
    cd "$FRONTEND_DIR"
    exec npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
  ) >>"$FRONTEND_LOG" 2>&1 &
}

restart_backend_for_recovery() {
  log "restarting backend for recovery"
  stop_backend
  start_backend
  sleep 3
}

retry_latest_transient_failed_task() {
  python3 - "$BACKEND_HOST" "$BACKEND_PORT" <<'PY'
import json
import sys
import urllib.error
import urllib.request

host = sys.argv[1]
port = sys.argv[2]
base = f"http://{host}:{port}"

try:
  with urllib.request.urlopen(f"{base}/api/tasks?status=failed", timeout=10) as response:
    tasks = json.load(response)
except Exception:
  sys.exit(1)

if not tasks:
  sys.exit(2)

tasks.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)
task = tasks[0]
error_message = task.get("error_message") or ""
if "opencode exited without completing or persisting tuning progress" not in error_message:
  sys.exit(3)

request = urllib.request.Request(
  f"{base}/api/tasks/{task['id']}/retry",
  method="POST",
)
with urllib.request.urlopen(request, timeout=10) as response:
  retried = json.load(response)
print(retried["id"])
PY
}

snapshot_json() {
  python3 - "$1" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=10) as response:
        print(response.read().decode())
except Exception:
    sys.exit(1)
PY
}

while true; do
  health_failures="${health_failures:-0}"
  backend_json=""
  if backend_json="$(snapshot_json "$HEALTH_URL" 2>/dev/null)"; then
    backend_ok=1
    health_failures=0
  else
    backend_ok=0
    health_failures=$((health_failures + 1))
  fi

  if [[ "$backend_ok" -eq 0 ]]; then
    log "backend health check failed count=$health_failures"
    if [[ "$health_failures" -lt "$HEALTH_FAILURE_THRESHOLD" ]]; then
      sleep 30
      continue
    fi
    if [[ "$(repo_pid_count 'uvicorn app.main:app')" -eq 0 ]]; then
      start_backend
      sleep 3
    else
      restart_backend_for_recovery
    fi
    health_failures=0
    sleep 30
    continue
  fi

  frontend_count="$(repo_pid_count '/frontend/node_modules/.bin/vite')"
  if [[ "$frontend_count" -eq 0 ]]; then
    log "frontend missing"
    start_frontend
    sleep 3
  fi

  read -r active_task_id running_count pending_count < <(
    HEALTH_JSON="$backend_json" python3 - <<'PY'
import json
import os

data = json.loads(os.environ['HEALTH_JSON'])
active = data.get('active_task_id')
counts = data.get('task_counts', {})
print(active if active is not None else 'none', counts.get('running', 0), counts.get('pending', 0))
PY
  )

  opencode_count="$(repo_pid_count 'opencode run --print-logs')"

  if [[ "$active_task_id" != "none" && "$opencode_count" -ne 1 ]]; then
    log "worker mismatch: active_task_id=$active_task_id opencode_count=$opencode_count"
    restart_backend_for_recovery
    sleep 30
    continue
  fi

  if [[ "$active_task_id" == "none" && "$running_count" -gt 0 ]]; then
    log "stale running task detected without active scheduler worker"
    restart_backend_for_recovery
    sleep 30
    continue
  fi

  if [[ "$active_task_id" == "none" && "$pending_count" -gt 0 && "$opencode_count" -gt 0 ]]; then
    log "pending tasks exist while orphan worker is running"
    restart_backend_for_recovery
    sleep 30
    continue
  fi

  if [[ "$active_task_id" == "none" && "$running_count" -eq 0 && "$pending_count" -eq 0 && "$opencode_count" -eq 0 ]]; then
    retried_task_id=""
    if retried_task_id="$(retry_latest_transient_failed_task 2>/dev/null)"; then
      log "auto-retried transient failed task as new task_id=$retried_task_id"
      sleep 30
      continue
    fi
  fi

  log "healthy active_task_id=$active_task_id running=$running_count pending=$pending_count opencode_count=$opencode_count"
  sleep 30
done