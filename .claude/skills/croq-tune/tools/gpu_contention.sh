#!/usr/bin/env bash
# gpu_contention.sh — Identify and optionally kill processes causing GPU contention.
#
# USAGE:
#   bash gpu_contention.sh                     # scan and report (dry run)
#   bash gpu_contention.sh --kill              # kill foreign GPU processes (spares our own)
#   bash gpu_contention.sh --kill --except <PID>  # kill all except one PID
#   bash gpu_contention.sh --threshold 15      # custom idle threshold (default 15%)
#   bash gpu_contention.sh --json              # machine-readable JSON only (no banner)
#   bash gpu_contention.sh --kill-ours         # also kill croq-tune owned processes
#
# EXIT CODES:
#   0 — GPU is idle (no contention) or all contenders killed successfully
#   1 — GPU contention detected but not killed (dry run)
#   2 — nvidia-smi not available
#   3 — some kills failed
#
# OUTPUT (stdout):
#   JSON: { "idle": bool, "gpu_util": N, "mem_util": N, "contenders": [...], "killed": [...], "failed": [...] }
#
# PROCESS CLASSIFICATION:
#   ours    — processes owned by the croq-tune harness: ncu, choreo, build_iter compiled
#             binaries living under the tuning workspace, cublas_baseline, etc.
#             These are SPARED by default. Use --kill-ours to include them.
#   foreign — everything else on the GPU (vllm, jupyter, other ML workloads).
#             These are always killed when --kill is passed.
#
# NOTES:
#   - No password is embedded. sudo must already be trusted for the current user.
#     To configure: run `sudo visudo` and add `<user> ALL=(ALL) NOPASSWD: /bin/kill`
#     Or simply ensure the user has a cached sudo token (run `sudo -v` first).
#   - Processes are identified by: pid, name, command line, owner user, gpu_mem, runtime.
#   - Self (this script's PID and bash ancestors) are always excluded from the kill list.
#   - Safe to run before any ncu/benchmark invocation to guarantee a clean GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/activity_trace.sh" 2>/dev/null || true

# ── argument parsing ─────────────────────────────────────────────────────────
KILL=false
KILL_OURS=false
EXCEPT_PID=""
UTIL_THRESHOLD=15
JSON_ONLY=false

# Resolve workspace root (two levels up from tools/)
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kill)        KILL=true; shift ;;
    --kill-ours)   KILL_OURS=true; shift ;;
    --except)      EXCEPT_PID="$2"; shift 2 ;;
    --threshold)   UTIL_THRESHOLD="$2"; shift 2 ;;
    --json)        JSON_ONLY=true; shift ;;
    --help|-h)
      # Print the leading comment block only (stop at first non-comment/non-blank line)
      awk 'NR==1 && /^#!/{next} /^#/{sub(/^# ?/,""); print; next} /^$/{print; next} {exit}' "$0"
      exit 0
      ;;
    *)
      echo "[gpu_contention] ERROR: unknown argument: $1" >&2
      echo "[SUGGESTION] Valid args: --kill  --kill-ours  --except <pid>  --threshold <util%>  --json" >&2
      exit 1
      ;;
  esac
done

# ── nvidia-smi guard ─────────────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
  echo '{"idle":false,"error":"nvidia-smi not found"}' 
  echo "[gpu_contention] ERROR: nvidia-smi not in PATH. Cannot detect GPU contention." >&2
  exit 2
fi

MY_PID=$$

# ── step 1: GPU utilisation ───────────────────────────────────────────────────
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu   --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ') || GPU_UTIL=0
MEM_UTIL=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ') || MEM_UTIL=0
GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used   --format=csv,noheader        2>/dev/null | head -1 | tr -d ' ') || GPU_MEM_USED="?"
GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader        2>/dev/null | head -1 | tr -d ' ') || GPU_MEM_TOTAL="?"
GPU_NAME=$(nvidia-smi --query-gpu=name              --format=csv,noheader        2>/dev/null | head -1)               || GPU_NAME="unknown"
GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu   --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ') || GPU_TEMP="?"
GPU_PWR=$(nvidia-smi --query-gpu=power.draw         --format=csv,noheader        2>/dev/null | head -1 | tr -d ' ')  || GPU_PWR="?"

IDLE=false
if [[ "$GPU_UTIL" -lt "$UTIL_THRESHOLD" ]]; then
  IDLE=true
fi

# ── step 2: compute-app processes ────────────────────────────────────────────
RAW_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
              --format=csv,noheader 2>/dev/null || echo "")

# Build contenders array as JSON via python3
# RAW_PROCS is written to a temp file to avoid heredoc-inside-subshell stdin conflicts
_PROCS_TMP=$(mktemp /tmp/gpu_contention_procs.XXXXXX)
echo "$RAW_PROCS" > "$_PROCS_TMP"
trap 'rm -f "$_PROCS_TMP"' EXIT

CONTENDERS_JSON=$(python3 - "$WORKSPACE_ROOT" "$MY_PID" "$EXCEPT_PID" "$UTIL_THRESHOLD" "$_PROCS_TMP" <<'PYEOF'
import sys, json, os, re, time, pwd

workspace_root = sys.argv[1].rstrip("/")
my_pid         = int(sys.argv[2])
except_pid_str = sys.argv[3]
except_pid     = int(except_pid_str) if except_pid_str.strip() else None
threshold      = int(sys.argv[4])
procs_file     = sys.argv[5]

# ── Collect parent PIDs of this script so we never kill our own shell tree ──
def ancestors(pid):
    result = set()
    try:
        p = pid
        for _ in range(20):
            with open(f"/proc/{p}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        p = int(line.split()[1])
                        if p <= 1:
                            return result
                        result.add(p)
                        break
    except Exception:
        pass
    return result

safe_pids = {my_pid} | ancestors(my_pid)

# ── Known croq-tune harness process signatures ───────────────────────────────
# A process is "ours" if ANY of the following is true:
#   1. Its cmdline contains a path inside workspace_root
#   2. Its cwd (from /proc/<pid>/cwd) is under workspace_root
#   3. Its binary name matches known harness executables
#   4. Its cmdline matches harness-specific patterns
OURS_CMD_PATTERNS = [
    re.compile(r"ncu\b.*--set\s+full"),        # ncu profiling launched by ncu_profile.sh
    re.compile(r"ncu\b.*--import"),            # ncu report export
    re.compile(r"\bchoreo\b"),                 # choreo compiler
    re.compile(r"\bnvcc\b"),                   # nvcc compile step (build_iter.sh)
    re.compile(r"cublas_baseline"),            # cublas baseline runner
    re.compile(r"build_iter"),                 # explicit build_iter invocation
    re.compile(r"/iter\d+_\w+"),              # compiled kernel binaries (iter021_warp2x4)
    re.compile(r"croq[-_]tune"),               # anything explicitly named croq-tune
    re.compile(r"\.claude/skills/croq-tune"),  # any tool from our toolchain
]
OURS_BINARY_NAMES = {"ncu", "choreo", "nvcc", "nsight-compute"}

def is_ours(pid, name, cmd, cwd):
    # Check binary name
    if name.lower() in OURS_BINARY_NAMES:
        return True
    # Check cwd under workspace
    if cwd and cwd.startswith(workspace_root):
        return True
    # Check cmdline contains workspace path
    if workspace_root in cmd:
        return True
    # Check regex patterns
    for pat in OURS_CMD_PATTERNS:
        if pat.search(cmd):
            return True
    return False

# Read raw process list from temp file
with open(procs_file) as f:
    raw = f.read()

contenders = []
for line in raw.strip().splitlines():
    parts = [p.strip() for p in line.split(',')]
    if len(parts) < 3:
        continue
    try:
        pid = int(parts[0])
    except ValueError:
        continue
    name = parts[1]
    mem  = parts[2]

    if pid in safe_pids:
        continue  # never include ourselves or our shell ancestors

    # ── enrich from /proc ────────────────────────────────────────────────
    owner = "unknown"
    cmd   = "unknown"
    cwd   = ""
    elapsed = 0

    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw_cmd = f.read().replace(b"\x00", b" ").decode(errors="replace").strip()
            cmd = raw_cmd[:200] if raw_cmd else "unknown"
    except Exception:
        pass

    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except Exception:
        pass

    try:
        st = os.stat(f"/proc/{pid}")
        owner = pwd.getpwuid(st.st_uid).pw_name
    except Exception:
        pass

    try:
        hz = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", 2)) or 100
        with open(f"/proc/{pid}/stat") as sf:
            fields = sf.read().split()
        stat_start = int(fields[21])
        boot_s = 0
        with open("/proc/stat") as pf:
            for ln in pf:
                if ln.startswith("btime"):
                    boot_s = int(ln.split()[1])
                    break
        elapsed = max(0, int(time.time()) - boot_s - stat_start // hz)
    except Exception:
        elapsed = 0

    ours   = is_ours(pid, name, cmd, cwd)
    ex     = (pid == except_pid)

    contenders.append({
        "pid":       pid,
        "name":      name,
        "gpu_mem":   mem,
        "owner":     owner,
        "cmd":       cmd,
        "cwd":       cwd,
        "runtime_s": elapsed,
        "is_ours":   ours,
        "is_except": ex,
    })

print(json.dumps(contenders))
PYEOF
)

CONTENDER_COUNT=$(python3 -c "import sys,json; print(len(json.loads(sys.argv[1])))" "$CONTENDERS_JSON" 2>/dev/null || echo 0)
# killable = not our process (unless --kill-ours) AND not excepted
KILLABLE_COUNT=$(python3 -c "
import sys, json
kill_ours = sys.argv[2] == 'true'
data = json.loads(sys.argv[1])
print(sum(1 for c in data if not c['is_except'] and (kill_ours or not c['is_ours'])))
" "$CONTENDERS_JSON" "$KILL_OURS" 2>/dev/null || echo 0)

# ── step 3: banner (unless --json) ───────────────────────────────────────────
if [[ "$JSON_ONLY" == "false" ]]; then
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  gpu_contention.sh — GPU Contention Report                  ║"
  echo "╚══════════════════════════════════════════════════════════════╝"
  echo ""
  echo "  GPU    : ${GPU_NAME}"
  echo "  Util   : ${GPU_UTIL}% compute  ${MEM_UTIL}% memory"
  echo "  VRAM   : ${GPU_MEM_USED} / ${GPU_MEM_TOTAL}"
  echo "  Temp   : ${GPU_TEMP}°C   Power: ${GPU_PWR}"
  echo "  Idle   : ${IDLE}  (threshold=${UTIL_THRESHOLD}%)"
  echo ""

  if [[ "$CONTENDER_COUNT" -eq 0 ]]; then
    echo "  No GPU compute processes found."
  else
    printf "  %-8s %-12s %-12s %-8s %-8s %-8s  %s\n" "PID" "OWNER" "GPU_MEM" "RUNTIME" "OURS?" "EXCEPT?" "COMMAND"
    printf "  %-8s %-12s %-12s %-8s %-8s %-8s  %s\n" "---" "-----" "-------" "-------" "-----" "-------" "-------"
    python3 -c "
import json, sys
kill_ours = sys.argv[2] == 'true'
data = json.loads(sys.argv[1])
for c in data:
    mins = c['runtime_s'] // 60
    secs = c['runtime_s'] % 60
    runtime = f\"{mins}m{secs:02d}s\"
    ours = 'yes' if c['is_ours'] else ''
    ex   = 'yes' if c['is_except'] else ''
    will_kill = not c['is_except'] and (kill_ours or not c['is_ours'])
    fate = '[KILL]' if will_kill else '[spare]'
    cmd_short = c['cmd'][:55]
    print(f\"  {c['pid']:<8} {c['owner']:<12} {c['gpu_mem']:<12} {runtime:<8} {ours:<8} {ex:<8}  {fate} {cmd_short}\")
" "$CONTENDERS_JSON" "$KILL_OURS"
    echo ""
    echo "  [spare] = croq-tune harness process (safe to keep); [KILL] = foreign GPU consumer"
  fi
  echo ""
fi

# ── step 4: kill ─────────────────────────────────────────────────────────────
KILLED_JSON="[]"
FAILED_JSON="[]"

if [[ "$KILL" == "true" && "$KILLABLE_COUNT" -gt 0 ]]; then
  if [[ "$JSON_ONLY" == "false" ]]; then
    echo "  [--kill] Terminating ${KILLABLE_COUNT} GPU process(es)..."
    echo ""
  fi

  KILLED=()
  FAILED=()

  while IFS= read -r line; do
    pid=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['pid'])")
    is_except=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['is_except'])")
    is_ours_proc=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['is_ours'])")
    name=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['name'])")

    if [[ "$is_except" == "True" ]]; then
      [[ "$JSON_ONLY" == "false" ]] && echo "  [skip]  PID=${pid} (${name}) — excluded by --except"
      continue
    fi

    if [[ "$is_ours_proc" == "True" && "$KILL_OURS" == "false" ]]; then
      [[ "$JSON_ONLY" == "false" ]] && echo "  [spare] PID=${pid} (${name}) — croq-tune harness process (use --kill-ours to override)"
      continue
    fi

    # Try SIGTERM first, then SIGKILL, with sudo if needed
    if kill -15 "$pid" 2>/dev/null; then
      [[ "$JSON_ONLY" == "false" ]] && echo "  [kill]  PID=${pid} (${name}) — SIGTERM sent"
      sleep 1
      # Escalate to SIGKILL if still alive
      if kill -0 "$pid" 2>/dev/null; then
        if sudo kill -9 "$pid" 2>/dev/null; then
          [[ "$JSON_ONLY" == "false" ]] && echo "  [kill]  PID=${pid} (${name}) — SIGKILL (escalated)"
          KILLED+=("$pid")
        else
          [[ "$JSON_ONLY" == "false" ]] && echo "  [FAIL]  PID=${pid} (${name}) — SIGKILL failed" >&2
          FAILED+=("$pid")
        fi
      else
        KILLED+=("$pid")
      fi
    elif sudo kill -9 "$pid" 2>/dev/null; then
      [[ "$JSON_ONLY" == "false" ]] && echo "  [kill]  PID=${pid} (${name}) — SIGKILL via sudo"
      KILLED+=("$pid")
    else
      [[ "$JSON_ONLY" == "false" ]] && echo "  [FAIL]  PID=${pid} (${name}) — could not kill" >&2
      FAILED+=("$pid")
    fi

  done < <(python3 -c "
import json, sys
data = json.loads(sys.argv[1])
for c in data:
    print(json.dumps(c))
" "$CONTENDERS_JSON")

  # Wait a moment for GPU to quiesce, then re-check utilisation
  sleep 2
  GPU_UTIL_AFTER=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ') || GPU_UTIL_AFTER="?"

  if [[ "$JSON_ONLY" == "false" ]]; then
    echo ""
    echo "  GPU util after kill: ${GPU_UTIL_AFTER}%"
    echo ""
  fi

  # Build JSON arrays for output
  if [[ ${#KILLED[@]} -gt 0 ]]; then
    KILLED_JSON=$(printf '%s\n' "${KILLED[@]}" | python3 -c "import sys,json; print(json.dumps([int(l.strip()) for l in sys.stdin]))")
  fi
  if [[ ${#FAILED[@]} -gt 0 ]]; then
    FAILED_JSON=$(printf '%s\n' "${FAILED[@]}" | python3 -c "import sys,json; print(json.dumps([int(l.strip()) for l in sys.stdin]))")
  fi

  trace_event "gpu_contention" "kill: killed=${#KILLED[@]} failed=${#FAILED[@]} gpu_util_before=${GPU_UTIL} after=${GPU_UTIL_AFTER:-?}" "info"

elif [[ "$KILL" == "false" && "$KILLABLE_COUNT" -gt 0 && "$JSON_ONLY" == "false" ]]; then
  echo "  [dry-run] ${KILLABLE_COUNT} process(es) would be killed. Pass --kill to terminate them."
  echo ""
fi

# ── step 5: emit final JSON ───────────────────────────────────────────────────
python3 - \
  "$CONTENDERS_JSON" "$KILLED_JSON" "$FAILED_JSON" \
  "$IDLE" "$GPU_UTIL" "$MEM_UTIL" \
  "${GPU_MEM_USED} / ${GPU_MEM_TOTAL}" "$GPU_NAME" \
  "$GPU_TEMP" "$GPU_PWR" "$UTIL_THRESHOLD" "$KILL" "$KILL_OURS" \
<<'PYEOF3'
import json, sys

contenders  = json.loads(sys.argv[1])
killed      = json.loads(sys.argv[2])
failed      = json.loads(sys.argv[3])
idle        = sys.argv[4] == "true"
gpu_util    = int(sys.argv[5])
mem_util    = int(sys.argv[6])
gpu_mem     = sys.argv[7]
gpu_name    = sys.argv[8]
temp_c      = sys.argv[9]
power       = sys.argv[10]
threshold   = int(sys.argv[11])
dry_run     = sys.argv[12] != "true"
kill_ours   = sys.argv[13] == "true"

# Annotate each contender with what would happen under current flags
for c in contenders:
    c["would_kill"] = (
        not c["is_except"]
        and (kill_ours or not c["is_ours"])
        and not dry_run
    )

result = {
    "idle":       idle,
    "gpu_util":   gpu_util,
    "mem_util":   mem_util,
    "gpu_mem":    gpu_mem,
    "gpu_name":   gpu_name,
    "temp_c":     temp_c,
    "power":      power,
    "threshold":  threshold,
    "kill_ours":  kill_ours,
    "contenders": contenders,
    "killed":     killed,
    "failed":     failed,
    "dry_run":    dry_run,
}
print(json.dumps(result, indent=2))
PYEOF3

# ── step 6: exit code ─────────────────────────────────────────────────────────
if [[ "$IDLE" == "true" && "$CONTENDER_COUNT" -eq 0 ]]; then
  exit 0
elif [[ "$KILL" == "true" ]]; then
  FAIL_COUNT=$(echo "$FAILED_JSON" | python3 -c "import sys,json; print(len(json.loads(sys.stdin.read())))")
  if [[ "$FAIL_COUNT" -gt 0 ]]; then
    exit 3
  fi
  exit 0
else
  # dry run, contention detected
  exit 1
fi
