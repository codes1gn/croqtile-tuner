#!/usr/bin/env bash
# validate_env.sh — Validate DSL-specific environment variables and toolchain.
#
# USAGE:
#   bash .claude/skills/croq-tune/tools/validate_env.sh --dsl <dsl>
#
# Runs standard GPU/ncu/nvcc checks, then DSL-specific checks.
#
# EXIT CODES:
#   0 — all checks pass
#   1 — one or more checks failed (details on stderr, summary JSON on stdout)

set -euo pipefail

DSL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dsl) DSL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$DSL" ]] && { echo "[validate_env] ERROR: --dsl required" >&2; exit 1; }

ERRORS=()
WARNINGS=()

check_cmd() {
  local label="$1" cmd="$2"
  if command -v "$cmd" &>/dev/null; then
    echo "[validate_env] OK: $label ($cmd found)" >&2
  else
    ERRORS+=("$label: $cmd not found")
    echo "[validate_env] FAIL: $label ($cmd not found)" >&2
  fi
}

check_env_var() {
  local var="$1" validate_cmd="${2:-}"
  if [[ -z "${!var:-}" ]]; then
    ERRORS+=("$var not set")
    echo "[validate_env] FAIL: \$$var not set" >&2
    return
  fi
  echo "[validate_env] OK: \$$var=${!var}" >&2
  if [[ -n "$validate_cmd" ]]; then
    if eval "$validate_cmd" &>/dev/null; then
      echo "[validate_env] OK: $var validation passed" >&2
    else
      ERRORS+=("$var validation failed: $validate_cmd")
      echo "[validate_env] FAIL: $var validation failed" >&2
    fi
  fi
}

# --- Standard checks (all DSLs) ---

check_cmd "nvidia-smi" "nvidia-smi"
check_cmd "nvcc" "nvcc"

if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
  GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
  echo "[validate_env] GPU: $GPU_NAME ($GPU_MEM)" >&2
fi

PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
if [[ "$PARANOID" != "unknown" && "$PARANOID" -le 2 ]]; then
  echo "[validate_env] OK: perf_event_paranoid=$PARANOID" >&2
else
  WARNINGS+=("perf_event_paranoid=$PARANOID (should be <= 2 for ncu)")
  echo "[validate_env] WARN: perf_event_paranoid=$PARANOID (ncu may fail)" >&2
fi

if command -v ncu &>/dev/null; then
  NCU_VER=$(ncu --version 2>/dev/null | head -1 || echo "unknown")
  echo "[validate_env] OK: ncu found ($NCU_VER)" >&2
else
  NCU_PATH=$(ls /usr/local/cuda*/bin/ncu 2>/dev/null | head -1 || echo "")
  if [[ -n "$NCU_PATH" ]]; then
    echo "[validate_env] OK: ncu at $NCU_PATH (not in PATH)" >&2
    WARNINGS+=("ncu not in PATH but found at $NCU_PATH")
  else
    ERRORS+=("ncu not found")
    echo "[validate_env] FAIL: ncu not found" >&2
  fi
fi

# --- DSL-specific checks ---

case "$DSL" in
  croqtile)
    check_env_var "CHOREO_HOME" "test -x \${CHOREO_HOME}/build/choreo || test -x \${CHOREO_HOME}/choreo"
    check_env_var "CUTE_HOME" "test -d \${CUTE_HOME}/include"
    check_env_var "CUDA_HOME" "test -x \${CUDA_HOME}/bin/nvcc"
    if [[ -n "${CHOREO_HOME:-}" ]]; then
      CHOREO_BIN="${CHOREO_HOME}/build/choreo"
      [[ ! -x "$CHOREO_BIN" ]] && CHOREO_BIN="${CHOREO_HOME}/choreo"
      if [[ -x "$CHOREO_BIN" ]]; then
        "$CHOREO_BIN" --help &>/dev/null && echo "[validate_env] OK: choreo --help works" >&2 \
          || { ERRORS+=("choreo --help failed"); echo "[validate_env] FAIL: choreo --help failed" >&2; }
      fi
    fi
    ;;
  cuda)
    check_env_var "CUDA_HOME" "test -x \${CUDA_HOME}/bin/nvcc"
    echo 'int main(){}' > /tmp/_croq_env_test.cu
    if nvcc /tmp/_croq_env_test.cu -o /tmp/_croq_env_test 2>/dev/null; then
      echo "[validate_env] OK: nvcc trivial compile works" >&2
      rm -f /tmp/_croq_env_test.cu /tmp/_croq_env_test
    else
      ERRORS+=("nvcc trivial compile failed")
      echo "[validate_env] FAIL: nvcc trivial compile failed" >&2
      rm -f /tmp/_croq_env_test.cu /tmp/_croq_env_test
    fi
    ;;
  triton)
    if python3 -c "import triton; print(triton.__version__)" &>/dev/null; then
      TRITON_VER=$(python3 -c "import triton; print(triton.__version__)" 2>/dev/null)
      echo "[validate_env] OK: triton $TRITON_VER" >&2
    else
      ERRORS+=("python3 'import triton' failed")
      echo "[validate_env] FAIL: triton not importable" >&2
    fi
    ;;
  cute|cutile)
    if python3 -c "import cutlass; print(cutlass.__version__)" &>/dev/null; then
      CUTLASS_VER=$(python3 -c "import cutlass; print(cutlass.__version__)" 2>/dev/null)
      echo "[validate_env] OK: cutlass $CUTLASS_VER" >&2
    else
      ERRORS+=("python3 'import cutlass' failed")
      echo "[validate_env] FAIL: cutlass not importable" >&2
    fi
    ;;
  helion)
    if python3 -c "import helion; print(helion.__version__)" &>/dev/null; then
      HELION_VER=$(python3 -c "import helion; print(helion.__version__)" 2>/dev/null)
      echo "[validate_env] OK: helion $HELION_VER" >&2
    else
      ERRORS+=("python3 'import helion' failed")
      echo "[validate_env] FAIL: helion not importable" >&2
    fi
    ;;
  tilelang)
    if python3 -c "import tilelang; print(tilelang.__version__)" &>/dev/null; then
      TILELANG_VER=$(python3 -c "import tilelang; print(tilelang.__version__)" 2>/dev/null)
      echo "[validate_env] OK: tilelang $TILELANG_VER" >&2
    else
      ERRORS+=("python3 'import tilelang' failed")
      echo "[validate_env] FAIL: tilelang not importable" >&2
    fi
    ;;
  *)
    WARNINGS+=("Unknown DSL '$DSL' — no DSL-specific checks available")
    echo "[validate_env] WARN: unknown DSL '$DSL'" >&2
    ;;
esac

# --- Summary JSON ---

ERROR_COUNT=${#ERRORS[@]}
WARN_COUNT=${#WARNINGS[@]}

ERRORS_JSON="[]"
if [[ $ERROR_COUNT -gt 0 ]]; then
  ERRORS_JSON=$(printf '%s\n' "${ERRORS[@]}" | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin]))")
fi

WARNINGS_JSON="[]"
if [[ $WARN_COUNT -gt 0 ]]; then
  WARNINGS_JSON=$(printf '%s\n' "${WARNINGS[@]}" | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin]))")
fi

cat <<EOF
{"dsl":"$DSL","pass":$([ $ERROR_COUNT -eq 0 ] && echo true || echo false),"errors":$ERRORS_JSON,"warnings":$WARNINGS_JSON}
EOF

[[ $ERROR_COUNT -eq 0 ]] && exit 0 || exit 1
