#!/usr/bin/env bash
# co2cu.sh — Two-phase croqtile harness: compile .co -> extract .cu from .cute.result
#
# USAGE (from repo root):
#   bash .claude/skills/croq-tune/tools/co2cu.sh \
#       --co   tuning/<gpu>/croqtile/srcs/<key>/<model>/iter<NNN>_<tag>.co \
#       --arch sm_86 \
#       [--flags "--use-warpspec"]
#
# WHAT IT DOES:
#   1. Runs choreo -gs to produce the .cute.result script
#   2. Extracts the __choreo_cute_*.cu heredoc from .cute.result
#   3. Writes it next to the .co as iter<NNN>_<tag>.cu
#   4. Extracts CFLAGS from the .cute.result for nvcc reference
#   5. Emits JSON with all paths + suggested nvcc flags
#
# OUTPUT (stdout): one-line JSON
#   {"co":"...","cu":"...","result":"...","nvcc_flags":"...","headers_dir":"...","status":"ready"}
#
# EXIT CODES:
#   0  — success, .cu extracted and ready for Phase 2
#   1  — usage error
#   2  — choreo compilation failed
#   3  — .cu extraction from .cute.result failed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HEADERS_DIR="${SCRIPT_DIR}/choreo_headers"

# ── argument parsing ──────────────────────────────────────────────────────────
CO=""
ARCH=""
CHOREO_FLAGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --co)    CO="$2";            shift 2 ;;
        --arch)  ARCH="$2";         shift 2 ;;
        --flags) CHOREO_FLAGS="$2"; shift 2 ;;
        *) echo "[co2cu] ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CO" || -z "$ARCH" ]]; then
    echo "[co2cu] ERROR: --co and --arch are required" >&2
    echo "Usage: bash co2cu.sh --co <path.co> --arch <sm_XX>" >&2
    exit 1
fi

if [[ ! -f "$CO" ]]; then
    echo "[co2cu] ERROR: .co file not found: $CO" >&2
    exit 1
fi

# ── resolve paths ─────────────────────────────────────────────────────────────
CO_DIR="$(dirname "$CO")"
CO_BASE="$(basename "$CO" .co)"

# .cute.result goes under cmd/ (parallel to srcs/)
RESULT_DIR="${CO_DIR/\/srcs\//\/cmd\/}"
mkdir -p "$RESULT_DIR"
RESULT="${RESULT_DIR}/${CO_BASE}.cute.result"

CU="${CO_DIR}/${CO_BASE}.cu"

# ── validate environment ─────────────────────────────────────────────────────
if [[ -z "${CHOREO_HOME:-}" ]]; then
    export CHOREO_HOME=/home/albert/workspace/croqtile
fi

CHOREO_BIN="${CHOREO_HOME}/build/choreo"
if [[ ! -x "$CHOREO_BIN" ]]; then
    CHOREO_BIN="${CHOREO_HOME}/choreo"
fi
if [[ ! -x "$CHOREO_BIN" ]]; then
    echo "[co2cu] ERROR: choreo compiler not found at ${CHOREO_HOME}/{build/choreo,choreo}" >&2
    exit 2
fi

if [[ ! -f "${HEADERS_DIR}/choreo.h" ]]; then
    echo "[co2cu] ERROR: choreo.h not found at ${HEADERS_DIR}/choreo.h" >&2
    exit 1
fi

# ── Phase 1: choreo -gs → .cute.result ────────────────────────────────────────
echo "[co2cu] Phase 1: compiling ${CO} -> ${RESULT}" >&2

BUILD_LOG="${CO_DIR/\/srcs\//\/perf\/}/build_${CO_BASE}.txt"
mkdir -p "$(dirname "$BUILD_LOG")"

# shellcheck disable=SC2086
if ! "$CHOREO_BIN" -gs -t cute -arch="$ARCH" $CHOREO_FLAGS \
        "$CO" -o "$RESULT" 2>&1 | tee "$BUILD_LOG" >&2; then
    echo "[co2cu] ERROR: choreo compilation failed. See ${BUILD_LOG}" >&2
    # Clean up stray a.out that choreo may leave on failure
    rm -f a.out
    exit 2
fi

# Clean up stray a.out that choreo may leave in cwd even with -o
rm -f a.out

if [[ ! -f "$RESULT" ]]; then
    echo "[co2cu] ERROR: choreo did not produce ${RESULT}" >&2
    exit 2
fi

# ── Phase 1.5: extract .cu from .cute.result heredoc ─────────────────────────
echo "[co2cu] Extracting .cu from ${RESULT}" >&2

python3 - "$RESULT" "$CU" <<'PYEOF'
import sys, re

result_path = sys.argv[1]
cu_path = sys.argv[2]

with open(result_path, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

# Find the __choreo_cute_*.cu heredoc block
cu_start = None
cu_end = None
for i, line in enumerate(lines):
    if cu_start is None and re.match(r"cat <<'EOF' > .+/__choreo_cute_.+\.cu$", line.strip()):
        cu_start = i + 1
    elif cu_start is not None and cu_end is None and line.strip() == 'EOF':
        cu_end = i
        break

if cu_start is None or cu_end is None:
    print(f"ERROR: could not find __choreo_cute_*.cu heredoc in {result_path}", file=sys.stderr)
    sys.exit(1)

cu_content = ''.join(lines[cu_start:cu_end])
with open(cu_path, 'w') as f:
    f.write(cu_content)

print(f"Extracted {cu_end - cu_start} lines to {cu_path}", file=sys.stderr)
PYEOF

EXTRACT_EXIT=$?
if [[ $EXTRACT_EXIT -ne 0 || ! -f "$CU" ]]; then
    echo "[co2cu] ERROR: .cu extraction failed" >&2
    exit 3
fi

# ── Extract CFLAGS from .cute.result ──────────────────────────────────────────
NVCC_FLAGS=$(python3 - "$RESULT" "$ARCH" "$HEADERS_DIR" <<'PYEOF'
import sys, re

result_path = sys.argv[1]
arch = sys.argv[2]
headers_dir = sys.argv[3]

with open(result_path, 'r') as f:
    content = f.read()

# Extract the raw CFLAGS line
match = re.search(r'export CFLAGS="([^"]+)"', content)
if not match:
    # Fallback: construct minimal flags
    print(f"-arch {arch} -std=c++17 -O3 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ -I{headers_dir}")
    sys.exit(0)

raw = match.group(1)

# Replace ${nv_arch} with actual arch, remove temp-dir -I paths, add headers_dir
flags = raw.replace('${nv_arch}', arch)
# Remove -I pointing to old srcs dir (will be re-added by build script)
flags = re.sub(r'-I/[^ ]+/srcs/[^ ]+', '', flags)
# Remove ${EXTRA_TARGET_CFLAGS} placeholder
flags = flags.replace('${EXTRA_TARGET_CFLAGS}', '')
# Add headers dir
flags = flags.strip() + f' -I{headers_dir}'
# Collapse whitespace
flags = re.sub(r'\s+', ' ', flags).strip()

print(flags)
PYEOF
)

# ── Emit JSON result ──────────────────────────────────────────────────────────
python3 - "$CO" "$CU" "$RESULT" "$NVCC_FLAGS" "$HEADERS_DIR" <<'PYEOF'
import sys, json

co, cu, result, nvcc_flags, headers_dir = sys.argv[1:6]

output = {
    "co": co,
    "cu": cu,
    "result": result,
    "nvcc_flags": nvcc_flags,
    "headers_dir": headers_dir,
    "status": "ready"
}
print(json.dumps(output))
PYEOF

echo "" >&2
echo "[READY] Phase 2: .cu at ${CU}. Edit the kernel section, then build with nvcc." >&2
echo "[READY] Headers at: ${HEADERS_DIR}" >&2
echo "[READY] Suggested nvcc flags: ${NVCC_FLAGS}" >&2
