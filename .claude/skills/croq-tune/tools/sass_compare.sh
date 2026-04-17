#!/usr/bin/env bash
# sass_compare.sh — Reverse-engineer baseline & custom kernel SASS for comparison.
#
# Three modes:
#
#   dump-baseline  Extract SASS from the cuBLAS kernel dispatched for this shape.
#                  Primary: ncu --print-source sass on torch.mm benchmark.
#                  Fallback: cuobjdump --dump-sass on libcublas.so + grep.
#
#   dump-custom    Extract SASS from a custom kernel binary via cuobjdump.
#
#   compare        Diff baseline vs custom SASS: instruction mix, register usage,
#                  tensor core types, memory widths. Outputs JSON to stdout.
#
# USAGE:
#   bash .claude/skills/croq-tune/tools/sass_compare.sh dump-baseline \
#       --gpu sm86_NVIDIA_GeForce_RTX_3070 --dsl croqtile \
#       --shape-key matmul_bf16fp32_512x16384x16384 --model opus-4 \
#       --dtype bf16fp32 --m 512 --n 16384 --k 16384
#
#   bash .claude/skills/croq-tune/tools/sass_compare.sh dump-custom \
#       --gpu sm86_NVIDIA_GeForce_RTX_3070 --dsl croqtile \
#       --shape-key matmul_bf16fp32_512x16384x16384 --model opus-4 \
#       --iter iter045_myidea
#
#   bash .claude/skills/croq-tune/tools/sass_compare.sh compare \
#       --gpu sm86_NVIDIA_GeForce_RTX_3070 --dsl croqtile \
#       --shape-key matmul_bf16fp32_512x16384x16384 --model opus-4 \
#       --iter iter045_myidea
#
# EXIT CODES:
#   0 — success
#   1 — argument error
#   2 — tool not found (ncu/cuobjdump)
#   3 — extraction failed
#   4 — comparison failed (missing SASS files)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/activity_trace.sh"

MODE="${1:-}"
if [[ -z "$MODE" || ! "$MODE" =~ ^(dump-baseline|dump-custom|compare)$ ]]; then
    echo "[sass_compare] ERROR: first arg must be 'dump-baseline', 'dump-custom', or 'compare'" >&2
    echo "Usage: $0 dump-baseline|dump-custom|compare [--gpu ...] [--dsl ...] ..." >&2
    echo "[SUGGESTION] Use your judgement to decide autonomously. Choose one mode: dump-baseline (extract cuBLAS reference SASS), dump-custom (extract your kernel SASS), or compare (diff baseline vs custom). Run dump-baseline and dump-custom before compare." >&2
    exit 1
fi
shift

# ── argument parsing ──────────────────────────────────────────────────────────
GPU=""
DSL=""
SHAPE_KEY=""
MODEL=""
ITER=""
DTYPE=""
M=""
N=""
K=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)       GPU="$2";       shift 2 ;;
        --dsl)       DSL="$2";       shift 2 ;;
        --shape-key) SHAPE_KEY="$2"; shift 2 ;;
        --model)     MODEL="$2";     shift 2 ;;
        --iter)      ITER="$2";      shift 2 ;;
        --dtype)     DTYPE="$2";     shift 2 ;;
        --m)         M="$2";        shift 2 ;;
        --n)         N="$2";        shift 2 ;;
        --k)         K="$2";        shift 2 ;;
        *) echo "[sass_compare] ERROR: unknown arg: $1" >&2; echo "[SUGGESTION] Use your judgement to decide autonomously. Remove '$1' and retry. Valid args: --gpu --dsl --shape-key --model --iter --dtype --m --n --k" >&2; exit 1 ;;
    esac
done

# Auto-detect GPU if not provided
if [[ -z "$GPU" ]]; then
    DETECT_SCRIPT="$SCRIPT_DIR/detect_gpu.sh"
    GPU=$(bash "$DETECT_SCRIPT" 2>/dev/null || echo "sm00_unknown")
fi

# Validate common required args
if [[ -z "$DSL" || -z "$SHAPE_KEY" || -z "$MODEL" ]]; then
    echo "[sass_compare] ERROR: --dsl, --shape-key, and --model are required" >&2
    echo "[SUGGESTION] Use your judgement to decide autonomously. Provide --dsl (cuda/croqtile/triton/etc), --shape-key (e.g. matmul_bf16fp32_512x512x512), and --model (e.g. opus-4)." >&2
    exit 1
fi

trace_init --gpu "$GPU" --dsl "$DSL" --shape-key "$SHAPE_KEY" --model "$MODEL"

# ── paths ─────────────────────────────────────────────────────────────────────
PERF_DIR="tuning/${GPU}/${DSL}/perf/${SHAPE_KEY}/${MODEL}"
BIN_DIR="tuning/${GPU}/${DSL}/bin/${SHAPE_KEY}/${MODEL}"
mkdir -p "$PERF_DIR"

# ── find ncu ──────────────────────────────────────────────────────────────────
find_ncu() {
    local ncu=""
    for candidate in \
        /usr/local/cuda-12.*/nsight-compute-*/target/linux-desktop-glibc_2_11_3-x64/ncu \
        /usr/local/cuda/nsight-compute-*/target/linux-desktop-glibc_2_11_3-x64/ncu \
        /usr/local/cuda*/bin/ncu; do
        if [[ -x "$candidate" ]]; then
            ncu="$candidate"
            break
        fi
    done
    if [[ -z "$ncu" ]] && command -v ncu &>/dev/null; then
        ncu=$(command -v ncu)
    fi
    echo "$ncu"
}

# ══════════════════════════════════════════════════════════════════════════════
# MODE: dump-baseline
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "dump-baseline" ]]; then
    if [[ -z "$DTYPE" || -z "$M" || -z "$N" || -z "$K" ]]; then
        echo "[sass_compare] ERROR: dump-baseline requires --dtype, --m, --n, --k" >&2
        echo "[SUGGESTION] Use your judgement to decide autonomously. Provide --dtype (bf16fp32/f16/f32/etc) and matrix dimensions --m --n --k matching your shape-key." >&2
        exit 1
    fi

    BASELINE_SASS="$PERF_DIR/sass_baseline.txt"
    trace_event "sass_compare" "Extracting baseline SASS for ${DTYPE} ${M}x${N}x${K}"

    # ── Primary approach: ncu profile + extract SASS ──────────────────────────
    # Two-step: (1) profile torch.mm to .ncu-rep, (2) import and print SASS.
    # Uses torch.empty (no init kernels) + warmup mm + --launch-skip 1 to
    # capture the exact cuBLAS GEMM kernel dispatched for this shape/dtype.
    NCU=$(find_ncu)
    PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "99")
    NCU_OK=false

    if [[ -n "$NCU" && "$PARANOID" -le 2 ]]; then
        echo "[sass_compare] Primary: ncu SASS extraction" >&2

        NCU_REP_TMP=$(mktemp /tmp/sass_baseline_XXXXXX)
        NCU_SASS_TMP=$(mktemp /tmp/sass_baseline_out_XXXXXX.txt)
        trap "rm -f '$NCU_REP_TMP' '${NCU_REP_TMP}.ncu-rep' '$NCU_SASS_TMP'" EXIT

        BENCH_SCRIPT=$(cat <<'PYEOF'
import sys, torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dtype_arg = sys.argv[1]
M, N, K = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
dtype_map = {"f16": torch.float16, "bf16": torch.bfloat16, "bf16fp32": torch.bfloat16,
             "f16fp32": torch.float16, "f32": torch.float32}
dt = dtype_map.get(dtype_arg, torch.float16)
A = torch.empty(M, K, device="cuda", dtype=dt)
B = torch.empty(K, N, device="cuda", dtype=dt)
torch.mm(A, B)
torch.cuda.synchronize()
torch.mm(A, B)
torch.cuda.synchronize()
PYEOF
)
        # Step 1: profile to .ncu-rep (skip warmup gemm, capture second)
        if "$NCU" --section SourceCounters --launch-skip 1 --launch-count 1 \
            -o "$NCU_REP_TMP" \
            python3 -c "$BENCH_SCRIPT" "$DTYPE" "$M" "$N" "$K" \
            2>/dev/null; then

            NCU_REP_FILE="${NCU_REP_TMP}.ncu-rep"
            [[ ! -f "$NCU_REP_FILE" ]] && NCU_REP_FILE="$NCU_REP_TMP"

            # Step 2: import and extract SASS
            if [[ -f "$NCU_REP_FILE" ]]; then
                "$NCU" --import "$NCU_REP_FILE" --page source --print-source sass \
                    > "$NCU_SASS_TMP" 2>/dev/null || true

                if [[ -s "$NCU_SASS_TMP" ]]; then
                    cp "$NCU_SASS_TMP" "$BASELINE_SASS"
                    NCU_OK=true
                    LINE_COUNT=$(wc -l < "$BASELINE_SASS")
                    echo "[sass_compare] Primary OK: $LINE_COUNT lines of baseline SASS" >&2
                    trace_event "sass_compare" "Baseline SASS extracted via ncu ($LINE_COUNT lines)"
                fi
            fi
        fi
        rm -f "$NCU_REP_TMP" "${NCU_REP_TMP}.ncu-rep" "$NCU_SASS_TMP"
        trap - EXIT
    fi

    # ── Fallback: cuobjdump on libcublas.so / libcublasLt.so ───────────────
    if [[ "$NCU_OK" == "false" ]]; then
        echo "[sass_compare] Fallback: cuobjdump on cuBLAS libraries" >&2
        trace_event "sass_compare" "Falling back to cuobjdump on cuBLAS libs" "warn"

        if ! command -v cuobjdump &>/dev/null; then
            echo "[sass_compare] ERROR: neither ncu nor cuobjdump available" >&2
            echo "[SUGGESTION] Use your judgement to decide autonomously. Install CUDA toolkit to get cuobjdump, or ensure ncu is available and perf_event_paranoid <= 2. Without these tools, skip SASS comparison and rely on profile_extract.sh bottleneck classification instead." >&2
            trace_event "sass_compare" "No SASS extraction tool available" "error"
            exit 2
        fi

        # Extract SM arch from GPU identifier (e.g. sm86_NVIDIA... -> sm_86)
        SM_NUM=$(echo "$GPU" | grep -oP 'sm\K\d+' | head -1)
        SM_ARCH="sm_${SM_NUM:-86}"

        # Try libcublasLt first (has the actual GEMM kernels), then libcublas
        CUBLAS_PATH=""
        for lib_name in libcublasLt.so libcublas.so; do
            CUBLAS_PATH=$(ldconfig -p 2>/dev/null | grep -m1 "$lib_name " | awk '{print $NF}')
            if [[ -z "$CUBLAS_PATH" ]]; then
                CUBLAS_PATH=$(find /usr/local/cuda*/targets/*/lib /usr/local/cuda*/lib64 -name "$lib_name" 2>/dev/null | head -1)
            fi
            [[ -n "$CUBLAS_PATH" && -f "$CUBLAS_PATH" ]] && break
            CUBLAS_PATH=""
        done

        if [[ -z "$CUBLAS_PATH" ]]; then
            echo "[sass_compare] ERROR: libcublas*.so not found" >&2
            echo "[SUGGESTION] Use your judgement to decide autonomously. cuBLAS library not found in standard paths. Try setting LD_LIBRARY_PATH to include your CUDA lib directory, or skip SASS comparison and use profile_extract.sh bottleneck classification." >&2
            trace_event "sass_compare" "cuBLAS library not found" "error"
            exit 3
        fi

        echo "[sass_compare] Found: $CUBLAS_PATH (targeting $SM_ARCH)" >&2

        FULL_SASS_TMP=$(mktemp /tmp/sass_cublas_full_XXXXXX.txt)
        trap "rm -f '$FULL_SASS_TMP'" EXIT

        # Dump SASS (can be very large — timeout after 60s)
        timeout 60 cuobjdump --dump-sass "$CUBLAS_PATH" > "$FULL_SASS_TMP" 2>/dev/null || true

        if [[ -s "$FULL_SASS_TMP" ]]; then
            python3 - "$FULL_SASS_TMP" "$BASELINE_SASS" "$SM_ARCH" "$DTYPE" <<'PYEOF'
import sys, re

full_path, out_path, sm_arch, dtype_arg = sys.argv[1:5]

with open(full_path, errors='replace') as f:
    content = f.read()

# Split into per-architecture + per-kernel sections
# cuobjdump output format: "Fatbin elf code:" header, then "arch = sm_XX",
# then "code for sm_XX", then function bodies
arch_blocks = re.split(r'(Fatbin elf code:)', content)

target_blocks = []
current_arch = ""
for i, block in enumerate(arch_blocks):
    if block.strip() == "Fatbin elf code:":
        rest = arch_blocks[i + 1] if i + 1 < len(arch_blocks) else ""
        arch_match = re.search(r'arch\s*=\s*(sm_\d+)', rest)
        if arch_match:
            current_arch = arch_match.group(1)
        if current_arch == sm_arch:
            target_blocks.append(rest)

if not target_blocks:
    # Fall back to any available arch
    target_blocks = [content]
    print(f"WARNING: no {sm_arch} sections found, using all content", file=sys.stderr)

combined = "\n".join(target_blocks)

# Further filter for gemm-related kernel functions
dtype_lower = dtype_arg.lower()
dtype_patterns = {
    "bf16": ["bf16", "bfloat", "gemm"],
    "bf16fp32": ["bf16", "bfloat", "fp16", "half", "gemm"],
    "f16": ["fp16", "f16", "half", "hgemm", "gemm"],
    "f16fp32": ["fp16", "f16", "half", "hgemm", "gemm"],
    "f32": ["sgemm", "f32", "fp32", "gemm"],
}
hints = dtype_patterns.get(dtype_lower, ["gemm", "matmul"])

# Look for function sections
func_re = re.compile(r'(\.text\.[^\n]+)', re.MULTILINE)
sections = func_re.split(combined)

matched = []
for i in range(1, len(sections), 2):
    header = sections[i].lower()
    body = sections[i + 1] if i + 1 < len(sections) else ""
    if any(h in header for h in hints):
        matched.append(sections[i] + body)

if matched:
    matched.sort(key=len, reverse=True)
    result = matched[0]
    if len(matched) > 1:
        result += f"\n\n/* {len(matched) - 1} other matching kernel(s) omitted */\n"
    print(f"Extracted {len(result)} chars from {len(matched)} matching kernel(s) [{sm_arch}]")
else:
    # Take the raw output for the target arch (truncated)
    result = combined[:100000]
    print(f"No kernel-specific match; saved {len(result)} chars of raw {sm_arch} SASS")

with open(out_path, 'w') as f:
    f.write(result)
PYEOF
            LINE_COUNT=$(wc -l < "$BASELINE_SASS")
            echo "[sass_compare] Fallback OK: $LINE_COUNT lines of baseline SASS" >&2
            trace_event "sass_compare" "Baseline SASS extracted via cuobjdump ($LINE_COUNT lines)"
        else
            echo "[sass_compare] ERROR: cuobjdump produced empty output" >&2
            echo "[SUGGESTION] Use your judgement to decide autonomously. cuobjdump could not extract SASS from the cuBLAS library. This may happen with stripped binaries. Skip SASS comparison and rely on profile_extract.sh bottleneck classification for your next IDEA." >&2
            trace_event "sass_compare" "cuobjdump produced empty output" "error"
            rm -f "$FULL_SASS_TMP"
            trap - EXIT
            exit 3
        fi

        rm -f "$FULL_SASS_TMP"
        trap - EXIT
    fi

    echo "[sass_compare] Baseline SASS: $BASELINE_SASS" >&2
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# MODE: dump-custom
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "dump-custom" ]]; then
    if [[ -z "$ITER" ]]; then
        echo "[sass_compare] ERROR: dump-custom requires --iter" >&2
        echo "[SUGGESTION] Use your judgement to decide autonomously. Provide --iter with the iteration tag of the kernel whose SASS you want to extract (e.g. iter045_swizzle)." >&2
        exit 1
    fi

    trace_event "sass_compare" "Extracting custom SASS for $ITER"

    # Find the binary
    BIN=""
    if [[ -f "$BIN_DIR/$ITER" ]]; then
        BIN="$BIN_DIR/$ITER"
    elif [[ -d "$BIN_DIR/$ITER" ]]; then
        BIN=$(find "$BIN_DIR/$ITER" -type f -executable 2>/dev/null | head -1)
    fi

    # Also check for iter_name as filename pattern
    if [[ -z "$BIN" ]]; then
        BIN=$(find "$BIN_DIR" -maxdepth 1 -name "${ITER}*" -type f 2>/dev/null | head -1)
    fi

    if [[ -z "$BIN" || ! -f "$BIN" ]]; then
        echo "[sass_compare] ERROR: binary not found for $ITER in $BIN_DIR" >&2
        echo "[SUGGESTION] Use your judgement to decide autonomously. The compiled binary for $ITER was not found. Ensure you built the kernel first. Check $BIN_DIR for available binaries. If the build failed, fix the compilation error and rebuild before retrying." >&2
        trace_event "sass_compare" "Binary not found for $ITER" "error"
        exit 3
    fi

    if ! command -v cuobjdump &>/dev/null; then
        echo "[sass_compare] ERROR: cuobjdump not found" >&2
        echo "[SUGGESTION] Use your judgement to decide autonomously. cuobjdump is needed to extract SASS from custom kernels. Install CUDA toolkit or add its bin directory to PATH. Without it, skip SASS comparison and use profile_extract.sh." >&2
        exit 2
    fi

    CUSTOM_SASS="$PERF_DIR/sass_${ITER}.txt"
    cuobjdump --dump-sass "$BIN" > "$CUSTOM_SASS" 2>/dev/null

    if [[ ! -s "$CUSTOM_SASS" ]]; then
        echo "[sass_compare] ERROR: cuobjdump produced empty output for $BIN" >&2
        echo "[SUGGESTION] Use your judgement to decide autonomously. The binary may be stripped or in an unexpected format. Skip SASS extraction for this kernel and proceed with profile_extract.sh bottleneck analysis." >&2
        trace_event "sass_compare" "cuobjdump empty output for $ITER" "error"
        exit 3
    fi

    LINE_COUNT=$(wc -l < "$CUSTOM_SASS")
    echo "[sass_compare] Custom SASS: $CUSTOM_SASS ($LINE_COUNT lines)" >&2
    trace_event "sass_compare" "Custom SASS extracted for $ITER ($LINE_COUNT lines)"
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# MODE: compare
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "compare" ]]; then
    if [[ -z "$ITER" ]]; then
        echo "[sass_compare] ERROR: compare requires --iter" >&2
        echo "[SUGGESTION] Use your judgement to decide autonomously. Provide --iter with the iteration tag you want to compare against baseline (e.g. iter045_swizzle)." >&2
        exit 1
    fi

    BASELINE_SASS="$PERF_DIR/sass_baseline.txt"
    CUSTOM_SASS="$PERF_DIR/sass_${ITER}.txt"
    COMPARE_OUT="$PERF_DIR/sass_compare_${ITER}.json"

    if [[ ! -f "$BASELINE_SASS" ]]; then
        echo "[sass_compare] ERROR: baseline SASS not found at $BASELINE_SASS" >&2
        echo "[SUGGESTION] Use your judgement to decide autonomously. Run 'sass_compare.sh dump-baseline' first with --dtype --m --n --k to extract the cuBLAS SASS, then retry compare." >&2
        exit 4
    fi

    if [[ ! -f "$CUSTOM_SASS" ]]; then
        echo "[sass_compare] ERROR: custom SASS not found at $CUSTOM_SASS" >&2
        echo "[SUGGESTION] Use your judgement to decide autonomously. Run 'sass_compare.sh dump-custom --iter $ITER' first to extract the custom kernel SASS, then retry compare." >&2
        exit 4
    fi

    trace_event "sass_compare" "Comparing baseline vs $ITER SASS"

    python3 - "$BASELINE_SASS" "$CUSTOM_SASS" "$ITER" "$COMPARE_OUT" <<'PYEOF'
import sys, json, re
from collections import Counter

baseline_path, custom_path, iter_tag, out_path = sys.argv[1:5]

def parse_sass(path):
    """Extract instruction mnemonics and metadata from SASS dump."""
    with open(path, errors='replace') as f:
        lines = f.readlines()

    instructions = Counter()
    total_instr = 0
    register_max = 0
    barriers = 0
    memory_ops = Counter()
    tensor_ops = Counter()

    # Regex to match SASS instruction lines: leading whitespace + address + mnemonic
    instr_re = re.compile(r'^\s*(?:\/\*[^*]*\*\/\s*)?(?:0x[0-9a-f]+\s+)?([A-Z][A-Z0-9_.]+)')
    reg_re = re.compile(r'R(\d+)')

    for line in lines:
        m = instr_re.match(line)
        if not m:
            continue
        mnemonic = m.group(1)
        total_instr += 1

        # Classify instruction
        base_op = mnemonic.split('.')[0]
        instructions[base_op] += 1

        # Track register references
        for rm in reg_re.finditer(line):
            rn = int(rm.group(1))
            register_max = max(register_max, rn)

        # Barriers / sync
        if base_op in ('BAR', 'MEMBAR', 'DEPBAR', 'BSYNC', 'BSSY'):
            barriers += 1

        # Memory operations with width
        if base_op in ('LDG', 'STG', 'LDS', 'STS', 'LDL', 'STL', 'LDGSTS', 'ATOMS'):
            width = ''
            for part in mnemonic.split('.'):
                if part in ('8', '16', '32', '64', '128', 'E', 'U', 'CONSTANT'):
                    width = part
            memory_ops[f"{base_op}.{width}" if width else base_op] += 1

        # Tensor / MMA ops
        if base_op in ('HMMA', 'IMMA', 'DMMA', 'QMMA', 'GMMA', 'WGMMA', 'MMA'):
            tensor_ops[mnemonic] += 1

    return {
        'total_instructions': total_instr,
        'instruction_mix': dict(instructions.most_common(30)),
        'register_max': register_max,
        'estimated_registers': register_max + 1,
        'barrier_count': barriers,
        'memory_ops': dict(memory_ops.most_common(20)),
        'tensor_ops': dict(tensor_ops.most_common(10)),
    }

baseline = parse_sass(baseline_path)
custom = parse_sass(custom_path)

# Build comparison
divergences = []

# 1. Tensor core instruction comparison
b_tc = baseline['tensor_ops']
c_tc = custom['tensor_ops']
if b_tc != c_tc:
    b_tc_total = sum(b_tc.values())
    c_tc_total = sum(c_tc.values())
    divergences.append({
        'category': 'tensor_core',
        'severity': 'high',
        'detail': f"Baseline uses {b_tc} ({b_tc_total} total), custom uses {c_tc} ({c_tc_total} total)",
    })

# 2. Register pressure
reg_diff = custom['estimated_registers'] - baseline['estimated_registers']
if abs(reg_diff) > 16:
    divergences.append({
        'category': 'register_pressure',
        'severity': 'high' if reg_diff > 32 else 'medium',
        'detail': f"Custom uses {custom['estimated_registers']} regs vs baseline {baseline['estimated_registers']} ({'+' if reg_diff > 0 else ''}{reg_diff})",
    })

# 3. Memory access patterns
b_mem = baseline['memory_ops']
c_mem = custom['memory_ops']
for op in set(list(b_mem.keys()) + list(c_mem.keys())):
    b_count = b_mem.get(op, 0)
    c_count = c_mem.get(op, 0)
    if b_count > 0 and c_count == 0:
        divergences.append({
            'category': 'memory_pattern',
            'severity': 'medium',
            'detail': f"Baseline has {op} ({b_count}x) but custom does not",
        })
    elif c_count > 0 and b_count == 0 and c_count > 3:
        divergences.append({
            'category': 'memory_pattern',
            'severity': 'medium',
            'detail': f"Custom has {op} ({c_count}x) but baseline does not",
        })

# 4. Instruction density (barrier/sync ratio)
b_ratio = baseline['barrier_count'] / max(baseline['total_instructions'], 1)
c_ratio = custom['barrier_count'] / max(custom['total_instructions'], 1)
if abs(b_ratio - c_ratio) > 0.02:
    divergences.append({
        'category': 'synchronization',
        'severity': 'medium',
        'detail': f"Barrier density: baseline {b_ratio:.3f} vs custom {c_ratio:.3f}",
    })

# 5. Overall instruction count ratio
if baseline['total_instructions'] > 0 and custom['total_instructions'] > 0:
    ratio = custom['total_instructions'] / baseline['total_instructions']
    if ratio > 1.5 or ratio < 0.5:
        divergences.append({
            'category': 'code_size',
            'severity': 'medium',
            'detail': f"Custom has {ratio:.1f}x the instructions of baseline ({custom['total_instructions']} vs {baseline['total_instructions']})",
        })

# 6. Top instruction mix divergences
all_ops = set(list(baseline['instruction_mix'].keys()) + list(custom['instruction_mix'].keys()))
mix_diffs = []
for op in all_ops:
    b_pct = baseline['instruction_mix'].get(op, 0) / max(baseline['total_instructions'], 1) * 100
    c_pct = custom['instruction_mix'].get(op, 0) / max(custom['total_instructions'], 1) * 100
    if abs(b_pct - c_pct) > 3.0:
        mix_diffs.append((op, b_pct, c_pct, abs(b_pct - c_pct)))
mix_diffs.sort(key=lambda x: -x[3])
for op, b_pct, c_pct, diff in mix_diffs[:5]:
    divergences.append({
        'category': 'instruction_mix',
        'severity': 'low',
        'detail': f"{op}: baseline {b_pct:.1f}% vs custom {c_pct:.1f}% (diff {diff:.1f}%)",
    })

# Sort divergences by severity
severity_order = {'high': 0, 'medium': 1, 'low': 2}
divergences.sort(key=lambda d: severity_order.get(d['severity'], 3))

# Build actionable summary
summary_lines = []
for d in divergences[:5]:
    summary_lines.append(f"[{d['severity'].upper()}] {d['category']}: {d['detail']}")

result = {
    'iter': iter_tag,
    'baseline': {
        'total_instructions': baseline['total_instructions'],
        'estimated_registers': baseline['estimated_registers'],
        'tensor_ops': baseline['tensor_ops'],
        'top_instructions': dict(list(baseline['instruction_mix'].items())[:10]),
        'memory_ops': baseline['memory_ops'],
    },
    'custom': {
        'total_instructions': custom['total_instructions'],
        'estimated_registers': custom['estimated_registers'],
        'tensor_ops': custom['tensor_ops'],
        'top_instructions': dict(list(custom['instruction_mix'].items())[:10]),
        'memory_ops': custom['memory_ops'],
    },
    'divergences': divergences,
    'summary': summary_lines,
    'actionable_insights': (
        "Use the divergences above to guide your next IDEA. "
        "High-severity items (tensor_core, register_pressure) usually have the biggest impact. "
        "Focus on matching the baseline's instruction patterns where feasible."
    ),
}

with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
PYEOF

    COMPARE_EXIT=$?
    if [[ $COMPARE_EXIT -ne 0 ]]; then
        echo "[sass_compare] ERROR: comparison analysis failed" >&2
        echo "[SUGGESTION] Use your judgement to decide autonomously. SASS comparison analysis encountered an internal error. Skip SASS comparison for this iteration and rely on profile_extract.sh bottleneck classification for your IDEA step." >&2
        trace_event "sass_compare" "Comparison analysis failed" "error"
        exit 4
    fi

    trace_event "sass_compare" "SASS comparison complete for $ITER (saved to $COMPARE_OUT)"
    echo "" >&2
    echo "[sass_compare] Comparison saved: $COMPARE_OUT" >&2
    echo "[sass_compare] Use the divergences to inform your next IDEA." >&2
    exit 0
fi
