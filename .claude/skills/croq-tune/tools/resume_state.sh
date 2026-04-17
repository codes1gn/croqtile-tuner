#!/usr/bin/env bash
# resume_state.sh — Auto-reload tuning session state on startup/resume.
#
# PURPOSE:
#   Eliminates the agent having to manually grep through 4-5 files to
#   reconstruct where a tuning session left off. Reads all canonical state
#   sources and emits a single JSON snapshot to stdout.
#
# USAGE (from repo root):
#   bash .claude/skills/croq-tune/tools/resume_state.sh \
#       --gpu sm90_H100 --dsl cuda --shape-key matmul_bf16fp32_512x16384x16384 \
#       --model opus-4
#
# The --gpu value is emitted by:
#   bash .claude/skills/croq-tune/tools/detect_gpu.sh
#
# OUTPUT (stdout): JSON with:
#   {
#     "dsl": "cuda",
#     "shape_key": "...",
#     "current_best_tflops": 35.2,
#     "current_best_kernel": "iter044_best",
#     "current_best_iter": "iter044",
#     "last_round": 68,
#     "last_iter": "iter068_ptxmma",
#     "last_decision": "DISCARD",
#     "last_bottleneck": "compute_bound",
#     "next_iter_number": 69,
#     "src_count": 268,
#     "open_checkpoint": null,         # or the checkpoint JSON if unverified
#     "memory_files_ok": true,
#     "warning": []
#   }
#
# EXIT:
#   0 — state loaded; JSON on stdout
#   1 — bad args
#   2 — no tuning artifacts found for this dsl/shape-key

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/activity_trace.sh"

GPU=""
DSL=""
SHAPE_KEY=""
MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)       GPU="$2";       shift 2 ;;
        --dsl)       DSL="$2";       shift 2 ;;
        --shape-key) SHAPE_KEY="$2"; shift 2 ;;
        --model)     MODEL="$2";     shift 2 ;;
        *) echo "[resume_state] ERROR: unknown arg: $1" >&2; echo "[SUGGESTION] Use your judgement to decide autonomously. Remove the unknown argument '$1' and retry. Valid args: --gpu --dsl --shape-key --model" >&2; exit 1 ;;
    esac
done

if [[ -z "$GPU" || -z "$DSL" || -z "$SHAPE_KEY" || -z "$MODEL" ]]; then
    echo "[resume_state] ERROR: --gpu, --dsl, --shape-key, and --model are required" >&2
    echo "[SUGGESTION] Use your judgement to decide autonomously. Provide all four required arguments. Use detect_gpu.sh to get --gpu value. --dsl is cuda/croqtile/triton/etc. --shape-key is the full shape key like matmul_bf16fp32_512x512x512. --model is the model name." >&2
    exit 1
fi

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR="tuning/${GPU}/${DSL}/srcs/${SHAPE_KEY}/${MODEL}"
LOG_DIR="tuning/${GPU}/${DSL}/logs/${SHAPE_KEY}/${MODEL}"
MEM_DIR="tuning/${GPU}/${DSL}/memory/${SHAPE_KEY}/${MODEL}"
CP_FILE="tuning/${GPU}/${DSL}/checkpoints/${SHAPE_KEY}/${MODEL}/current_idea.json"

trace_init --gpu "$GPU" --dsl "$DSL" --shape-key "$SHAPE_KEY" --model "$MODEL"
trace_event "resume_state" "Loading tuning state for $DSL/$SHAPE_KEY"

# Validate something exists
if [[ ! -d "$SRC_DIR" && ! -d "$LOG_DIR" ]]; then
    trace_event "resume_state" "No tuning artifacts found for $DSL/$SHAPE_KEY" "error"
    echo "[resume_state] ERROR: no tuning artifacts found for $DSL/$SHAPE_KEY" >&2
    echo "[SUGGESTION] Use your judgement to decide autonomously. This is a fresh tuning session with no prior artifacts. This is expected for new tasks. Proceed with the first iteration: run cublas_baseline.sh to get the baseline TFLOPS, then start your first IDEA step." >&2
    exit 2
fi

# ── run the state extraction in Python for reliability ───────────────────────
python3 - "$DSL" "$SHAPE_KEY" "$SRC_DIR" "$LOG_DIR" "$CP_FILE" <<'PYEOF'
import sys, json, re, os
from pathlib import Path

dsl, shape_key, src_dir, log_dir, cp_file = sys.argv[1:]

warnings = []
state = {
    "dsl": dsl,
    "shape_key": shape_key,
    "current_best_tflops": None,
    "current_best_kernel": None,
    "current_best_iter": None,
    "last_round": None,
    "last_iter": None,
    "last_decision": None,
    "last_bottleneck": None,
    "next_iter_number": 1,
    "src_count": 0,
    "open_checkpoint": None,
    "memory_files_ok": False,
    "warnings": []
}

# ── 1. Count source files and find highest iter number ─────────────────────
ITER_RE = re.compile(r'^iter(\d{3})_[a-z][a-z0-9_]{1,15}\.[a-z]+$')
src_path = Path(src_dir)
if src_path.exists():
    max_iter_num = 0
    count = 0
    for f in src_path.iterdir():
        m = ITER_RE.match(f.name)
        if m:
            count += 1
            n = int(m.group(1))
            max_iter_num = max(max_iter_num, n)
    state["src_count"] = count
    state["next_iter_number"] = max_iter_num + 1
else:
    warnings.append(f"src_dir not found: {src_dir}")

# ── 2. Parse results.tsv + idea-log.jsonl for best/last state ───────────────
tsv_path = Path(log_dir) / "results.tsv"
idea_log_path = Path(log_dir) / "idea-log.jsonl"

best_tflops = -1.0
last_round = 0
last_iter = None
last_decision = None
last_bottleneck = None

if tsv_path.exists():
    state["memory_files_ok"] = True
    row_count = 0
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split('\t')
            if not parts:
                continue
            first = parts[0].lower()
            if first in ("iter", "round"):
                continue

            row_count += 1

            # Detect format: croqtile (col0 is int round) vs cuda (col0 is iter name)
            try:
                int(parts[0])
                tflops_idx, decision_idx, kernel_idx = 3, 4, 2
            except ValueError:
                tflops_idx, decision_idx, kernel_idx = 2, 3, 1

            if len(parts) <= tflops_idx:
                continue

            kernel = parts[kernel_idx].strip() if len(parts) > kernel_idx else ""
            try:
                tflops = float(parts[tflops_idx].strip())
            except (ValueError, IndexError):
                tflops = 0.0
            decision = parts[decision_idx].strip().upper() if len(parts) > decision_idx else ""
            bottleneck = parts[decision_idx + 1].strip() if len(parts) > decision_idx + 1 else ""

            last_round = row_count
            last_iter = kernel or parts[0].strip()
            last_decision = decision
            last_bottleneck = bottleneck

            # Extract iter number from kernel name
            m = re.match(r'iter(\d{3})', kernel)
            if m:
                state["next_iter_number"] = max(state["next_iter_number"], int(m.group(1)) + 1)

            if decision == "KEEP" and tflops > best_tflops and kernel.startswith("iter") and not kernel.startswith("iter000"):
                best_tflops = tflops
                state["current_best_tflops"] = tflops
                state["current_best_kernel"] = kernel
                m2 = re.match(r'(iter\d{3})', kernel)
                state["current_best_iter"] = m2.group(1) if m2 else kernel

    state["last_round"] = last_round if last_round > 0 else None
    state["last_iter"] = last_iter
    state["last_decision"] = last_decision
    state["last_bottleneck"] = last_bottleneck
else:
    warnings.append(f"results.tsv not found: {tsv_path}")

# Enrich with idea-log.jsonl round numbers if available
if idea_log_path.exists():
    try:
        last_idea_round = 0
        with open(idea_log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rn = obj.get("round")
                    if rn is not None:
                        last_idea_round = max(last_idea_round, int(rn))
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
        if last_idea_round > (state["last_round"] or 0):
            state["last_round"] = last_idea_round
    except OSError:
        pass

# ── 3. Check for open (PLANNED but not VERIFIED) checkpoint ───────────────
cp_path = Path(cp_file)
if cp_path.exists():
    try:
        with open(cp_path) as f:
            cp = json.load(f)
        if cp.get("status") == "PLANNED":
            state["open_checkpoint"] = cp
            warnings.append(
                f"Open checkpoint detected: {cp.get('iter')} was planned but not verified — "
                f"run checkpoint_write.sh verify or discard and re-run IDEA"
            )
    except Exception:
        warnings.append("checkpoint file exists but is not valid JSON")

# ── 4. Log files presence check ───────────────────────────────────────────
log_path = Path(log_dir)
missing_files = []
for fname in ["results.tsv", "idea-log.jsonl"]:
    if not (log_path / fname).exists():
        missing_files.append(fname)
if missing_files:
    warnings.append(f"Missing log files: {', '.join(missing_files)}")
    state["memory_files_ok"] = False

state["warnings"] = warnings
print(json.dumps(state, indent=2))
PYEOF
