#!/usr/bin/env bash
# resume_state.sh — Auto-reload tuning session state on startup/resume.
#
# PURPOSE:
#   Eliminates the agent having to manually grep through 4-5 files to
#   reconstruct where a tuning session left off. Reads all canonical state
#   sources and emits a single JSON snapshot to stdout.
#
# USAGE (from repo root):
#   bash .claude/skills/croq-resume/resume_state.sh \
#       --gpu sm90_H100 --dsl cuda --shape-key matmul_bf16fp32_512x16384x16384
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

GPU=""
DSL=""
SHAPE_KEY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)       GPU="$2";       shift 2 ;;
        --dsl)       DSL="$2";       shift 2 ;;
        --shape-key) SHAPE_KEY="$2"; shift 2 ;;
        *) echo "[resume_state] ERROR: unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$GPU" || -z "$DSL" || -z "$SHAPE_KEY" ]]; then
    echo "[resume_state] ERROR: --gpu, --dsl and --shape-key are required" >&2
    exit 1
fi

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR="tuning/${GPU}/${DSL}/srcs/${SHAPE_KEY}"
LOG_DIR="tuning/${GPU}/${DSL}/logs/${SHAPE_KEY}"
MEM_DIR="tuning/${GPU}/${DSL}/memory/${SHAPE_KEY}"
CP_FILE="tuning/${GPU}/${DSL}/checkpoints/${SHAPE_KEY}/current_idea.json"

# Validate something exists
if [[ ! -d "$SRC_DIR" && ! -d "$MEM_DIR" ]]; then
    echo "[resume_state] ERROR: no tuning artifacts found for $DSL/$SHAPE_KEY" >&2
    exit 2
fi

# ── run the state extraction in Python for reliability ───────────────────────
python3 - "$DSL" "$SHAPE_KEY" "$SRC_DIR" "$LOG_DIR" "$MEM_DIR" "$CP_FILE" <<'PYEOF'
import sys, json, re, os
from pathlib import Path

dsl, shape_key, src_dir, log_dir, mem_dir, cp_file = sys.argv[1:]

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

# ── 2. Parse rounds.raw.jsonl for best result and last round ───────────────
raw_path = Path(mem_dir) / "rounds.raw.jsonl"
if raw_path.exists():
    state["memory_files_ok"] = True
    best_tflops = -1.0
    last_round = None
    last_iter = None
    last_decision = None
    last_bottleneck = None

    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") == "note":
                    continue

                tflops = obj.get("tflops")
                decision = obj.get("decision", "")
                kernel = obj.get("kernel", "")
                it = obj.get("iter", "")
                bottleneck = obj.get("bottleneck", "")
                round_n = obj.get("round")

                # Track best (KEEP decisions only, skip baseline/non-iter kernels)
                if decision == "KEEP" and tflops is not None and it.startswith("iter") and it != "iter000":
                    try:
                        t = float(tflops)
                        if t > best_tflops:
                            best_tflops = t
                            state["current_best_tflops"] = t
                            state["current_best_kernel"] = kernel
                            state["current_best_iter"] = it
                    except (ValueError, TypeError):
                        pass

                # Track last round
                if round_n is not None:
                    try:
                        rn = int(round_n)
                        if last_round is None or rn >= last_round:
                            last_round = rn
                            last_iter = it
                            last_decision = decision
                            last_bottleneck = bottleneck
                    except (ValueError, TypeError):
                        pass

            except json.JSONDecodeError:
                warnings.append(f"malformed JSON in rounds.raw.jsonl")

    state["last_round"] = last_round
    state["last_iter"] = last_iter
    state["last_decision"] = last_decision
    state["last_bottleneck"] = last_bottleneck

    # Also cross-check with results.tsv for the highest iter number
    tsv_path = Path(log_dir) / "results.tsv"
    if tsv_path.exists():
        tsv_max = 0
        with open(tsv_path) as f:
            next(f, None)  # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    m = re.match(r'^iter(\d{3})', parts[1])  # kernel column
                    if m:
                        tsv_max = max(tsv_max, int(m.group(1)))
        # next_iter_number should be max of src files and tsv entries
        state["next_iter_number"] = max(state["next_iter_number"], tsv_max + 1)
else:
    warnings.append(f"rounds.raw.jsonl not found: {raw_path}")

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

# ── 4. Memory files presence check ───────────────────────────────────────
mem_path = Path(mem_dir)
missing_files = []
for fname in ["rounds.raw.jsonl", "rounds.md"]:
    if not (mem_path / fname).exists():
        missing_files.append(fname)
log_path = Path(log_dir)
for fname in ["results.tsv", "idea-log.jsonl"]:
    if not (log_path / fname).exists():
        missing_files.append(fname)
if missing_files:
    warnings.append(f"Missing memory/log files: {', '.join(missing_files)}")
    state["memory_files_ok"] = False

state["warnings"] = warnings
print(json.dumps(state, indent=2))
PYEOF
