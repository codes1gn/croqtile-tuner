#!/usr/bin/env bash
# discover_baseline.sh — Scan for usable starting kernels.
#
# Scans reference examples and tuning/ for kernels matching the target operator,
# excluding _aitune_ artifacts. Outputs a JSON list of candidates for the agent
# to compile and benchmark (top 2-3).
#
# USAGE:
#   bash .cursor/skills/cursor-croq-tune/tools/discover_baseline.sh \
#       --dsl <dsl> --operator <op> [--dtype <dtype>] [--gpu <gpu_key>]
#
# OUTPUT (stdout): JSON with tiers of candidates
#   {
#     "tier": "reference" | "tuning" | "scratch",
#     "candidates": [ { "path": "...", "source": "reference|tuning", "tags": [...] } ],
#     "recommendation": "compile_and_benchmark" | "implement_from_scratch"
#   }
#
# EXIT CODES:
#   0 — candidates found or scratch recommendation emitted
#   1 — argument error

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$SCRIPT_DIR/activity_trace.sh"

DSL=""
OPERATOR=""
DTYPE=""
GPU=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dsl)      DSL="$2";      shift 2 ;;
    --operator) OPERATOR="$2"; shift 2 ;;
    --dtype)    DTYPE="$2";    shift 2 ;;
    --gpu)      GPU="$2";      shift 2 ;;
    *) echo "[discover_baseline] ERROR: unknown arg: $1" >&2; echo "[SUGGESTION] Use your judgement to decide autonomously. Remove '$1' and retry. Valid args: --dsl --operator --dtype --gpu" >&2; exit 1 ;;
  esac
done

[[ -z "$DSL" ]]      && { echo "[discover_baseline] ERROR: --dsl required" >&2; echo "[SUGGESTION] Use your judgement to decide autonomously. Provide --dsl (cuda/croqtile/triton/cute/cutile/helion/tilelang)." >&2; exit 1; }
[[ -z "$OPERATOR" ]] && { echo "[discover_baseline] ERROR: --operator required" >&2; echo "[SUGGESTION] Use your judgement to decide autonomously. Provide --operator with the operation type (matmul/gemm/spmm/conv/attention)." >&2; exit 1; }

# Detect GPU if not provided
if [[ -z "$GPU" && -f "$SCRIPT_DIR/detect_gpu.sh" ]]; then
  GPU=$(bash "$SCRIPT_DIR/detect_gpu.sh" 2>/dev/null || echo "")
fi

# Map DSL to file extension
case "$DSL" in
  croqtile) EXT="co" ;;
  cuda)     EXT="cu" ;;
  triton|helion|tilelang) EXT="py" ;;
  cute|cutile) EXT="cu" ;;
  *) EXT="*" ;;
esac

# Map operator to reference directory patterns
op_lower=$(echo "$OPERATOR" | tr '[:upper:]' '[:lower:]')
case "$op_lower" in
  matmul|gemm|mm)      REF_PATTERNS=("matmul" "gemm" "matmul-tests" "gemm-tests") ;;
  spmm|sparse*|gemm_sp) REF_PATTERNS=("gemm_sp" "gemm-sp-tests") ;;
  conv*)                REF_PATTERNS=("conv") ;;
  attn*|flash*)         REF_PATTERNS=("attention" "flash") ;;
  *)                    REF_PATTERNS=("$op_lower") ;;
esac

if [[ -n "$GPU" ]]; then
    trace_init --gpu "$GPU" --dsl "$DSL" --shape-key "$OPERATOR"
fi
trace_event "discover_baseline" "Scanning for $OPERATOR ($DSL) candidates"

CANDIDATES=()
SOURCES=()
TAGS_LIST=()

# -------------------------------------------------------------------
# TIER 1: Scan reference examples (cursor-choreo-kernel-examples/)
# -------------------------------------------------------------------
REF_DIR="$REPO_ROOT/.cursor/skills/cursor-choreo-kernel-examples"
TIER1_COUNT=0

if [[ -d "$REF_DIR" ]]; then
  for pattern in "${REF_PATTERNS[@]}"; do
    search_dir="$REF_DIR/$pattern"
    [[ ! -d "$search_dir" ]] && continue

    while IFS= read -r -d '' f; do
      basename_f=$(basename "$f")

      # Skip _aitune_ artifacts
      if [[ "$basename_f" == *_aitune_* ]]; then
        echo "[discover_baseline] SKIP (aitune artifact): $basename_f" >&2
        continue
      fi

      # Apply dtype filter if specified
      if [[ -n "$DTYPE" ]]; then
        dtype_lower=$(echo "$DTYPE" | tr '[:upper:]' '[:lower:]')
        # Accept if filename contains dtype hint, or if no dtype in name (generic)
        fname_lower=$(echo "$basename_f" | tr '[:upper:]' '[:lower:]')
        has_dtype_in_name=false
        for dt in f16 f32 bf16 e4m3 e5m2 fp8 fp16 fp32; do
          if [[ "$fname_lower" == *"$dt"* ]]; then
            has_dtype_in_name=true
            break
          fi
        done
        if $has_dtype_in_name && [[ "$fname_lower" != *"$dtype_lower"* ]]; then
          continue
        fi
      fi

      rel_path="${f#$REPO_ROOT/}"
      tags=()
      [[ "$basename_f" == *warpspec* ]]  && tags+=("warpspec")
      [[ "$basename_f" == *prepack* ]]   && tags+=("prepack")
      [[ "$basename_f" == *swizzle* ]]   && tags+=("swizzle")
      [[ "$basename_f" == *persis* ]]    && tags+=("persistent")
      [[ "$basename_f" == *dyn* ]]       && tags+=("dynamic")
      [[ "$basename_f" == *sm90* ]]      && tags+=("sm90")
      [[ "$basename_f" == *sm80* ]]      && tags+=("sm80")
      [[ "$basename_f" == *sm86* ]]      && tags+=("sm86")
      [[ "$basename_f" == *1p1c* ]]      && tags+=("1p1c")
      [[ "$basename_f" == *1p2c* ]]      && tags+=("1p2c")
      [[ "$basename_f" == *1p3c* ]]      && tags+=("1p3c")
      [[ "$basename_f" == *stmatrix* ]]  && tags+=("stmatrix")
      [[ "$basename_f" == *regctrl* ]]   && tags+=("regctrl")

      CANDIDATES+=("$rel_path")
      SOURCES+=("reference")
      tags_str=$(IFS=,; echo "${tags[*]}")
      TAGS_LIST+=("$tags_str")
      TIER1_COUNT=$((TIER1_COUNT + 1))
    done < <(find "$search_dir" -maxdepth 1 -name "*.$EXT" -print0 2>/dev/null | sort -z)
  done
fi

echo "[discover_baseline] Tier 1 (reference): $TIER1_COUNT candidates" >&2

# -------------------------------------------------------------------
# TIER 2: Scan tuning/ for same-operator, different shapes
# -------------------------------------------------------------------
TIER2_COUNT=0

if [[ -n "$GPU" ]]; then
  TUNING_SRCS="$REPO_ROOT/tuning/$GPU/$DSL/srcs"
  if [[ -d "$TUNING_SRCS" ]]; then
    while IFS= read -r -d '' shape_dir; do
      shape_name=$(basename "$shape_dir")

      # Only consider same operator
      if [[ "$shape_name" != ${op_lower}_* ]]; then
        continue
      fi

      # Find the best iter (highest number) in any model subdirectory
      while IFS= read -r -d '' model_dir; do
        best_iter=""
        best_num=0
        while IFS= read -r -d '' src_file; do
          basename_src=$(basename "$src_file")
          # Extract iter number
          if [[ "$basename_src" =~ iter([0-9]+) ]]; then
            iter_num=${BASH_REMATCH[1]}
            iter_num=$((10#$iter_num))
            if [[ $iter_num -gt $best_num ]]; then
              best_num=$iter_num
              best_iter="$src_file"
            fi
          fi
        done < <(find "$model_dir" -maxdepth 1 -name "iter*.$EXT" -print0 2>/dev/null | sort -z)

        if [[ -n "$best_iter" ]]; then
          rel_path="${best_iter#$REPO_ROOT/}"
          CANDIDATES+=("$rel_path")
          SOURCES+=("tuning")
          TAGS_LIST+=("shape:$shape_name,best_iter:iter$(printf '%03d' $best_num)")
          TIER2_COUNT=$((TIER2_COUNT + 1))
        fi
      done < <(find "$shape_dir" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | sort -z)
    done < <(find "$TUNING_SRCS" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | sort -z)
  fi
fi

echo "[discover_baseline] Tier 2 (tuning/): $TIER2_COUNT candidates" >&2

# -------------------------------------------------------------------
# Build JSON output
# -------------------------------------------------------------------

TOTAL=$((TIER1_COUNT + TIER2_COUNT))

if [[ $TOTAL -eq 0 ]]; then
  trace_event "discover_baseline" "No candidates found — recommend from scratch"
  echo "[discover_baseline] No candidates found — recommend implement_from_scratch" >&2
  cat <<EOF
{"tier":"scratch","candidates":[],"recommendation":"implement_from_scratch","scratch_guidance":"Use web search to research ${op_lower} GPU kernel implementation patterns. Implement a version that uses MMA/tensor core instructions (not scalar loops). Target: correct + faster than naive scalar baseline."}
EOF
  exit 0
fi

# Determine tier
if [[ $TIER1_COUNT -gt 0 ]]; then
  TIER="reference"
else
  TIER="tuning"
fi

# Build JSON candidates array
JSON_CANDIDATES="["
first=true
for i in "${!CANDIDATES[@]}"; do
  tags_csv="${TAGS_LIST[$i]}"
  tags_json="[]"
  if [[ -n "$tags_csv" ]]; then
    tags_json=$(echo "$tags_csv" | tr ',' '\n' | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))")
  fi

  $first || JSON_CANDIDATES+=","
  first=false
  JSON_CANDIDATES+="{\"path\":\"${CANDIDATES[$i]}\",\"source\":\"${SOURCES[$i]}\",\"tags\":$tags_json}"
done
JSON_CANDIDATES+="]"

cat <<EOF
{"tier":"$TIER","candidates":$JSON_CANDIDATES,"recommendation":"compile_and_benchmark","total":$TOTAL}
EOF
