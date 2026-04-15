#!/usr/bin/env bash
# _rename_bare_iters.sh — One-time rename of bare/bad-tag iter source files.
# Run from the repo root:
#   bash tuning/sm90_H100/cuda/srcs/matmul_bf16fp32_512x16384x16384/_rename_bare_iters.sh
#
# Strategy:
#   1. Bare files (iter<NNN>.cu): extract tag from first-line comment.
#   2. Bad-tag files (tag starts with digit or has uppercase): lowercased + normalized.
#   3. Rename matching bin/ directory if it exists.
#   4. Dry-run by default — pass --apply to actually rename.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SRC="tuning/sm90_H100/cuda/srcs/matmul_bf16fp32_512x16384x16384"
BIN="tuning/sm90_H100/cuda/bin/matmul_bf16fp32_512x16384x16384"
DRY=1
[[ "${1:-}" == "--apply" ]] && DRY=0

TAG_REGEX='^[a-z][a-z0-9_]{1,15}$'
RENAMES=0
SKIP=0
ERRORS=0

do_rename() {
    local old_src="$1"
    local new_src="$2"
    local old_name=$(basename "$old_src" .cu)
    local new_name=$(basename "$new_src" .cu)

    if [[ "$old_src" == "$new_src" ]]; then
        return
    fi

    if [[ -f "$new_src" ]]; then
        echo "  SKIP  $old_name → $new_name  (target already exists)"
        SKIP=$((SKIP + 1))
        return
    fi

    if [[ "$DRY" -eq 1 ]]; then
        echo "  WOULD RENAME  $old_name.cu → $new_name.cu"
    else
        git mv "$old_src" "$new_src"
        echo "  RENAMED  $old_name.cu → $new_name.cu"
    fi
    RENAMES=$((RENAMES + 1))

    # Rename bin/ directory if it exists (bare name only, no extension)
    old_bin_dir="$BIN/$old_name"
    new_bin_dir="$BIN/$new_name"
    if [[ -d "$old_bin_dir" ]]; then
        if [[ "$DRY" -eq 1 ]]; then
            echo "  WOULD RENAME  bin/$old_name/ → bin/$new_name/"
        else
            git mv "$old_bin_dir" "$new_bin_dir"
            echo "  RENAMED  bin/$old_name/ → bin/$new_name/"
        fi
    fi
}

normalize_tag() {
    # lowercase, replace spaces/hyphens with underscores, strip leading digits
    local tag="$1"
    tag=$(echo "$tag" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr ' ' '_')
    # Strip leading digits
    tag=$(echo "$tag" | sed 's/^[0-9_]*//')
    # Truncate to 15 chars (max is 16 total including first char)
    tag="${tag:0:15}"
    # Strip trailing underscores
    tag="${tag%_}"
    echo "$tag"
}

echo "=== Rename Plan (DSL: cuda, Shape: matmul_bf16fp32_512x16384x16384) ==="
[[ "$DRY" -eq 1 ]] && echo "  DRY RUN — pass --apply to execute"
echo ""

# ── Category 1: Truly bare files (iter<NNN>.cu, no underscore) ────────────────
echo "--- Bare iter files (extract tag from first-line comment) ---"
for f in "$SRC"/iter[0-9][0-9][0-9].cu; do
    [[ -f "$f" ]] || continue
    base=$(basename "$f" .cu)            # e.g. iter076
    num=$(echo "$base" | grep -oE '[0-9]{3}')

    # Extract tag from first-line comment: "// iter076_swizzle.cu - ..."
    comment_tag=$(head -1 "$f" | grep -oE "iter${num}_[A-Za-z][A-Za-z0-9_]+" | \
                  sed "s/iter${num}_//" | head -1 || echo "")

    if [[ -z "$comment_tag" ]]; then
        echo "  NO COMMENT TAG: $base.cu — skipping (manual rename needed)"
        SKIP=$((SKIP + 1))
        continue
    fi

    tag=$(normalize_tag "$comment_tag")
    if ! echo "$tag" | grep -qE "$TAG_REGEX"; then
        echo "  BAD TAG: $base.cu → derived '$tag' still invalid — skipping"
        SKIP=$((SKIP + 1))
        continue
    fi

    new_name="iter${num}_${tag}"
    do_rename "$SRC/${base}.cu" "$SRC/${new_name}.cu"
done

echo ""
echo "--- Bad-tag iter files (uppercase/digit-leading tag → normalized) ---"
for f in "$SRC"/iter[0-9][0-9][0-9]_*.cu; do
    [[ -f "$f" ]] || continue
    base=$(basename "$f" .cu)            # e.g. iter016_wideN
    num=$(echo "$base" | grep -oE '^iter([0-9]{3})' | grep -oE '[0-9]{3}')
    raw_tag="${base#iter${num}_}"        # e.g. wideN or 2warp

    # Already valid?
    if echo "$raw_tag" | grep -qE "$TAG_REGEX"; then
        continue   # fine — skip
    fi

    tag=$(normalize_tag "$raw_tag")
    if ! echo "$tag" | grep -qE "$TAG_REGEX"; then
        echo "  BAD TAG: $base.cu → derived '$tag' still invalid — skipping"
        SKIP=$((SKIP + 1))
        continue
    fi

    if [[ "$tag" == "$raw_tag" ]]; then
        continue
    fi

    new_name="iter${num}_${tag}"
    do_rename "$SRC/${base}.cu" "$SRC/${new_name}.cu"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Renames: $RENAMES   Skipped: $SKIP   Errors: $ERRORS"
[[ "$DRY" -eq 1 ]] && echo "  Run with --apply to execute."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
