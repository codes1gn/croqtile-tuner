#!/usr/bin/env bash
# Pull tuning data + refresh skills/kernels from a sibling croktile_paper checkout.
# Usage: from CroqTuner repo root:
#   ./scripts/sync_paper_assets.sh [path/to/croktile_paper]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
SRC="${1:-$(dirname "$ROOT")/croktile_paper}"
if [[ ! -d "$SRC" ]]; then
  echo "Source not found: $SRC" >&2
  exit 1
fi
echo "Syncing from: $SRC -> $ROOT"
rsync -a "$SRC/tuning/" "$ROOT/tuning/"
rsync -a "$SRC/.claude/skills/" "$ROOT/.claude/skills/"
rsync -a "$SRC/kernels/" "$ROOT/kernels/"
echo "Done. Re-seed the task DB if needed:"
echo "  cd backend && source .venv/bin/activate && python ../scripts/seed_from_state.py"
