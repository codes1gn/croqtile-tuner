#!/usr/bin/env bash
# Fix: opencode fails with:
#   EACCES: permission denied, open '/home/.../.opencode/package.json'
# when ~/.opencode was partially installed or updated as root.
#
# Usage (one-time):
#   bash scripts/fix-opencode-permissions.sh

set -euo pipefail

OC="${HOME}/.opencode"
if [[ ! -d "$OC" ]]; then
  echo "No directory: $OC (is opencode installed?)"
  exit 1
fi

owner=$(stat -c '%U' "$OC/package.json" 2>/dev/null || echo "missing")
echo "Current owner of ~/.opencode/package.json: $owner"

if [[ "$owner" != "$(whoami)" && "$owner" != "missing" ]]; then
  echo ""
  echo "Fix: give your user ownership of the install tree (requires sudo once):"
  echo "  sudo chown -R $(whoami):$(whoami) \"$OC\""
  echo ""
  echo "Then verify:"
  echo "  opencode --version"
  echo "  opencode providers list"
  exit 2
fi

echo "Ownership looks OK. Testing..."
opencode --version
echo "OK."
