#!/usr/bin/env bash
# validate_artifact_names.sh — PostToolUse Guard hook for croq-tune.
#
# Fires after every Write or Bash tool call.
# Warns (exit 1 = non-blocking) when it detects:
#   • A source file being written without a _tag suffix (bare iter<NNN>.cu)
#   • A ncu CSV being written without a _tag suffix (ncu_iter<NNN>.csv)
#
# Wired via .claude/settings.json PostToolUse on Write|Edit|Bash.
# Exit code 1 = non-blocking warning (agent sees stderr, operation still runs).
# Exit code 2 = blocking error (use only for security-critical violations).
#
# We use exit 1 here — we WARN the agent, we don't block, because a blocking
# hook on every Write would break legitimate operations. The agent should
# self-correct on seeing the warning.

set -euo pipefail

INPUT=$(cat)

# Extract tool name and relevant fields from the JSON payload
TOOL=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_name',''))" 2>/dev/null || echo "")
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('path',''))" 2>/dev/null || echo "")
CMD=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('command',''))" 2>/dev/null || echo "")

# ── check Write/Edit tool: validate the target file path ─────────────────────
if [[ "$TOOL" == "Write" || "$TOOL" == "Edit" || "$TOOL" == "StrReplace" ]]; then
    BASENAME=$(basename "$FILE_PATH" 2>/dev/null || echo "")

    # Pattern: bare iter file — iter<NNN>.<ext> with no _tag
    # Matches: iter012.cu, iter003.co, iter099.py
    # Does NOT match: iter012_myidea.cu (correct)
    if echo "$BASENAME" | grep -qE '^iter[0-9]{3}\.[a-z]+$'; then
        cat >&2 <<EOF
[validate_artifact_names] WARNING: Bare iteration filename detected.
  File: $FILE_PATH
  Name: $BASENAME
  Rule: Every iter source file MUST include a descriptive _tag suffix.
  Fix:  Rename to iter$(echo "$BASENAME" | grep -oE '[0-9]{3}')_<tag>.$(echo "$BASENAME" | grep -oE '[^.]+$')
  Ref:  .cursor/skills/croq-artifacts/SKILL.md
EOF
        exit 1
    fi

    # Pattern: bare ncu CSV — ncu_iter<NNN>.csv with no _tag
    if echo "$BASENAME" | grep -qE '^ncu_iter[0-9]{3}\.csv$'; then
        cat >&2 <<EOF
[validate_artifact_names] WARNING: Bare ncu CSV filename detected.
  File: $FILE_PATH
  Name: $BASENAME
  Rule: ncu CSVs MUST be named ncu_iter<NNN>_<tag>.csv to match the source iter.
  Fix:  Rename to ncu_iter$(echo "$BASENAME" | grep -oE '[0-9]{3}')_<tag>.csv
  Ref:  .cursor/skills/croq-profile/SKILL.md
EOF
        exit 1
    fi
fi

# ── check Bash tool: look for mv/cp/touch that creates bare iter files ────────
if [[ "$TOOL" == "Bash" && -n "$CMD" ]]; then
    # Flag any command that would create/rename to a bare iter filename
    if echo "$CMD" | grep -qE '(mv|cp|touch|nvcc|croqc|python)[^;|]*iter[0-9]{3}\.(cu|co|py|ptx)\b'; then
        # Exclude correctly tagged names
        if ! echo "$CMD" | grep -qE 'iter[0-9]{3}_[a-z][a-z0-9_]{1,15}\.(cu|co|py|ptx)'; then
            cat >&2 <<EOF
[validate_artifact_names] WARNING: Command may produce a bare iter filename.
  Command: $(echo "$CMD" | head -c 200)
  Rule: All iter source files must include a _tag suffix (iter<NNN>_<tag>.<ext>).
  Ref:  .cursor/skills/croq-artifacts/SKILL.md
EOF
            exit 1
        fi
    fi
fi

exit 0
