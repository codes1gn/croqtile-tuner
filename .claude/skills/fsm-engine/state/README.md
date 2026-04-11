Legacy root-level JSON files in this directory are preserved snapshots from the
pre-migration FSM layout.

For new tuning sessions, use only per-DSL active state files:

- `.claude/skills/fsm-engine/state/<dsl>/loop-state.json`

The per-DSL directories and active state files are created lazily by `state-transition.sh INIT`.

`iter000` is the trivial measured baseline for a session. Mode-specific root-level files and other root-level snapshots are legacy artifacts kept for reproducibility.

Do not delete the legacy files unless the user explicitly asks. They are kept for
reproducibility and postmortem analysis.