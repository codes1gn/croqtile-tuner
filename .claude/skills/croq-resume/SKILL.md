---
name: croq-resume
description: On-demand startup and resume contract for croq-tune. Use only when deciding fresh start vs resume or rehydrating a live tuning session after interruption.
---

# Croq-Resume

Use this skill only at startup, restart, or compaction recovery.

## Responsibilities

- decide resume vs fresh start from local repo state only
- validate that the active tuning session stays inside the current repo
- rehydrate only the minimum hot context needed to continue
- keep resume/cleanup helper tooling self-contained under `.claude/skills/croq-resume/tools/`

## Rules

- do not inspect sibling repos, copied worktrees, or external manifests
- do not trust stale summaries without validating local artifacts
- do not rebuild history from scratch if a valid local active session exists

## Continuation Node Before Compaction/Resume

Before proactive compaction or explicit resume handoff, ensure a continuation node exists:

- `Continue /croq-tune <dsl> <dtype> [shape_key]`

Framework execution:

1. Cursor IDE: use `TodoWrite` with id `continue-croq-tune` and status `in_progress`
2. OpenCode: use exposed todo tool if available; otherwise write equivalent state into `.agent/todo.json`
3. Copilot VSCode IDE: maintain `.agent/todo.json` with `continue-croq-tune` as `in_progress`

Use the same id/payload semantics defined in `croq-tune` -> `Continuation Anchor (Framework-Specific)`.

This node is a workflow anchor only; resume correctness still depends on validated local artifacts.

## Validation Path

Before resuming, run:

```bash
python3 .claude/skills/croq-resume/tools/validate_tuning_session.py --dsl "$CROQTUNER_DSL"
```

If validation fails, clean invalid local work state:

```bash
python3 .claude/skills/croq-resume/tools/clean_kernel_work_state.py --dsl "$CROQTUNER_DSL" --invalid-only
```

If the user asked for a fresh restart, or the local state is stale or ambiguous, reset local stateful artifacts:

```bash
python3 .claude/skills/croq-resume/tools/clean_kernel_work_state.py --dsl "$CROQTUNER_DSL" --reset-all
```

## Resume Read Set

After choosing resume, keep the read set minimal:

1. active session state
2. current best kernel
3. recent results
4. recent round log entries
5. active checkpoint or summary

Do not reload the entire protocol tree just to resume one active round.
