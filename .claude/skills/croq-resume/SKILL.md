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

## Resume Read Set — Use the Harness (MANDATORY)

**DO NOT manually read rounds.raw.jsonl, results.tsv, etc. to reconstruct state.**  
Use `resume_state.sh` instead. It reads all sources and emits a single JSON snapshot:

```bash
STATE=$(bash .claude/skills/croq-resume/resume_state.sh \
    --dsl <dsl> --shape-key <shape_key>)
echo "$STATE"
```

The JSON contains everything needed to continue:

| Field | Meaning |
|---|---|
| `current_best_tflops` | Best custom kernel performance so far |
| `current_best_kernel` | Name of the best kernel file |
| `last_round` | Round number of the last stored result |
| `last_iter` | Last iter name (may be DISCARD) |
| `last_decision` | KEEP or DISCARD |
| `last_bottleneck` | Bottleneck from last round |
| `next_iter_number` | What number to use for the next `iter<NNN>` |
| `src_count` | Total iter source files present |
| `open_checkpoint` | Non-null if IDEA wrote a plan but VERIFY was not completed |
| `memory_files_ok` | true if all 4 memory/log files are present |
| `warnings` | List of issues detected (malformed JSON, missing files, etc.) |

**After loading state:**

1. If `open_checkpoint` is non-null: the last IDEA wrote a plan but implementation was interrupted.
   Resume from IMPLEMENT using that checkpoint (call `checkpoint_write.sh read`).
2. If `warnings` is non-empty: address each warning before continuing the loop.
3. Otherwise: pick up from `last_round + 1`, using `next_iter_number` for the next IMPLEMENT.

Do not reload the entire protocol tree just to resume one active round.
