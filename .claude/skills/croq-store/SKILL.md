---
name: croq-store
description: On-demand result serialization contract for croq-tune. Use only when storing a round outcome, naming artifacts, or writing checkpoints and summaries.
---

# Croq-Store

Use this skill only when serializing the result of a round.

## Load Order

Before deciding artifact names or locations, load:

1. `.claude/skills/croq-artifacts/SKILL.md`

Do not keep artifact naming rules in entry-skill hot context.

## Store Rule

Every round must serialize its outcome, including DISCARD rounds.

For a public measured `iter<NNN>` result, persist:

- source snapshot
- build and run scripts
- timing output
- profile output when profiling ran
- results row
- checkpoint or session summary update
- round log entry

For a failed internal `attempt<AAAA>`, persist:

- attempted source snapshot
- build script
- build log or stderr
- failure metadata

Attempt records do not consume the public measured iteration sequence.

## Post-STORE Continuation Update

After STORE finishes, refresh workflow continuation state:

1. mark `round-step` as `completed`
2. ensure `continue-croq-tune` is `in_progress`

Use the framework-specific ids/payload rules defined in `croq-tune` -> `Continuation Anchor (Framework-Specific)`.

## Path Rule

- keep artifacts under `tuning/aitune/<dsl>/...`
- never invent external artifact roots
- never serialize resume-critical state outside the current repo

This skill owns naming and serialization detail so the entry skill can stay focused on the infinite loop.
