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
- results row in `logs/<key>/results.tsv`
- checkpoint update in `checkpoints/<key>.json`
- **MANDATORY memory updates** (see Memory Files section below)

For a failed internal `attempt<AAAA>`, persist:

- attempted source snapshot
- build script
- build log or stderr
- failure metadata in `logs/<key>/attempt-log.jsonl`
- **MANDATORY memory updates** (see Memory Files section below)

Attempt records do not consume the public measured iteration sequence.

## Memory Files (MANDATORY)

**Every STORE step MUST update these files:**

1. `memory/<key>/rounds.raw.jsonl` — Append one JSON line:
   ```json
   {"iter": "iter<NNN>", "kernel": "<kernel_name>", "tflops": <float>, "decision": "<KEEP|DISCARD|SEGFAULT|HANG>", "bottleneck": "<bottleneck>", "idea": "<idea_summary>", "timestamp": "<ISO8601>"}
   ```

2. `memory/<key>/rounds.md` — Append markdown section:
   ```markdown
   ## iter<NNN> - <timestamp>
   - kernel: `<kernel_name>`
   - tflops: `<tflops>`
   - decision: **<decision>**
   - bottleneck: `<bottleneck>`
   - idea: <idea_summary>
   ```

3. `logs/<key>/idea-log.jsonl` — Append one JSON line:
   ```json
   {"round": <N>, "iter": "iter<NNN>", "bottleneck": "<bottleneck>", "idea": "<idea>", "category": "<category>", "expected_gain": "<gain>", "timestamp": "<ISO8601>"}
   ```

**Failure to update these files breaks session resumption and history tracking.**

## Raw Session Transcript (MANDATORY)

**At session end or context compaction, MUST preserve the raw chat transcript:**

1. Locate the current session JSONL file:
   ```
   ~/.cursor/projects/<project-slug>/agent-transcripts/<session-id>/<session-id>.jsonl
   ```

2. Copy to tuning memory:
   ```
   memory/<key>/sessions/<session-id>.jsonl
   ```

3. Create/update session index `memory/<key>/sessions/index.jsonl`:
   ```json
   {"session_id": "<uuid>", "start_iter": <N>, "end_iter": <M>, "tokens_approx": <bytes/4>, "timestamp": "<ISO8601>"}
   ```

**Why:** Raw transcripts enable post-hoc analysis of token usage, reasoning quality, and idea generation patterns across tuning runs.

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
