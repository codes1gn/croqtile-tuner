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

## Memory Files — Use the Harness (MANDATORY)

**Every STORE step MUST call `store_round.sh`.** Do NOT manually append to individual files.
The harness writes all 4 mandatory files atomically and verifies them before returning.

```bash
bash .claude/skills/croq-store/store_round.sh \
  --dsl        <dsl> \
  --shape-key  <shape_key> \
  --iter       iter<NNN> \
  --kernel     iter<NNN>_<tag> \
  --tflops     <float> \
  --decision   <KEEP|DISCARD|SEGFAULT|HANG|COMPILE_FAIL> \
  --bottleneck <bottleneck> \
  --idea       "<one-line idea summary>" \
  --round      <N> \
  --category   "<tiling|pipeline|memory|compute|misc>" \
  --expected-gain "<e.g. +5% TFLOPS>"
```

The harness writes and verifies:
1. `memory/<key>/rounds.raw.jsonl` — JSON line per round
2. `memory/<key>/rounds.md` — markdown section per round
3. `logs/<key>/idea-log.jsonl` — JSON line per round
4. `logs/<key>/results.tsv` — TSV row per round (creates header if missing)

If the harness exits non-zero, fix the error before proceeding to CONTINUE.
Do NOT proceed to CONTINUE until the harness prints `[store_round] STORE complete`.

**NEVER bypass the harness by writing files manually.** Manual writes have been
the root cause of missing memory files across sessions.

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
