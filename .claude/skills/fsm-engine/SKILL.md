---
name: fsm-engine
description: CroqTuner FSM engine with per-DSL state/memory isolation, strict non-stop looping, and per-round transcript persistence. Use as the execution engine for croq-tuner and all DSL-specific ai-tune skills.
argument-hint: <dsl: croqtile|cuda|cute|triton|tilelang|helion|cutile> <dtype: f16|e4m3|all> [shape_key]
---

# CroqTuner FSM Engine (Migrated + Refined)

This is the execution engine for non-stop tuning. It does not stop at shape boundaries and it does not rely on chat memory.

## Explicit Read Requirement

Skill names mentioned in this file are not auto-loaded by the agent runtime. When this
file says to load another skill or protocol file, explicitly open and read that file
before proceeding.

## Mandatory Skill Load Order (Do This First)

On each tuning invocation, load these skills in order:

1. `.claude/skills/ai-tune-artifacts/SKILL.md`
2. `.claude/skills/fsm-engine/protocol/identity.md`
3. `.claude/skills/fsm-engine/protocol/loop-contract.md`
4. `.claude/skills/fsm-engine/protocol/iteration-execution.md`
5. `.claude/skills/fsm-engine/protocol/step-checklists.md`
6. `.claude/skills/fsm-engine/protocol/idea-diversity-rules.md`

Additionally, if `CROQTUNER_DSL=croqtile`, load:

7. `.claude/skills/choreo-syntax/SKILL.md`

Do not skip this load order.
Do not replace this load order with a paraphrase from memory.

## Non-Stop Keyphrase (Backup Rule)

Always keep this literal phrase in active context while running:

`NON_STOP_CONTINUE_WITHOUT_WAIT`

If anything conflicts, this keyphrase wins: continue the loop immediately.

## Per-DSL Isolation (Required)

Before any FSM command, set:

```bash
export CROQTUNER_DSL="<croqtile|cuda|cute|triton|tilelang|helion|cutile>"
# Optional override:
# export CROQTUNER_STATE_FILE=".claude/skills/fsm-engine/state/$CROQTUNER_DSL/loop-state.json"
```

State files are isolated per DSL:

- `.claude/skills/fsm-engine/state/<dsl>/loop-state.json`

Legacy root-level files in `.claude/skills/fsm-engine/state/` are archival snapshots.
Preserve them for reproducibility, but do not resume new sessions from them.

Artifacts are also isolated per DSL under:
- `tuning/aitune/<dsl>/logs/<key>/`
- `tuning/aitune/<dsl>/srcs/<key>/`
- `tuning/aitune/<dsl>/perf/<key>/`
- `tuning/aitune/<dsl>/checkpoints/<key>.json`
- `tuning/aitune/<dsl>/memory/<key>/rounds.raw.jsonl`
- `tuning/aitune/<dsl>/memory/<key>/rounds.md`

Resume is valid only from the current repository.
Reject any checkpoint, best-kernel path, or source include that resolves outside the
current git root, including sibling repositories such as `*-paper` workspaces.

Mandatory resume guard before continuing an `in_progress` shape:

```bash
python3 scripts/validate_tuning_session.py --dsl "$CROQTUNER_DSL"
```

If the guard fails, clean invalid work state first:

```bash
python3 scripts/clean_kernel_work_state.py --dsl "$CROQTUNER_DSL" --invalid-only
```

The execution model is single-workflow only:
- resume the latest valid state for the selected DSL
- keep pushing the current best forward
- do not introduce alternate tuning modes

## Executing a Step

For every state:

1) Pre-check:
```bash
bash .claude/skills/fsm-engine/scripts/pre-step-check.sh <STATE>
```

2) Execute the state actions from `protocol/step-checklists.md`.

3) Update guard flags/metrics via `SET`:
```bash
bash .claude/skills/fsm-engine/scripts/state-transition.sh SET <key=value ...>
```

4) Post-check:
```bash
bash .claude/skills/fsm-engine/scripts/post-step-check.sh <STATE>
```

5) Transition:
```bash
bash .claude/skills/fsm-engine/scripts/state-transition.sh <NEXT_STATE> [key=value ...]
```

Then continue immediately. Do not pause for summaries.

## Canonical Loop Body (Must Be Followed)

For every iteration:

1. `PROFILE` — run profiling and identify bottleneck
2. `IDEATE` — raise one novel, data-grounded idea and use online/vendor references when profiler data or local history alone are insufficient
3. `IMPLEMENT` — implement and compile (up to 3 retries)
4. `MEASURE` — benchmark and parse TFLOPS
5. `DECIDE` — KEEP or DISCARD
6. `STORE` — save artifacts + records even on DISCARD

Do not skip steps. A discarded iteration still persists artifacts and logs.

`iter000` is always the trivial, simplest correct baseline implementation plus timing, verification, and environment scaffolding.
Compile-failed attempts remain internal `attempt<AAAA>` records and do not consume public `iter<NNN>` ids.

## Branch Policy (Required)

Exactly one long-lived tuning branch per DSL:
- `aitune/croqtile`
- `aitune/cuda`
- `aitune/cute`
- `aitune/triton`
- `aitune/tilelang`
- `aitune/helion`
- `aitune/cutile`

No date branches, no resume suffix branches.

## Per-Round Session History Persistence (Required)

At every STORE step, save BOTH:

- Raw conversation history (JSONL): `paths.round_memory_raw_log`
- Full text markdown transcript: `paths.round_memory_md_log`

Set:
- `round_memory_raw_saved=true`
- `round_memory_md_saved=true`

`post-step-check.sh STORE` enforces both files and current-iteration entries.

## Early-Break Diagnosis and Fixes

Observed early-stop causes:
1. Shared state files across sessions/modes caused cross-session clobber.
2. Shape completion was treated as a natural stopping point by higher-level skills.
3. Round-memory persistence was not enforced, so resume quality degraded.

Refinements now enforced in this engine:
- Per-DSL/per-mode state files by default.
- Non-stop loop mandate encoded in state (`non_stop_keyphrase`) and transition flow.
- STORE step requires raw + markdown round-history persistence.
- SHAPE_COMPLETE no longer allows relaxed early completion for regular sweep modes.

## Protocol Files Reference

| File | When to Read | Content |
|---|---|---|
| `protocol/identity.md` | Every invocation | Inviolable non-stop constraints |
| `protocol/loop-contract.md` | Fresh start or confusion | FSM transitions and no-stop rules |
| `protocol/iteration-execution.md` | Before IMPLEMENT/STORE | Attempt-vs-iteration execution contract |
| `protocol/step-checklists.md` | Before each step | Required actions and guard flags |
| `protocol/compaction-protocol.md` | After compaction | Resume order with per-DSL state |
| `protocol/idea-diversity-rules.md` | Before IDEATE | Dedup and escalation rules |

Always resume from the current state file first, then continue immediately.
If the current state file is missing, only resume from local checkpoint + local artifacts
after validation passes; otherwise invalidate the broken work state and restart from INIT.

## Experimental Loop Integration

The same FSM engine governs:
- the live tuning workflow
- experimental validation loops, including A/B skill experiments

Experimental harnesses may simulate or audit the loop, but the protocol files in this directory remain the single source of truth.
