---
name: croq-tuner
description: Unified AI-guided kernel tuning entrypoint. Dispatches tuning/status/summarize for CroqTile and other DSLs using a single non-stop FSM engine with per-DSL isolation.
---

# CroqTuner Unified Dispatcher

This is the main entrypoint for tuning requests.

## Explicit Read Requirement

Referenced skills are not auto-loaded just because their paths are mentioned here.
For every tuning invocation, explicitly open and read the files listed below before
starting or resuming the loop.

## Required Read Set

Always read, in this order:

1. `.claude/skills/ai-tune-artifacts/SKILL.md`
2. `.claude/skills/fsm-engine/SKILL.md`
3. `.claude/skills/fsm-engine/protocol/identity.md`
4. `.claude/skills/fsm-engine/protocol/loop-contract.md`
5. `.claude/skills/fsm-engine/protocol/iteration-execution.md`
6. `.claude/skills/fsm-engine/protocol/step-checklists.md`
7. `.claude/skills/fsm-engine/protocol/idea-diversity-rules.md`

Additionally, when `dsl=croqtile`, read:

8. `.claude/skills/choreo-syntax/SKILL.md`

Do not substitute a summary of these files for actually reading them.

## Non-Stop Keyphrase (Backup Rule)

Always keep this literal phrase in context while tuning:

`NON_STOP_CONTINUE_WITHOUT_WAIT`

Never stop at shape boundaries. Never summarize-and-wait between loop rounds.

## Intent Types

Classify user request into:

- `TUNE` (default): start or resume non-stop tuning
- `STATUS`: report current FSM state for a DSL
- `SUMMARIZE`: summarize results already collected

If ambiguous, use `TUNE`.

## DSL and Branch Mapping (Single Branch Per DSL)

Use exactly one long-lived branch per DSL:

| DSL | Branch |
|---|---|
| `croqtile` | `aitune/croqtile` |
| `cuda` | `aitune/cuda` |
| `cute` | `aitune/cute` |
| `triton` | `aitune/triton` |
| `tilelang` | `aitune/tilelang` |
| `helion` | `aitune/helion` |
| `cutile` | `aitune/cutile` |

No dated branches. No `-resume-N` branches.

## Per-DSL State and Memory

Before any FSM call:

```bash
export CROQTUNER_DSL="<dsl>"
```

Defaults:
- `dsl=croqtile`

State file:
- `.claude/skills/fsm-engine/state/<dsl>/loop-state.json`

Artifacts:
- `tuning/aitune/<dsl>/...`

Resume sources are valid only if they stay inside the current git repository.
Never resume from sibling repositories, copied worktrees, or absolute-source includes
that point outside this repo root.

Before resuming any `in_progress` shape, run:

```bash
python3 scripts/validate_tuning_session.py --dsl "$CROQTUNER_DSL"
```

If validation fails, clean invalid work state and restart from `INIT` using only local
artifacts in this repo:

```bash
python3 scripts/clean_kernel_work_state.py --dsl "$CROQTUNER_DSL" --invalid-only
```

The workflow is always the same:
- resume the latest valid progress for the selected DSL
- keep pushing the current best forward
- never branch behavior by artificial tuning modes

## TUNE Flow

1. Parse request: `dsl`, `dtype`, optional shape.
2. Checkout/create the mapped DSL branch (`aitune/<dsl>`).
3. Read the active state file for the selected DSL.
4. Validate that the active loop-state, checkpoint, best-kernel path, and source includes
  all resolve inside the current repo.
5. If no active loop-state exists, do not blindly trust `tuning/aitune/<dsl>/state.json`.
  Only resume if the local checkpoint and local artifacts validate cleanly; otherwise
  clean invalid work state and restart from `INIT`.
6. If no state: initialize FSM with `INIT`.
7. Explicitly read the full required skill set above.
8. Execute FSM loop through `fsm-engine` continuously.
9. Do not pause after a shape; transition to NEXT_SHAPE and continue.

`fsm-engine` is the authoritative runtime loop for both:
- live tuning work
- validation or experimental loops such as A/B skill experiments

Experimental tasks must validate against this FSM contract, not redefine it.

## Canonical Iteration Loop

Every tuning iteration must follow the same loop body:

1. `PROFILE` — inspect the current kernel and identify the bottleneck.
2. `IDEATE` — raise one new, data-grounded idea, including online/vendor research when local evidence is insufficient or the search is stalled.
3. `IMPLEMENT` — apply the idea and compile, with up to 3 fix attempts.
4. `MEASURE` — benchmark and parse TFLOPS.
5. `DECIDE` — KEEP or DISCARD.
6. `STORE` — persist artifacts, checkpoint, and round transcript even on DISCARD.

No dispatcher mode may skip `STORE` for a discarded iteration.

Additional invariant:
- `iter000` is the trivial reference baseline for the selected DSL, using the simplest correct scalar-loop implementation plus timing, verification, and environment setup.
- compile-failed attempts do not consume public `iter<NNN>` ids.

## DSL Implementation Guardrails

- If `dsl=croqtile`:
  1. Try implementing each idea in pure `.co` first.
  2. If needed, use `__cpp__` blocks in `.co` for inline CUDA/CuTe/PTX fragments.
  3. Only if unsupported in `.co` (for example detailed sync-point mutation), edit generated `.cu`.
- If `dsl!=croqtile`:
  - Do not tune through other DSLs; stay strictly in the selected target DSL.

## STATUS Flow

Read the DSL state file and report:
- `fsm.current_state`, `fsm.iteration`, `fsm.max_iteration`
- best TFLOPS, baseline TFLOPS, discard streak
- last bottleneck
- memory log paths

## Legacy State Note

Root-level files under `.claude/skills/fsm-engine/state/*.json` are legacy snapshots
kept for reproducibility. New tuning sessions must use only:

- `.claude/skills/fsm-engine/state/<dsl>/loop-state.json`

## SUMMARIZE Flow

Summarize existing artifacts only. Do not interrupt active non-stop loops.

## Required Companion Skill

Always use:
- `.claude/skills/fsm-engine/SKILL.md`

The FSM engine is the authority for transition rules and enforcement.
