---
name: ai-tune-croqtile
description: Non-stop tuning skill for CroqTile/Choreo DSL on branch aitune/croqtile with isolated state and memory.
---

# AI Tune CroqTile

Use this for Choreo/CroqTile kernel tuning.

## Non-Stop Keyphrase

`NON_STOP_CONTINUE_WITHOUT_WAIT`

## Required setup

```bash
export CROQTUNER_DSL="croqtile"
```

Branch:
- `aitune/croqtile` only

State and memory are isolated automatically by `fsm-engine`.

## Required Read Set

Referenced skills are not auto-loaded. Explicitly read:

1. `.claude/skills/ai-tune-artifacts/SKILL.md`
2. `.claude/skills/fsm-engine/SKILL.md`
3. `.claude/skills/fsm-engine/protocol/identity.md`
4. `.claude/skills/fsm-engine/protocol/loop-contract.md`
5. `.claude/skills/fsm-engine/protocol/iteration-execution.md`
6. `.claude/skills/fsm-engine/protocol/step-checklists.md`
7. `.claude/skills/fsm-engine/protocol/idea-diversity-rules.md`
8. `.claude/skills/choreo-syntax/SKILL.md`

## Per-Iteration Implementation Policy

The live loop is always `croq-tuner -> fsm-engine`. Experimental or A/B validators should audit the same protocol files rather than define a separate workflow.

For each idea:
1. Try pure `.co` first.
2. If needed, use `__cpp__` inside `.co` for inline CUDA/CuTe/PTX.
3. Only when unsupported in `.co` (e.g. detailed sync-point mutation), modify generated `.cu`.

Additional rules:
- `iter000` is the trivial scalar-loop CroqTile baseline used to establish timing, verification, and environment scaffolding.
- Compile-failed attempts stay as `attempt<AAAA>` and do not consume public `iter<NNN>` ids.
- During IDEATE, use current external references when profiler data alone is not enough to justify the next CroqTile change.

Always load:
- `.claude/skills/choreo-syntax/SKILL.md`
- `.claude/skills/ai-tune-artifacts/SKILL.md`

Then run via:
- `.claude/skills/croq-tuner/SKILL.md`
- `.claude/skills/fsm-engine/SKILL.md`
