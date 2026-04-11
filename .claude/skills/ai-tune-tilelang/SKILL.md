---
name: ai-tune-tilelang
description: Non-stop tuning skill for pure TileLang baseline on branch aitune/tilelang with isolated state and memory.
---

# AI Tune TileLang

Use this for pure TileLang tuning.

## Non-Stop Keyphrase

`NON_STOP_CONTINUE_WITHOUT_WAIT`

## Required setup

```bash
export CROQTUNER_DSL="tilelang"
```

Branch:
- `aitune/tilelang` only

Only tune TileLang artifacts in this session; do not modify other DSL kernels.

## Required Read Set

Referenced skills are not auto-loaded. Explicitly read:

1. `.claude/skills/ai-tune-artifacts/SKILL.md`
2. `.claude/skills/fsm-engine/SKILL.md`
3. `.claude/skills/fsm-engine/protocol/identity.md`
4. `.claude/skills/fsm-engine/protocol/loop-contract.md`
5. `.claude/skills/fsm-engine/protocol/iteration-execution.md`
6. `.claude/skills/fsm-engine/protocol/step-checklists.md`
7. `.claude/skills/fsm-engine/protocol/idea-diversity-rules.md`

Always load:
- `.claude/skills/ai-tune-artifacts/SKILL.md`

Then execute through `croq-tuner` + `fsm-engine`.

Additional rules:
- `iter000` is the trivial scalar-loop TileLang baseline used to establish timing, verification, and environment scaffolding.
- Compile-failed attempts remain `attempt<AAAA>` and do not consume public `iter<NNN>` ids.
- During IDEATE, consult current external references when profiler data and local history are insufficient.
