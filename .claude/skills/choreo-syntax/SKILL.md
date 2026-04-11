---
name: choreo-syntax
description: CroqTile/Choreo `.co` implementation policy and syntax guidance for tuning loops.
---

# Choreo Syntax and Implementation Policy

Load this skill whenever `dsl=croqtile` and an iteration may modify `.co`.

## Core Policy (Per Iteration)

For each new idea:

1. First ask: can this idea be implemented in pure `.co`?
2. If yes, implement in `.co` first.
3. If partial low-level logic is needed, use inlined `__cpp__` inside `.co` (CUDA/CuTe/PTX snippets).
4. If lower-level control is still needed, compile the chosen `.co` base and modify the generated `.cu` from that base.
5. Only if the idea is not representable through `.co` plus generation, modify the selected base `.cu` directly.

Do not jump directly to `.cu` for CroqTile unless step 1-3 are truly infeasible.

## Scope Guardrail

This skill is only for `dsl=croqtile`.
When the target DSL is not CroqTile, do not use `.co` as the implementation language.

## Documentation Requirement

When fallback to `.cu` is used, record reason in `idea-log.jsonl`:
- `co_supported: false`
- `fallback_reason: "<why .co/__cpp__ is insufficient>"`
- `fallback_base: "generated_cu_from_co|existing_iter_base_cu"`
