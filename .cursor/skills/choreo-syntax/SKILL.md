---
name: choreo-syntax
description: Reference CroqTile/Choreo `.co` syntax, implementation policy, and imported optimization patterns for croqtile tuning.
---

# Choreo Syntax and Implementation Policy

Use this skill whenever `dsl=croqtile` and an iteration may modify `.co`.

## Required Reference

Before editing `.co`, also read:

- `.claude/skills/choreo-syntax/choreo-syntax-reference.md`

That reference was imported from the sibling CroqTile repo so this tuner repo has the same syntax and performance guidance available locally.

## Core Behavior

- Treat this skill as CroqTile-specific guidance layered on top of the shared FSM loop.
- Match existing CroqTile/Choreo syntax instead of inventing new spellings.
- Prefer repo-observed patterns for tiling, copy primitives, and MMA forms.
- Keep the target DSL pure: this skill only applies when `dsl=croqtile`.

## CroqTile Implementation Priority

For each new idea:

1. First ask whether the idea can be implemented in pure `.co`.
2. If yes, implement it in `.co` first.
3. If partial low-level logic is needed, use inlined `__cpp__` inside `.co` for CUDA/CuTe/PTX fragments.
4. If lower-level control is still needed, compile the chosen `.co` base and modify the generated `.cu` from that base.
5. Only if the idea is not representable through `.co` plus generation should you modify the selected base `.cu` directly.

Do not jump directly to `.cu` for CroqTile unless steps 1-4 are truly infeasible.

## Performance-Critical Guidance

- Prefer `mma.op` as the compute primitive for new GEMM-like CroqTile kernels unless a legacy form must be preserved.
- Prefer tile-level `tma.copy` or `dma.copy` programming over scalar `.at()` in hot paths.
- Use the imported CroqTile benchmark references for staging, swizzle, event, and warp-specialized pipeline patterns.
- Reuse reference patterns before guessing on async copies, barriers, or tiled view construction.

## Ask Instead of Guessing

Stop and ask only when one answer would materially change the code shape, such as:

- pure `.co` vs `.co` plus `__cpp__`
- async vs sync pipeline
- `dma` vs `tma`
- the `mma.op` shape or scale semantics
- tile-level copy/view form vs conservative scalar fallback

Do all non-blocked work first.

## Documentation Requirement

When fallback to `.cu` is used, record the reason in `idea-log.jsonl`:

- `co_supported: false`
- `fallback_reason: "<why .co/__cpp__ is insufficient>"`
- `fallback_base: "generated_cu_from_co|existing_iter_base_cu"`

## Scope Guardrail

When the target DSL is not CroqTile, do not use `.co` as the implementation language.
