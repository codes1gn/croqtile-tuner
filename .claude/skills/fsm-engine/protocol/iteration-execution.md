# CroqTuner — Iteration Execution Contract

This file defines the execution semantics between `IDEATE` and `STORE`.
Read it together with `loop-contract.md` and `step-checklists.md`.

## Core Rule

There is only one tuning workflow:

- resume the latest valid progress
- push it forward continuously
- never branch behavior by artificial workflow forks

## Infinite Runtime Loop (Authoritative)

The runtime loop is intentionally non-stop and must behave like this:

```text
while true:
  read active per-DSL state file
  execute current FSM state exactly
  persist required artifacts for this state
  transition immediately to next state

  if state == SHAPE_COMPLETE:
    transition immediately to NEXT_SHAPE -> INIT for next pending shape
    continue

  stop only when:
    1) all scheduled shapes are done, or
    2) user explicitly interrupts, or
    3) systemic GPU failure blocks progress
```

Completing one shape is never a stopping point.
Compile failures are never a stopping point.
Discard streaks are never a stopping point.

## Public Iteration Ids vs Internal Attempts

`iter<NNN>` ids are reserved for compile-passed, correctness-passed, benchmarked candidates.

- `iter000` is the measured baseline
- `iter<NNN>` is for candidates that reached MEASURE
- compile-failed or pre-benchmark discarded tries are `attempt<AAAA>` records, not public iterations

Failed attempts must still be saved, but they do not consume the measured iteration sequence.
Only promote to `iter<NNN>` after compile and correctness both pass and the candidate is runnable.

## Golden Per-Idea Flow

For every new idea:

1. Start from exactly one current best kernel.
2. Prefer the highest-level implementation path available.
3. Compile and verify before assigning a new public iteration id.
4. Only after the candidate is runnable and correctness-clean may it be promoted to `iter<NNN>`.
5. Benchmark, decide KEEP or DISCARD, then persist artifacts.
6. Continue immediately.

## CroqTile-First Implementation Priority

When `dsl=croqtile`, each idea must follow this order:

1. Pure `.co`
2. `.co` with inlined `__cpp__` fragments
3. Generated `.cu` produced from the chosen `.co` base for this idea
4. Direct edits on the selected base `.cu` only when the idea is not supportable through CroqTile generation

Do not jump directly to raw CUDA if the idea can still be expressed in CroqTile.

## Attempt Artifact Contract

Every failed compile/debug attempt must save:

- the attempted source snapshot
- the build shell script
- the build log or stderr capture
- the failure reason in checkpoint or idea log metadata

Attempt artifacts do not fabricate a measured `results.tsv` row.

## Measured Iteration Artifact Contract

Every compile-passed, benched `iter<NNN>` must save:

- source file
- build shell script
- run shell script
- timing output
- profile shell script and profile output when profiling ran
- `results.tsv` row with KEEP or DISCARD decision
- checkpoint update and round-memory persistence

## Profiling Cadence

The loop must profile continuously, but not always with maximum overhead.

### Early iterations

Prefer lighter profiling signal first when it is sufficient:

- compiler diagnostics
- runtime timing deltas
- targeted metric collection
- short or focused profiler runs

### Mandatory full-profile triggers

Run a full `ncu` pass when any of these happens:

1. baseline characterization / first real iteration
2. a new best kernel is found
3. `consecutive_discards >= 3`
4. the agent is out of grounded ideas or feels stuck
5. before changing optimization category due to stall
6. before PTX/SASS-level work

If a new best is found, re-profile that new best before the next ideation round.
If the search stalls, re-profile the current best before proposing another idea.

## Late-Stage Escalation

When the search becomes long-horizon, for example `iter > 300`, escalate from source-level tuning into instruction-level inspection:

1. inspect PTX and SASS
2. check issue mix, dependency chains, register pressure, barriers, and memory ops
3. consider inline PTX / inline ASM only after higher-level CroqTile or CUDA changes stop helping
4. treat this as last-mile optimization, not a replacement for earlier source-level exploration

## Stop Conditions (Only Three)

A run may stop only when one of these is true:

1. all shapes scheduled for this DSL are `done`
2. user explicitly interrupts
3. systemic GPU failure persists after remediation attempts

## Non-Stop Rule

A failure to improve is not a stopping condition.

- keep looping after discards
- keep looping after compile failures
- keep looping after shape completion
- only the user, total sweep completion, or a genuine systemic failure may stop the process