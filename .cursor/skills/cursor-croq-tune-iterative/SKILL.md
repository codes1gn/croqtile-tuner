---
name: cursor-croq-tune-iterative
description: Durable wrapper for kernel tuning in this repository. Use when the user asks for /croq-tune-iterative, wants to combine durable-request with croq-tune, wants checkpointed resume-or-continue control around autonomous tuning runs, or wants human confirmation between autonomous tuning invocation leases while keeping the base /croq-tune protocol unchanged.
argument-hint: <dsl: croqtile|cuda|cute|triton|tilelang|helion|cutile> <dtype: f16|e4m3|all> [shape_key]
---

# Croq-Tune-Iterative — Durable Boundary Wrapper

`cursor-croq-tune-iterative` is a thin wrapper around the existing tuning system.
It exists so `/croq-tune` can stay fully autonomous and unchanged while this
variant adds durable, user-visible checkpoints at invocation boundaries.

## Design Goal

Keep the original tuning protocol authoritative.

- The inner tuning loop is still owned by `croqtile-tuner` plus `cursor-croq-tune`
- This wrapper owns only user checkpoints, invocation framing, and resume/continue control
- Do not fork or paraphrase the tuning loop if delegation is available

## Required Inputs

Parse the same invocation contract as `/croq-tune`:

- `dsl`
- `dtype`
- optional `shape_key`
- optional `--model`
- required `--task-uid` for fresh runs
- any user-stated round budget

Use the exact `shape_key` string from the user. Do not normalize or reorder it.

## Load Order

1. Read this file first
2. If a durable checkpoint skill is active, use its checkpoint mechanism
3. Treat `.cursor/skills/cursor-croq-tune/SKILL.md` as the inner tuning contract
4. Treat `.cursor/skills/cursor-croq-dsl-<dsl>/SKILL.md` as the inner DSL contract when needed by the delegated tuner

## Core Rule

One autonomous tuning invocation lease is the atomic task boundary for this wrapper.

That means:

- Per-round transitions inside `PROFILE -> IDEA -> IMPLEMENT -> VERIFY -> MEASURE -> DECIDE -> STORE` are not checkpoint boundaries
- `reinforce.sh` output is not a checkpoint boundary
- A single KEEP or DISCARD is not a checkpoint boundary
- A resumable inner return is a checkpoint boundary
- A valid stop condition is a checkpoint boundary
- A human-requested switch of shape, model, or task is a checkpoint boundary

## Wrapper Workflow

### 1. Pre-Launch Checkpoint

Before the first tuning launch, present a durable checkpoint that summarizes:

- parsed `dsl`, `dtype`, `shape_key`
- parsed `--model`
- whether `--task-uid` is present
- the target round budget for this invocation

Offer context-specific next actions such as:

- start tuning now
- edit invocation arguments
- cancel

If the user chooses to edit, revise the invocation before launching.

### 2. Delegate The Real Work

When the user chooses to start or continue, delegate the tuning run to the existing
`croqtile-tuner` subagent.

Delegate with the original tuning arguments and explicit framing that:

- this is a `/croq-tune-iterative` wrapper session
- the inner run must still follow the existing `/croq-tune` protocol unchanged
- the inner run must not add user checkpoints mid-loop
- the inner run should return only on a meaningful boundary: invocation budget met, valid hard stop, or genuine blocker

Do not inline a replacement tuning loop unless delegation is unavailable.

### 3. Post-Return Checkpoint

Whenever the delegated tuner returns, checkpoint before ending the outer turn.

Summarize only the high-value state:

- fresh start or resume
- completed rounds in that invocation
- current best kernel versus baseline
- blocker, if any
- exact valid stop condition, if any

Then offer likely next actions such as:

- continue the same tuning session
- inspect blocker or state in more detail
- switch task
- stop here

### 4. Continue Loop If Requested

If the user chooses to continue, invoke `croqtile-tuner` again with the same
session arguments so it resumes from disk state.

The outer wrapper may repeat this cycle indefinitely:

checkpoint -> delegate -> checkpoint -> delegate

## Allowed Checkpoint Boundaries

Interactive checkpoints are allowed only at these boundaries:

- before first launch
- after a delegated tuning invocation returns
- before switching to a different shape, dtype, model, or task-uid
- when a blocking external input is required

## Forbidden Behavior

- Do not edit `.cursor/skills/cursor-croq-tune/SKILL.md`
- Do not modify the inner loop semantics of `/croq-tune`
- Do not insert `AskQuestion` inside an active inner tuning run
- Do not hand-edit tuning artifacts that belong to the delegated tuner
- Do not summarize an inner run as finished if it returned early without a valid stop condition; offer resume first
- Do not duplicate the harness workflow in this wrapper

## Compatibility Contract

This wrapper is the compatibility layer for combining durable checkpoints with tuning.

- `/croq-tune` remains the autonomous, no-checkpoint entrypoint
- `/croq-tune-iterative` is the checkpointed entrypoint
- Both entrypoints share the same underlying tuning harness and artifact layout

## Return Discipline

If the delegated tuner reports a blocker that truly requires the human, surface it.
Otherwise, bias the first option toward continuing the same session.
