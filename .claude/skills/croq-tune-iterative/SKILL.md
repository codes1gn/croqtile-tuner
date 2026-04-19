---
name: croq-tune-iterative
description: Durable wrapper for GPU kernel tuning. Use when the user wants to combine durable-request with croq-tune, asks for a checkpointed tuning workflow, or wants human confirmation between autonomous tuning invocation leases while keeping the base /croq-tune protocol unchanged.
argument-hint: <dsl: croqtile|cuda|cute|triton|tilelang|helion|cutile> <dtype: f16|e4m3|all> [shape_key]
---

# Croq-Tune-Iterative — Durable Boundary Wrapper

`croq-tune-iterative` is a wrapper entrypoint.
It preserves the existing autonomous `/croq-tune` protocol and adds durable
checkpoints only at invocation boundaries.

## Purpose

Use this skill when the user wants both of these properties at once:

- the inner tuning loop remains autonomous and protocol-driven
- the outer conversation does not end silently between invocation leases

## Wrapper Contract

1. Parse the same invocation fields as `/croq-tune`
2. Present a durable checkpoint before the first launch
3. Delegate actual tuning work to the existing `croqtile-tuner` agent or equivalent inner tuning executor
4. Do not inject user checkpoints inside the active tuning loop
5. When the inner run returns, present a durable checkpoint with continue, inspect, switch, or stop options
6. If the user chooses continue, re-enter the same session by delegating again

## Boundary Semantics

Treat one autonomous tuning invocation lease as the atomic task boundary.

Not a checkpoint boundary:

- a single round
- `reinforce.sh`
- one KEEP or DISCARD
- any mid-loop milestone that remains autonomously resumable

Checkpoint boundary:

- before first launch
- after inner return
- before changing invocation arguments
- on a true blocker that requires human input

## Forbidden Behavior

- Do not modify `.claude/skills/croq-tune/SKILL.md`
- Do not paraphrase the tuning loop into a second implementation
- Do not ask the human questions from inside the active inner tuning loop
- Do not hand-edit tuning state files owned by the inner tuner

## Compatibility Outcome

- `/croq-tune` remains autonomous and unchanged
- `croq-tune-iterative` becomes the durable, checkpointed wrapper variant
- both variants share the same underlying tuning harness and artifacts
