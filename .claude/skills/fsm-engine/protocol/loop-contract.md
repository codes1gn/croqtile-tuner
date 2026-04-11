# CroqTuner — Loop Contract (FSM Definition)

## State Machine

The tuning loop is a finite state machine. The current state is stored in the per-DSL state file under `fsm.current_state`. You MUST only perform actions corresponding to the current state, then transition to the next state using `state-transition.sh`.

```
┌──────┐     ┌──────────┐     ┌─────────┐     ┌────────┐     ┌───────────┐
│ INIT │────▶│ BASELINE │────▶│ PROFILE │────▶│ IDEATE │────▶│ IMPLEMENT │
└──────┘     └──────────┘     └─────────┘     └────────┘     └─────┬─────┘
                                   ▲                               │
                                   │                               ▼
                              ┌────┴──┐     ┌────────┐     ┌──────────┐
                              │ STORE │◀────│ DECIDE │◀────│ MEASURE  │
                              └───┬───┘     └────────┘     └──────────┘
                                  │
                    ┌─────────────┼─────────────────┐
                    ▼             ▼                  ▼
              ┌──────────┐  ┌───────────────┐  ┌────────────┐
              │ PROFILE  │  │ SHAPE_COMPLETE│  │ NEXT_SHAPE │
              │ (loop)   │  │ (iter >= max) │  │   → INIT   │
              └──────────┘  └───────────────┘  └────────────┘
```

## States

| State | Purpose | Next State |
|---|---|---|
| `INIT` | Create directories, write the trivial `iter000` reference kernel, and set up timing/verify/env scaffolding | `BASELINE` |
| `BASELINE` | Compile + verify + measure the trivial `iter000` baseline | `PROFILE` |
| `PROFILE` | Run ncu (or skip if allowed), identify bottleneck | `IDEATE` |
| `IDEATE` | Propose ONE novel optimization idea grounded in local data and, when needed, current external references | `IMPLEMENT` or `PROFILE` |
| `IMPLEMENT` | Create an internal attempt, compile, verify correctness, and only then promote to a public iteration candidate | `MEASURE` |
| `MEASURE` | Run timing benchmark, capture TFLOPS | `DECIDE` |
| `DECIDE` | Compare to current best: KEEP or DISCARD | `STORE` |
| `STORE` | Append results.tsv, write checkpoint, git commit, update FSM | `PROFILE` or `SHAPE_COMPLETE` |
| `SHAPE_COMPLETE` | Register best kernel, update state.json, commit | `NEXT_SHAPE` |
| `NEXT_SHAPE` | Pick next shape from schedule, reset FSM | `INIT` |

## Iteration Semantics

**`fsm.iteration` = the next public measured iteration id currently being worked on (or just completed).**

- After INIT: `iteration = 0` (baseline)
- During BASELINE: working on iter 0
- Transition BASELINE → PROFILE sets `iteration = 1`
- After STORE for a measured public iter 1: `iteration = 1`
- Transition STORE → PROFILE auto-increments only if the stored candidate reached MEASURE and produced TFLOPS
- Compile-failed or pre-benchmark failures are `attempt<AAAA>` and do not advance `fsm.iteration`
- Files are named with `iter<NNN>` where NNN = the public measured iteration id (e.g., `iter001_warpn128.cu`)

`iter000` is always the simplest correct scalar-loop baseline for the selected DSL. It exists to establish correctness, environment capture, timing utilities, and a reproducible reference floor before optimization begins.

## Transition Rules

After STORE:
- If `iteration < max_iteration` → transition to `PROFILE`
- If `iteration >= max_iteration` → transition to `SHAPE_COMPLETE`

Additional loop rule:
- `IDEATE -> PROFILE` is legal when the current best needs re-profiling, when ideas are exhausted, or when stall-triggered category switching requires fresher data.

After SHAPE_COMPLETE:
- If more shapes remain in sweep → transition to `NEXT_SHAPE` then `INIT` **IMMEDIATELY — do NOT pause, summarize, or wait for user input**
- If no shapes remain → sweep is done, report to user
- **Completing a shape is NOT a stopping point.** There are ~260 shapes. Keep going.

## Per-DSL State File

Default location:

` .claude/skills/fsm-engine/state/<dsl>/loop-state.json `

## loop-state Schema

```json
{
  "schema_version": 2,
  "fsm": {
    "current_state": "<STATE>",
    "iteration": 0,
    "max_iteration": 30,
    "dsl": "croqtile",
    "branch_name": "aitune/croqtile",
    "shape_key": "f16_4096x16384x16384",
    "dtype": "f16",
    "shape": [4096, 16384, 16384]
  },
  "guard_flags": {
    "gpu_health_checked": false,
    "ncu_ran_this_iter": false,
    "bottleneck_identified": false,
    "idea_is_novel": false,
    "idea_logged": false,
    "compile_succeeded": false,
    "correctness_verified": false,
    "timing_captured": false,
    "decision_made": false,
    "results_appended": false,
    "checkpoint_written": false,
    "git_committed": false,
    "round_memory_raw_saved": false,
    "round_memory_md_saved": false
  },
  "metrics": {
    "baseline_tflops": null,
    "current_best_tflops": null,
    "current_best_iter": null,
    "current_best_kernel": null,
    "this_iter_tflops": null,
    "this_iter_decision": null,
    "consecutive_discards": 0,
    "last_bottleneck": null,
    "last_idea_category": null
  },
  "paths": {
    "shape_dir_logs": "tuning/aitune/<dsl>/logs/<KEY>",
    "shape_dir_srcs": "tuning/aitune/<dsl>/srcs/<KEY>",
    "shape_dir_perf": "tuning/aitune/<dsl>/perf/<KEY>",
    "shape_dir_cmd": "tuning/aitune/<dsl>/cmd/<KEY>",
    "checkpoint_file": "tuning/aitune/<dsl>/checkpoints/<KEY>.json",
    "idea_log": "tuning/aitune/<dsl>/logs/<KEY>/idea-log.jsonl",
    "attempt_log": "tuning/aitune/<dsl>/logs/<KEY>/attempt-log.jsonl",
    "round_memory_raw_log": "tuning/aitune/<dsl>/memory/<KEY>/rounds.raw.jsonl",
    "round_memory_md_log": "tuning/aitune/<dsl>/memory/<KEY>/rounds.md"
  },
  "completion_promise": "never stop at shape boundary; continue NEXT_SHAPE until user interrupts",
  "non_stop_keyphrase": "NON_STOP_CONTINUE_WITHOUT_WAIT",
  "last_updated": "2026-04-03T00:00:00Z"
}
```

## How to Transition

Run:
```bash
bash .claude/skills/fsm-engine/scripts/state-transition.sh <NEXT_STATE> [key=value ...]
```

Example:
```bash
bash .claude/skills/fsm-engine/scripts/state-transition.sh IDEATE
bash .claude/skills/fsm-engine/scripts/state-transition.sh STORE decision_made=true this_iter_tflops=450.2
```

The script atomically updates the active per-DSL state file, resets guard flags for the new state, and validates the transition is legal.
