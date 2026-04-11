# CroqTile Tuner

This repo contains the tuning control plane, FSM skills, stress-test harness, and
per-DSL work directories for non-stop GPU kernel tuning.

## Standard Usage

Use `/perf-tune` as the only tuning entrypoint.

Examples:

```text
/perf-tune
/perf-tune cuda f16
/perf-tune croqtile f16 4096x8192x4096
```

The expected workflow is always:

```text
PROFILE -> IDEATE -> IMPLEMENT -> MEASURE -> DECIDE -> STORE
```

Measured KEEP/DISCARD iterations must always execute `STORE`.
Compile-failed attempts are stored as attempt records and do not consume public iteration ids.

## Resume Safety

Only resume from artifacts inside the current git repository.

Forbidden resume sources:
- sibling repos such as `*-paper`
- absolute includes that resolve outside this repo
- stale `state.json` entries without a valid local checkpoint and local active loop-state

Before resuming any in-progress tuning session, validate the resume source:

```bash
python3 scripts/validate_tuning_session.py --dsl cuda
python3 scripts/validate_tuning_session.py --dsl all
```

If validation fails, clean invalid work state and restart from `INIT`:

```bash
python3 scripts/clean_kernel_work_state.py --dsl cuda --invalid-only
python3 scripts/clean_kernel_work_state.py --dsl all --invalid-only
```

To intentionally clear all active in-progress kernel work state before starting over:

```bash
python3 scripts/clean_kernel_work_state.py --dsl all
```

## What Must Persist Every Iteration

At every `STORE` step, the system must update:

- `tuning/aitune/<dsl>/logs/<key>/results.tsv`
- `tuning/aitune/<dsl>/checkpoints/<key>.json`
- `tuning/aitune/<dsl>/memory/<key>/rounds.raw.jsonl`
- `tuning/aitune/<dsl>/memory/<key>/rounds.md`
- `.claude/skills/fsm-engine/state/<dsl>/compaction-summary.md`

These files are what make crash-safe resume and post-mortem debugging possible.

## Stress Testing

The mock harness simulates long tuning sessions and failure modes:

```bash
python3 scripts/mock_skill_ab_test.py \
  --agents-per-dsl 24 \
  --round-target 336 \
  --seed 20260411 \
  --run-name strict_ab_20260411_336iters
```

The strict variant models:
- explicit `iter000` baseline
- attempt-only compile failures
- win/stall-triggered profiling
- research escalation
- raw/md round-history persistence
- active-state and resume-source validation at long-run checkpoints

## Working-State Cleanup

If a tuning session behaved strangely, do this in order:

1. `python3 scripts/validate_tuning_session.py --dsl <dsl>`
2. `python3 scripts/clean_kernel_work_state.py --dsl <dsl> --invalid-only`
3. restart with `/perf-tune <dsl> <dtype> [shape]`

This avoids carrying forward polluted `seed.cu`, stale checkpoint state, or external-source includes.