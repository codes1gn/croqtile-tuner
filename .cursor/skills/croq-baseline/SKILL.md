---
name: croq-baseline
description: One-time baseline preparation contract for croq-tune, with persistent workspace path, naming, and schema.
---

# Croq-Baseline

Use this skill only in `PREPARATION_ONCE`, before round counting starts for a new shape/session.

## Preparation Contract

1. Detect environment and baseline library readiness using:
   - `.claude/skills/croq-baseline/tools/prepare_baseline_env.py`
2. Check baseline readout in the persistent baseline workspace
3. If readout exists, reuse it directly
4. If missing, run baseline readout and persist it before first tuning round

This preparation is outside tuning round count.

## Trigger Rules

Run this skill when:

1. entering `PREPARATION_ONCE` for a new `(dsl, operator, dtype, shape_key)`
2. baseline env report is missing or stale
3. user explicitly asks for a dependency setup check

Dependency selection policy:

- default mode is `--libs auto` (agent decides from `operator` and `kernel`)
- user can override with explicit `--libs cublas,torch-cuda,cusparselt`

## CLI Usage

Auto-select libs from target op/kernel:

```bash
python3 .claude/skills/croq-baseline/tools/prepare_baseline_env.py \
  --dsl cuda \
  --operator spmm \
  --kernel sparse_gemm \
  --shape-key f16_4096x8192x8192 \
  --libs auto
```

Explicit library set:

```bash
bash .claude/skills/croq-baseline/tools/prepare_baseline_env.sh \
  --dsl cuda \
  --operator spmm \
  --kernel sparse_gemm \
  --shape-key f16_4096x8192x8192 \
  --libs cublas,torch-cuda,cusparselt
```

Output:

- report file: `baseline-workspace/_env/prep_<timestamp>.json`
- latest symlink-style copy: `baseline-workspace/_env/latest.json`

## Baseline Workspace Contract

Persistent root:

- `baseline-workspace/<dsl>/<operator>/<dtype>/`

Naming:

- `shape_key`: `<dtype>_<M>x<N>x<K>` (for example `f16_4096x8192x8192`)
- readout file: `readouts/<shape_key>.json`
- optional profile file: `profiles/<shape_key>/ncu_baseline.txt`
- summary table: `tables/readouts.tsv`

### Form Schema (`forms/readout_request.json`)

```json
{
  "schema": "baseline-readout-request-v1",
  "dsl": "cuda",
  "operator": "spmm",
  "dtype": "f16",
  "shape_key": "f16_4096x8192x8192",
  "shape": { "m": 4096, "n": 8192, "k": 8192 },
  "warmup": 10,
  "iters": 50,
  "samples": 5
}
```

### Data Schema (`readouts/<shape_key>.json`)

```json
{
  "schema": "baseline-readout-v1",
  "dsl": "cuda",
  "operator": "spmm",
  "dtype": "f16",
  "shape_key": "f16_4096x8192x8192",
  "shape": { "m": 4096, "n": 8192, "k": 8192 },
  "kernel": "iter000_baseline",
  "verification": { "passed": true, "max_abs_err": 0.0, "max_rel_err": 0.0 },
  "samples": [{ "idx": 1, "time_ms": 0.0, "tflops": 0.0 }],
  "median": { "time_ms": 0.0, "tflops": 0.0 },
  "commands": { "build": "", "run": "", "verify": "" },
  "artifacts": { "stdout": "", "stderr": "", "profile": "" },
  "recorded_at": "ISO-8601"
}
```
