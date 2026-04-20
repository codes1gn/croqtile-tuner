# Tuning Directory Conventions

This document describes the structure and naming conventions for the `tuning/` directory,
used by the croq-tune system for GPU kernel optimization experiments.

## Root Layout

```
tuning/<gpu>/<dsl>/<category>/<shape_key>/<model>/...
```

### Components

| Component | Description | Example |
|-----------|-------------|---------|
| `<gpu>` | GPU identifier from `detect_gpu.sh` | `sm90_NVIDIA_H800_PCIe`, `sm86_NVIDIA_GeForce_RTX_3070` |
| `<dsl>` | Domain-specific language | `croqtile`, `cuda`, `triton`, `tilelang`, `helion`, `cutile`, `cute` |
| `<category>` | Artifact category | `srcs`, `cmd`, `perf`, `logs`, `baseline`, `checkpoints`, `memory`, `bin` |
| `<shape_key>` | Problem shape identifier | `matmul_fp16fp32_16384x16384x16384` |
| `<model>` | AI model used for tuning | `opus-4`, `sonnet-4`, `gpt-5`, `claude-4-5-opus-high` |

## Shape Key Format

```
<operator>_<dtype>_<dimensions>
```

### Examples

- `matmul_fp16fp32_16384x16384x16384` — FP16 input, FP32 accumulator matmul, M=16384, N=16384, K=16384
- `blockscale_gemm_e4m3fp32_16384x16384x16384` — E4M3 block-scale GEMM
- `gdn_f16_B2_T4_H8_K128_V128` — Grouped dense attention kernel with batch/head dimensions

**CRITICAL**: Dimension order is **canonical** and NEVER permuted. For matmul: `MxNxK`.
`matmul_bf16fp32_16384x16384x512` ≠ `matmul_bf16fp32_512x16384x16384`.

## Model Naming

- Lowercase alphanumeric + hyphens only
- Examples: `opus-4`, `sonnet-4-6`, `gpt-5-4`, `claude-4-5-opus-high`, `qwen35_moe_fp8`
- Model name is passed via `--model` flag to all harness scripts

## Category Directories

### `srcs/` — Kernel Source Code

```
srcs/<shape_key>/<model>/iter<NNN>_<tag>.<ext>
srcs/<shape_key>/<model>/attempt<AAAA>_<tag>.<ext>
```

- `iter<NNN>` — Successfully measured iterations (3-digit, zero-padded)
- `attempt<AAAA>` — Failed compile attempts (4-digit, do not consume iter sequence)
- `<tag>` — **Required**, 2-31 chars, lowercase alphanumeric + underscores, describes the idea
- `<ext>` — `.co` (Choreo), `.cu` (CUDA), `.py` (Triton/TileLang/Helion)

**Special iterations:**
- `iter000` — cuBLAS/library reference baseline
- `iter001` — First custom kernel (from discovery or scratch)

### `cmd/` — Build and Run Scripts

```
cmd/<shape_key>/<model>/build_iter<NNN>.sh
cmd/<shape_key>/<model>/run_iter<NNN>.sh
cmd/<shape_key>/<model>/profile_iter<NNN>.sh
cmd/<shape_key>/<model>/iter<NNN>_<tag>.cute.result  # CroqTile: generated run command
```

### `perf/` — Performance Artifacts

```
perf/<shape_key>/<model>/build_iter<NNN>.txt         # Compiler output
perf/<shape_key>/<model>/timing_iter<NNN>.txt        # Benchmark timing
perf/<shape_key>/<model>/ncu_iter<NNN>_<tag>_round<R>.csv      # ncu CSV export
perf/<shape_key>/<model>/ncu_iter<NNN>_<tag>_round<R>.ncu-rep  # ncu binary report
perf/<shape_key>/<model>/ncu_iter<NNN>_<tag>_round<R>.metrics.csv  # Optional detailed metrics
```

### `logs/` — Session Logs

```
logs/<shape_key>/<model>/task_config.json   # Task metadata (task_uid, model, device, status)
logs/<shape_key>/<model>/results.tsv        # TSV with one row per measured iteration
logs/<shape_key>/<model>/idea-log.jsonl     # JSONL with one entry per round
logs/<shape_key>/<model>/attempt-log.jsonl  # JSONL for failed attempts (optional)
```

#### `results.tsv` Format

| Column | Description |
|--------|-------------|
| `iter` | Iteration name (`iter000`, `iter001`, ...) |
| `kernel` | Full kernel name with tag (`iter001_draft`) |
| `tflops` | Measured performance (TFLOPS) |
| `decision` | `KEEP`, `DISCARD`, `COMPILE_FAIL`, `SEGFAULT`, `HANG`, `RUNTIME_FAIL`, `TESTED` |
| `bottleneck` | `baseline`, `compute_bound`, `memory_bound`, `latency_bound`, `launch_bound`, `mixed` |
| `idea_summary` | Human-readable summary of the optimization idea |

#### `idea-log.jsonl` Format

Each line is a JSON object with fields:
- `round` (int) — Round number (0 = baseline)
- `iter` (string) — Iteration name
- `bottleneck` (string) — Identified bottleneck category
- `idea` (string) — Description of the optimization tried
- `category` (string) — `tiling`, `pipeline`, `memory`, `compute`, `misc`, `baseline`
- `expected_gain` (string) — Expected improvement (e.g., "+50 TFLOPS")
- `decision` (string) — KEEP/DISCARD/etc.
- `tflops` (float) — Measured performance
- `timestamp` (string) — ISO 8601 timestamp

### `baseline/` — cuBLAS Reference Results

```
baseline/<shape_key>/<model>/cublas_result.json
```

Contains:
```json
{
  "tflops": 421.6146,
  "dtype": "fp16fp32",
  "m": 16384, "n": 16384, "k": 16384,
  "warmup": 10, "iters": 50,
  "avg_ms": 20.8629,
  "status": "ok",
  "gpu": "sm90_NVIDIA_H800_PCIe",
  "dsl": "croqtile",
  "shape_key": "matmul_fp16fp32_16384x16384x16384",
  "model": "sonnet46"
}
```

### `checkpoints/` — IDEA Checkpoints

```
checkpoints/<shape_key>/<model>/current_idea.json
```

Used by `checkpoint_write.sh` to persist the current optimization plan between IDEA and IMPLEMENT phases.

### `memory/` — Session Memory

```
memory/<shape_key>/<model>/activity.jsonl
memory/<shape_key>/<model>/sessions/<session-id>.jsonl
```

Raw session transcripts and activity logs.

### `bin/` — Compiled Binaries

```
bin/<shape_key>/<model>/iter<NNN>_<tag>
```

Compiled kernel binaries for DSLs that produce executables.

## Archive Directory

```
tuning/.archive/<shape_key>/...
```

Old tuning sessions that don't follow the current `<gpu>/<dsl>/` layout are moved here.

## Iteration Naming Rules

1. **iter000** — Always reserved for cuBLAS/library baseline
2. **iter001** — First custom kernel (from discovery or scratch implementation)
3. **iter002+** — Subsequent tuning iterations
4. **attemptAAAA** — Failed compile attempts (4-digit, e.g., `attempt0001`)

### Tag Requirements

- **Mandatory**: Every iteration file must have a tag (`iter<NNN>_<tag>.<ext>`)
- **Length**: 2-31 characters
- **Characters**: Lowercase alphanumeric + underscores only
- **Purpose**: Brief description of the optimization idea

**Good tags**: `tm192`, `wn256_st3`, `cluster2`, `maxreg128`, `arrive_once`, `wait1`

**Bad tags**: `test`, `v2`, `final` (not descriptive)

## Task Config Format

```json
{
  "task_uid": "24fjf20",
  "dsl": "croqtile",
  "model": "gpt-5-4",
  "device": "NVIDIA H800 PCIe",
  "status": "waiting",
  "task_id": 11,
  "variant": "high",
  "agent_type": "opencode",
  "mode": "opencode",
  "request_budget": 3,
  "request_number": 2,
  "max_iterations": 30,
  "best_tflops": 333.4,
  "baseline_tflops": 432.0588,
  "best_kernel": "iter031_tm192"
}
```

## Directory Tree Example

```
tuning/
├── sm90_NVIDIA_H800_PCIe/
│   └── croqtile/
│       ├── baseline/
│       │   └── matmul_fp16fp32_16384x16384x16384/
│       │       └── sonnet46/
│       │           └── cublas_result.json
│       ├── bin/
│       │   └── matmul_fp16fp32_16384x16384x16384/
│       │       └── sonnet46/
│       │           └── iter001_draft
│       ├── checkpoints/
│       │   └── matmul_fp16fp32_16384x16384x16384/
│       │       └── sonnet46/
│       │           └── current_idea.json
│       ├── cmd/
│       │   └── matmul_fp16fp32_16384x16384x16384/
│       │       └── sonnet46/
│       │           ├── build_iter001.sh
│       │           ├── run_iter001.sh
│       │           └── iter001_draft.cute.result
│       ├── logs/
│       │   └── matmul_fp16fp32_16384x16384x16384/
│       │       └── sonnet46/
│       │           ├── task_config.json
│       │           ├── results.tsv
│       │           └── idea-log.jsonl
│       ├── memory/
│       │   └── matmul_fp16fp32_16384x16384x16384/
│       │       └── sonnet46/
│       │           └── activity.jsonl
│       ├── perf/
│       │   └── matmul_fp16fp32_16384x16384x16384/
│       │       └── sonnet46/
│       │           ├── build_iter001.txt
│       │           ├── timing_iter001.txt
│       │           └── ncu_iter001_draft_round1.csv
│       └── srcs/
│           └── matmul_fp16fp32_16384x16384x16384/
│               └── sonnet46/
│                   ├── iter001_draft.co
│                   └── iter001_draft.cu
└── .archive/
    └── matmul_fp16fp32_16384x16384x16384/
        └── ... (old format sessions)
```

## Migration Guide

When migrating artifacts from another branch or location:

1. **Preserve results.tsv data** — Never re-run benchmarks if results exist
2. **Match GPU identifier exactly** — Use `detect_gpu.sh` to get canonical name
3. **Preserve timestamps** — Keep original timestamps in idea-log.jsonl
4. **Move, don't copy** — To avoid duplicates
5. **Update task_config.json** — If task_uid changes

### Merging results.tsv

When combining data from multiple sources:
1. Sort by iteration number
2. Preserve all columns
3. Remove exact duplicates (same iter + kernel + tflops)
4. Keep the higher tflops for same iter if different runs exist

## Harness Scripts Reference

| Script | Purpose |
|--------|---------|
| `detect_gpu.sh` | Get canonical GPU identifier |
| `store_baseline.sh` | Measure and store cuBLAS baseline |
| `store_round.sh` | Store iteration results atomically |
| `resume_state.sh` | Get current tuning session state |
| `next_iter.sh` | Get next iteration name |
| `checkpoint_write.sh` | Persist/read IDEA checkpoints |
| `reinforce.sh` | Post-STORE reinforcement |

All scripts are located in `.cursor/skills/cursor-croq-tune/tools/`.
