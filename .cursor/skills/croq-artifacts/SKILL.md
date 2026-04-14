---
name: croq-artifacts
description: Canonical artifact naming and storage rules for croq-tune non-stop loops across all DSLs.
---

# Croq Artifact Rules (DSL-Isolated)

This file is the source of truth for where iteration artifacts are stored.

Top-level tuning skills must explicitly read this file before tuning. Mentioning this
path in another skill is not sufficient.

## Root Layout

All tuning artifacts must be kept under:

`tuning/aitune/<dsl>/`

## Shape Key Naming Convention

Shape keys MUST include the operator/kernel type:

```
<operator>_<dtype>_<dimensions>
```

Examples:
- `matmul_f16_512x16384x16384` (dense matmul)
- `spmm_f16_4096x8192x8192` (sparse matmul)
- `conv2d_f16_128x64x3x3` (2D convolution)
- `attention_f16_2048x64x64` (attention kernel)

Components:
- `operator`: kernel type (matmul, spmm, conv2d, attention, etc.)
- `dtype`: data type (f16, e4m3, bf16, f32)
- `dimensions`: shape parameters (MxNxK for matmul, etc.)

This naming convention ensures:
1. Clear identification of what operation is being tuned
2. No confusion between different operator types with same dimensions
3. Easier filtering and grouping of tuning results

Per shape key `<key>`:

- `logs/<key>/results.tsv`
- `logs/<key>/idea-log.jsonl`
- `logs/<key>/attempt-log.jsonl`
- `logs/<key>/env_iter000.txt`
- `srcs/<key>/iter000_ref.<co|cu>`
- `srcs/<key>/attempt<AAAA>_<tag>.<co|cu>`
- `srcs/<key>/iter<NNN>_<tag>.<co|cu>`
- `cmd/<key>/build_iter000.sh`
- `cmd/<key>/run_iter000.sh`
- `cmd/<key>/profile_iter000.sh`
- `cmd/<key>/build_attempt<AAAA>.sh`
- `cmd/<key>/build_iter<NNN>.sh`
- `cmd/<key>/run_iter<NNN>.sh`
- `cmd/<key>/profile_iter<NNN>.sh`
- `perf/<key>/build_attempt<AAAA>.txt`
- `perf/<key>/verify_iter000.txt`
- `perf/<key>/timing_iter000_baseline.txt`
- `perf/<key>/timing_iter<NNN>.txt`
- `perf/<key>/ncu_iter<NNN>.txt`
- `checkpoints/<key>.json`
- `memory/<key>/rounds.raw.jsonl`
- `memory/<key>/rounds.md`

Global per-DSL:

- `state.json`

## Iteration Naming

- `iter000` is the trivial measured baseline.
- Public measured iteration ids are 3-digit: `iter001`, `iter002`, ...
- Compile-failed or pre-benchmark discarded tries are `attempt<AAAA>`.
- Failed attempts must be saved, but they do not consume the measured iteration sequence.
- Keep all attempts and all measured iterations, including discarded measured candidates.

## STORE-Step Requirements

Every STORE step (KEEP or DISCARD) must:

1. Persist the source snapshot and command scripts for whatever was executed
2. Append `results.tsv` for measured public iterations, or `attempt-log.jsonl` for failed attempts
3. Update `idea-log.jsonl`
4. Write checkpoint file
5. Persist round transcript:
   - raw JSONL entry in `rounds.raw.jsonl`
   - markdown section in `rounds.md`
6. Update `state.json`
7. Commit all changed files

Every compile-passed, benched `iter<NNN>` must retain:
- source file
- build shell script
- run shell script
- timing output
- profile shell script and profile output when profiling ran

Every failed `attempt<AAAA>` must retain:
- attempted source snapshot
- build shell script
- build output / stderr capture
- failure note in attempt log or checkpoint metadata

## Branch Rule

Use only one long-lived branch per DSL: `aitune/<dsl>`.

No date branches and no resume suffixes.
