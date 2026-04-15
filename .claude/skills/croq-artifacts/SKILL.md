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

`tuning/<gpu>/<dsl>/`

where `<gpu>` is the GPU key emitted by:
```
bash .claude/skills/croq-tune/tools/detect_gpu.sh
```
e.g. `sm90_H100`, `sm80_A100`, `sm89_L40S`.

## Shape Key Naming Convention

Shape keys MUST include the operator/kernel type:

```
<operator>_<dtype>_<dimensions>
```

### Dimension Ordering Is CANONICAL — NEVER Permute

**For matmul**, dimensions are ALWAYS ordered **MxNxK** (rows of A × cols of B × inner dimension).

- `matmul_bf16fp32_16384x16384x512` → M=16384, N=16384, K=512
- `matmul_bf16fp32_512x16384x16384` → M=512, N=16384, K=16384

These are **completely different shapes** with different memory access patterns, tile strategies,
and performance characteristics. They MUST NEVER be treated as interchangeable or equivalent,
even though they use the same set of numbers. The user's requested shape dimensions are the
ground truth — always map them left-to-right as M, N, K.

**CRITICAL AGENT RULE**: Before any resume decision, compare the user's requested dimensions
character-by-character against existing `shape_key` directories. A mismatch in any digit or
position means a fresh session is required for the new shape. Do NOT infer equivalence from
"same numbers, different order."

Examples:
- `matmul_bf16fp32_16384x16384x512` (M=16384, N=16384, K=512 — wide square output, small K)
- `matmul_bf16fp32_512x16384x16384` (M=512, N=16384, K=16384 — tall-skinny A, large K)
- `matmul_f16_4096x8192x8192` (M=4096, N=8192, K=8192)
- `spmm_f16_4096x8192x8192` (sparse matmul)
- `conv2d_f16_128x64x3x3` (2D convolution)
- `attention_f16_2048x64x64` (attention kernel)

Components:
- `operator`: kernel type (matmul, spmm, conv2d, attention, etc.)
- `dtype`: data type (f16, e4m3, bf16, bf16fp32, f32)
- `dimensions`: shape parameters in canonical order (MxNxK for matmul)

This naming convention ensures:
1. Clear identification of what operation is being tuned
2. No confusion between different operator types with same dimensions
3. No confusion between same dimensions in different orders (e.g. `16384x16384x512` ≠ `512x16384x16384`)
4. Easier filtering and grouping of tuning results

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

### Tag Is MANDATORY (INVIOLABLE)

Every source artifact for a public `iter<NNN>` MUST include a descriptive `<tag>`:

```
iter<NNN>_<tag>.<ext>     ← CORRECT
iter<NNN>.<ext>            ← FORBIDDEN — no tag
```

The tag MUST be:
- Lowercase alphanumeric + underscores only
- 2–16 characters
- Descriptive of the optimization idea (e.g. `swizzle`, `pipeline`, `ldmatrix`, `maxreg`)

**If the agent writes `iter<NNN>.cu` (no tag), it MUST immediately rename it** before committing:
```bash
mv iter<NNN>.cu iter<NNN>_<descriptive_tag>.cu
# update build/run/profile scripts to match
```

This rule exists because bare-number filenames prevent post-hoc reconstruction of what each iteration tested. The tag is the only human-readable record of the idea associated with each source file.

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

**Tuning directly on `main` is allowed.** No branch ceremony required.

Commit tuning progress directly to `main` after each measured iter:

```bash
git add -A
git commit -m "tune(<dsl>): <op> <dtype> <shape> - iter<NNN> <X> TFLOPS"
git push origin main
```

No date branches, no DSL branches, no resume suffixes.
