---
name: base-tune
description: Launch an infinite AI-driven kernel optimization loop for GPU kernels. Use when the user asks to "tune", "ai-tune", "optimize", or "auto-tune" a kernel. Profiles with ncu, iterates optimizations indefinitely until interrupted.
argument-hint: <kernel-folder-path e.g. .claude/skills/choreo-kernel-examples/matmul/>
---

# Base-Tune: Infinite Kernel Optimization Loop

Kick off an autonomous, ncu-guided optimization experiment on the kernel folder: `$ARGUMENTS`.

## Required Environment Variables

Set these before starting a tuning session:

```bash
# Path to the choreo compiler repo
export CHOREO_HOME=/home/albert/workspace/croqtile

# CUTLASS/CuTe headers (required by choreo)
export CUTE_HOME=$CHOREO_HOME/extern/cutlass

# CUDA toolkit (defaults to /usr/local/cuda)
export CUDA_HOME=/usr/local/cuda
```

Validation: run the environment harness and confirm no errors:
```bash
bash .claude/skills/croq-tune/tools/validate_env.sh --dsl croqtile
```

**Note**: Sparse GEMM kernels with TMA require CUDA 12.9+. Check `nvcc --version`.

## Pre-flight

1. **Read this entire file** — it defines the loop protocol, mandatory rules, and constraints. Follow it exactly.

2. **Read the syntax reference**: Load the `choreo-syntax` skill (`.claude/skills/choreo-syntax/SKILL.md`) before editing any `.co` file.

3. **Identify the target folder**: The user specifies a kernel folder (e.g. `.claude/skills/choreo-kernel-examples/matmul/`). All kernel `.co` files live there.

4. **Determine the kernel mnemonic**: Derive a short mnemonic from the folder name (e.g. `matmul`, `gemm_sp_e4m3`). The user may specify a more specific mnemonic if the folder contains multiple kernel families.

5. **Detect the GPU**:
   ```bash
   GPU=$(bash .claude/skills/croq-tune/tools/detect_gpu.sh)
   echo "$GPU"
   ```
   Result is cached at `/tmp/croq_gpu_key`.

## Branch Setup

**Tuning directly on `main` is allowed.** No branch ceremony required. **Do NOT push** — commit locally only. Parallel tuning sessions on the same repo would conflict on push.

```bash
git add -A
git commit -m "tune(croqtile): <op> <dtype> <shape> - iter<NNN> <X> TFLOPS"
```

## Baseline Discovery

Before any optimization, determine the current best kernel(s) in the target folder:

1. **List all `.co` files** in the folder. Exclude files with `_aitune_` in the name — those are prior tuning outputs, not clean references.
2. **Compile and benchmark the top 2-3 candidates** to find the actual best performer. Use the DSL-specific build workflow from `.claude/skills/croq-tune/tools/`:

   ```bash
   # Phase 1: compile .co to extracted .cu
   bash .claude/skills/croq-tune/tools/co2cu.sh \
       --co <target_folder>/<kernel>.co \
       --arch sm_90a \
       [--flags "--use-warpspec --use-prepack"]

   # Phase 2: compile .cu to binary
   bash .claude/skills/croq-tune/tools/build_iter.sh \
       --cu <target_folder>/<kernel>.cu \
       --out tuning/${GPU}/croqtile/bin/<shape_key>/<model>/<kernel>

   # Phase 3: run benchmark
   CHOREO_TIMING_WARMUP=10 CHOREO_TIMING_REPEAT=50 \
       tuning/${GPU}/croqtile/bin/<shape_key>/<model>/<kernel>
   ```

   **[LEGACY — for older `.co` files that pre-date `co2cu.sh`]** If `build_iter.sh` is
   unavailable or the kernel does not extract cleanly, fall back to:
   ```bash
   $CHOREO_HOME/build/choreo -gs -t cute -arch=sm_90a [flags] <kernel>.co \
       -o /tmp/<kernel>.cute.result
   CHOREO_TIMING_WARMUP=5 CHOREO_TIMING_REPEAT=50 \
       bash /tmp/<kernel>.cute.result --execute
   ```
   Prefer the `co2cu.sh` + `build_iter.sh` path for all new tuning sessions.

3. **Record the baseline** via the harness. First, identify the shape dimensions (M, N, K) from the kernel, then:
   ```bash
   bash .claude/skills/croq-tune/tools/store_baseline.sh \
       --dsl croqtile --shape-key <shape_key> --model <model> \
       --dtype <dtype> --m <M> --n <N> --k <K> \
       --task-uid <task_uid>
   ```
   The harness writes `task_config.json`, measures cuBLAS TFLOPS, and records `iter000_cublas`
   in `tuning/${GPU}/croqtile/logs/<shape_key>/<model>/results.tsv`.

4. **Select the best** as the starting kernel for optimization. Copy it to:
   ```
   tuning/${GPU}/croqtile/srcs/<shape_key>/<model>/iter001_draft.co
   tuning/${GPU}/croqtile/srcs/<shape_key>/<model>/iter001_draft.cu
   ```
   Build scripts go to:
   ```
   tuning/${GPU}/croqtile/cmd/<shape_key>/<model>/build_iter001.sh
   tuning/${GPU}/croqtile/cmd/<shape_key>/<model>/run_iter001.sh
   ```

## Optimization Loop (Infinite)

### Step 1 — Profile current best (MANDATORY)

Profile the current best kernel with ncu. Read the report to identify the bottleneck.
If you made changes in a previous iteration, ALWAYS re-profile from this step.

**First, kill GPU contention:**
```bash
bash .claude/skills/croq-tune/tools/gpu_contention.sh --kill 2>/dev/null || true
```

**Profile using the harness wrapper (MANDATORY — do NOT call ncu directly):**
```bash
NCU_BASE="tuning/${GPU}/croqtile/perf/<shape_key>/<model>"
TAG="<iter_tag>"

bash .claude/skills/croq-tune/tools/ncu_profile.sh \
    --out "${NCU_BASE}/ncu_${TAG}_round<R>" \
    --cmd tuning/${GPU}/croqtile/bin/<shape_key>/<model>/<iter_binary>
```
Produces `ncu_<TAG>_round<R>.ncu-rep` + `ncu_<TAG>_round<R>.csv` atomically.

**Classify the bottleneck:**
```bash
PROFILE_JSON=$(bash .claude/skills/croq-tune/tools/profile_extract.sh \
    --csv  "${NCU_BASE}/ncu_${TAG}_round<R>.csv" \
    --iter "${TAG}")
echo "$PROFILE_JSON"
```
Output is one-line JSON with `bottleneck`, `confidence`, `evidence.key_metrics`. The `bottleneck`
field MUST be one of: `memory_bound`, `compute_bound`, `latency_bound`, `launch_bound`, `mixed`.
`unknown` is FORBIDDEN.

Key metrics to interpret manually when confidence is medium:

| Bottleneck symptom | ncu indicator | Typical fix |
|---|---|---|
| SM underutilized | sm__throughput.avg.pct < 80% | Increase CTAs, check occupancy |
| Memory bound | sm__pipe_tensor_op_pct < 50%, high l1tex ld | Larger tiles, TMA async, more stages |
| L2 thrash | lts__lookup_miss.sum high vs total | Change tile shape (M/N/K ratio) |
| WGMMA latency hidden | low wgmma__ops issue rate | Better producer/consumer overlap, more stages |
| Register pressure | sm__occ_pct_of_peak_smem_per_block_at_block_limit low | Reduce tile size, fewer threads per CTA |

### Step 2 — Raise an optimization idea

Based on the ncu analysis from Step 1, identify ONE concrete bottleneck and propose ONE
targeted optimization. The idea must be specific: not "make it faster" but "reduce X
bottleneck by doing Y because Z".

**Before forming the idea, run a mandatory web search:**
Search for the identified bottleneck (e.g. "CUDA WGMMA shared memory bank conflict matmul fp16").
Collect 1-3 external inspirations. Merge with local ncu evidence into exactly one testable idea.

**Brainstorming principles:**
- Ideas come from ncu data, not from guessing.
- Each iteration should test ONE idea. Multiple ideas in one commit make it impossible
  to know what worked.
- If an idea was tried before (see `idea-log.jsonl`), do NOT repeat the same change.
- Ideas may involve: tuning `#define` constants, changing tile shapes, adding/removing
  pipeline stages, warp-specialization ratio changes (1p1c → 1p3c), switching from
  sync to async TMA, persistent CTA vs static CTA, blockscale vs non-blockscale,
  swizzle factor, K-tiling depth, register blocking, smem bank conflicts, etc.
- Ideas may also involve Choreo DSL syntax: a new primitive form that doesn't exist
  yet in the compiler. If so, implement the compiler change first, rebuild, then use
  it in the kernel. Treat both as ONE atomic change.

**What NOT to do:**
- Do not change the host harness (main, timing, verification) — it is infrastructure.
- Do not change problem size (M/N/K) unless explicitly asked.
- Do not disable verification or timing to "cheat" a better score.
- Do not submit ideas identical to something already in `idea-log.jsonl`.
- Do not guess without ncu data. Profile first, hypothesize second.

**Write the checkpoint (MANDATORY — last action of IDEA):**
```bash
bash .claude/skills/croq-tune/tools/checkpoint_write.sh write \
    --dsl croqtile --shape-key <shape_key> --model <model> \
    --iter <planned_iter> \
    --bottleneck <bottleneck> \
    --idea "<one-line description>" \
    --expected-gain "<+X TFLOPS estimate>" \
    --levers "<comma-separated parameter names>"
```

### Step 3 — Implement and debug

**First action in IMPLEMENT — read back the checkpoint:**
```bash
bash .claude/skills/croq-tune/tools/checkpoint_write.sh read \
    --dsl croqtile --shape-key <shape_key> --model <model>
```
Build exactly what was planned.

**Second action — get canonical iteration name:**
```bash
ITER=$(bash .claude/skills/croq-tune/tools/next_iter.sh \
    --dsl croqtile --shape-key <shape_key> --model <model> --tag <short_idea_tag>)
echo "$ITER"
```

For each iteration, create a new versioned candidate:
```
tuning/${GPU}/croqtile/srcs/<shape_key>/<model>/${ITER}.co
tuning/${GPU}/croqtile/srcs/<shape_key>/<model>/${ITER}.cu
```
Do not mutate the current best kernel in place. Keep every tried candidate for traceability.

**Compile using co2cu + build_iter:**
```bash
# Phase 1: .co → .cu extraction
bash .claude/skills/croq-tune/tools/co2cu.sh \
    --co tuning/${GPU}/croqtile/srcs/<shape_key>/<model>/${ITER}.co \
    --arch sm_90a \
    [--flags "--use-warpspec --use-prepack"]

# Phase 2: .cu → binary
bash .claude/skills/croq-tune/tools/build_iter.sh \
    --cu  tuning/${GPU}/croqtile/srcs/<shape_key>/<model>/${ITER}.cu \
    --out tuning/${GPU}/croqtile/bin/<shape_key>/<model>/${ITER}
```
Save build log to `tuning/${GPU}/croqtile/perf/<shape_key>/<model>/build_${ITER}.txt`.

**Run functional test (no timing):**
```bash
tuning/${GPU}/croqtile/bin/<shape_key>/<model>/${ITER}
```

If it fails to compile:
- Read the choreo compiler error. Fix the `.co` source.
- Common issues: shape mismatch, wrong swizzle factor, WGMMA constraint violations,
  invalid event indexing in staged pipeline.
- Each fix attempt must try a genuinely different approach — do NOT blindly retry the same thing.
- Save failed attempt source as `attempt<AAAA>_<tag>.co` (does not consume iter sequence).

If it compiles but fails verification:
- Read the error. Common issues: wrong tile shape causing out-of-bounds, accumulator
  precision loss, race condition in producer/consumer.

**Hard debugging protocol:** If after 5 distinct fix attempts the kernel still doesn't work,
ABANDON the idea (mark as `COMPILE_FAIL`), STORE, and return to Step 2 with a new idea.

### Step 4 — Verify, Profile, and Decide

**Verify implementation matches checkpoint plan:**
```bash
bash .claude/skills/croq-tune/tools/checkpoint_write.sh verify \
    --dsl croqtile --shape-key <shape_key> --model <model> --iter <actual_iter_built>
```
Exit code 3 = significant drift from plan — investigate before continuing.

**Profile the candidate:**
```bash
bash .claude/skills/croq-tune/tools/gpu_contention.sh --kill 2>/dev/null || true

bash .claude/skills/croq-tune/tools/ncu_profile.sh \
    --out "tuning/${GPU}/croqtile/perf/<shape_key>/<model>/ncu_${ITER}_round<R>" \
    --cmd tuning/${GPU}/croqtile/bin/<shape_key>/<model>/${ITER}
```

**Benchmark (timing run):**
```bash
bash .claude/skills/croq-tune/tools/gpu_contention.sh --kill 2>/dev/null || true
CHOREO_TIMING_WARMUP=10 CHOREO_TIMING_REPEAT=50 \
    tuning/${GPU}/croqtile/bin/<shape_key>/<model>/${ITER} \
    2>&1 | tee tuning/${GPU}/croqtile/perf/<shape_key>/<model>/timing_${ITER}.txt
```

Extract `TFLOPS` from the output.

**Contention sanity check:** If measured TFLOPS is < 50% of current best, re-run
`gpu_contention.sh --kill` and re-benchmark before accepting the result.

**Decision rule:**
- If TFLOPS > current best TFLOPS: **KEEP** — proceed to STORE with `--decision KEEP`.
- If TFLOPS ≤ current best: **DISCARD** — proceed to STORE with `--decision DISCARD`.

### Step 5 — Store and iterate

**MANDATORY — STORE executes on every round (KEEP, DISCARD, COMPILE_FAIL):**
```bash
bash .claude/skills/croq-tune/tools/store_round.sh \
    --dsl croqtile --shape-key <shape_key> --model <model> \
    --iter iter<NNN> --kernel iter<NNN>_<tag> \
    --tflops <float> --decision <KEEP|DISCARD|COMPILE_FAIL|SEGFAULT|HANG> \
    --bottleneck <bottleneck> --idea "<summary>" --round <N> \
    --category "<tiling|pipeline|memory|compute|misc>" \
    --expected-gain "<e.g. +5% TFLOPS>"
```

The harness writes and verifies:
1. `tuning/${GPU}/croqtile/logs/<shape_key>/<model>/idea-log.jsonl`
2. `tuning/${GPU}/croqtile/logs/<shape_key>/<model>/results.tsv`

Do NOT proceed until the harness prints `[store_round] STORE complete`.
**NEVER bypass the harness by writing files manually.**

**After STORE, call reinforce (MANDATORY):**
```bash
bash .claude/skills/croq-tune/tools/reinforce.sh \
    --dsl croqtile --shape-key <shape_key> --model <model>
```
After reinforce, you MUST re-read `.claude/skills/base-tune/SKILL.md` before the next round.
This is the continuation contract.

Then **repeat from Step 1** using the improved kernel as the new baseline.

---

## Mandatory Verification (NEVER SKIP)

**Every iteration MUST pass verification before it can be KEPT.** A kernel that produces wrong results is worthless regardless of TFLOPS.

### Tolerance Guidelines

| Precision | base_tol | rel_tol | Notes |
|-----------|----------|---------|-------|
| FP16 input, FP32 accum | 1.0 | 0.01 | Standard |
| FP16 input, FP16 accum | 16.0 | 0.05 | Higher error from FP16 accumulation |
| FP8 E4M3 input, FP16 accum | 0.5 | 0.01 | FP8 inputs are lower magnitude |

### Requirements

- Host code MUST include verification block when `skip_verify` is false.
- **NEVER print "Test Passed" without actual numerical comparison.** Running kernel twice is NOT verification.
- **Keep `lhs_dense_h` alive** in host code — needed for CPU reference.

---

## Iteration File Naming

- `.co` iterations: `iter<NNN>_<brief-tag>.co` (e.g. `iter001_draft.co`, `iter012_stg4_swiz64.co`)
- `.cu` iterations (extracted CUDA, mandatory for croqtile): `iter<NNN>_<brief-tag>.cu`
- Failed attempts (do NOT consume iter sequence): `attempt<AAAA>_<tag>.co`
- Keep ALL iteration files (even discarded) for traceability.

Tag: 2-31 chars, lowercase alphanumeric + underscores, descriptive of the idea.
Bare `iter<NNN>` without a tag is rejected by the harness.

## Artifact Layout

All artifacts under: `tuning/${GPU}/croqtile/`

`${GPU}` is the string emitted by `bash .claude/skills/croq-tune/tools/detect_gpu.sh`
(e.g. `sm90_NVIDIA_H800_PCIe`, `sm90_H100`).

```
tuning/${GPU}/croqtile/srcs/<shape_key>/<model>/iter<NNN>_<tag>.co
tuning/${GPU}/croqtile/srcs/<shape_key>/<model>/iter<NNN>_<tag>.cu
tuning/${GPU}/croqtile/srcs/<shape_key>/<model>/attempt<AAAA>_<tag>.co
tuning/${GPU}/croqtile/bin/<shape_key>/<model>/iter<NNN>_<tag>           ← compiled binary
tuning/${GPU}/croqtile/cmd/<shape_key>/<model>/build_iter<NNN>.sh
tuning/${GPU}/croqtile/cmd/<shape_key>/<model>/run_iter<NNN>.sh
tuning/${GPU}/croqtile/perf/<shape_key>/<model>/build_iter<NNN>.txt
tuning/${GPU}/croqtile/perf/<shape_key>/<model>/timing_iter<NNN>.txt
tuning/${GPU}/croqtile/perf/<shape_key>/<model>/ncu_iter<NNN>_<tag>_round<R>.csv
tuning/${GPU}/croqtile/perf/<shape_key>/<model>/ncu_iter<NNN>_<tag>_round<R>.ncu-rep
tuning/${GPU}/croqtile/logs/<shape_key>/<model>/results.tsv
tuning/${GPU}/croqtile/logs/<shape_key>/<model>/idea-log.jsonl
tuning/${GPU}/croqtile/baseline/<shape_key>/<model>/cublas_result.json
tuning/${GPU}/croqtile/checkpoints/<shape_key>/<model>/current_idea.json
```

**Iteration numbering:**
- `iter000` = cuBLAS/library reference baseline (from `store_baseline.sh`)
- `iter001` = first custom kernel (from folder discovery)
- `iter002`, `iter003`, ... = measured tuning iterations
- `attempt<AAAA>` = compile-failed attempts (alphabetical: `attemptAAAA`, `attemptAAAB`, ...)

## Artifact Commit Protocol

- **Every KEEP iteration**: commit `iter<NNN>_<tag>.co`, `iter<NNN>_<tag>.cu`, and all
  associated perf/cmd artifacts immediately (local commit, no push):
  ```bash
  git add -A
  git commit -m "tune(croqtile): <shape_key> - iter<NNN>_<tag> <X> TFLOPS (KEEP)"
  ```
- **Every DISCARD**: recorded in `idea-log.jsonl` by the harness. Source may be included in the
  next batch commit tagged `(DISCARD)`.
- **On context compaction**: commit all pending work, note current state in commit message.
  After compaction, re-read this file and `idea-log.jsonl` to resume.

---

## Results Tracking

The harness maintains two files per `(shape_key, model)` pair:

**`logs/<shape_key>/<model>/results.tsv`** — one row per measured iter:
```
round   iter    kernel  tflops  decision    bottleneck  idea_summary
```

**`logs/<shape_key>/<model>/idea-log.jsonl`** — one JSON line per round:
```json
{"round": N, "iter": "iter<NNN>", "kernel": "iter<NNN>_<tag>", "tflops": X.X,
 "decision": "KEEP", "bottleneck": "...", "idea": "...", "category": "..."}
```

Both files are written by `store_round.sh`. **NEVER bypass the harness to write these directly.**
Before raising a new idea, read `idea-log.jsonl` to avoid repeating failed combinations.

---

## Mandatory Behavioral Rules (INVIOLABLE)

These rules are ABSOLUTE. Violating any of them renders the experiment worthless.

### Rule 1: PROFILE BEFORE EVERY IDEA — no exceptions

Before proposing ANY optimization idea, you MUST run `ncu_profile.sh` on the current best kernel
and classify the bottleneck with `profile_extract.sh`. The `bottleneck` field in `idea-log.jsonl`
MUST contain a real category (`memory_bound`, `compute_bound`, `latency_bound`, `launch_bound`,
`mixed`). The value `unknown` is FORBIDDEN. If you cannot profile, STOP and report the issue —
do NOT proceed with a guess.

### Rule 2: Hill-climb from structurally distinct top candidates

Maintain up to 2 "active bases" — the top-performing kernels that differ **structurally**
(different tiling scheme, persistent vs non-persistent, different warp specialization topology).
Each new candidate MUST derive from one of these active bases.

**Structural differences** (qualify for separate active base):
- Different tiling dimensions (BM/BN/BK)
- Persistent kernel vs static CTA launch
- Different warp specialization topology (1p1c vs 1p2c vs 1p3c)
- Different memory hierarchy strategy (shared-only vs TMA async pipeline)

**Non-structural differences** (do NOT qualify — same base):
- warpgroup_arrive placement, swizzle factor changes, register control tweaks, stage count

When a candidate beats its base: it becomes the new base best.
When it doesn't: revert to that base's best. Do NOT randomly hop between unrelated kernels.

### Rule 3: DIVERSE optimizations — macro sweeps alone are not optimization

Changing only `#define` macro values (e.g. WARP_N, STAGES) across iterations is
a grid search, not optimization. After at most 2 consecutive macro-only changes,
you MUST try a STRUCTURAL change. Structural changes include:
- Adding/removing/reordering synchronization fences (wgmma.fence, warpgroup_arrive)
- Changing register control (setmaxnreg / regctrl)
- Reordering load/store operations (e.g. RHS before LHS, metadata first)
- Adding compiler flags (--hoist-offset, --hoist-scale, --stmatrix)
- Changing output store pattern (shared padding, transpose)
- Modifying pipeline structure (event placement, commit placement)
- Switching warp specialization topology (1p1c ↔ 1p2c ↔ 1p3c)
- Adding explicit inline asm intrinsics (nanosleep, fence_proxy_async)
- Changing copy-shaping patterns (view().from(), subspan().step().at())

### Rule 4: NEVER repeat a failed combination

Before every iteration, read `idea-log.jsonl` and check whether the exact combination
(base kernel × parameter set × structural change) has been tried before. If it has, choose a
different idea. If a combo already failed, trying the same combo again is FORBIDDEN.

### Rule 5: ABANDON stuck ideas after 5 attempts

If an optimization idea fails to compile or pass verification after 5 distinct fix attempts,
ABANDON it entirely. Mark as `COMPILE_FAIL`, call `store_round.sh`, revert to current best,
and propose a completely different idea.

**Code bugs vs DSL/compiler limitations** (different retry budgets):
- **Code bug** (your kernel has a typo, wrong index, missing sync): fix and retry, up to 5 total.
- **DSL/compiler limitation** (unsupported instruction, codegen bug on valid input): work around
  it with a genuinely different approach each retry. Up to 5 workaround attempts.

### Rule 6: UNDERSTAND the kernel before mutating it

Before making any change, you MUST read and understand:
1. The current best `.co` kernel source code (full kernel function)
2. The generated `.cu` output (at least the kernel launch signature)
3. The ncu profiling data from Step 1

Each mutation must be accompanied by a 1-sentence hypothesis explaining WHY the change
should improve performance, grounded in specific ncu metrics.

### Rule 7: COMMIT messages must encode the optimization

Every commit message and `idea-log.jsonl` entry MUST include:
1. What was changed (e.g. "add wgmma.fence before K-loop")
2. Why it was expected to help (e.g. "to reduce WGMMA serialization")
3. The measured TFLOPS result
4. KEEP or DISCARD decision

The idea_summary MUST be a human-readable description, NOT a raw command line.

### Rule 8: USE the harness workflows — do NOT call tools directly

Compile and profile using the harness scripts in `.claude/skills/croq-tune/tools/`.
Do NOT call `ncu`, `nvcc`, or `$CHOREO_HOME/build/choreo` directly. You need the harness
for atomic artifact creation, GPU contention management, and consistent output formats.

### Rule 9: TRACK iteration counter monotonically

Each iteration gets a unique, monotonically increasing number via `next_iter.sh`.
Do not reuse numbers. Do not skip large ranges.

### Rule 10: gemm_sp-specific constraints

For gemm_sp (sparse GEMM) f16 kernels on SM90:
- `SPMM_WARP_M` MUST be 64 (WGMMA constraint — never change)
- `SPMM_WARP_K` MUST be 32 for f16 (never change)
- `SPMM_TILE_K` MUST equal `2 * SPMM_PACKED_TILE_K`
- `SPMM_META_TILE_COLS` MUST equal `SPMM_TILE_K / 32`
- Compiler flags: pass `--flags "--use-warpspec --use-prepack"` to `co2cu.sh`

---

## Anti-pattern Checklist (verify before every iteration)

Before submitting each iteration, verify you are NOT doing any of these:
- [ ] Changing only macros without any structural change (after 2 consecutive)
- [ ] Using `unknown` as the bottleneck category
- [ ] Repeating a combination already in `idea-log.jsonl`
- [ ] Starting from a different base kernel than the current best
- [ ] Submitting a raw command line as the idea_summary
- [ ] Skipping ncu profiling (calling `profile_extract.sh` without running `ncu_profile.sh` first)
- [ ] Bypassing `store_round.sh` by writing results files manually
- [ ] Fabricating TFLOPS numbers without a real run

---

## Stop Conditions

The ONLY valid reasons to stop the loop:
1. User manually stops the loop.
2. Compiler crashes repeatedly (likely a bug — escalate to user).
3. GPU failure that cannot be remediated by `gpu_check.sh --reset`.

**NEVER STOP otherwise.** User may leave the experiment overnight. Do NOT ask for
confirmation. Make all decisions autonomously. On context compaction, commit
pending work, then re-read this file and `idea-log.jsonl` to resume.

---

## Tunable Parameters Reference

The following are canonical degrees of freedom (`#define` macros):

| Macro | Typical values | Effect |
|---|---|---|
| `MATMUL_WARP_M` | 64 | Warp-group height |
| `MATMUL_WARP_N` | 64, 128, 192, 256 | Warp-group width |
| `MATMUL_TILE_M` | 64, 128, 192 | CTA tile height |
| `MATMUL_TILE_K` | 64 | CTA tile K-depth |
| `MATMUL_WARP_K` | 16 | WGMMA K-step per MMA instruction |
| `MATMUL_SWIZ` | 32, 64, 128 | TMA swizzle factor |
| `MATMUL_STAGES` | 2, 3, 4 | Producer/consumer pipeline depth |
| `NUM_SMS` | 114 (H800) | CTA count for persistent kernels |
| `MATMUL_DEFAULT_M/N/K` | 2048+ | Problem size |

---

## Available MMA / Copy Primitives (Choreo DSL)

```
# Data movement
dma.copy src => dst;
f = dma.copy.async src => shared after i_shared;
tma.copy.swiz<N> src => dst;
tma.copy.async<event>.swiz<N> src => dst;

# Compute
mma.fill.f16 0.0f;
mma.load.swiz<N> buffer;
mma.row.row accumulator, a, b;
mma.row.row.scale accumulator, a, b, scale_a, scale_b;
mma.row.col.sp accumulator, a, b;    # sparse MMA
mma.op <shape> accumulator, a, b;
mma.commit;
mma.store accumulator, output_s;
mma.store.transp accumulator, output_s;

# Control
parallel {p, q} by [a, b] : block;
parallel p by n : group-4;
inthreads.async (condition) { ... };
foreach {i} in [N] { ... };
wait event;
trigger event;
```

---

## Reference Kernels (read before editing)

Study these kernels to understand Choreo DSL patterns.
See `.claude/skills/choreo-kernel-examples/` for the full collection.

- `matmul/matmul_f16_dyn_sm90.co` — baseline FP16 GEMM with TMA+swizzle
- `matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co` — 1-producer/1-consumer warp-specialized GEMM with staged events
- `matmul/matmul_f16_dyn_sm90_warpspec_1p3c.co` — 1-producer/3-consumer warp-specialized GEMM (wider CTA tile)
- `matmul/matmul_f16_dyn_persis_sta_sm90.co` — persistent CTA with `.step().at()` tile iteration
- `gemm_sp/gemm_sp_f16_dyn_sm90_warpspec_1p2c_swizzle128_128_prepack.co` — sparse GEMM with warp-spec and prepack

---

## Harness Scripts Reference

All scripts under `.claude/skills/croq-tune/tools/`.

| Script | When to call | Required args |
|---|---|---|
| `detect_gpu.sh` | Pre-flight | (none) — result cached in `/tmp/croq_gpu_key` |
| `validate_env.sh` | Pre-flight | `--dsl` |
| `store_baseline.sh` | Baseline Discovery step 3 | `--dsl --shape-key --model --dtype --m --n --k --task-uid` |
| `co2cu.sh` | IMPLEMENT Phase 1 | `--co --arch [--flags]` |
| `build_iter.sh` | IMPLEMENT Phase 2 | `--cu --out` |
| `checkpoint_write.sh write` | End of IDEA | `--dsl --shape-key --model --iter --bottleneck --idea --expected-gain --levers` |
| `checkpoint_write.sh read` | Start of IMPLEMENT | `--dsl --shape-key --model` |
| `checkpoint_write.sh verify` | Start of VERIFY | `--dsl --shape-key --model --iter` |
| `next_iter.sh` | Start of IMPLEMENT | `--dsl --shape-key --model --tag` |
| `ncu_profile.sh` | PROFILE step | `--out --cmd` |
| `profile_extract.sh` | After ncu CSV | `--csv --iter` |
| `gpu_contention.sh` | Before PROFILE and MEASURE | (none) scan; `--kill` kill foreign |
| `gpu_check.sh` | GPU state check | (none), or `--wait --timeout 120`, or `--reset` |
| `store_round.sh` | STORE step | `--dsl --shape-key --model --iter --kernel --tflops --decision --bottleneck --idea --round` |
| `reinforce.sh` | After STORE (MANDATORY) | `--dsl --shape-key --model` |

---

## Related Skills

- `.claude/skills/choreo-syntax/SKILL.md` — DSL reference for `.co` editing (load before any `.co` change)
- `.claude/skills/choreo-kernel-examples/` — Reference kernel collection (study before editing)
- `.claude/skills/croq-tune/SKILL.md` — Full production tuning entrypoint (uses same harness)
- `.claude/skills/croq-dsl-croqtile/SKILL.md` — CroqTile-specific DSL contracts
- `.claude/skills/perf-nsight-compute-analysis/SKILL.md` — Systematic ncu report analysis

---

**IMPORTANT: DO NOT STOP THE LOOP UNTIL USER MANUALLY STOPS. USER MAY LEAVE THE EXPERIMENT OVERNIGHT. DO NOT ASK FOR CONFIRMATION. MAKE ALL DECISIONS AUTONOMOUSLY. ON CONTEXT COMPACTION, COMMIT PENDING WORK, THEN RE-READ THIS FILE AND `idea-log.jsonl` TO RESUME. LOOP INFINITELY.**
