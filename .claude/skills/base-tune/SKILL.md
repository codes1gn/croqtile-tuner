---
name: base-tune
description: Launch an infinite AI-driven kernel optimization loop for GPU kernels. Use when the user asks to "tune", "ai-tune", "optimize", or "auto-tune" a kernel. Profiles with ncu, iterates optimizations indefinitely until interrupted.
argument-hint: <kernel-folder-path e.g. .claude/skills/choreo-kernel-examples/gemm_sp/>
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

Validation: `$CHOREO_HOME/build/choreo --help` should print usage.

**Note**: Sparse GEMM kernels with TMA require CUDA 12.9+. Check `nvcc --version`.

## Pre-flight

1. **Read this entire file** — it defines the loop protocol, mandatory rules, and constraints. Follow it exactly.

2. **Read the syntax reference**: Load the `choreo-syntax` skill before editing any `.co` file.

3. **Identify the target folder**: The user specifies a kernel folder (e.g. `.claude/skills/choreo-kernel-examples/gemm_sp/`). All kernel `.co` files live there.

4. **Determine the kernel mnemonic**: Derive a short mnemonic from the folder name (e.g. `gemm_sp`, `gemm_sp_e4m3`). The user may specify a more specific mnemonic if the folder contains multiple kernel families.

## Branch Setup

1. **Generate the branch name**: `ai-tune/<today's date YYYY-MM-DD>/<kernel-mnemonic>`.
2. **Check uniqueness**: Run `git branch -a | grep ai-tune` and verify no branch with the same name exists. If a collision exists, append a numeric suffix (e.g. `-2`, `-3`).
3. **Create and switch**:
   ```bash
   git checkout -b ai-tune/<date>/<mnemonic>
   ```

## Baseline Discovery

Before any optimization, find the current best kernel(s) in the target folder:

1. **List all `.co` files** in the folder and identify candidates by name patterns (warpspec, prepack, stages, swizzle, etc.).
2. **Compile and benchmark the top 2-3 candidates** to find the actual best performer. Use:
   ```bash
   $CHOREO_HOME/choreo -gs -t cute -arch=sm_90a [flags] <kernel>.co -o /tmp/<kernel>.cute.result
   CHOREO_TIMING_WARMUP=5 CHOREO_TIMING_REPEAT=50 CUDA_VISIBLE_DEVICES=<free-gpu> \
     bash /tmp/<kernel>.cute.result --execute
   ```
   Set `CHOREO_HOME` to the choreo/croqtile repo path (e.g. `/home/user/croqtile`).
   Choose appropriate choreo flags for the kernel type (e.g. `--use-warpspec --use-prepack` for sparse GEMM).
3. **Record the baseline** in `results.tsv` with the full `run_command`.
4. **Select the best** as the starting genome for optimization.

## Optimization Loop (Infinite)

### Step 1 — Profile current best (MANDATORY)

Profile the current best kernel with ncu. Read the report to identify the bottleneck.
If you made changes in a previous iteration, ALWAYS re-profile from this step.

```bash
KERNEL=<path_to_current_best>.co
KERNEL_OUT=<path_to_current_best>

# Always recompile first if source changed
$CHOREO_HOME/choreo -gs -t cute -arch=sm_90a $KERNEL -o ${KERNEL_OUT}.cute.result
# you may need --use-warpspec --use-prepack or other optional switches

# ncu profiling
/usr/local/cuda/bin/ncu --set full --target-processes all \
  -o ${KERNEL_OUT}_ncu_iter<N> \
  bash ${KERNEL_OUT}.cute.result --execute
```

Read the `.ncu-rep` file. Key metrics to interpret:

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

**Brainstorming principles:**
- Ideas come from ncu data, not from guessing.
- Each iteration should test ONE idea. Multiple ideas in one commit make it impossible
  to know what worked.
- If an idea was tried before (see results.tsv), do NOT repeat the same change.
- Ideas may involve: tuning #define constants, changing tile shapes, adding/removing
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
- Do not submit ideas identical to something already in results.tsv.
- Do not guess without ncu data. Profile first, hypothesize second.

### Step 3 — Implement and debug

For each iteration, create a new versioned candidate `.co` file and edit that candidate only.
Do not mutate the current best kernel file in place. Keep every tried candidate file for traceability.

```bash
# Edit the kernel
$EDITOR <target_folder>/<your_kernel>.co

# Compile (set CHOREO_HOME to your choreo repo, e.g. /home/user/croqtile)
$CHOREO_HOME/choreo -gs -t cute -arch=sm_90a <target_folder>/<your_kernel>.co \
  -o <target_folder>/<your_kernel>.cute.result

# Run functional test (no timing)
bash <target_folder>/<your_kernel>.cute.result --execute
```

If it fails to compile:
- Read the choreo compiler error. Fix the `.co` source.
- Common issues: shape mismatch, wrong swizzle factor, WGMMA constraint violations,
  invalid event indexing in staged pipeline.

If it compiles but fails verification:
- Read the error. Common issues: wrong tile shape causing out-of-bounds, accumulator
  precision loss, race condition in producer/consumer.

**Hard debugging protocol:** If after 3 distinct fix attempts within the same idea
the kernel still doesn't work, ABANDON the idea. Do not spend more than 3 iterations
debugging one idea. Discard the broken code, revert to last known-good state, and go
back to Step 2 with a new idea.

### Step 4 — Profile and decide

```bash
# Profile
/usr/local/cuda/bin/ncu --set full --target-processes all \
  -o ${KERNEL_OUT}_ncu_iter<N> \
  bash ${KERNEL_OUT}.cute.result --execute

# Timing run
bash ${KERNEL_OUT}.cute.result --execute
```

Extract `TFLOPS` and `HW efficiency %` from the output.

**Decision rule:**
- If TFLOPS > current best TFLOPS: **KEEP** — commit, update best, go to Step 1.
- If TFLOPS ≤ current best: **DISCARD** — revert to last best, go to Step 2.

### Step 5 — Commit and iterate

After a successful (kept) optimization:

```bash
# Commit on ai-tune/<tag> branch
git add <target_folder>/<kernel>.co
# Also add any compiler changes if applicable
git commit -m "iter<N>: <brief description> — TFLOPS: X -> Y"

# Update results.tsv (append one row; include reproducible run command)
echo -e "iter<N>\t<KERNEL>\t<ARCH>\t<TFLOPS>\t<EFF%>\t<BOTTLENECK>\t<RUN_COMMAND>\t<IDEA>" >> results.tsv
```

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
- **Keep `lhs_dense_h` alive** in host code - needed for CPU reference.

---

## Iteration File Naming

- `.co` iterations: `<kernel-base>_iter<NNN>_<brief-tag>.co` (e.g. `blockscale_gemm_e4m3_iter001_tma_meta.co`)
- `.cu` iterations (when modifying generated CUDA): `<kernel-base>_iter<NNN>_<brief-tag>.cu`
- Keep ALL iteration files (even discarded ones) for traceability.

## Artifact Management

- **Every KEEP iteration**: `git add` the new `.co` (and `.cu` if applicable), `results.tsv`, and any compiler changes. Commit with: `iter<NNN>: <description> - TFLOPS: X -> Y (KEEP)`
- **Every DISCARD**: Note in `results.tsv` but do not commit the failed kernel file. Or commit with `(DISCARD)` tag for traceability.
- **Periodic pushes**: Push to the remote branch every 5-10 iterations or after significant wins.
- **On context compaction**: Before the context window fills, commit all pending work, push, and note the current state in the commit message. After compaction, re-read this file and `results.tsv` to resume.

---

## Results Tracking

Create and maintain `results.tsv` in repo root with columns:

```
iter	kernel	arch	tflops	eff%	bottleneck_before	run_command	idea_summary
```

Append one row per iteration. This is the experiment log — it is the agent's memory.
Before raising a new idea, consult this file to avoid repeating ideas.

---

## Mandatory Behavioral Rules (INVIOLABLE)

These rules are ABSOLUTE. Violating any of them renders the experiment worthless.

### Rule 1: PROFILE BEFORE EVERY IDEA — no exceptions

Before proposing ANY optimization idea, you MUST run ncu on the current best kernel
and read the report. The `bottleneck_before` column in results.tsv MUST contain a
real bottleneck category (e.g. `wgmma_serialized`, `smem_pressure`, `memory_bound`,
`l1tex_stall`, `occupancy`). The value `unknown` is FORBIDDEN. If you cannot profile,
STOP and report the issue — do NOT proceed with a guess.

### Rule 2: ONE current best, hill-climb from it — no random base-hopping

At any point there is exactly ONE file designated as the "current best". Every new
candidate MUST be derived from that file. You are FORBIDDEN from randomly picking a
different base kernel each iteration. When a candidate beats the current best, it
becomes the new current best. When it doesn't, you revert to the current best.

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

Before every iteration, you MUST read results.tsv and check whether the exact
combination (base kernel × parameter set × structural change) has been tried before.
If it has, you MUST choose a different idea. Specifically:
- If a parameter combo already failed, trying the same combo again is FORBIDDEN.
- If a parameter combo failed with a specific exit code (e.g. smem overflow),
  do NOT retry without first addressing the root cause.
- Keep a mental blacklist of failed combinations.

### Rule 5: ABANDON stuck ideas after 3 attempts

If an optimization idea fails to compile or pass verification after 3 distinct
fix attempts, ABANDON it entirely. Do not spend iteration 4, 5, 6... on the same
idea. Revert to the current best and propose a completely different idea.

### Rule 6: UNDERSTAND the kernel before mutating it

Before making any change, you MUST read and understand:
1. The current best `.co` kernel source code (the full kernel function)
2. The generated `.cute.result` output (at least the kernel launch signature)
3. The ncu profiling data from Step 1

You are FORBIDDEN from treating the build/verify pipeline as a black box. Each
mutation must be accompanied by a 1-sentence hypothesis explaining WHY the change
should improve performance, grounded in specific ncu metrics.

### Rule 7: COMMIT messages must encode the optimization

Every commit message and results.tsv entry MUST include:
1. What was changed (e.g. "add wgmma.fence before K-loop")
2. Why it was expected to help (e.g. "to reduce WGMMA serialization")
3. The measured TFLOPS result
4. KEEP or DISCARD decision

The idea_summary column MUST be a human-readable description, NOT a raw command line.

### Rule 8: USE the compile+run workflow from this file

Compile and run kernels using the workflow described here:
```
$CHOREO_HOME/choreo -gs -t cute -arch=sm_90a $KERNEL -o ${OUT}.cute.result
bash ${OUT}.cute.result --execute
```
Do NOT delegate to external wrapper scripts as black boxes. You need to see and
understand compiler output, error messages, and timing output directly.

### Rule 9: TRACK your iteration counter monotonically

Each iteration gets a unique, monotonically increasing number. Do not reuse numbers.
Do not skip large ranges. The iteration number in your commit and in results.tsv
must match.

### Rule 10: gemm_sp-specific constraints

For gemm_sp (sparse GEMM) f16 kernels on SM90:
- `SPMM_WARP_M` MUST be 64 (WGMMA constraint — never change)
- `SPMM_WARP_K` MUST be 32 for f16 (never change)
- `SPMM_TILE_K` MUST equal `2 * SPMM_PACKED_TILE_K`
- `SPMM_META_TILE_COLS` MUST equal `SPMM_TILE_K / 32`
- Compiler flags: `-t cute -arch=sm_90a --use-warpspec --use-prepack`

---

## Anti-pattern Checklist (verify before every iteration)

Before submitting each iteration, verify you are NOT doing any of these:
- [ ] Changing only macros without any structural change (after 2 consecutive)
- [ ] Using `unknown` as the bottleneck category
- [ ] Repeating a combination already in results.tsv
- [ ] Starting from a different base kernel than the current best
- [ ] Submitting a raw command line as the idea_summary
- [ ] Skipping ncu profiling

---

## Stop Conditions

The ONLY valid reasons to stop the loop:
1. User manually stops the loop.
2. 10 consecutive discarded ideas (stuck in local minimum — report to user).
3. Compiler crashes repeatedly (likely a bug, escalate to user).

**NEVER STOP otherwise.** User may leave the experiment overnight. Do NOT ask for
confirmation. Make all decisions autonomously. On context compaction, commit
pending work, then re-read this file and results.tsv to resume.

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

Study these kernels to understand Choreo DSL patterns. See `.claude/skills/choreo-kernel-examples/` for the full collection.

- `matmul/matmul_f16_dyn_sm90.co` — baseline FP16 GEMM with TMA+swizzle
- `matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co` — 1-producer/1-consumer warp-specialized GEMM with staged events
- `matmul/matmul_f16_dyn_sm90_warpspec_1p3c.co` — 1-producer/3-consumer warp-specialized GEMM (wider CTA tile)
- `matmul/matmul_f16_dyn_persis_sta_sm90.co` — persistent CTA with `.step().at()` tile iteration
- `gemm_sp/gemm_sp_f16_dyn_sm90_warpspec_1p2c_swizzle128_128_prepack.co` — sparse GEMM with warp-spec and prepack

---

## Related Skills

- `choreo-syntax` — DSL reference for `.co` editing
- `choreo-kernel-examples` — Reference kernel collection (study before editing)
- `compile-and-test` — build/run workflows
- `develop-compiler` — when compiler changes are needed
- `profiling` — ncu invocation and metric interpretation
- `performance-bottleneck-analysis` — interpreting ncu reports

---

**IMPORTANT: DO NOT STOP THE LOOP UNTIL USER MANUALLY STOPS. USER MAY LEAVE THE EXPERIMENT OVERNIGHT. DO NOT ASK FOR CONFIRMATION. MAKE ALL DECISIONS AUTONOMOUSLY. ON CONTEXT COMPACTION, COMMIT PENDING WORK, THEN RE-READ THIS FILE AND results.tsv TO RESUME. LOOP INFINITELY.**
