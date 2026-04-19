---
name: cursor-croq-tune
description: Cursor-only kernel tuning entrypoint for GPU kernels. Use when the user asks to /croq-tune, continue tuning, resume tuning, ai-tune, auto-tune, optimize, or perf-tune a kernel in this repository. Profiles with ncu, iterates optimizations indefinitely until interrupted.
argument-hint: <dsl: croqtile|cuda|cute|triton|tilelang|helion|cutile> <dtype: f16|e4m3|all> [shape_key]
---

# Croq-Tune — Unified Kernel Tuning Loop

`cursor-croq-tune` is the Cursor-only tuning entrypoint. Load this file and one `cursor-croq-dsl-<dsl>` file per session — nothing else.

## Main Loop

1. Parse `dsl`, `dtype`, optional `shape_key`, `--model`, `--task-uid`, and any user-stated invocation round budget
2. Startup/resume decision (see "Resume Contract" below)
3. Load `cursor-croq-dsl-<dsl>` for DSL-specific commands (BUILD, RUN, PROFILE, IDEA menu)
4. Run `PREPARATION_ONCE` for this `(dsl, operator, dtype, shape_key)`
5. Repeat round loop until a valid stop condition is met

## Invocation Run Budget

Treat each tuning launch as a long-running execution lease, not a single-answer task.

- Default target: `50` completed STORE rounds per invocation.
- If the user explicitly asks for a larger round count, use that larger value.
- If the user asks to run continuously or indefinitely, treat `50` as the minimum floor before completion is eligible.
- A round counts only after the full pipeline completes and `store_round.sh` succeeds.
- Baseline setup, discovery, one KEEP, one DISCARD, one commit, one reinforce, or one profiling pass does not mean the invocation is done.

Track the launch-local budget as follows:

- Fresh start: `start_round = 0`
- Resume: `start_round = last_round` from `resume_state.sh` before new work
- `completed_this_invocation = current_stored_round - start_round`

Do not conclude the invocation until `completed_this_invocation >= target_completed_rounds`, unless a valid non-local-minimum stop condition fires.

To avoid premature local-minimum exits, escalate before the DISCARD ceiling:

- At 7 consecutive DISCARDs on the same active base: force baseline/custom SASS diff plus fresh baseline reprofiling.
- At 8-9 consecutive DISCARDs: force a structural reset or switch to the other active base; do not spend those rounds on minor macro sweeps.
- Do not drift into the local-minimum stop condition accidentally because of weak iteration planning.

**CRITICAL: Use the `shape_key` EXACTLY as given in the prompt.**
If the prompt says `tune cuda f16fp32 matmul_fp16fp32_16384x16384x512`, the shape_key is
`matmul_fp16fp32_16384x16384x512` — do NOT re-derive or rename it. This string is used
as-is for all folder paths and harness script `--shape-key` arguments.

**CRITICAL: Extract `--task-uid` from the prompt and pass it to `store_baseline.sh`.**
The task-uid is the unique identifier that connects this tuning session to the monitor
database. Without it, the monitor may create phantom duplicate tasks.

---

## Hard Constraints

1. Keep exactly one continuation node in progress: `continue-croq-tune`
2. Trigger proactive compaction at `>= 80%` context, but only after continuation node is present
3. Compile-fail retries: code bugs ≤ 5 fix attempts, DSL limitations ≤ 5 workaround attempts; then COMPILE_FAIL → STORE → IDEA
4. Exactly one new idea per round
5. STORE executes on both KEEP and DISCARD
6. **NO LIBRARY CALLS IN TUNING ITERATIONS** — see "Pure Implementation Rule"
7. **NEVER ASK THE HUMAN QUESTIONS** — Make all decisions autonomously. If you encounter ambiguity, choose the most reasonable default and proceed. Never use `AskQuestion`, `question`, or any interactive prompt tool during tuning execution.
8. Do not treat one successful round, one commit, or one profiling pass as completion; continue until the invocation run budget is satisfied.

---

## Mandatory Behavioral Rules

These rules ensure experiment validity. Violating any wastes the entire round.

### Rule 1: PROFILE BEFORE EVERY IDEA — no exceptions

Before proposing ANY optimization idea, you MUST run ncu on the current best kernel
and read the report. If you cannot profile, STOP and report — do NOT guess.

### Rule 2: Hill-climb from structurally distinct top candidates

Maintain up to 2 "active bases" — the top-performing kernels that differ **structurally**
(e.g. different tiling scheme, persistent vs non-persistent, different warp specialization
topology). Each new candidate MUST derive from one of these active bases.

**Structural differences** (qualify for separate active base):
- Different tiling dimensions (BM/BN/BK)
- Persistent kernel vs static CTA launch
- Different warp specialization topology (1p1c vs 1p2c vs 1p3c)
- Different memory hierarchy strategy (shared-only vs TMA async pipeline)

**Non-structural differences** (do NOT qualify — same base):
- warpgroup_arrive placement
- Swizzle factor changes
- Register control tweaks
- Pipeline stage count

When a candidate beats the best of its base: it replaces that base's best.
When it doesn't: revert to that base's best. Do NOT randomly hop between unrelated kernels.

### Rule 3: DIVERSE optimizations — macro sweeps alone are not optimization

After at most 2 consecutive macro-only changes, you MUST try a STRUCTURAL change.
See the loaded `cursor-croq-dsl-<dsl>` file for DSL-specific structural changes.

### Rule 4: NEVER repeat a failed combination

Before every iteration, check `idea-log.jsonl`. If the exact combination (base kernel
x parameter set x structural change) was tried before, choose a different idea.

### Rule 5: ABANDON stuck ideas after 5 attempts

If an optimization idea fails to compile or pass verification after 5 distinct fix
attempts, ABANDON it. Revert to the active base's best and propose a completely different idea.

**Code bugs vs DSL/compiler limitations** (different retry budgets):
- **Code bug** (your kernel has a typo, wrong index, missing sync): fix and retry,
  up to 5 attempts total. These are normal.
- **DSL/compiler limitation** (unsupported instruction for this arch, codegen bug on valid
  input): do NOT blindly retry the same approach. Work around it — use a different
  instruction, tiling, or code pattern. Up to 5 workaround attempts, each trying a
  genuinely different approach. Only fall back to a different variant/DSL after all 5 fail.
  See the `cursor-croq-dsl-<dsl>` skill for arch-specific workaround patterns.

### Rule 6: UNDERSTAND the kernel before mutating it

Before making any change, you MUST read and understand:
1. The current best kernel source code (full kernel function)
2. The generated output (at least kernel launch signature)
3. The ncu profiling data from the PROFILE step

Each mutation must be accompanied by a 1-sentence hypothesis explaining WHY.

### Rule 7: COMMIT messages must encode the optimization

Every commit message and `idea-log.jsonl` entry MUST include:
1. What was changed
2. Why it was expected to help
3. The measured TFLOPS result
4. KEEP or DISCARD decision

The idea_summary MUST be human-readable, NOT a raw command line.

### Rule 8: USE the compile+run workflow

Compile and run using the workflow described in `cursor-croq-dsl-<dsl>`. Do NOT delegate to
external wrapper scripts as black boxes. You need to see compiler output directly.

### Rule 9: TRACK iteration counter monotonically

Each iteration gets a unique, monotonically increasing number. Do not reuse numbers.
Do not skip large ranges. Use `next_iter.sh` to get the canonical name.

---

## Pure Implementation Rule

All tuning iterations (`iter001` and beyond) MUST use **pure kernel code**.
Per-DSL allowed/forbidden lists are in the loaded `cursor-croq-dsl-<dsl>` file.

**Allowed in all DSLs:** Raw kernel code, CUDA intrinsics, PTX inline asm, cooperative groups, warp-level primitives, DSL-specific primitives.

**FORBIDDEN in all DSLs:** cuBLAS, cuTLASS library GEMM, cuSPARSELt, cuDNN, PyTorch/TensorFlow ops, any code that delegates core compute to a library.

**Where library calls ARE allowed:** iter000 baseline only; verification reference computation. Never in iter001+.

**Enforcement:** If IMPLEMENT produces a library call for core compute, reject immediately, log as `attempt<AAAA>` with reason `library_call_forbidden`, return to IDEA.

---

## CRITICAL ANTI-PATTERNS (FORBIDDEN)

- **Fabricating iteration data** — Every `iter<NNN>` in `results.tsv` / `idea-log.jsonl` MUST correspond to a real kernel run with real TFLOPS measurement
- **Batch-generating fake results** — NEVER create multiple iteration records with identical timestamps or synthetic values
- **Skipping profiling** — If ncu fails, STOP; do NOT silently skip or guess
- **Proceeding without evidence** — Every IDEA must be based on real ncu profiling data
- **Bypassing baseline validation** — NEVER start tuning without environment validation passing
- **`unknown` as bottleneck** — MUST contain a real category from the classifier (`memory_bound`, `compute_bound`, `latency_bound`, `launch_bound`, `mixed`). `unknown` is FORBIDDEN.
- **Repeating failed combinations** — Before every IDEA, check `idea-log.jsonl`
- **Random base-hopping** — Hill-climb from active bases only (max 2 structurally distinct top kernels)
- **Macro-only sweeps** — After 2 consecutive macro-only changes, you MUST try a STRUCTURAL change
- **Changing host harness** — Do not change the host harness (main, timing, verification) — it is infrastructure
- **Changing problem shape (M/N/K)** — The problem dimensions are set by the user. Do NOT change M, N, K, or dtype to get a "better" score
- **Disabling verification** — Do not disable verification or timing to "cheat" a better score

---

## PREPARATION_ONCE (Per Shape, Outside Round Count)

Run once before round counting starts for a new `(dsl, operator, dtype, shape_key)`.

### 1. Environment Validation (MANDATORY, BLOCKING)

**BEFORE any tuning work begins**, run the environment validation harness:

```bash
bash .cursor/skills/cursor-croq-tune/tools/validate_env.sh --dsl <dsl>
```

The harness validates: nvidia-smi, nvcc, ncu, perf_event_paranoid, GPU availability,
plus DSL-specific checks (e.g. `CHOREO_HOME`/`CUTE_HOME`/`CUDA_HOME` for croqtile,
`import triton` for triton, etc.). Output is one-line JSON.

If the harness exits non-zero: **STOP and escalate to user.** Show the `errors` array
from the JSON output. Tuning CANNOT start if validation fails. No fallbacks.

### 2. Baseline Environment + cuBLAS Reference (MANDATORY)

**Step 2a — Check environment libraries:**
```bash
python3 .cursor/skills/cursor-croq-tune/tools/prepare_baseline_env.py \
  --dsl <dsl> --operator <op> --kernel <kernel> --shape-key <shape_key> --libs auto
```

**Step 2b — Measure + store cuBLAS reference (MANDATORY, BLOCKING):**

Before any tuning starts, you MUST measure a cuBLAS/torch.mm baseline to know
the hardware ceiling. Use `store_baseline.sh` which atomically measures, saves
artifacts, and records `iter000_cublas` in `results.tsv` — making it impossible
to accidentally skip the baseline.

```bash
# store_baseline.sh calls cublas_baseline.sh, which already runs gpu_contention.sh --kill
# automatically. For an explicit check before calling store_baseline.sh:
bash .cursor/skills/cursor-croq-tune/tools/gpu_contention.sh --kill 2>/dev/null || true

bash .cursor/skills/cursor-croq-tune/tools/store_baseline.sh \
    --dsl <dsl> --shape-key <shape_key> --model <model> \
    --dtype <dtype> --m <M> --n <N> --k <K> \
    --task-uid <task_uid>
```

**`--task-uid` is MANDATORY.** Extract `<task_uid>` from the `--task-uid` argument in the
original prompt (e.g., `tune cuda f16fp32 matmul_... --task-uid abc123def456`).
The script writes `task_config.json` with this uid as the first disk artifact, preventing
phantom task creation by the monitor.

The script:
1. Writes `task_config.json` with `task_uid` (earliest possible disk anchor)
2. Runs `cublas_baseline.sh` to measure TFLOPS
3. Saves a JSON artifact to `tuning/<gpu>/<dsl>/baseline/<shape_key>/<model>/cublas_result.json`
4. Stores `iter000_cublas` in `results.tsv` and `idea-log.jsonl` via `store_round.sh`
5. Is idempotent — re-running skips if baseline already exists

The baseline number is used for all KEEP/DISCARD context.

**NEVER skip this step.** Even for small shapes, the cuBLAS baseline gives context:
- If cuBLAS gets 0.5 TFLOPS and your kernel gets 0.4, that's 80% — good
- Without the baseline, 0.4 TFLOPS is meaningless

### 3. Starting Kernel Discovery (MANDATORY)

Before entering the round loop, find or create a starting kernel. Run the discovery harness:

```bash
DISCOVERY=$(bash .cursor/skills/cursor-croq-tune/tools/discover_baseline.sh \
    --dsl <dsl> --operator <operator> --dtype <dtype> --gpu "$GPU")
echo "$DISCOVERY"
```

The harness scans 2 tiers and returns one of 3 recommendations:

**Tier A — Reference examples** (`cursor-choreo-kernel-examples/`):
Scans for kernels matching the target operator/dtype. Files containing `_aitune_` in
the filename are **always excluded** (these are prior tuning outputs, not clean references).
If candidates found: compile and benchmark the top 2-3, select the best as `iter001`.

**Tier B — Prior tuning sessions** (`tuning/<gpu>/<dsl>/srcs/`):
Scans for same-operator kernels from different shapes (e.g. a matmul_f16 tuned at
4096x4096x4096 may be adaptable to 16384x16384x16384). If found: compile, verify
correctness at the target shape, benchmark, and use the best as `iter001` if it works.

**Tier C — Implement from scratch** (no candidates from A or B):
When no existing kernel is usable:
1. Run web search: research `<operator> GPU kernel <dsl> implementation` patterns
2. Collect 2-3 implementation references/papers
3. Implement a kernel that uses MMA/tensor-core instructions (NOT scalar loops, NOT naive per-thread accumulation)
4. Minimum bar: must use at least `mma` instructions (or DSL equivalent: `tl.dot` for Triton, `mma.op` for Choreo, etc.)
5. Must be correct (pass verification) and faster than a trivial scalar baseline

**Tier C — DSL/compiler limitation workaround:**
If the chosen DSL hits a limitation for the target operator+dtype+arch (e.g. compiler
codegen bug, unsupported instruction for the GPU arch), do NOT stop or ask the user.
Treat it as a **debug problem** and work around it:
1. Identify the specific unsupported feature (e.g. WGMMA on SM86, sparse codegen bug)
2. Find an alternative that IS supported (e.g. use MMA instead of WGMMA, use different
   tiling to avoid the codegen bug, use a supported dtype proxy)
3. You have up to **5 workaround attempts** to get a compiling+correct kernel
4. Each attempt should try a genuinely different approach, not just retry the same thing
5. If after 5 attempts no workaround works, THEN STORE as `COMPILE_FAIL` with
   `"reason": "dsl_limitation"` and fall back to the closest feasible variant:
   - If sparse is unsupported: fall back to dense GEMM of the same shape+dtype
   - If the dtype is unsupported: fall back to the closest supported dtype (e.g. e4m3→f16)
   - If the DSL itself cannot target this arch: switch to raw CUDA as DSL
6. Log the workaround decision in `idea-log.jsonl`
7. Continue the tuning loop — NEVER stop and wait for user input

**After discovery, create the iter001 artifacts:**
1. Source: `tuning/<gpu>/<dsl>/srcs/<shape_key>/<model>/iter001_draft.<cu|co|py>`
2. Build script: `tuning/<gpu>/<dsl>/cmd/<shape_key>/<model>/build_iter001.sh`
3. Run script: `tuning/<gpu>/<dsl>/cmd/<shape_key>/<model>/run_iter001.sh`

Use the BUILD/RUN templates from the loaded `cursor-croq-dsl-<dsl>` file.

---

## Resume Contract

### Shape-Key Exact-Match Guard (MANDATORY, RUNS FIRST)

**Before ANY resume decision**, verify the requested shape exactly matches an existing session character-for-character. Dimension order is part of the identity — `16384x16384x512` and `512x16384x16384` are different shapes and MUST NOT be conflated.

```
requested_key = matmul_<dtype>_<M>x<N>x<K>   ← from user input, left-to-right
existing_keys = ls tuning/<gpu>/<dsl>/srcs/    ← from disk
```

If `requested_key` is NOT in `existing_keys` → **fresh start**. Do not resume a session with a different key.

### Resume State — Use the Harness (MANDATORY)

**DO NOT manually read results.tsv, idea-log.jsonl, etc. to reconstruct state.**

```bash
GPU=$(bash .cursor/skills/cursor-croq-tune/tools/detect_gpu.sh)
STATE=$(bash .cursor/skills/cursor-croq-tune/tools/resume_state.sh \
    --gpu "$GPU" --dsl <dsl> --shape-key <shape_key> --model <model>)
echo "$STATE"
```

| Field | Meaning |
|---|---|
| `dsl` | DSL being tuned |
| `shape_key` | Shape key being tuned |
| `current_best_tflops` | Best custom kernel performance so far |
| `current_best_kernel` | Name of the best kernel file |
| `current_best_iter` | Iter name of the best kernel |
| `last_round` | Round number of the last stored result |
| `last_iter` | Last iter name (may be DISCARD) |
| `last_decision` | KEEP or DISCARD |
| `last_bottleneck` | Bottleneck from last round |
| `next_iter_number` | Number for the next `iter<NNN>` |
| `src_count` | Total iter source files present |
| `open_checkpoint` | Non-null if IDEA wrote a plan but VERIFY not completed |
| `memory_files_ok` | true if log files (results.tsv, idea-log.jsonl) present |
| `warnings` | Issues detected (malformed JSON, missing files, etc.) |

**After loading state:**
1. If `open_checkpoint` is non-null: resume from IMPLEMENT using that checkpoint (`checkpoint_write.sh read`).
2. If `warnings` is non-empty: address each warning before continuing.
3. Otherwise: pick up from `last_round + 1`.

### Validation Before Resume

```bash
GPU=$(bash .cursor/skills/cursor-croq-tune/tools/detect_gpu.sh)
python3 .cursor/skills/cursor-croq-tune/tools/validate_tuning_session.py --gpu "$GPU" --dsl <dsl>
```

If validation fails:
```bash
python3 .cursor/skills/cursor-croq-tune/tools/clean_kernel_work_state.py --gpu "$GPU" --dsl <dsl> --invalid-only
```

For fresh restart:
```bash
python3 .cursor/skills/cursor-croq-tune/tools/clean_kernel_work_state.py --gpu "$GPU" --dsl <dsl> --reset-all
```

---

## Per-Round Contract

**Round-loop preamble — FIRST ACTIONS on every round (before PROFILE):**

1. **TodoWrite durable anchor** — set the round-step todo to `in_progress` AND keep `continue-croq-tune` as `in_progress`. Both MUST be present before any other action:
   ```
   TodoWrite([
     { id: "continue-croq-tune", content: "Continue /croq-tune <dsl> <dtype> <shape_key>", status: "in_progress" },
     { id: "round-step",         content: "Round <N>: PROFILE → IDEA → IMPLEMENT → VERIFY → MEASURE → DECIDE → STORE", status: "in_progress" }
   ], merge=true)
   ```
2. **TodoWrite cleanup** — if todo count > 20, run cleanup first:
   ```bash
   echo '$CURRENT_TODOS_JSON' | bash ~/.cursor/skills/durable-request/todo-cleanup.sh
   ```
   Then `TodoWrite({ todos: <cleaned>, merge: false })`.
3. If context usage is `>= 80%`, trigger proactive compaction (the two todos above must already be written first).

### 1) PROFILE

**Every PROFILE step MUST run `ncu` on the current best kernel.** No exceptions. No timing-only shortcuts.

**Step 0 — GPU Contention Check (MANDATORY before profiling AND benchmarking):**

`ncu_profile.sh`, `cublas_baseline.sh`, and `store_round.sh` all run `gpu_contention.sh` automatically. For manual invocation or when running benchmark binaries directly:

```bash
# Scan — see who is on the GPU (dry run)
bash .cursor/skills/cursor-croq-tune/tools/gpu_contention.sh

# Kill foreign processes (spares our own ncu/choreo/iter* — no password needed)
bash .cursor/skills/cursor-croq-tune/tools/gpu_contention.sh --kill
```

`gpu_contention.sh` classifies each GPU process as **ours** (croq-tune harness: ncu, choreo, nvcc, compiled iter* binaries, cublas_baseline) or **foreign** (vllm, jupyter, other ML workloads). Foreign processes are killed; ours are spared. Use `--kill-ours` only as a last resort.

If you suspect the GPU is still busy after the kill:
1. Wait: `bash .cursor/skills/cursor-croq-tune/tools/gpu_check.sh --wait --timeout 120`
2. Reset (last resort): `bash .cursor/skills/cursor-croq-tune/tools/gpu_check.sh --reset`
3. If reset fails: STOP and report to user — GPU requires manual intervention

**NEVER trust TFLOPS numbers collected while the GPU is under contention.** A sudden extreme drop in performance (e.g. 50%+ below the previous best) is a strong signal of GPU contention. When this happens, re-run `gpu_contention.sh` and re-benchmark.

**Step 1 — Profile + Export:**
```bash
NCU_BASE="tuning/<gpu>/<dsl>/perf/<shape_key>/<model>"
TAG="<iter_tag>"

bash .cursor/skills/cursor-croq-tune/tools/ncu_profile.sh \
    --out  "${NCU_BASE}/ncu_${TAG}" \
    --cmd  <kernel_binary> [args]
```
Produces `ncu_<TAG>.ncu-rep` + `ncu_<TAG>.csv` atomically. Do NOT call ncu manually — use this wrapper.

**Step 2 — Classify Bottleneck:**
```bash
PROFILE_JSON=$(bash .cursor/skills/cursor-croq-tune/tools/profile_extract.sh \
    --csv  "${NCU_BASE}/ncu_${TAG}.csv" \
    --iter "${TAG}")
```
Output is one-line JSON with `bottleneck`, `confidence`, `evidence.key_metrics`, and `evidence.ncu_rep` (path to the full `.ncu-rep` report). Pass verbatim to IDEA. The script handles classification — do not override its output. If exit code != 0, re-run ncu or STOP.

If the output includes a `hint` field (confidence is medium), consider deeper analysis before forming IDEA:
- `ncu --import <ncu_rep> --page details` for warp stall breakdown, L1/L2 hit rates
- `sass_compare.sh dump-baseline` + `dump-custom` + `compare` for baseline vs custom SASS comparison (see "Reverse Engineering" in IDEA section)
- Load `perf-nsight-compute-analysis` skill for systematic ncu report analysis

**ncu Failure Handling:**
- Transient (GPU busy, timeout): run `gpu_check.sh --wait`, then retry once
- Permission errors: STOP, report to user, do NOT continue
- Persistent failure: STOP the tuning loop, log error, report to user

### 2) IDEA

- Generate one model-proposed idea from the current bottleneck and local history
- **MANDATORY web search** — always run at least one targeted web search before forming the final idea. Search for the identified bottleneck (e.g. "CUDA shared memory bank conflict matmul bf16"). Collect 1-3 external inspirations. Do NOT skip this.
- Merge model-proposed idea with web search findings into exactly one testable idea
- If an idea involves compiler changes (e.g. new Choreo primitive), implement the compiler change first, rebuild, then use it in the kernel. Treat both as ONE atomic change.

#### Reverse Engineering (optional, when stuck)

When standard bottleneck analysis is insufficient — typically after **3+ consecutive DISCARDs**
on the same active base, when `profile_extract` returns medium confidence, or when the
optimization problem is clearly hard (e.g. <50% of baseline TFLOPS after 5+ iterations) —
use SASS-level reverse engineering to understand what the baseline is doing differently:

1. **Extract baseline SASS** (once per shape, result is cached):
   ```bash
   bash .cursor/skills/cursor-croq-tune/tools/sass_compare.sh dump-baseline \
       --dsl <dsl> --shape-key <key> --model <model> \
       --dtype <dtype> --m <M> --n <N> --k <K>
   ```
   Primary: captures cuBLAS kernel SASS via `ncu --print-source sass`.
   Fallback: extracts from `libcublas.so` via `cuobjdump --dump-sass`.

2. **Extract custom kernel SASS**:
   ```bash
   bash .cursor/skills/cursor-croq-tune/tools/sass_compare.sh dump-custom \
       --dsl <dsl> --shape-key <key> --model <model> --iter <current_best_iter>
   ```

3. **Compare and analyze divergences**:
   ```bash
   bash .cursor/skills/cursor-croq-tune/tools/sass_compare.sh compare \
       --dsl <dsl> --shape-key <key> --model <model> --iter <current_best_iter>
   ```
   Outputs JSON with instruction mix, register usage, tensor core types, memory
   patterns, and actionable divergences ranked by severity (high/medium/low).

**What to look for in the comparison:**
- **Tensor core instructions**: HMMA shape/precision mismatches (e.g. baseline uses
  HMMA.16816.F32 but custom uses HMMA.1688.F16)
- **Register pressure**: custom using significantly more registers → lower occupancy
- **Memory access width**: baseline using LDG.128 (coalesced 128B) vs custom LDG.32
- **Barrier density**: excessive synchronization overhead vs baseline
- **Instruction count ratio**: custom >1.5x baseline → redundant computation

Form the IDEA targeting the **top 1-2 high-severity divergences** from the comparison.
This is additive context — the mandatory web search still applies.

- **Last action of IDEA** — write the checkpoint (MANDATORY):
  ```bash
  bash .cursor/skills/cursor-croq-tune/tools/checkpoint_write.sh write \
      --dsl <dsl> --shape-key <key> --model <model> \
      --iter <planned_iter> \
      --bottleneck <bottleneck> \
      --idea "<one-line description>" \
      --expected-gain "<+X TFLOPS estimate>" \
      --levers "<comma-separated parameter names>"
  ```

### 3) IMPLEMENT

- **First action** — read back the checkpoint:
  ```bash
  bash .cursor/skills/cursor-croq-tune/tools/checkpoint_write.sh read \
      --dsl <dsl> --shape-key <key> --model <model>
  ```
  Build exactly what was planned. Not something else.

- **Second action** — get canonical iteration name:
  ```bash
  ITER=$(bash .cursor/skills/cursor-croq-tune/tools/next_iter.sh \
      --dsl <dsl> --shape-key <key> --model <model> --tag <short_idea_tag>)
  ```
  For compile-fail retries, use `--attempt` flag.

- Apply only the current round's single idea
- If editing `.co`, load `choreo-syntax` before changing code
- Compile and run using the BUILD/RUN templates from `cursor-croq-dsl-<dsl>`
- If retries exhausted, mark as failed attempt (`COMPILE_FAIL`), STORE, and return to IDEA

#### Compile-fail debug: ANALYZE first, then classify

Every compile failure requires a structured debug cycle. Do NOT blindly retry:

1. **ANALYZE** — Gather evidence before classifying. Read the full compiler error output.
   Check web search and DSL docs/examples. Read generated intermediate files (`.cu`,
   `.cute.result`) if the error is unclear. The loaded `cursor-croq-dsl-<dsl>` skill has
   DSL-specific debug procedures and arch-specific workaround patterns.
2. **CLASSIFY** — Based on the evidence: code bug (your fault) or DSL/compiler limitation
   (the tool's fault). Do NOT classify without evidence from step 1.
3. **FIX** — Apply a targeted fix. For DSL limitations, find a supported alternative
   (e.g. different instruction for the target arch, `.cu` bypass). Each fix attempt must
   try a genuinely different approach.

**Retry budget: 5 attempts** for both code bugs and DSL limitations. After 5 failed
attempts: COMPILE_FAIL → STORE → return to IDEA.

### 4) VERIFY

- **First action** — verify implementation matches plan:
  ```bash
  bash .cursor/skills/cursor-croq-tune/tools/checkpoint_write.sh verify \
      --dsl <dsl> --shape-key <key> --model <model> --iter <actual_iter_built>
  ```
  Exit 3 = significant drift — investigate before continuing.
- Correctness must pass before performance measurement can count

### 5) MEASURE

- **GPU contention check before benchmarking** — `store_round.sh` runs `gpu_contention.sh` automatically before recording results. When running the benchmark binary directly, call `bash .cursor/skills/cursor-croq-tune/tools/gpu_contention.sh --kill` first if GPU is not idle.
- Run benchmark using DSL-specific defaults from `cursor-croq-dsl-<dsl>` (default: 10 warmup + 50 timed)
- Use CUDA event timing, never wall-clock time
- Minimum 3 timed iterations for stability
- Compute and record TFLOPS; include raw timing in STORE payload
- **Contention sanity check:** If measured TFLOPS is < 50% of the current best, suspect GPU contention. Re-run `gpu_contention.sh`, wait for idle, then re-measure before accepting the result.

### 6) DECIDE

- KEEP only if candidate beats current best
- Otherwise DISCARD

### 7) STORE

**MANDATORY — runs on every round outcome without exception (KEEP, DISCARD, compile-failed).**

Call the harness to write all 4 mandatory files atomically:
```bash
bash .cursor/skills/cursor-croq-tune/tools/store_round.sh \
  --dsl <dsl> --shape-key <shape_key> --model <model> \
  --iter iter<NNN> --kernel iter<NNN>_<tag> \
  --tflops <float> --decision <KEEP|DISCARD|SEGFAULT|HANG|COMPILE_FAIL> \
  --bottleneck <bottleneck> --idea "<summary>" --round <N> \
  --category "<tiling|pipeline|memory|compute|misc>" \
  --expected-gain "<e.g. +5% TFLOPS>"
```

The harness writes and verifies:
1. `logs/<key>/<model>/idea-log.jsonl`
2. `logs/<key>/<model>/results.tsv`

Do NOT proceed to CONTINUE until the harness prints `[store_round] STORE complete`.
**NEVER bypass the harness by writing files manually.**

For a public measured `iter<NNN>` result, also persist: source snapshot, build/run scripts, timing output, profile output.
For a failed `attempt<AAAA>`, also persist: attempted source, build script, build log/stderr.

**Post-STORE — mandatory sequence (in this exact order):**

1. **Mark current round done AND anchor next round** — single `TodoWrite` call:
   ```
   TodoWrite([
     { id: "round-step",         status: "completed" },
     { id: "continue-croq-tune", content: "Continue /croq-tune <dsl> <dtype> <shape_key>", status: "in_progress" },
     { id: "round-step",         content: "Round <N+1>: PROFILE → IDEA → IMPLEMENT → VERIFY → MEASURE → DECIDE → STORE", status: "in_progress" }
   ], merge=true)
   ```
   Writing the next `round-step` as `in_progress` **before** calling reinforce ensures context compaction cannot silently end the loop — the `in_progress` anchor is always present.

2. **Call reinforce** (MANDATORY):
   ```bash
   bash .cursor/skills/cursor-croq-tune/tools/reinforce.sh \
     --dsl <dsl> --shape-key <shape_key> --model <model>
   ```
   The script prints a one-line progress summary and **mandates an immediate full re-read of both skill files**. You MUST read `.cursor/skills/cursor-croq-tune/SKILL.md` and `.cursor/skills/cursor-cursor-croq-dsl-<dsl>/SKILL.md` before starting the next round. This is the continuation contract — reading the full protocol is the confirmation that the loop will proceed.

**Session Transcript as Memory:** The raw chat session JSONL is the primary memory — NOT summarized round logs. At session end or context compaction, copy the full session JSONL from `~/.cursor/projects/<slug>/agent-transcripts/<id>/` to `memory/<key>/<model>/sessions/`. Do NOT generate summarized `rounds.md` — the raw transcript IS the memory.

### 8) CONTINUE

Advance immediately to next round (or next shape).

---

## Artifact Naming and Path Layout

### Root Layout

All artifacts under: `tuning/<gpu>/<dsl>/`

`<gpu>` is emitted by `bash .cursor/skills/cursor-croq-tune/tools/detect_gpu.sh` (e.g. `sm90_H100`, `sm86_NVIDIA_GeForce_RTX_3070`).

### Model Level

```
tuning/<gpu>/<dsl>/<category>/<shape_key>/<model>/...
```

Model naming: lowercase alphanumeric + hyphens (`opus-4`, `sonnet-4`, `gpt-5`).
Model name is passed via `--model` flag to all harness scripts. `--model` is **required**.

### Shape Key Format

```
<operator>_<dtype>_<dimensions>
```

Dimension order is CANONICAL — NEVER permute. For matmul: MxNxK.
`matmul_bf16fp32_16384x16384x512` != `matmul_bf16fp32_512x16384x16384`.

### Per Shape Key + Model — Artifact Listing

All paths below are relative to `tuning/<gpu>/<dsl>/`:

- `srcs/<key>/<model>/iter<NNN>_<tag>.<co|cu|py>` — kernel source
- `srcs/<key>/<model>/iter<NNN>_<tag>.cu` — extracted + fine-tuned CUDA source (croqtile: mandatory)
- `srcs/<key>/<model>/attempt<AAAA>_<tag>.<co|cu|py>` — failed attempt source
- `bin/<key>/<model>/iter<NNN>_<tag>` — compiled binary (compiled-binary DSLs)
- `cmd/<key>/<model>/build_iter<NNN>.sh` — build script
- `cmd/<key>/<model>/run_iter<NNN>.sh` — run script
- `cmd/<key>/<model>/profile_iter<NNN>.sh` — profile script
- `cmd/<key>/<model>/iter<NNN>_<tag>.cute.result` — generated run script (croqtile)
- `perf/<key>/<model>/build_iter<NNN>.txt` — build output/log
- `perf/<key>/<model>/timing_iter<NNN>.txt` — timing output
- `perf/<key>/<model>/ncu_iter<NNN>_<tag>_round<R>.csv` — ncu CSV
- `perf/<key>/<model>/ncu_iter<NNN>_<tag>_round<R>.ncu-rep` — ncu report
- `perf/<key>/<model>/ncu_iter<NNN>_<tag>_round<R>.metrics.csv` — ncu metrics (optional)
- `logs/<key>/<model>/results.tsv` — TSV row per measured iter
- `logs/<key>/<model>/idea-log.jsonl` — JSON line per round
- `logs/<key>/<model>/attempt-log.jsonl` — JSON line per failed attempt
- `checkpoints/<key>/<model>/current_idea.json` — current IDEA checkpoint
- `memory/<key>/<model>/sessions/<session-id>.jsonl` — raw session transcript (primary memory)

### Iteration Naming

- `iter000` = cuBLAS/library reference baseline (from `cublas_baseline.sh`)
- `iter001` = first custom kernel (from `discover_baseline.sh` or implemented from scratch)
- Public measured: 3-digit `iter002`, `iter003`, ...
- Compile-failed: `attempt<AAAA>` (do not consume iter sequence)

**Tag is required:** `iter<NNN>_<tag>.<ext>` — bare `iter<NNN>.<ext>` without a tag is rejected by the harness.
Tag: 2-31 chars, lowercase alphanumeric + underscores, descriptive of the idea.

---

## Harness Scripts Reference

| Script | When to call | Required args |
|---|---|---|
| `detect_gpu.sh` | Start of session | (none) — result cached in `/tmp/croq_gpu_key` |
| `validate_env.sh` | PREPARATION_ONCE step 1 | `--dsl` |
| `discover_baseline.sh` | PREPARATION_ONCE step 3 | `--dsl --operator [--dtype] [--gpu]` |
| `resume_state.sh` | Resume decision | `--gpu --dsl --shape-key --model` |
| `checkpoint_write.sh` | End of IDEA, start of IMPLEMENT, start of VERIFY | `--dsl --shape-key --model` + action-specific |
| `next_iter.sh` | Start of IMPLEMENT | `--dsl --shape-key --model --tag` |
| `store_round.sh` | STORE step | `--dsl --shape-key --model --iter --kernel --tflops --decision --bottleneck --idea --round` |
| `reinforce.sh` | After STORE (MANDATORY) | `--dsl --shape-key --model` |
| `co2cu.sh` | IMPLEMENT Phase 1 (croqtile only) | `--co --arch [--flags]` |
| `ncu_profile.sh` | PROFILE step | `--out --cmd` |
| `profile_extract.sh` | After ncu CSV | `--csv --iter` |
| `gpu_check.sh` | Before PROFILE and MEASURE (lightweight poll) | (none), or `--wait`, `--kill-others`, `--reset` |
| `gpu_contention.sh` | Before PROFILE/MEASURE — auto-called by ncu_profile, cublas_baseline, store_round | (none) scan; `--kill` kill foreign; `--kill-ours` last resort; `--json` agent-friendly; `--except <pid>` spare one PID |
| `store_baseline.sh` | PREPARATION_ONCE step 2b | `--dsl --shape-key --model --dtype --m --n --k --task-uid [--warmup --iters]` |
| `cublas_baseline.sh` | Called by store_baseline.sh | `--dtype --m --n --k [--warmup --iters]` |
| `prepare_baseline_env.py` | PREPARATION_ONCE step 2a | `--dsl --operator --kernel --shape-key --libs` |

All scripts under `.cursor/skills/cursor-croq-tune/tools/`. `--gpu` is optional for store/checkpoint/next_iter (auto-detected); **required** for `resume_state.sh`.

---

## Continuation Anchor

Keep one persistent continuation node: `Continue /croq-tune <dsl> <dtype> [shape_key]`

Ids: `continue-croq-tune` (`in_progress`), `round-step` (`completed` after STORE).

For framework-specific payload templates (Cursor TodoWrite, OpenCode, Copilot), see `todo-payloads.md` in this skill directory.

---

## How to Treat Unexpected Events

**Timeout:** Each experiment ~5 minutes. If >10 minutes, kill and treat as failure.

**Crashes:** If easy to fix (typo, missing import), fix and re-run. If fundamentally broken, skip, log "crash", move on.

---

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human. Do NOT ask "should I keep going?". The human might be asleep. You are autonomous. If you run out of ideas, think harder — read papers, re-read kernel code, try combining near-misses, try radical changes. The loop runs until interrupted.

Unless the user explicitly requested a smaller run, a single launch should normally complete at least `50` STORE-finished rounds before stopping for success/completion.

## Valid Stop Conditions Only

1. User explicitly interrupts
2. All scheduled shapes are done
3. Systemic GPU failure blocks progress after remediation attempts
4. 10 consecutive discarded ideas after the invocation has already completed at least `target_completed_rounds`, or after the mandatory 7/8/9 DISCARD escalation path has already been exhausted without opening a new viable direction (stuck in local minimum — report to user)

---

## Branch Strategy

**Tuning directly on `main` is allowed.** No branch ceremony required. **Do NOT push** — commit locally only. Parallel tuning sessions on the same repo would conflict on push.

```bash
git add -A
git commit -m "tune(<dsl>): <op> <dtype> <shape> - iter<NNN> <X> TFLOPS"
```

- Each `iter<NNN>` that passes VERIFY: commit immediately (local only)
- Failed `attempt<AAAA>` sources: batch commit with next passing iter
- One commit per measured iter minimum
- On context compaction: commit all pending work, note current state in commit message
