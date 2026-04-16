---
name: croq-tune
description: Launch an infinite AI-driven kernel optimization loop for GPU kernels. Use when the user asks to "tune", "ai-tune", "optimize", "auto-tune", or "perf-tune" a kernel. Profiles with ncu, iterates optimizations indefinitely until interrupted.
argument-hint: <dsl: croqtile|cuda|cute|triton|tilelang|helion|cutile> <dtype: f16|e4m3|all> [shape_key]
---

# Croq-Tune — Unified Kernel Tuning Loop

`croq-tune` is the **only** tuning entrypoint. Load this file and one `croq-dsl-<dsl>` file per session — nothing else.

## Main Loop

1. Parse `dsl`, `dtype`, optional `shape_key`
2. Startup/resume decision (see "Resume Contract" below)
3. Load `croq-dsl-<dsl>` for DSL-specific commands (BUILD, RUN, PROFILE, IDEA menu)
4. Run `PREPARATION_ONCE` for this `(dsl, operator, dtype, shape_key)`
5. Repeat round loop until a valid stop condition is met

---

## Hard Constraints

1. Keep exactly one continuation node in progress: `continue-croq-tune`
2. Trigger proactive compaction at `>= 80%` context, but only after continuation node is present
3. Compile-fail debug/fix retries bounded to 4-7 (default: 6); then discard attempt and return to IDEA
4. Exactly one new idea per round
5. STORE executes on both KEEP and DISCARD
6. **NO LIBRARY CALLS IN TUNING ITERATIONS** — see "Pure Implementation Rule"

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
See the loaded `croq-dsl-<dsl>` file for DSL-specific structural changes.

### Rule 4: NEVER repeat a failed combination

Before every iteration, check `idea-log.jsonl`. If the exact combination (base kernel
x parameter set x structural change) was tried before, choose a different idea.

### Rule 5: ABANDON stuck ideas after 3 attempts

If an optimization idea fails to compile or pass verification after 3 distinct fix
attempts, ABANDON it. Revert to the active base's best and propose a completely different idea.

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

Compile and run using the workflow described in `croq-dsl-<dsl>`. Do NOT delegate to
external wrapper scripts as black boxes. You need to see compiler output directly.

### Rule 9: TRACK iteration counter monotonically

Each iteration gets a unique, monotonically increasing number. Do not reuse numbers.
Do not skip large ranges. Use `next_iter.sh` to get the canonical name.

---

## Pure Implementation Rule

All tuning iterations (`iter001` and beyond) MUST use **pure kernel code**.
Per-DSL allowed/forbidden lists are in the loaded `croq-dsl-<dsl>` file.

**Allowed in all DSLs:** Raw kernel code, CUDA intrinsics, PTX inline asm, cooperative groups, warp-level primitives, DSL-specific primitives.

**FORBIDDEN in all DSLs:** cuBLAS, cuTLASS library GEMM, cuSPARSELt, cuDNN, PyTorch/TensorFlow ops, any code that delegates core compute to a library.

**Where library calls ARE allowed:** iter000 baseline only; verification reference computation. Never in iter001+.

**Enforcement:** If IMPLEMENT produces a library call for core compute, reject immediately, log as `attempt<AAAA>` with reason `library_call_forbidden`, return to IDEA.

---

## CRITICAL ANTI-PATTERNS (FORBIDDEN)

- **Fabricating iteration data** — Every `iter<NNN>` in `rounds.raw.jsonl` MUST correspond to a real kernel run with real TFLOPS measurement
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
bash .claude/skills/croq-tune/tools/validate_env.sh --dsl <dsl>
```

The harness validates: nvidia-smi, nvcc, ncu, perf_event_paranoid, GPU availability,
plus DSL-specific checks (e.g. `CHOREO_HOME`/`CUTE_HOME`/`CUDA_HOME` for croqtile,
`import triton` for triton, etc.). Output is one-line JSON.

If the harness exits non-zero: **STOP and escalate to user.** Show the `errors` array
from the JSON output. Tuning CANNOT start if validation fails. No fallbacks.

### 2. Baseline Environment Setup

Detect environment and baseline library readiness:
```bash
python3 .claude/skills/croq-tune/tools/prepare_baseline_env.py \
  --dsl <dsl> --operator <op> --kernel <kernel> --shape-key <shape_key> --libs auto
```
Check baseline readout in `baseline-workspace/<dsl>/<operator>/<dtype>/`. If readout exists, reuse it. If missing, run baseline and persist before first round.

### 3. Starting Kernel Discovery (MANDATORY)

Before entering the round loop, find or create a starting kernel. Run the discovery harness:

```bash
DISCOVERY=$(bash .claude/skills/croq-tune/tools/discover_baseline.sh \
    --dsl <dsl> --operator <operator> --dtype <dtype> --gpu "$GPU")
echo "$DISCOVERY"
```

The harness scans 2 tiers and returns one of 3 recommendations:

**Tier A — Reference examples** (`choreo-kernel-examples/`):
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

**After discovery, create the iter001 artifacts:**
1. Source: `tuning/<gpu>/<dsl>/srcs/<shape_key>/<model>/iter001_draft.<cu|co|py>`
2. Build script: `tuning/<gpu>/<dsl>/cmd/<shape_key>/<model>/build_iter001.sh`
3. Run script: `tuning/<gpu>/<dsl>/cmd/<shape_key>/<model>/run_iter001.sh`

Use the BUILD/RUN templates from the loaded `croq-dsl-<dsl>` file.

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

**DO NOT manually read rounds.raw.jsonl, results.tsv, etc. to reconstruct state.**

```bash
GPU=$(bash .claude/skills/croq-tune/tools/detect_gpu.sh)
STATE=$(bash .claude/skills/croq-tune/tools/resume_state.sh \
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
| `memory_files_ok` | true if all 4 memory/log files present |
| `warnings` | Issues detected (malformed JSON, missing files, etc.) |

**After loading state:**
1. If `open_checkpoint` is non-null: resume from IMPLEMENT using that checkpoint (`checkpoint_write.sh read`).
2. If `warnings` is non-empty: address each warning before continuing.
3. Otherwise: pick up from `last_round + 1`.

### Validation Before Resume

```bash
GPU=$(bash .claude/skills/croq-tune/tools/detect_gpu.sh)
python3 .claude/skills/croq-tune/tools/validate_tuning_session.py --gpu "$GPU" --dsl <dsl>
```

If validation fails:
```bash
python3 .claude/skills/croq-tune/tools/clean_kernel_work_state.py --gpu "$GPU" --dsl <dsl> --invalid-only
```

For fresh restart:
```bash
python3 .claude/skills/croq-tune/tools/clean_kernel_work_state.py --gpu "$GPU" --dsl <dsl> --reset-all
```

---

## Per-Round Contract

Round-loop preamble on every round:
1. Ensure there is an in-progress todo: `Continue /croq-tune <dsl> <dtype> [shape_key]`
2. If context usage is `>= 80%`, trigger proactive compaction (add continuation todo first, then compact)

### 1) PROFILE

**Every PROFILE step MUST run `ncu` on the current best kernel.** No exceptions. No timing-only shortcuts.

**Step 1 — Profile + Export:**
```bash
NCU_BASE="tuning/<gpu>/<dsl>/perf/<shape_key>/<model>"
TAG="<iter_tag>"

bash .claude/skills/croq-tune/tools/ncu_profile.sh \
    --out  "${NCU_BASE}/ncu_${TAG}" \
    --cmd  <kernel_binary> [args]
```
Produces `ncu_<TAG>.ncu-rep` + `ncu_<TAG>.csv` atomically. Do NOT call ncu manually — use this wrapper.

**Step 2 — Classify Bottleneck:**
```bash
PROFILE_JSON=$(bash .claude/skills/croq-tune/tools/profile_extract.sh \
    --csv  "${NCU_BASE}/ncu_${TAG}.csv" \
    --iter "${TAG}")
```
Output is one-line JSON with `bottleneck`, `confidence`, `evidence.key_metrics`, and `evidence.ncu_rep` (path to the full `.ncu-rep` report). Pass verbatim to IDEA. The script handles classification — do not override its output. If exit code != 0, re-run ncu or STOP.

If the output includes a `hint` field (confidence is medium), consider deeper analysis before forming IDEA:
- `ncu --import <ncu_rep> --page details` for warp stall breakdown, L1/L2 hit rates
- `cuobjdump --dump-sass <binary>` for instruction-level hotspots
- Load `perf-nsight-compute-analysis` skill for systematic ncu report analysis

**ncu Failure Handling:**
- Transient (GPU busy, timeout): wait 10s, retry once
- Permission errors: STOP, report to user, do NOT continue
- Persistent failure: STOP the tuning loop, log error, report to user

### 2) IDEA

- Generate one model-proposed idea from the current bottleneck and local history
- **MANDATORY web search** — always run at least one targeted web search before forming the final idea. Search for the identified bottleneck (e.g. "CUDA shared memory bank conflict matmul bf16"). Collect 1-3 external inspirations. Do NOT skip this.
- Merge model-proposed idea with web search findings into exactly one testable idea
- If an idea involves compiler changes (e.g. new Choreo primitive), implement the compiler change first, rebuild, then use it in the kernel. Treat both as ONE atomic change.
- **Last action of IDEA** — write the checkpoint (MANDATORY):
  ```bash
  bash .claude/skills/croq-tune/tools/checkpoint_write.sh write \
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
  bash .claude/skills/croq-tune/tools/checkpoint_write.sh read \
      --dsl <dsl> --shape-key <key> --model <model>
  ```
  Build exactly what was planned. Not something else.

- **Second action** — get canonical iteration name:
  ```bash
  ITER=$(bash .claude/skills/croq-tune/tools/next_iter.sh \
      --dsl <dsl> --shape-key <key> --model <model> --tag <short_idea_tag>)
  ```
  For compile-fail retries, use `--attempt` flag.

- Apply only the current round's single idea
- If editing `.co`, load `choreo-syntax` before changing code
- Compile and run using the BUILD/RUN templates from `croq-dsl-<dsl>`
- If compile fails, debug/fix with bounded retry budget (target: 6, range: 4-7)
- If retries exhausted, mark as failed attempt and return to IDEA

### 4) VERIFY

- **First action** — verify implementation matches plan:
  ```bash
  bash .claude/skills/croq-tune/tools/checkpoint_write.sh verify \
      --dsl <dsl> --shape-key <key> --model <model> --iter <actual_iter_built>
  ```
  Exit 3 = significant drift — investigate before continuing.
- Correctness must pass before performance measurement can count

### 5) MEASURE

- Run benchmark using DSL-specific defaults from `croq-dsl-<dsl>` (default: 10 warmup + 50 timed)
- Use CUDA event timing, never wall-clock time
- Minimum 3 timed iterations for stability
- Compute and record TFLOPS; include raw timing in STORE payload

### 6) DECIDE

- KEEP only if candidate beats current best
- Otherwise DISCARD

### 7) STORE

**MANDATORY — runs on every round outcome without exception (KEEP, DISCARD, compile-failed).**

Call the harness to write all 4 mandatory files atomically:
```bash
bash .claude/skills/croq-tune/tools/store_round.sh \
  --dsl <dsl> --shape-key <shape_key> --model <model> \
  --iter iter<NNN> --kernel iter<NNN>_<tag> \
  --tflops <float> --decision <KEEP|DISCARD|SEGFAULT|HANG|COMPILE_FAIL> \
  --bottleneck <bottleneck> --idea "<summary>" --round <N> \
  --category "<tiling|pipeline|memory|compute|misc>" \
  --expected-gain "<e.g. +5% TFLOPS>"
```

The harness writes and verifies:
1. `memory/<key>/<model>/rounds.raw.jsonl`
2. `memory/<key>/<model>/rounds.md`
3. `logs/<key>/<model>/idea-log.jsonl`
4. `logs/<key>/<model>/results.tsv`

Do NOT proceed to CONTINUE until the harness prints `[store_round] STORE complete`.
**NEVER bypass the harness by writing files manually.**

For a public measured `iter<NNN>` result, also persist: source snapshot, build/run scripts, timing output, profile output.
For a failed `attempt<AAAA>`, also persist: attempted source, build script, build log/stderr.

**Post-STORE:** mark `round-step` as `completed`, ensure `continue-croq-tune` is `in_progress`.

**Session Transcript:** At session end or context compaction, copy the session JSONL from `~/.cursor/projects/<slug>/agent-transcripts/<id>/` to `memory/<key>/<model>/sessions/`.

### 8) CONTINUE

Advance immediately to next round (or next shape).

---

## Artifact Naming and Path Layout

### Root Layout

All artifacts under: `tuning/<gpu>/<dsl>/`

`<gpu>` is emitted by `bash .claude/skills/croq-tune/tools/detect_gpu.sh` (e.g. `sm90_H100`, `sm86_NVIDIA_GeForce_RTX_3070`).

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
- `memory/<key>/<model>/rounds.raw.jsonl` — JSON line per round
- `memory/<key>/<model>/rounds.md` — markdown section per round
- `memory/<key>/<model>/sessions/<session-id>.jsonl` — raw session transcript

### Iteration Naming

- `iter000` = trivial measured baseline
- Public measured: 3-digit `iter001`, `iter002`, ...
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
| `co2cu.sh` | IMPLEMENT Phase 1 (croqtile only) | `--co --arch [--flags]` |
| `ncu_profile.sh` | PROFILE step | `--out --cmd` |
| `profile_extract.sh` | After ncu CSV | `--csv --iter` |
| `prepare_baseline_env.py` | PREPARATION_ONCE step 2 | `--dsl --operator --kernel --shape-key --libs` |

All scripts under `.claude/skills/croq-tune/tools/`. `--gpu` is optional for store/checkpoint/next_iter (auto-detected); **required** for `resume_state.sh`.

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

## Valid Stop Conditions Only

1. User explicitly interrupts
2. All scheduled shapes are done
3. Systemic GPU failure blocks progress after remediation attempts
4. 10 consecutive discarded ideas (stuck in local minimum — report to user)

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
