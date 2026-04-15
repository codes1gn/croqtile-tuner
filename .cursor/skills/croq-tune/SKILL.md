---
name: croq-tune
description: Canonical tuning entry skill with a concrete round contract and TodoWrite-driven continuation guard.
argument-hint: <dsl: croqtile|cuda|cute|triton|tilelang|helion|cutile> <dtype: f16|e4m3|all> [shape_key]
---

# Croq-Tune

`croq-tune` is the only tuning entrypoint.
Do not use `croq-tuner` or `ai-tune-*`.

## Main Loop (Authoritative)

1. Parse `dsl`, `dtype`, optional `shape_key`
2. Startup/resume decision (load `croq-resume`)
3. Run `PREPARATION_ONCE` for this `(dsl, operator, dtype, shape_key)` (load `croq-baseline`)
4. Repeat round loop until a valid stop condition is met

## Hard Constraints

1. Keep exactly one continuation node in progress: `continue-croq-tune`
2. Trigger proactive compaction at `>= 80%` context, but only after continuation node is present
3. Compile-fail debug/fix retries are bounded to 4-7 (default target: 6); then discard attempt and return to `IDEA`
4. Exactly one new idea per round
5. `STORE` executes on both KEEP and DISCARD
6. **NO LIBRARY CALLS IN TUNING ITERATIONS** — see "Pure Implementation Rule" below

## Pure Implementation Rule (INVIOLABLE)

All tuning iterations (`iter001` and beyond) MUST use **pure kernel code**:

### Allowed in Tuning Iterations

- Raw CUDA C++ kernel code
- CUDA intrinsics (`__shfl_sync`, `__ldg`, etc.)
- PTX inline assembly (`asm volatile`)
- Cooperative groups, warp-level primitives
- CuTe/CUTLASS **primitives only** (MMA atoms, copy atoms, layouts) — not library GEMM calls
- Choreo DSL primitives for `dsl=croqtile`

### FORBIDDEN in Tuning Iterations

- cuBLAS calls (`cublasSgemm`, `cublasGemmEx`, etc.)
- cuTLASS library GEMM calls (`cutlass::gemm::device::Gemm`)
- cuSPARSELt, cuDNN, or any other library compute calls
- PyTorch/TensorFlow ops
- Any code that delegates the core compute to an external library

### Where Library Calls ARE Allowed

- **iter000 baseline only**: Framework/library calls are permitted for baseline measurement
- **Verification reference**: Library calls can be used to generate ground truth for correctness checking
- **Never in iter001+**: The tuning loop optimizes YOUR code, not library code

### Enforcement

If an IMPLEMENT step produces code that calls a library for the core compute:
1. Reject the implementation immediately
2. Log as `attempt<AAAA>` with reason `library_call_forbidden`
3. Return to IDEA with explicit guidance to use pure kernel code

This rule exists because **the goal is to tune custom kernel implementations**, not to benchmark existing libraries.

Round-loop preamble on every round:

1. Ensure there is an in-progress todo: `Continue /croq-tune <dsl> <dtype> [shape_key]`
2. If context usage is `>= 80%`, trigger proactive compaction:
   - if the continuation todo is missing, add it first
   - then trigger compaction
   - after resume, re-issue `/croq-tune ...` from that todo

## Continuation Anchor (Framework-Specific)

Keep one persistent continuation node:

- `Continue /croq-tune <dsl> <dtype> [shape_key]`

Keep at most one round-step node in progress (`round-step`).

### Cursor IDE (with `TodoWrite`)

Use `TodoWrite` with ids:

- `continue-croq-tune` (`in_progress`)
- `round-step` (`completed` after STORE)

### OpenCode

Use same ids and refresh semantics.
If no todo tool is available, persist equivalent state in `.agent/todo.json`.

### Copilot VSCode IDE

Use file-backed state in `.agent/todo.json` with the same ids/statuses.

Todo state controls workflow continuity only; it does not replace persisted tuning artifacts.

Exact payload templates are defined in:

- `.claude/skills/croq-tune/todo-payloads.md`

## Per-Round Contract (Detailed)

### 1) PROFILE

- Load `croq-profile`
- Gather the lightest evidence that answers current bottleneck
- Produce explicit `bottleneck` and `confidence`

### 2) IDEA

- Generate one model-proposed idea from the current bottleneck and local history
- Run targeted web search for the same bottleneck to collect 1-3 external inspirations
- Merge into exactly one testable idea with expected gain and risk

### 3) IMPLEMENT

- Apply only the current round's single idea
- If editing `.co`, load `choreo-syntax` before changing code
- Compile and run verification
- If compile fails, debug/fix and retry with bounded budget:
  - target budget: 6 retries (allowed range: 4-7 by error severity)
- If retries exhausted, mark as failed attempt and return to `IDEA`

### 4) VERIFY

- Correctness must pass before performance measurement can count

### 5) MEASURE

- Run benchmark, collect stable timing samples, compute TFLOPS

### 6) DECIDE

- KEEP only if candidate beats current best
- Otherwise DISCARD

### 7) STORE

- Load `croq-store` (and `croq-artifacts` through it)
- Persist round outcome for both KEEP and DISCARD
- Compile-failed attempts use `attempt<AAAA>` and do not consume public `iter<NNN>`

### 8) CONTINUE

- Advance immediately to next round (or next shape)

## PREPARATION_ONCE (Per Shape, Outside Round Count)

Preparation is one-time and outside round counting.
Load `croq-baseline`, run its env-prep CLI, then follow its baseline workspace contract.

## Step-to-Skill References

- Startup/resume: `croq-resume`
- Preparation once: `croq-baseline`
- Profile step: `croq-profile`
- Store step: `croq-store` -> `croq-artifacts`
- `.co` implementation path only: `choreo-syntax`

## How to Treat Unexpected Events

Timeout: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

Crashes: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

## MOST IMPORTANT!!!!

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Valid Stop Conditions Only

1. User explicitly interrupts
2. All scheduled shapes are done
3. Systemic GPU failure blocks progress after remediation attempts

---

## Environment Prerequisites (Validated at Baseline)

The `croq-baseline` skill validates these BEFORE tuning starts:

1. **ncu profiling** — `perf_event_paranoid <= 2`
2. **CUDA compiler** — nvcc available and working
3. **GPU availability** — nvidia-smi reports GPU

**If baseline validation passes, the tuning loop assumes all tools work.**

If any tool fails unexpectedly during tuning:
- STOP immediately
- This indicates baseline was bypassed or environment changed
- Escalate to user

**CRITICAL ANTI-PATTERNS (FORBIDDEN):**

- **Fabricating iteration data** — Every `iter<NNN>` in `rounds.raw.jsonl` MUST correspond to a real kernel run with real TFLOPS measurement
- **Batch-generating fake results** — NEVER create multiple iteration records with identical timestamps or synthetic values
- **Skipping profiling** — If ncu fails, STOP; do NOT silently skip or guess
- **Proceeding without evidence** — Every IDEA must be based on real ncu profiling data
- **Bypassing baseline validation** — NEVER start tuning without environment validation passing

---

## Branch Strategy

This workspace is a container for all tuning results and visualization.
**Tuning directly on `main` is allowed.** No branch ceremony required.

### Workflow

Commit tuning progress directly to `main`:

```bash
git add -A
git commit -m "tune(<dsl>): <op> <dtype> <shape> - iter<NNN> <X> TFLOPS"
git push origin main
```

### Commit Message Format

```
tune(<dsl>): <op> <dtype> <shape> - iter<NNN> <X> TFLOPS
```

Examples:
- `tune(cuda): matmul bf16fp32 512x16384x16384 - iter044 35.20 TFLOPS (best)`
- `tune(triton): attention f16 2048x64 - iter012 182.3 TFLOPS`

### When to Commit

- Each `iter<NNN>` that passes VERIFY: commit immediately
- Failed `attempt<AAAA>` sources: batch commit with the next passing iter
- After each STORE step, a commit should exist for the round

### Rules

1. **Push to main** — no long-lived feature branches
2. **Commit after each measured iter** — ensures no work is lost
3. **One commit per measured iter minimum** — do not batch multiple measured iters