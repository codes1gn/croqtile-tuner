# CroqTuner — Step Checklists (Per-Step Entry/Exit Conditions)

Each FSM state has mandatory entry conditions, actions, exit conditions, and failure handlers. The validation scripts (`pre-step-check.sh`, `post-step-check.sh`) enforce these mechanically.

Referenced protocol files are not auto-loaded. When a state checklist says to use another
protocol or artifact rule, explicitly read that file first.

Non-stop backup keyphrase: `NON_STOP_CONTINUE_WITHOUT_WAIT`

---

## INIT

### Entry Conditions
- Shape key is determined (from sweep schedule or resume)
- No active per-DSL state file for a different shape (finish it first)

### Mandatory Actions
1. Set `fsm.shape_key`, `fsm.dtype`, `fsm.shape`, `fsm.max_iteration`
2. Create DSL-isolated directories: `tuning/aitune/<dsl>/{logs,srcs,perf,cmd,memory}/<KEY>/`
3. Create the trivial `iter000` reference implementation in the target DSL:
   - simplest correct scalar computations and loops
   - no speculative optimizations
   - if `dsl=croqtile`, use pure `.co` first
4. Write baseline scaffolding:
   - `cmd/<KEY>/build_iter000.sh`
   - `cmd/<KEY>/run_iter000.sh`
   - `cmd/<KEY>/profile_iter000.sh`
5. Set up verification and environment capture artifacts:
   - verification utility or command for iter000
   - environment snapshot file for execution reproducibility
6. Initialize `results.tsv` with header
7. Initialize `idea-log.jsonl` and `attempt-log.jsonl`
8. The active per-DSL state file is created by `state-transition.sh INIT` with `current_state: "INIT"`, `iteration: 0`

### Exit → BASELINE
- Directories exist, `iter000` source exists, timing/verify/env scaffolding exists, results.tsv initialized

### On Failure
- If directories already exist (resume case): skip creation, read existing state instead

---

## BASELINE

### Entry Conditions
- `fsm.current_state == "BASELINE"`
- `iter000` reference kernel exists in `tuning/aitune/<dsl>/srcs/<KEY>/`

### Mandatory Actions
1. Run GPU health check: `nvidia-smi --query-gpu=...`
2. Set `guard_flags.gpu_health_checked = true`
3. Build the trivial `iter000` kernel using `cmd/<KEY>/build_iter000.sh`
4. Run verification for `iter000`
5. Run timing benchmark for `iter000` (for example `CHOREO_TIMING_WARMUP=10 CHOREO_TIMING_REPEAT=500`)
6. Capture output to `tuning/aitune/<dsl>/perf/<KEY>/timing_iter000_baseline.txt`
7. Record baseline TFLOPS
8. Capture execution environment snapshot if not already written
7. Set `metrics.baseline_tflops`, `metrics.current_best_tflops`, `metrics.current_best_iter = 0`
8. Append iter 0 row to results.tsv
9. Set `fsm.iteration = 0`

### Exit → PROFILE
- `guard_flags.gpu_health_checked == true`
- `metrics.baseline_tflops != null`
- Timing file exists on disk

### On Failure
- Compile error: check NVCC flags, include paths. Report error details.
- GPU unhealthy: run GPU health remediation protocol, retry once.
- Suspiciously low TFLOPS (<10% of expected): re-check GPU, re-run once.

---

## PROFILE

### Entry Conditions
- `fsm.current_state == "PROFILE"`
- Previous iteration fully committed (or this is the first iteration)

### Mandatory Actions
1. Decide the lightest profiling signal that still answers the current question:
   - compiler diagnostics
   - runtime timing deltas
   - targeted metric collection
   - full `ncu`
2. A full `ncu` pass is mandatory when any of these are true:
   - baseline characterization / first real iteration
   - the previous round found a new best
   - `consecutive_discards >= 3`
   - the search is out of grounded ideas
   - optimization category is changing due to stall
   - PTX/SASS/inline-ASM work is being considered
3. Save profiler output to `tuning/aitune/<dsl>/perf/<KEY>/ncu_iter<NNN>.txt` or another clearly named profile artifact
4. Parse bottleneck category from the best available evidence. Categories:
   - `smem_throughput`
   - `l2_throughput`
   - `dram_throughput`
   - `compute_bound`
   - `latency_bound`
   - `occupancy_limited`
   - `instruction_fetch`
5. Set `guard_flags.ncu_ran_this_iter = true`
6. Set `guard_flags.bottleneck_identified = true`
7. Set `metrics.last_bottleneck = <category>`

### Exit → IDEATE
- `guard_flags.ncu_ran_this_iter == true`
- `guard_flags.bottleneck_identified == true`
- `metrics.last_bottleneck != null`

### On Failure
- ncu hangs (>5 min): kill ncu process, set `last_bottleneck = "ncu_timeout"`, proceed
- ncu crash: retry once, if still fails set `last_bottleneck = "ncu_error"`, proceed
- GPU unhealthy during ncu: remediate, retry

---

## IDEATE

### Entry Conditions
- `fsm.current_state == "IDEATE"`
- `guard_flags.ncu_ran_this_iter == true` (profiling done or waived)
- `guard_flags.bottleneck_identified == true`

### Mandatory Actions
1. Read `idea-log.jsonl` for this shape
2. Count idea categories used so far
3. Check diversity requirement (see `idea-diversity-rules.md`)
4. Collect supporting evidence for the next idea. This may combine:
   - ncu bottleneck data (preferred)
   - Compiler output / register usage
   - TFLOPS trends from results.tsv
   - Known shape-sensitivity patterns from reference kernel
   - Current external references such as NVIDIA CUDA/Nsight/CUTLASS documentation, papers, or closely related public examples when local evidence is insufficient or stale
5. Propose ONE optimization idea grounded in that evidence
6. Verify idea is novel (not in idea-log.jsonl)
7. Set `guard_flags.idea_is_novel = true`
8. Log idea to `idea-log.jsonl`:
   ```jsonl
   {"iter": N, "idea": "<description>", "category": "<macro|structural|choreo|ncu_micro>", "bottleneck_before": "<category>", "justification": "<why this idea>", "research_basis": ["<source or note>"]}
   ```
9. Set `guard_flags.idea_logged = true`

### Exit → IMPLEMENT
- `guard_flags.idea_is_novel == true`
- `guard_flags.idea_logged == true`

### On Failure
- No novel idea found: widen category search. If truly stuck after 3 attempts, transition back to PROFILE, refresh the data, consult new external references, and only then return to IDEATE.

---

## IMPLEMENT

### Entry Conditions
- `fsm.current_state == "IMPLEMENT"`
- `guard_flags.idea_logged == true`

### Mandatory Actions
1. Read current best kernel source
2. Enforce DSL purity: only tune the target DSL for this session; do not edit kernels of other DSLs.
3. Start a new internal `attempt<AAAA>` record before any edit that may fail.
4. Save attempt artifacts immediately as they are created:
   - attempted source snapshot
   - build shell script
   - build stderr/log capture
   - failure note if relevant
5. Implement the optimization idea using DSL-specific priority:
   - **If `dsl=croqtile`**: first try pure `.co` implementation.
   - If needed, use `__cpp__` blocks inside `.co` to inline CUDA/CuTe/PTX fragments.
   - Only if idea is truly unsupported by CroqTile syntax (for example, fine-grained sync/wait mutation) then edit generated `.cu`.
   - **If `dsl!=croqtile`**: do not use CroqTile `.co` as implementation language; stay purely in target DSL.
6. Compile for target shape
7. If compile fails: fix and retry up to 3 times without consuming a public iteration id
8. Set `guard_flags.compile_succeeded = true`
9. Run correctness verification
10. Set `guard_flags.correctness_verified = true`
11. Only after compile and correctness both pass, promote the candidate to a public source file: `tuning/aitune/<dsl>/srcs/<KEY>/iter<NNN>_<tag>.<co|cu>` and generate its build/run/profile shell scripts under `cmd/<KEY>/`

### Exit → MEASURE
- `guard_flags.compile_succeeded == true`
- `guard_flags.correctness_verified == true`
- New kernel file exists on disk

### On Failure
- Compile error (3 retries failed): log as DISCARD with reason `compile_failure`, keep it as `attempt<AAAA>`, skip MEASURE/DECIDE, and go directly to STORE without consuming a public iteration id
- Correctness failure: log as DISCARD with reason `incorrect`, keep it as `attempt<AAAA>`, skip MEASURE/DECIDE, and go to STORE without consuming a public iteration id

---

## MEASURE

### Entry Conditions
- `fsm.current_state == "MEASURE"`
- `guard_flags.compile_succeeded == true`

### Mandatory Actions
1. Run GPU health check (abbreviated: just check utilization + memory)
2. Run timing benchmark: `CHOREO_TIMING_WARMUP=10 CHOREO_TIMING_REPEAT=500`
3. Capture output to `tuning/aitune/<dsl>/perf/<KEY>/timing_iter<NNN>.txt`
4. Parse TFLOPS from output
5. Set `metrics.this_iter_tflops = <value>`
6. Set `guard_flags.timing_captured = true`
7. Apply sanity checks:
   - If TFLOPS > 1.5 × current_best: suspicious, re-run both kernels
   - If TFLOPS < 0.5 × current_best: suspect GPU contention, check health, re-run
   - Use re-run numbers if original was suspicious

### Exit → DECIDE
- `guard_flags.timing_captured == true`
- `metrics.this_iter_tflops != null`
- Timing file exists on disk

### On Failure
- Kernel crashes at runtime: log as DISCARD with "runtime_crash", go to STORE
- GPU unhealthy: remediate, re-run
- Timeout (>2 min for 500 repeats): kill, re-run with 100 repeats

---

## DECIDE

### Entry Conditions
- `fsm.current_state == "DECIDE"`
- `guard_flags.timing_captured == true`

### Mandatory Actions
1. Compare `metrics.this_iter_tflops` vs `metrics.current_best_tflops`
2. If `this_iter_tflops > current_best_tflops`:
   - Decision = KEEP
   - Update `current_best_tflops`, `current_best_iter`, `current_best_kernel`
   - Reset `consecutive_discards = 0`
3. If `this_iter_tflops <= current_best_tflops`:
   - Decision = DISCARD
   - Increment `consecutive_discards`
4. Set `metrics.this_iter_decision = "KEEP"` or `"DISCARD"`
5. Set `guard_flags.decision_made = true`

### Exit → STORE
- `guard_flags.decision_made == true`
- `metrics.this_iter_decision != null`

### On Failure
- None expected at this step (pure logic)

---

## STORE

### Entry Conditions
- `fsm.current_state == "STORE"`
- `guard_flags.decision_made == true`

### Mandatory Actions (ALL required, in order)
1. If this round produced a public measured `iter<NNN>`:
   - append row to `tuning/aitune/<dsl>/logs/<KEY>/results.tsv`
   - ensure source, build shell, run shell, timing output, and profile output (when profiling ran) are all persisted
2. If this round failed before MEASURE:
   - append an attempt entry to `attempt-log.jsonl`
   - ensure attempt source, build shell, and failure log are persisted
3. Set `guard_flags.results_appended = true`
4. Write checkpoint `tuning/aitune/<dsl>/checkpoints/<KEY>.json`
5. Set `guard_flags.checkpoint_written = true`
6. Update `tuning/aitune/<dsl>/state.json` with current progress
7. Update `compaction-summary.md` with latest iteration info
8. `git add tuning/aitune/<dsl>/logs/<KEY>/ tuning/aitune/<dsl>/srcs/<KEY>/ tuning/aitune/<dsl>/perf/<KEY>/ tuning/aitune/<dsl>/cmd/<KEY>/ tuning/aitune/<dsl>/checkpoints/<KEY>.json tuning/aitune/<dsl>/state.json`
9. `git commit -m "<KEY> iter<NNN-or-attempt>: <idea> — <KEEP|DISCARD>"`
10. Set `guard_flags.git_committed = true`
11. Persist loop-round conversation history (required):
    - Append JSON line for this iteration to `paths.round_memory_raw_log`
    - Append full markdown transcript section to `paths.round_memory_md_log`
    - Set `guard_flags.round_memory_raw_saved = true`
    - Set `guard_flags.round_memory_md_saved = true`
12. Determine next state:
    - If `iteration >= max_iteration` → transition to `SHAPE_COMPLETE`
    - Else → transition to `PROFILE`

### Exit → PROFILE or SHAPE_COMPLETE
- `guard_flags.results_appended == true`
- `guard_flags.checkpoint_written == true`
- `guard_flags.git_committed == true`
- `guard_flags.round_memory_raw_saved == true`
- `guard_flags.round_memory_md_saved == true`

### On Failure
- Git commit fails: check for uncommitted conflicts, retry
- File write fails: check disk space, retry

---

## SHAPE_COMPLETE

### Entry Conditions
- `fsm.iteration >= fsm.max_iteration`

### Mandatory Actions
1. Copy best kernel to `kernels/gemm_sp_<dtype>/<KEY>_best.cu`
2. Update `tuning/aitune/<dsl>/state.json`: set this shape to `status: "done"`
3. Final checkpoint write
4. Git commit: `"<KEY>: completed <mode>, best TFLOPS: X at iter<N>"`
5. Run `post-step-check.sh SHAPE_COMPLETE` to verify all conditions met
6. **IMMEDIATELY proceed to NEXT_SHAPE. Do NOT output a summary and wait. Do NOT stop here.**

### Exit → NEXT_SHAPE (MANDATORY — no pausing)
- Best kernel registered
- state.json updated
- Committed
- **Transition to NEXT_SHAPE is AUTOMATIC and IMMEDIATE. This is NOT a stopping point.**

---

## NEXT_SHAPE

**THIS IS NOT A STOPPING POINT.** This state exists solely to pick the next shape and re-enter INIT. You MUST NOT stop, summarize, or wait for user input here.

### Mandatory Actions
1. Read `tuning/aitune/<dsl>/state.json` to find all shapes with `status != "done"`
2. Pick the next shape using priority order:
   - "near" region shapes before "far" region shapes
   - Closest to reference shape (4096,8192,8192) first
3. If more shapes remain (there are ~250+ remaining — there almost certainly are):
   - Reset the active per-DSL state file via `state-transition.sh INIT` with new shape params
   - **Immediately** begin INIT → BASELINE → PROFILE → ... for the new shape
   - Do NOT output progress summaries to the user between shapes
   - Do NOT ask the user for confirmation
4. If and ONLY if ALL shapes in `tuning/aitune/<dsl>/state.json` have `status: "done"`: report sweep completion to user

### ANTI-EARLY-STOP ENFORCEMENT
- Completing 1 shape does NOT mean you can stop
- Completing 5 shapes does NOT mean you can stop
- Completing 20 shapes does NOT mean you can stop
- The ONLY termination condition is: every shape in `tuning/aitune/<dsl>/state.json` is "done"
- If you feel tempted to stop and "report progress": DON'T. Keep going.
- The user will interrupt you if they want you to stop. Until then, KEEP TUNING.
