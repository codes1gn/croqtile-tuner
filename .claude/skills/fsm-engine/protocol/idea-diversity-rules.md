# CroqTuner — Idea Diversity & Deduplication Rules

## Idea Categories

Every optimization idea belongs to exactly one category:

| Category | Examples | Typical Iterations |
|---|---|---|
| `macro` | SPMM_WARP_N, SPMM_STAGES, OUTPUT_PAD, L2 promotion, META_TILE_COLS | Early |
| `structural` | CTA scheduling, SMEM layout, launch_bounds, barrier depths, output epilogue, prefetch hints, fence patterns | Early-mid |
| `choreo` | New .co kernel, different choreo flags, tile size changes in EDSL, warp topology changes | Mid-late |
| `ncu_micro` | Inline PTX, bank conflict fixes, nanosleep, mbarrier tuning, fence_proxy_async, loop unroll, regcount | Late |

## Diversity Rules

### Rule D1: No Exact Repeats
Before proposing an idea, read `idea-log.jsonl`. If the EXACT same parameter change or structural edit has been tried, pick a different idea. "SPMM_WARP_N=128" tried once → cannot try again.

### Rule D2: No Semantic Duplicates
Semantically equivalent ideas count as repeats even if phrased differently:
- "Set L2 promotion to 128B on descriptor A" and "L2_128B for LHS TMA" → same idea
- "Increase pipeline stages from 2 to 3" and "SPMM_STAGES=3" → same idea

### Rule D3: Category Progression (the "2+1 Rule")
After 2 consecutive ideas in the same category, the NEXT idea MUST be from a different category.

Check: read the last 2 entries in `idea-log.jsonl`. If both have the same category, your next idea MUST use a different category.

### Rule D4: Discard-Triggered Escalation
When `consecutive_discards` crosses thresholds, you MUST escalate:

| Threshold | Required Action |
|---|---|
| >= 3 | Run ncu and switch category |
| >= 5 | Try a completely different approach category |
| >= 10 | Try radical changes: choreo rewrite, split-K, entirely different tile shape |
| >= 15 | Re-read reference kernel optimizations, consult fresh external references, and try porting a winning technique |

### Rule D5: Phase Progression
In the single forward-only tuning workflow, you MUST progress through optimization phases:

- By iter 10: at least 2 ideas should be `structural`
- By iter 25: at least 1 idea should be `choreo`
- By iter 40: at least 1 idea should be `ncu_micro`
- By iter 80: at least 3 ideas should be `choreo`
- By iter 120: at least 3 ideas should be `ncu_micro`

### Rule D6: Bottleneck-Responsive Ideas
Your idea should address the current bottleneck (`metrics.last_bottleneck`):

| Bottleneck | Relevant Ideas |
|---|---|
| `smem_throughput` | SMEM layout padding, swizzle changes, bank conflict elimination |
| `l2_throughput` | L2 promotion flags, CTA scheduling for L2 reuse, tile size for L2 residency |
| `dram_throughput` | Prefetch hints, TMA descriptor tuning, data layout changes |
| `compute_bound` | Warp topology (1p2c vs 2p1c), WGMMA tile, loop unrolling |
| `latency_bound` | Pipeline depth, barrier wait depth, nanosleep, fence patterns |
| `occupancy_limited` | Register pressure (regctrl, launch_bounds, maxrregcount) |

Exception: if the bottleneck-responsive ideas have been exhausted (all tried per idea-log), try an orthogonal idea that may shift the bottleneck itself.

### Rule D6b: Research-Backed Ideation

When profiler data and local repo history are not enough to justify the next move, explicitly consult current external references before proposing the idea. Good sources include:
- NVIDIA CUDA best-practice documentation
- Nsight Compute profiling guidance
- CUTLASS or related GEMM design references
- Recent papers or public examples directly relevant to the current bottleneck

Log the reason the source mattered in the idea entry. The goal is not to collect links for their own sake; it is to avoid unguided guessing.

### Rule D7: Re-profile on Win or Stall

- If the previous round produced a new best, the next round must re-profile that new best before ideation.
- If you cannot justify the next idea from current data, re-profile before proposing another one.
- If you still cannot justify the next idea after re-profiling, expand the evidence with external references before proposing another one.

### Rule D8: Late-Stage PTX/SASS Escalation

For long-horizon sessions, for example `iter > 300`:

- inspect PTX and SASS
- look for dependency chains, scoreboard stalls, wasted barriers, register pressure, and issue imbalance
- allow inline PTX / inline ASM only after higher-level source changes stop moving performance

## idea-log.jsonl Format

Each line is a JSON object:

```json
{"iter": 5, "idea": "L2_256B on both TMA descriptors", "category": "macro", "bottleneck_before": "l2_throughput", "justification": "ncu shows 78% L2 hit rate, promotion may improve reuse", "result": "DISCARD", "tflops": 415.3, "tflops_delta": -6.2}
```

Fields:
- `iter`: iteration number (monotonic)
- `idea`: short description of the optimization
- `category`: one of `macro`, `structural`, `choreo`, `ncu_micro`
- `bottleneck_before`: bottleneck category from ncu at this iteration
- `justification`: WHY this idea (data source)
- `research_basis`: optional list of external references or distilled notes used during ideation
- `result`: filled in after DECIDE — `"KEEP"`, `"DISCARD"`, `"DISCARD_COMPILE_FAIL"`, `"DISCARD_INCORRECT"`
- `tflops`: measured TFLOPS (null if compile/correctness failure)
- `tflops_delta`: delta vs current best (positive = improvement)

The `result`, `tflops`, and `tflops_delta` fields are updated after the DECIDE step by appending a corrected entry or by the STORE step updating the last line.

## Pre-IDEATE Validation (run by pre-step-check.sh)

The script checks:
1. `ncu_ran_this_iter == true` (profiling happened)
2. Last 2 idea categories — if same, warns that diversity is needed
3. Category phase progression — warns if behind schedule
4. `consecutive_discards` — triggers escalation warnings
