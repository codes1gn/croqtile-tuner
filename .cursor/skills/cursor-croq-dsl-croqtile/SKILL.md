---
name: cursor-croq-dsl-croqtile
description: Cursor-only DSL tuning contract for CroqTile and Choreo kernels. Loaded by cursor-croq-tune when dsl=croqtile.
---

# Croq-DSL: CroqTile (Choreo)

Source extension: `.co` -> `.cu` (two-phase) | Compiler: `choreo` + `nvcc` | Group: compiled-binary

## Required Environment Variables

```bash
export CHOREO_HOME=/home/albert/workspace/croqtile
export CUTE_HOME=$CHOREO_HOME/extern/cutlass
export CUDA_HOME=/usr/local/cuda
```

Validation: `$CHOREO_HOME/build/choreo --help` should print usage.

## Environment Validation

```bash
test -x $CHOREO_HOME/build/choreo && $CHOREO_HOME/build/choreo --help | head -1
```

## Two-Phase BUILD Pipeline

Every croqtile iteration uses a **two-phase** build. Phase 1 produces a `.cu` from the `.co`
via the harness. Phase 2 is where the agent fine-tunes the `.cu` and compiles with nvcc.

### Phase 1: .co -> .cu (harness tool)

Write or copy the `.co` file, then call the harness:

```bash
READY=$(bash .cursor/skills/cursor-croq-tune/tools/co2cu.sh \
    --co tuning/<gpu>/croqtile/srcs/<key>/<model>/iter<NNN>_<tag>.co \
    --arch <sm_arch> \
    --flags "<choreo flags>")
echo "$READY"  # JSON with paths: co, cu, result, nvcc_flags, headers_dir
```

The harness:
1. Runs `choreo -gs` to produce `.cute.result`
2. Extracts the `__choreo_cute_*.cu` heredoc from it
3. Writes the `.cu` next to the `.co` in `srcs/`
4. Reports paths and suggested nvcc flags as JSON

If the IDEA only needs `.cu` changes (not `.co` changes), copy the base iter's `.co` verbatim
before calling the harness. The `.co` must always exist for reproducibility.

### Phase 2: fine-tune .cu + nvcc build (agent)

Read the extracted `.cu`, modify the kernel section to implement the IDEA. Common
Phase 2 edits:
- Adding `__launch_bounds__`
- Custom PTX inline asm
- SMEM layout changes
- Loop unrolling pragmas
- Register-level optimizations

**build_iter\<NNN\>.sh** (agent writes this):
```bash
#!/usr/bin/env bash
set -e
export CHOREO_HOME=/home/albert/workspace/croqtile
export CUTE_HOME=$CHOREO_HOME/extern/cutlass
export CUDA_HOME=/usr/local/cuda
HEADERS=".cursor/skills/cursor-croq-tune/tools/choreo_headers"

SRC="tuning/<gpu>/croqtile/srcs/<key>/<model>/iter<NNN>_<tag>.cu"
BIN="tuning/<gpu>/croqtile/bin/<key>/<model>/iter<NNN>_<tag>"
LOG="tuning/<gpu>/croqtile/perf/<key>/<model>/build_iter<NNN>.txt"

mkdir -p "$(dirname "$BIN")"
nvcc -O3 -arch=<sm_arch> -std=c++17 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
    -D__USE_CUDA_TYPE__ \
    -I"$HEADERS" -I${CUTE_HOME}/include \
    -Xcompiler -static-libstdc++ \
    -L${CUDA_HOME}/lib64 -lcuda \
    -o "$BIN" "$SRC" \
    2>&1 | tee "$LOG"
```

**run_iter\<NNN\>.sh** (agent writes this):
```bash
#!/usr/bin/env bash
BIN="tuning/<gpu>/croqtile/bin/<key>/<model>/iter<NNN>_<tag>"
LOG="tuning/<gpu>/croqtile/perf/<key>/<model>/timing_iter<NNN>.txt"
"$BIN" "$@" 2>&1 | tee "$LOG"
```

**Phase 2 pass-through:** If the IDEA is purely a `.co`-level change and no `.cu` fine-tuning
is needed, use the extracted `.cu` as-is — still build with nvcc (not the `.cute.result`).

### Required artifacts per iteration

| Artifact | Path | Required |
|---|---|---|
| `.co` source | `srcs/<key>/<model>/iter<NNN>_<tag>.co` | Always |
| `.cu` source | `srcs/<key>/<model>/iter<NNN>_<tag>.cu` | Always |
| `.cute.result` | `cmd/<key>/<model>/iter<NNN>_<tag>.cute.result` | Always |
| build script | `cmd/<key>/<model>/build_iter<NNN>.sh` | Always |
| run script | `cmd/<key>/<model>/run_iter<NNN>.sh` | Always |
| binary | `bin/<key>/<model>/iter<NNN>_<tag>` | After build |

## PROFILE — ncu Command

Profile the **nvcc-compiled binary** (Phase 2 output), not the `.cute.result` script:

```bash
bash .cursor/skills/cursor-croq-tune/tools/ncu_profile.sh \
    --out tuning/<gpu>/croqtile/perf/<key>/<model>/ncu_iter<NNN>_<tag>_round<R> \
    --cmd tuning/<gpu>/croqtile/bin/<key>/<model>/iter<NNN>_<tag>
```

## Pure Implementation Rule

Follow `choreo-syntax` skill. **Study reference kernels in `cursor-choreo-kernel-examples/` before editing.**

**.co-First Philosophy:** Start every idea at the `.co` level. Phase 2 (.cu fine-tune) exists
for optimizations that `.co` cannot express — it does NOT replace `.co` authoring.

Typical optimization flow:
1. **Phase 1 only:** `.co` change is sufficient (tile sizes, MMA forms, pipeline stages, warp spec)
2. **Phase 1 + Phase 2:** `.co` gets close, then `.cu` fine-tune adds `__launch_bounds__`, PTX asm,
   custom SMEM layouts, or other constructs `.co` doesn't support
3. **Phase 2 heavy:** `.co` is unchanged from base iter, all work is in the `.cu` fine-tune

Log which phases were active in `idea-log.jsonl`: `co_changed: true/false`, `cu_finetuned: true/false`.

If an idea involves compiler changes (e.g. new Choreo primitive), implement the compiler change first, rebuild, then use it in the kernel. Treat both as ONE atomic change.

**Choreo flags (passed to `co2cu.sh --flags`):**
- Sparse GEMM (gemm_sp): `--use-warpspec --use-prepack`
- Dense GEMM: `--use-warpspec` (optional)

**gemm_sp Constraints (SM90A/WGMMA only — NOT for SM80/SM86):**
- `SPMM_WARP_M` MUST be 64 (WGMMA constraint)
- `SPMM_WARP_K` MUST be 32 for f16
- `SPMM_TILE_K` MUST equal `2 * SPMM_PACKED_TILE_K`
- `SPMM_META_TILE_COLS` MUST equal `SPMM_TILE_K / 32`

**gemm_sp on SM80/SM86 (Ampere MMA):** These WGMMA constraints do NOT apply. Use
`mma.sp.sync.aligned` with WARP_M=16, WARP_N=8, WARP_K=16 (f16)(not only this MMA shape, look at choreo.h for references). See
`cursor-choreo-kernel-examples/gemm-sp-tests/gemm_sp_sm80.co` for the correct pattern.

**Forbidden in `.co` and `.cu`:** cuBLAS calls, `cutlass::gemm::device::Gemm`, PyTorch ops.

## Tunable Parameters

| Macro | Typical values | Effect |
|---|---|---|
| `MATMUL_WARP_M` | 64 | Warp-group height (fixed for WGMMA) |
| `MATMUL_WARP_N` | 64, 128, 192, 256 | Warp-group width |
| `MATMUL_TILE_M` | 64, 128, 192 | CTA tile height |
| `MATMUL_TILE_K` | 64 | CTA tile K-depth |
| `MATMUL_WARP_K` | 16 | WGMMA K-step per MMA |
| `MATMUL_SWIZ` | 32, 64, 128 | TMA swizzle factor |
| `MATMUL_STAGES` | 2, 3, 4 | Pipeline depth |
| `NUM_SMS` | 114 (H800) | CTA count for persistent |
| `MATMUL_DEFAULT_M/N/K` | 2048+ | Problem size |

## CRITICAL: Accumulator Type Must Match Output dtype

**ALWAYS match the accumulator fill form to the output dtype. Using the wrong form causes WMMA compile errors.**

| Kernel dtype | Output tensor type | MMA fill form | Example |
|---|---|---|---|
| f16→f32 (matmul_f16f32) | `f32 [M,N] output` | `mma.fill.f32 0.0` | `mc = mma.fill.f32 0.0;` |
| f16→f16 (matmul_f16) | `f16 [M,N] output` | `mma.fill 0.0` | `mc = mma.fill 0.0;` |
| bf16→f32 (matmul_bf16fp32) | `f32 [M,N] output` | `mma.fill.f32 0.0` | `mc = mma.fill.f32 0.0;` |

**Error signature if wrong:** `no instance of overloaded function "nvcuda::wmma::store_matrix_sync" matches argument list` — this means you used f16 accumulator fill but the output is `float*`. Fix: change `mma.fill 0.0` to `mma.fill.f32 0.0` and change `f16 [M,N] output` to `f32 [M,N] output`.

Reference: `.cursor/skills/cursor-cursor-choreo-kernel-examples/mma-tests/wmma.co` shows both f16→f32 and f16→f16 patterns.

## Available DSL Primitives

```
# Data movement
dma.copy src => dst;
dma.copy.async src => shared after event;
tma.copy.swiz<N> src => dst;
tma.copy.async<event>.swiz<N> src => dst;

# Compute
mma.fill.f32 0.0;   # float32 accumulator (use for f16f32, bf16fp32 kernels)
mma.fill.f16 0.0f;  # f16 accumulator (use only for f16->f16 kernels)
mma.fill 0.0;       # default accumulator type (matches output dtype in simple kernels)
mma.load.swiz<N> buffer;
mma.row.row accumulator, a, b;
mma.row.row.scale accumulator, a, b, scale_a, scale_b;
mma.row.col.sp accumulator, a, b;  # sparse MMA
mma.op <shape> accumulator, a, b;   # preferred unified form
mma.commit;
mma.store accumulator, output;
mma.store.transp accumulator, output;

# Control
parallel {p, q} by [a, b] : block;
parallel p by n : group-4;  # warp specialization
inthreads.async (condition) { ... };
foreach {i} in [N] { ... };
wait event;
trigger event;
```

## IDEA Menu

| Bottleneck | ncu indicator | Ideas |
|---|---|---|
| SM underutilized | sm__throughput < 80% | Increase CTAs, check occupancy |
| Memory-bound | tensor_op < 50%, high l1tex | Increase tile grid; `tma.copy` vs `dma.copy`; add pipelining; larger tiles |
| Compute-bound | low wgmma issue rate | Switch MMA to `mma.op` with higher throughput shape; more register accumulators |
| Latency-bound | high wgmma stall | Add async pipeline stages; warp-specialized pipeline (1p1c -> 1p2c) |
| L2 thrash | lts__lookup_miss high | Change tile shape (M/N/K ratio); swizzle tile mapping |
| WGMMA latency | low wgmma__ops rate | Better producer/consumer overlap; more stages |
| Register pressure | low occupancy | Reduce tile size, fewer threads per CTA, regctrl |
| Bank conflicts | smem stalls | Swizzled shared layout in tile view; pad smem |

**Structural changes (MANDATORY after 2 consecutive macro-only changes):**

Phase 1 (.co level):
- Adding/removing/reordering synchronization fences (`wgmma.fence`, `warpgroup_arrive`)
- Changing register control (`setmaxnreg` / `regctrl`)
- Reordering load/store operations (e.g. RHS before LHS, metadata first)
- Adding compiler flags (`--hoist-offset`, `--hoist-scale`, `--stmatrix`)
- Changing output store pattern (shared padding, transpose)
- Modifying pipeline structure (event placement, commit placement)
- Switching warp specialization topology (`1p1c <-> 1p2c <-> 1p3c`)
- Changing copy-shaping patterns (`view().from()`, `subspan().step().at()`)

Phase 2 (.cu level — when .co can't express it):
- `__launch_bounds__(threads, minBlocks)` on kernel functions
- Custom PTX inline asm (`asm volatile(...)`)
- `#pragma unroll N` on critical loops
- SMEM bank-conflict-free layout adjustments
- Custom `__nanosleep()` or `__threadfence()` placement
- nvcc flags: `-maxrregcount`, `--use_fast_math`, `-Xptxas -dlcm=ca`

**Key levers:** Tile dimensions [BM, BN, BK], copy primitive (dma vs tma),
MMA form, pipeline stages, warp specialization (1p1c/1p2c/1p3c), swizzle layout,
regctrl, compiler flags, output epilogue.

Consult `choreo-syntax` and study `cursor-choreo-kernel-examples/` before editing `.co`.

## Compile-Fail Debug Procedure

**Choreo error messages are designed for AI consumption — read them carefully.**
They contain structured hints: the failing construct, expected vs actual types,
supported arch features, and suggested alternatives. Never ignore them.

When a compile fails, follow this ordered procedure:

### Step 1: ANALYZE — gather evidence first

Before classifying the failure, gather evidence from multiple sources:
- **Read the compiler error** thoroughly — choreo errors often say what went wrong and
  suggest what to use instead (e.g. "WGMMA requires SM90A, use mma.op for SM86")
- **Web search** for the specific error (e.g. "SM86 sparse MMA instructions",
  "mma.sp.sync.aligned Ampere", "CUDA SM86 tensor core capabilities")
- **Read compiler/DSL docs** — load `choreo-syntax` and `choreo-kernel-examples` to find
  supported patterns for the target arch
- If the error is unclear, **read the generated `.cu` or `.cute.result`** file to
  understand what the compiler actually produced

### Step 2: CLASSIFY — based on the evidence

With evidence in hand, classify the failure:
- **Code bug**: typo, wrong index, missing sync, shape mismatch in your `.co` code
- **DSL/compiler limitation**: unsupported instruction for this arch (e.g. WGMMA on
  SM86), missing primitive, codegen bug on valid input

### Step 3: FIX based on the analysis

- For **code bugs**: fix the `.co` and retry (within the 5-attempt budget)
- For **DSL limitations**: find an alternative that IS supported on the target arch:
  - WGMMA unsupported on SM86 → use `mma.op` or `mma.sp.sync.aligned` (Ampere MMA)
  - Sparse codegen bug on specific tile size → try different tile dimensions
  - TMA unsupported → use `dma.copy.async` instead
  - Instruction not available → use the closest supported equivalent
- For **persistent `.co` → `.cu` failures** (choreo keeps producing broken `.cu` output
  despite valid `.co` input), you may **bypass Phase 1**:
  1. Read the generated `.cu` or `.cute.result` file directly
  2. Hack/fix the `.cu` code by hand to repair the broken codegen output
  3. Compile the fixed `.cu` directly with nvcc, skipping the `.co` → `.cu` step
  4. Log this as `"co_bypassed": true` in the idea-log entry
  5. Still keep the `.co` file for reference (do NOT delete it)

### Arch-Specific Notes

| GPU Arch | Supported MMA | Sparse MMA | TMA | WGMMA |
|----------|---------------|------------|-----|-------|
| SM80 (A100) | `mma.sync.aligned` | `mma.sp.sync.aligned` | No | No |
| SM86 (RTX 3070) | `mma.sync.aligned` | `mma.sp.sync.aligned` | No | No |
| SM89 (RTX 4090) | `mma.sync.aligned` | `mma.sp.sync.aligned` | Yes | No |
| SM90a (H100) | `mma.sync.aligned` + WGMMA | `mma.sp.sync.aligned` + WGMMA | Yes | Yes |

When targeting SM80/SM86, the reference kernels in `cursor-choreo-kernel-examples/gemm-sp-tests/`
that require WGMMA or TMA are NOT usable. Look for `_sm80` suffix kernels or implement
from scratch using `mma.op` / `mma.sp.sync.aligned` forms.

## VERIFY / MEASURE

Verification tolerance same as compiled-binary group:

| Precision | base_tol | rel_tol |
|-----------|----------|---------|
| FP16 input, FP32 accum | 1.0 | 0.01 |
| FP16 input, FP16 accum | 16.0 | 0.05 |
| FP8 E4M3 input, FP16 accum | 0.5 | 0.01 |

**NEVER print "Test Passed" without actual numerical comparison.**
Timing via CUDA events. Default: 10 warmup + 50 timed.

## BASELINE (iter000 + iter001)

**iter000 — cuBLAS reference (PREPARATION_ONCE step 2b, MANDATORY):**

Before any kernel tuning, measure cuBLAS/torch.mm performance using the harness:
```bash
bash .cursor/skills/cursor-croq-tune/tools/cublas_baseline.sh \
    --dtype <dtype> --m <M> --n <N> --k <K>
```
Record the `tflops` output as `baseline_tflops` and store as iter000 in results.tsv.
This is your hardware ceiling — all subsequent iterations are compared against it.

**iter001 — Starting kernel (PREPARATION_ONCE step 3):**

Kernel discovery is handled by `discover_baseline.sh`. The harness scans
`cursor-choreo-kernel-examples/` (excluding `_aitune_` files) and `tuning/` for
same-operator kernels. If no candidates exist, implement from scratch using MMA primitives.

Library calls (cuBLAS/torch) allowed only in iter000 for reference performance comparison.

## Reference Kernels

**All reference `.co` kernels live in two locations — study both before editing:**

1. **`cursor-choreo-kernel-examples/`** — clean reference implementations organized by kernel family
   (`matmul/`, `gemm_sp/`, `matmul_bf16/`, etc.). Files with `_aitune_` in the name are
   prior tuning outputs and should NOT be used as starting points.
2. **`tuning/<gpu>/croqtile/srcs/`** — kernels from prior tuning sessions. The best iteration
   per shape is the highest-numbered `iter<NNN>` file in each model subdirectory.

Key reference kernels in `cursor-choreo-kernel-examples/`:

- `matmul/matmul_f16_dyn_sm90.co` — baseline FP16 GEMM with TMA+swizzle
- `matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co` — 1-producer/1-consumer warp-specialized
- `matmul/matmul_f16_dyn_sm90_warpspec_1p3c.co` — 1-producer/3-consumer warp-specialized
- `matmul/matmul_f16_dyn_persis_sta_sm90.co` — persistent CTA
- `gemm_sp/gemm_sp_f16_dyn_sm90_warpspec_1p2c_swizzle128_128_prepack.co` — sparse GEMM

## Related Skills

- `choreo-syntax` — Choreo DSL reference for `.co` editing (load before any `.co` changes)
- `choreo-kernel-examples` — Full reference kernel collection (study patterns before editing)
- `perf-nsight-compute-analysis` — Deep ncu report analysis (load when profile hints suggest deeper investigation)
