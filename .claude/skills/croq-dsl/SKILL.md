---
name: croq-dsl
description: DSL-specific tuning contract for all supported kernel DSLs. Load this skill at the start of any croq-tune session to get the correct build/run/profile commands, pure-implementation rules, and IDEA menus for the chosen DSL. Covers: cuda, croqtile, cutile (compiled-binary group) and triton, cute, helion, tilelang (Python-JIT group).
---

# Croq-DSL — Per-DSL Tuning Contract

Load this skill at the start of every `croq-tune` session, immediately after parsing `dsl`.
This skill overrides PROFILE, IDEA, IMPLEMENT, VERIFY, MEASURE for the active DSL.
All other steps (DECIDE, STORE, CONTINUE, branch management, artifact naming) follow
the main `croq-tune` contract.

---

## DSL Groups and Source Extensions

| DSL        | Group          | Source ext             | Compiler model              |
|------------|----------------|------------------------|-----------------------------|
| `cuda`     | compiled-binary | `.cu`                  | `nvcc` → static binary      |
| `croqtile` | compiled-binary | `.co` (→ `.gen.cu`)   | `croqc` + `nvcc` → binary   |
| `cutile`   | compiled-binary | `.cu` or `.cpp`        | `nvcc` (CUTLASS templates)  |
| `triton`   | python-jit      | `.py`                  | Triton JIT → PTX via LLVM   |
| `cute`     | python-jit      | `.py`                  | `cute.compile()` → PTX      |
| `helion`   | python-jit      | `.py`                  | Helion → Triton → PTX       |
| `tilelang` | python-jit      | `.py`                  | TileLang JIT → PTX via TVM  |

All source artifacts use the `iter<NNN>_<tag>.<ext>` naming from `croq-artifacts`.

---

## BUILD / RUN TEMPLATES

### Compiled-Binary Group (`cuda`, `cutile`)

**build_iter\<NNN\>.sh**:
```bash
#!/usr/bin/env bash
set -e
nvcc -O3 -arch=sm_90 -std=c++17 -I/usr/local/cuda/include \
     -o tuning/aitune/<dsl>/bin/<shape_key>/iter<NNN>_<tag> \
     tuning/aitune/<dsl>/srcs/<shape_key>/iter<NNN>_<tag>.cu \
     2>&1 | tee tuning/aitune/<dsl>/perf/<shape_key>/build_iter<NNN>.txt
```

**run_iter\<NNN\>.sh**:
```bash
#!/usr/bin/env bash
tuning/aitune/<dsl>/bin/<shape_key>/iter<NNN>_<tag> \
    2>&1 | tee tuning/aitune/<dsl>/perf/<shape_key>/timing_iter<NNN>.txt
```

Binary must print: `TFLOPS: <value>   time_ms: <value>`

### CroqTile — Two-Step Build

**build_iter\<NNN\>.sh**:
```bash
#!/usr/bin/env bash
set -e
# Step 1: .co → .gen.cu
croqc tuning/aitune/croqtile/srcs/<shape_key>/iter<NNN>_<tag>.co \
      -o tuning/aitune/croqtile/srcs/<shape_key>/iter<NNN>_<tag>.gen.cu \
      2>&1 | tee tuning/aitune/croqtile/perf/<shape_key>/build_iter<NNN>_co.txt
# Step 2: .gen.cu → binary
nvcc -O3 -arch=sm_90 -std=c++17 \
     -o tuning/aitune/croqtile/bin/<shape_key>/iter<NNN>_<tag> \
     tuning/aitune/croqtile/srcs/<shape_key>/iter<NNN>_<tag>.gen.cu \
     2>&1 | tee tuning/aitune/croqtile/perf/<shape_key>/build_iter<NNN>_cu.txt
```

Both `.co` (primary source) and `.gen.cu` (generated) are preserved as artifacts.

### Python-JIT Group (`triton`, `cute`, `helion`, `tilelang`)

**build_iter\<NNN\>.sh** (syntax/import check only):
```bash
#!/usr/bin/env bash
python3 -c "import ast; ast.parse(open('$SRC').read()); print('parse OK')"
python3 -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('kernel', '$SRC')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
print('import OK')
"
```

A JIT **compile error** surfaces at first run, not import. Treat first-run
JIT compile failure the same as a compile failure in the compiled-binary group
(bounded retry budget, then `attempt<AAAA>`).

**run_iter\<NNN\>.sh**:
```bash
#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 tuning/aitune/<dsl>/srcs/<shape_key>/iter<NNN>_<tag>.py \
    2>&1 | tee tuning/aitune/<dsl>/perf/<shape_key>/timing_iter<NNN>.txt
```

Script must print: `TFLOPS: <value>   time_ms: <value>`

---

## PROFILE STEP — ncu Command Per DSL Group

### Compiled-Binary Group

```bash
ncu --set full \
    --export tuning/aitune/<dsl>/perf/<shape_key>/ncu_iter<NNN>.ncu-rep \
    --force-overwrite \
    tuning/aitune/<dsl>/bin/<shape_key>/iter<NNN>_<tag> [args]

ncu --import tuning/aitune/<dsl>/perf/<shape_key>/ncu_iter<NNN>.ncu-rep \
    --csv --page raw \
    > tuning/aitune/<dsl>/perf/<shape_key>/ncu_iter<NNN>.csv
```

### Python-JIT Group

`ncu` wraps the **Python process**; the kernel script must call the kernel at least once:

```bash
ncu --target-processes all \
    --set full \
    --export tuning/aitune/<dsl>/perf/<shape_key>/ncu_iter<NNN>.ncu-rep \
    --force-overwrite \
    python3 tuning/aitune/<dsl>/srcs/<shape_key>/iter<NNN>_<tag>.py

ncu --import tuning/aitune/<dsl>/perf/<shape_key>/ncu_iter<NNN>.ncu-rep \
    --csv --page raw \
    > tuning/aitune/<dsl>/perf/<shape_key>/ncu_iter<NNN>.csv
```

**JIT-specific pre-profiling pins** (disable autotuning so ncu sees one kernel):

| DSL      | Pre-profiling requirement |
|----------|---------------------------|
| `helion` | `@helion.kernel(autotune_effort="none", config=helion.Config(...))` or `HELION_AUTOTUNE_EFFORT=none` |
| `cute`   | Use `cute.compile()` with explicit config params (not the autotuner) |
| `tilelang` | Pin `block_M`, `block_N`, `num_stages` explicitly; do not run `AutoTuner` |
| `triton` | Remove `@triton.autotune` decorator and use the best config's constants directly |

---

## PURE IMPLEMENTATION RULE — Per DSL

All iterations `iter001+` MUST use pure DSL primitives. No library compute calls.

### `cuda`

**Allowed:** Raw CUDA kernels, CUDA intrinsics (`__shfl_sync`, `__ldg`, etc.),
PTX inline asm (`asm volatile`), cooperative groups, CuTe/CUTLASS atoms only
(copy atoms, MMA atoms, layout algebra).

**Forbidden:** `cublasSgemm`, `cublasGemmEx`, `cutlass::gemm::device::Gemm`,
cuSPARSELt, cuDNN compute calls, PyTorch/TensorFlow ops.

### `croqtile`

Follow `choreo-syntax` skill. Fallback hierarchy (follow in order):
1. Pure `.co` using Choreo DSL primitives (preferred)
2. `.co` + `__cpp__` inline fragments
3. Compile `.co` base, modify the generated `.gen.cu`
4. Direct `.cu` modification (only if 1–3 are truly infeasible)

Log fallback reason in `idea-log.jsonl`: `co_supported`, `fallback_reason`, `fallback_base`.

**Forbidden in `.co`:** cuBLAS calls, `cutlass::gemm::device::Gemm`, PyTorch ops.

### `cutile`

**Allowed:** CUTLASS C++ tile primitives, CuTe atoms, raw CUDA intrinsics, PTX.

**Forbidden:** `cutlass::gemm::device::Gemm` (top-level library GEMM), cuBLAS.

### `triton`

**Allowed:** `@triton.jit` kernels, `tl.load/store/dot/atomic_*`, warp-level
intrinsics, `tl.inline_ptx_asm`, fixed-config launchers (no `@triton.autotune`
in iter001+).

**Forbidden:** `torch.ops.*` compute delegation, `triton.ops.matmul`, flash-attn
package calls. `@triton.autotune` is allowed only in iter000.

### `cute` (CuTe DSL)

**Allowed:** `@cute.jit`/`cute.compile()`, CuTe copy atoms, MMA atoms, layout
algebra, TMA copy primitives, warp specialization primitives.

**Forbidden:** `cutlass.gemm.device.Gemm`, cuBLAS bindings, `torch.mm`/`torch.bmm`.

### `helion`

**Allowed:** `@helion.kernel` with explicit `helion.Config`, `hl.tile`, `hl.grid`,
`hl.register_buffer`.

**Forbidden:** `torch.ops` compute delegation inside kernel body, xformers/flash-attn
library calls inside Helion kernel.

### `tilelang`

**Allowed:** `T.Pipelined`, `T.Parallel`, `T.use_swizzle`, `T.copy`, `T.mma`,
explicit `block_M/N/K`, `num_stages`, tile-level memory placement.

**Forbidden:** `tvm.relay.*` library GEMM, external library compute inside tile program.

---

## IDEA STEP — What to Tune Per DSL

Use the ncu bottleneck to pick one idea per round from the menu for the active DSL.

### `cuda` Idea Menu

| Bottleneck | Ideas |
|---|---|
| Memory-bound (DRAM) | Increase BM/BN/BK tiles; vectorize loads (`float4`/`uint4`); `cp.async`; `__ldg`; TMA (Hopper) |
| Compute-bound | Reduce smem; decrease register count (`--maxrregcount`); increase threads; split accumulator |
| Latency-bound | Add `cp.async` pipeline stages; `tma.load` on Hopper; interleave independent mem + compute |
| L2 locality | Threadblock swizzle (tile-ID remapping); change CTA launch order |
| Bank conflicts | Pad smem arrays (`+4`/`+8` elements/row); swizzled shared layout |
| Warp divergence | Eliminate conditional branches in hot paths; predicate with masks |

**Key levers:** BM/BN/BK (multiples of 16, typically 64–256), warp count (blockDim),
pipeline stages (1–5), smem padding, register pressure (`#pragma unroll N`, `--maxrregcount`),
vectorized loads, warp primitives, PTX inline (`ldmatrix`, `mma.sync`), TMA.

### `croqtile` Idea Menu

| Bottleneck | Ideas |
|---|---|
| Memory-bound | Increase tile grid; `tma.copy` vs `dma.copy`; add pipelining with event barriers |
| Compute-bound | Switch MMA to `mma.op` with higher throughput shape; more register accumulators |
| Latency-bound | Add async pipeline stages; warp-specialized pipeline pattern |
| L2 locality | Swizzle tile mapping; change CTA launch order |
| Bank conflicts | Swizzled shared layout in tile view |

**Key levers:** Tile dimensions `[BM, BN, BK]`, copy primitive (`dma` vs `tma`),
MMA form, pipeline stages, warp specialization, swizzle layout.

Consult `choreo-syntax` before editing `.co`.

### `cutile` Idea Menu

| Bottleneck | Ideas |
|---|---|
| Memory-bound | Increase `ThreadblockShape`; add pipeline stages; TMA (Hopper) |
| Compute-bound | Change `WarpShape`; increase `InstructionShape` |
| Latency-bound | Increase `kStages`; use persistent kernel |
| L2 locality | Apply threadblock swizzle epilogue |

### `triton` Idea Menu

| Bottleneck | Ideas |
|---|---|
| Memory-bound | Increase `BLOCK_M/N/K`; add `num_stages`; `eviction_policy="evict_last"` |
| Compute-bound | Reduce register pressure; tune `num_warps` (4/8); `tl.dot(allow_tf32=True)` |
| Latency-bound | Increase `num_stages`; prefetch hints; warp specialization (`warp_specialize=True`) |
| Launch-bound | Persistent kernel: grid-stride loop; operator fusion |
| Swizzle / L2 | Program-ID swizzle: `pid = (pid % GROUP_M) * (N // BLOCK_N) + pid // GROUP_M` |

**Key levers:** `BLOCK_M/N/K` (32–256, powers of 2), `num_warps` (4/8/16),
`num_stages` (1–7), `num_ctas` (multi-CTA, Hopper), `allow_tf32`, swizzle, warp spec.

### `cute` (CuTe DSL) Idea Menu

| Bottleneck | Ideas |
|---|---|
| Memory-bound | Increase `mma_tiler_mn`; add TMA pipeline stages |
| Compute-bound | Change MMA instruction shape; `use_2cta_instrs=True` (Blackwell); tune `cluster_shape_mn` |
| Latency-bound | Increase pipeline depth; interleave MMA and copy |
| L2 locality | Swizzled layout; change `cluster_shape_mn` |
| Register pressure | Reduce accumulator fragmentation; split MMA tiles |

**Key levers:** `mma_tiler_mn` (e.g. `(64,64)`, `(128,128)`, `(256,128)`),
`cluster_shape_mn` (e.g. `(1,1)`, `(2,1)`), `use_2cta_instrs`, `use_tma_store`,
number of pipeline stages.

### `helion` Idea Menu

Each iteration tries one `helion.Config`. Document configs in `idea-log.jsonl`.
View generated Triton with `@helion.kernel(print_output_code=True, config=cfg)`.

| Bottleneck | Ideas |
|---|---|
| Memory-bound | Increase `block_sizes`; change `loop_orders` |
| Compute-bound | Larger block sizes; higher `num_warps` |
| Latency-bound | Increase `num_stages`; `indexing="block"` |
| L2 locality | Reorder block_sizes combination; `reduce_dim` reordering |

**Key Config levers:**
```python
helion.Config(
    block_sizes=[BLOCK_M, BLOCK_N, BLOCK_K],
    num_warps=4,       # 4, 8, 16
    num_stages=3,      # pipeline depth
    loop_orders=[0, 1, 2],
    indexing="block",  # or "element"
)
```

### `tilelang` Idea Menu

| Bottleneck | Ideas |
|---|---|
| Memory-bound | Increase `block_M/N`; add `num_stages` |
| Compute-bound | Use `T.mma` with larger tile; tune `thread_num` |
| Latency-bound | Increase `num_stages`; `T.Pipelined` with prefetch |
| L2 locality | `T.use_swizzle` with different swizzle factors |
| Shared memory | Pad smem (`pad=N`) to avoid bank conflicts |

**Key levers:**
```python
def matmul(M, N, K,
           block_M=128,   # tune: 64, 128, 256
           block_N=128,   # tune: 64, 128, 256
           block_K=32,    # tune: 16, 32, 64
           num_stages=3,  # tune: 1–5
           thread_num=128,
           enable_rasterization=True):
```
Use `tilelang.Carver` for hardware-aware starting-point recommendations.

---

## VERIFY / MEASURE — Python-JIT Group

For JIT DSLs the kernel script must contain `verify()` and `bench()` functions:

**verify()**: compute with kernel + reference (torch.float32), assert `max_abs_err < tol`
(1e-2 for bf16, 1e-3 for f16), print `VERIFY: PASS` or `VERIFY: FAIL max_abs_err=<v>`.

**bench()**: CUDA event timing (never `time.time()`):
```python
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(iters):  kernel_fn(...)
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end) / iters
tflops = 2 * M * N * K / elapsed_ms / 1e9
print(f"TFLOPS: {tflops:.2f}   time_ms: {elapsed_ms:.3f}")
```

Default: 10 warmup + 50 timed iterations.

---

## VERIFY / MEASURE — Compiled-Binary Group

The binary must accept `--verify` (or run verify by default) and:
1. Compute with custom kernel
2. Compute reference (cuBLAS in harness, or prior verified iter)
3. Print `VERIFY: PASS` or `VERIFY: FAIL max_abs_err=<v>`

Timing via CUDA events in the harness (not Python wall-clock). Default: 10 warmup + 50 timed.
Print: `TFLOPS: <v>   time_ms: <v>`

---

## BASELINE (iter000) Per DSL

| DSL        | iter000 baseline approach |
|------------|---------------------------|
| `cuda`     | cuBLAS harness (`.cu`) — calls `cublasGemmEx` |
| `croqtile` | cuBLAS harness (`.cu`) or CroqTile reference `.co` |
| `cutile`   | cuBLAS harness (`.cu`) |
| `triton`   | `torch.matmul` in Python script |
| `cute`     | `torch.matmul` in Python script |
| `helion`   | `torch.matmul` in Python script |
| `tilelang` | `torch.matmul` in Python script |

Library calls are allowed **only** in iter000 and in correctness verification references.

---

## ENVIRONMENT VALIDATION — Additional Checks Per DSL

Beyond the standard `croq-baseline` checks (ncu, nvcc, GPU):

| DSL        | Additional validation |
|------------|-----------------------|
| `croqtile` | `which croqc && croqc --version` |
| `triton`   | `python3 -c "import triton; print(triton.__version__)"` |
| `cute`     | `python3 -c "import cutlass; print(cutlass.__version__)"` |
| `helion`   | `python3 -c "import helion; print(helion.__version__)"` |
| `tilelang` | `python3 -c "import tilelang; print(tilelang.__version__)"` |
| `cuda`     | No additional check |
| `cutile`   | `python3 -c "import cutlass"` (optional, for Python template instantiation) |

If the DSL runtime is missing, STOP and escalate to user before tuning starts.
