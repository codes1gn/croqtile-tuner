---
name: croq-dsl-tilelang
description: DSL-specific tuning contract for TileLang kernels. Loaded by croq-tune when dsl=tilelang.
---

# Croq-DSL: TileLang

Source extension: `.py` | Compiler: TileLang JIT -> PTX via TVM | Group: python-jit

## Environment Validation

```bash
python3 -c "import tilelang; print(tilelang.__version__)"
```

## BUILD / RUN Templates

**build_iter\<NNN\>.sh** (syntax/import check):
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

JIT compile errors surface at first run, not import. Treat first-run JIT failure
the same as a compile failure (bounded retry budget, then `attempt<AAAA>`).

**run_iter\<NNN\>.sh:**
```bash
#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 tuning/<gpu>/tilelang/srcs/<shape_key>/<model>/iter<NNN>_<tag>.py \
    2>&1 | tee tuning/<gpu>/tilelang/perf/<shape_key>/<model>/timing_iter<NNN>.txt
```

Script must print: `TFLOPS: <value>   time_ms: <value>`

## PROFILE — ncu Command

```bash
ncu --target-processes all \
    --set full \
    --export tuning/<gpu>/tilelang/perf/<shape_key>/<model>/ncu_iter<NNN>.ncu-rep \
    --force-overwrite \
    python3 tuning/<gpu>/tilelang/srcs/<shape_key>/<model>/iter<NNN>_<tag>.py

ncu --import tuning/<gpu>/tilelang/perf/<shape_key>/<model>/ncu_iter<NNN>.ncu-rep \
    --csv --page raw \
    > tuning/<gpu>/tilelang/perf/<shape_key>/<model>/ncu_iter<NNN>.csv
```

**Pre-profiling pin:** Pin `block_M`, `block_N`, `num_stages` explicitly; do not run `AutoTuner`.

## Pure Implementation Rule

**Allowed:** `T.Pipelined`, `T.Parallel`, `T.use_swizzle`, `T.copy`, `T.mma`,
explicit `block_M/N/K`, `num_stages`, tile-level memory placement.

**Forbidden:** `tvm.relay.*` library GEMM, external library compute inside tile program.

## IDEA Menu

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
           num_stages=3,  # tune: 1-5
           thread_num=128,
           enable_rasterization=True):
```
Use `tilelang.Carver` for hardware-aware starting-point recommendations.

## VERIFY / MEASURE

Kernel script must contain `verify()` and `bench()` functions:

**verify():** compute with kernel + reference (torch.float32), assert `max_abs_err < tol`
(1e-2 for bf16, 1e-3 for f16), print `VERIFY: PASS` or `VERIFY: FAIL max_abs_err=<v>`.

**bench():** CUDA event timing (never `time.time()`):
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

## BASELINE (iter000)

`torch.matmul` in Python script. Library calls allowed only in iter000.
