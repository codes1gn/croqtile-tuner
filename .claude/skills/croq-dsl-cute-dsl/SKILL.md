---
name: croq-dsl-cute-dsl
description: DSL-specific tuning contract for CuTe DSL (Python JIT) kernels. Loaded by croq-tune when dsl=cute-dsl.
---

# Croq-DSL: CuTe DSL (Python JIT)

Source extension: `.py` | Compiler: `cute.compile()` -> PTX | Group: python-jit

> **Note:** This is the **Python** CuTe DSL interface (`pip install nvidia-cutlass-dsl`).
> For C++ CUTLASS templates, use `cute-cpp` instead.

## Environment Validation

```bash
python3 -c "import cutlass; print(cutlass.__version__)"
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
python3 tuning/<gpu>/cute-dsl/srcs/<shape_key>/<model>/iter<NNN>_<tag>.py \
    2>&1 | tee tuning/<gpu>/cute-dsl/perf/<shape_key>/<model>/timing_iter<NNN>.txt
```

Script must print: `TFLOPS: <value>   time_ms: <value>`

## PROFILE — ncu Command

```bash
ncu --target-processes all \
    --set full \
    --export tuning/<gpu>/cute-dsl/perf/<shape_key>/<model>/ncu_iter<NNN>.ncu-rep \
    --force-overwrite \
    python3 tuning/<gpu>/cute-dsl/srcs/<shape_key>/<model>/iter<NNN>_<tag>.py

ncu --import tuning/<gpu>/cute-dsl/perf/<shape_key>/<model>/ncu_iter<NNN>.ncu-rep \
    --csv --page raw \
    > tuning/<gpu>/cute-dsl/perf/<shape_key>/<model>/ncu_iter<NNN>.csv
```

**Pre-profiling pin:** Use `cute.compile()` with explicit config params (not the autotuner).

## Pure Implementation Rule

**Allowed:** `@cute.jit`/`cute.compile()`, CuTe copy atoms, MMA atoms, layout
algebra, TMA copy primitives, warp specialization primitives.

**Forbidden:** `cutlass.gemm.device.Gemm`, cuBLAS bindings, `torch.mm`/`torch.bmm`.

## IDEA Menu

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
