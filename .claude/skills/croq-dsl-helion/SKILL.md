---
name: croq-dsl-helion
description: DSL-specific tuning contract for Helion kernels. Loaded by croq-tune when dsl=helion.
---

# Croq-DSL: Helion

Source extension: `.py` | Compiler: Helion -> Triton -> PTX | Group: python-jit

## Environment Validation

```bash
python3 -c "import helion; print(helion.__version__)"
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
python3 tuning/<gpu>/helion/srcs/<shape_key>/<model>/iter<NNN>_<tag>.py \
    2>&1 | tee tuning/<gpu>/helion/perf/<shape_key>/<model>/timing_iter<NNN>.txt
```

Script must print: `TFLOPS: <value>   time_ms: <value>`

## PROFILE — ncu Command

```bash
ncu --target-processes all \
    --set full \
    --export tuning/<gpu>/helion/perf/<shape_key>/<model>/ncu_iter<NNN>.ncu-rep \
    --force-overwrite \
    python3 tuning/<gpu>/helion/srcs/<shape_key>/<model>/iter<NNN>_<tag>.py

ncu --import tuning/<gpu>/helion/perf/<shape_key>/<model>/ncu_iter<NNN>.ncu-rep \
    --csv --page raw \
    > tuning/<gpu>/helion/perf/<shape_key>/<model>/ncu_iter<NNN>.csv
```

**Pre-profiling pin:** `@helion.kernel(autotune_effort="none", config=helion.Config(...))` or `HELION_AUTOTUNE_EFFORT=none`.

## Pure Implementation Rule

**Allowed:** `@helion.kernel` with explicit `helion.Config`, `hl.tile`, `hl.grid`,
`hl.register_buffer`.

**Forbidden:** `torch.ops` compute delegation inside kernel body, xformers/flash-attn
library calls inside Helion kernel.

## IDEA Menu

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
