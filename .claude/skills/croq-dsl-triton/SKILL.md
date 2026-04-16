---
name: croq-dsl-triton
description: DSL-specific tuning contract for Triton kernels. Loaded by croq-tune when dsl=triton.
---

# Croq-DSL: Triton

Source extension: `.py` | Compiler: Triton JIT -> PTX via LLVM | Group: python-jit

## Environment Validation

```bash
python3 -c "import triton; print(triton.__version__)"
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
python3 tuning/<gpu>/triton/srcs/<shape_key>/<model>/iter<NNN>_<tag>.py \
    2>&1 | tee tuning/<gpu>/triton/perf/<shape_key>/<model>/timing_iter<NNN>.txt
```

Script must print: `TFLOPS: <value>   time_ms: <value>`

## PROFILE — ncu Command

```bash
ncu --target-processes all \
    --set full \
    --export tuning/<gpu>/triton/perf/<shape_key>/<model>/ncu_iter<NNN>.ncu-rep \
    --force-overwrite \
    python3 tuning/<gpu>/triton/srcs/<shape_key>/<model>/iter<NNN>_<tag>.py

ncu --import tuning/<gpu>/triton/perf/<shape_key>/<model>/ncu_iter<NNN>.ncu-rep \
    --csv --page raw \
    > tuning/<gpu>/triton/perf/<shape_key>/<model>/ncu_iter<NNN>.csv
```

**Pre-profiling pin:** Remove `@triton.autotune` decorator and use the best config's constants directly.

## Pure Implementation Rule

**Allowed:** `@triton.jit` kernels, `tl.load/store/dot/atomic_*`, warp-level
intrinsics, `tl.inline_ptx_asm`, fixed-config launchers (no `@triton.autotune` in iter001+).

**Forbidden:** `torch.ops.*` compute delegation, `triton.ops.matmul`, flash-attn
package calls. `@triton.autotune` is allowed only in iter000.

## IDEA Menu

| Bottleneck | Ideas |
|---|---|
| Memory-bound | Increase `BLOCK_M/N/K`; add `num_stages`; `eviction_policy="evict_last"` |
| Compute-bound | Reduce register pressure; tune `num_warps` (4/8); `tl.dot(allow_tf32=True)` |
| Latency-bound | Increase `num_stages`; prefetch hints; warp specialization (`warp_specialize=True`) |
| Launch-bound | Persistent kernel: grid-stride loop; operator fusion |
| Swizzle / L2 | Program-ID swizzle: `pid = (pid % GROUP_M) * (N // BLOCK_N) + pid // GROUP_M` |

**Key levers:** `BLOCK_M/N/K` (32-256, powers of 2), `num_warps` (4/8/16),
`num_stages` (1-7), `num_ctas` (multi-CTA, Hopper), `allow_tf32`, swizzle, warp spec.

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
