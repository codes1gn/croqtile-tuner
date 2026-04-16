---
name: croq-dsl-cuda
description: DSL-specific tuning contract for CUDA kernels. Loaded by croq-tune when dsl=cuda.
---

# Croq-DSL: CUDA

Source extension: `.cu` | Compiler: `nvcc` -> static binary | Group: compiled-binary

## Environment Validation

No additional checks beyond the standard ncu/nvcc/GPU validation.

## BUILD / RUN Templates

**build_iter\<NNN\>.sh:**
```bash
#!/usr/bin/env bash
set -e
nvcc -O3 -arch=sm_90 -std=c++17 -I/usr/local/cuda/include \
     -o tuning/<gpu>/cuda/bin/<shape_key>/<model>/iter<NNN>_<tag> \
     tuning/<gpu>/cuda/srcs/<shape_key>/<model>/iter<NNN>_<tag>.cu \
     2>&1 | tee tuning/<gpu>/cuda/perf/<shape_key>/<model>/build_iter<NNN>.txt
```

**run_iter\<NNN\>.sh:**
```bash
#!/usr/bin/env bash
tuning/<gpu>/cuda/bin/<shape_key>/<model>/iter<NNN>_<tag> \
    2>&1 | tee tuning/<gpu>/cuda/perf/<shape_key>/<model>/timing_iter<NNN>.txt
```

Binary must print: `TFLOPS: <value>   time_ms: <value>`

## PROFILE — ncu Command

```bash
ncu --set full \
    --export tuning/<gpu>/cuda/perf/<shape_key>/<model>/ncu_iter<NNN>.ncu-rep \
    --force-overwrite \
    tuning/<gpu>/cuda/bin/<shape_key>/<model>/iter<NNN>_<tag> [args]

ncu --import tuning/<gpu>/cuda/perf/<shape_key>/<model>/ncu_iter<NNN>.ncu-rep \
    --csv --page raw \
    > tuning/<gpu>/cuda/perf/<shape_key>/<model>/ncu_iter<NNN>.csv
```

## Pure Implementation Rule

**Allowed:** Raw CUDA kernels, CUDA intrinsics (`__shfl_sync`, `__ldg`, etc.),
PTX inline asm (`asm volatile`), cooperative groups, CuTe/CUTLASS atoms only
(copy atoms, MMA atoms, layout algebra).

**Forbidden:** `cublasSgemm`, `cublasGemmEx`, `cutlass::gemm::device::Gemm`,
cuSPARSELt, cuDNN compute calls, PyTorch/TensorFlow ops.

## IDEA Menu

| Bottleneck | Ideas |
|---|---|
| Memory-bound (DRAM) | Increase BM/BN/BK tiles; vectorize loads (`float4`/`uint4`); `cp.async`; `__ldg`; TMA (Hopper) |
| Compute-bound | Reduce smem; decrease register count (`--maxrregcount`); increase threads; split accumulator |
| Latency-bound | Add `cp.async` pipeline stages; `tma.load` on Hopper; interleave independent mem + compute |
| L2 locality | Threadblock swizzle (tile-ID remapping); change CTA launch order |
| Bank conflicts | Pad smem arrays (`+4`/`+8` elements/row); swizzled shared layout |
| Warp divergence | Eliminate conditional branches in hot paths; predicate with masks |

**Key levers:** BM/BN/BK (multiples of 16, typically 64-256), warp count (blockDim),
pipeline stages (1-5), smem padding, register pressure (`#pragma unroll N`, `--maxrregcount`),
vectorized loads, warp primitives, PTX inline (`ldmatrix`, `mma.sync`), TMA.

## VERIFY / MEASURE

The binary must accept `--verify` (or run verify by default) and:
1. Compute with custom kernel
2. Compute reference (cuBLAS in harness, or prior verified iter)
3. Print `VERIFY: PASS` or `VERIFY: FAIL max_abs_err=<v>`

**Verification Tolerance:**

| Precision | base_tol | rel_tol | Notes |
|-----------|----------|---------|-------|
| FP16 input, FP32 accum | 1.0 | 0.01 | Standard |
| FP16 input, FP16 accum | 16.0 | 0.05 | Higher error from FP16 accumulation |
| FP8 E4M3 input, FP16 accum | 0.5 | 0.01 | FP8 inputs are lower magnitude |

**NEVER print "Test Passed" without actual numerical comparison.**
Keep reference tensors alive in host code. Timing via CUDA events (10 warmup + 50 timed).

## BASELINE (iter000)

cuBLAS harness (`.cu`) — calls `cublasGemmEx`. Library calls allowed only in iter000.
