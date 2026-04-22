---
name: croq-dsl-cute-cpp
description: DSL-specific tuning contract for CuTe/CUTLASS C++ template kernels. Loaded by croq-tune when dsl=cute-cpp.
---

# Croq-DSL: CuTe C++ (CUTLASS Templates)

Source extension: `.cu` or `.cpp` | Compiler: `nvcc` (CUTLASS templates) | Group: compiled-binary

> **Note:** This is the **C++** CuTe/CUTLASS template interface.
> For Python JIT via CuTe DSL, use `cute-dsl` instead.

## Environment Validation

```bash
python3 -c "import cutlass"  # optional, for Python template instantiation
```

## BUILD / RUN Templates

**build_iter\<NNN\>.sh:**
```bash
#!/usr/bin/env bash
set -e
nvcc -O3 -arch=sm_90 -std=c++17 -I/usr/local/cuda/include \
     -o tuning/<gpu>/cute-cpp/bin/<shape_key>/<model>/iter<NNN>_<tag> \
     tuning/<gpu>/cute-cpp/srcs/<shape_key>/<model>/iter<NNN>_<tag>.cu \
     2>&1 | tee tuning/<gpu>/cute-cpp/perf/<shape_key>/<model>/build_iter<NNN>.txt
```

**run_iter\<NNN\>.sh:**
```bash
#!/usr/bin/env bash
tuning/<gpu>/cute-cpp/bin/<shape_key>/<model>/iter<NNN>_<tag> \
    2>&1 | tee tuning/<gpu>/cute-cpp/perf/<shape_key>/<model>/timing_iter<NNN>.txt
```

Binary must print: `TFLOPS: <value>   time_ms: <value>`

## PROFILE — ncu Command

```bash
ncu --set full \
    --export tuning/<gpu>/cute-cpp/perf/<shape_key>/<model>/ncu_iter<NNN>.ncu-rep \
    --force-overwrite \
    tuning/<gpu>/cute-cpp/bin/<shape_key>/<model>/iter<NNN>_<tag> [args]

ncu --import tuning/<gpu>/cute-cpp/perf/<shape_key>/<model>/ncu_iter<NNN>.ncu-rep \
    --csv --page raw \
    > tuning/<gpu>/cute-cpp/perf/<shape_key>/<model>/ncu_iter<NNN>.csv
```

## Pure Implementation Rule

**Allowed:** CUTLASS C++ tile primitives, CuTe atoms, raw CUDA intrinsics, PTX.

**Forbidden:** `cutlass::gemm::device::Gemm` (top-level library GEMM), cuBLAS.

## IDEA Menu

| Bottleneck | Ideas |
|---|---|
| Memory-bound | Increase `ThreadblockShape`; add pipeline stages; TMA (Hopper) |
| Compute-bound | Change `WarpShape`; increase `InstructionShape` |
| Latency-bound | Increase `kStages`; use persistent kernel |
| L2 locality | Apply threadblock swizzle epilogue |

## VERIFY / MEASURE

The binary must accept `--verify` (or run verify by default) and:
1. Compute with custom kernel
2. Compute reference (cuBLAS in harness, or prior verified iter)
3. Print `VERIFY: PASS` or `VERIFY: FAIL max_abs_err=<v>`

**Verification Tolerance:**

| Precision | base_tol | rel_tol |
|-----------|----------|---------|
| FP16 input, FP32 accum | 1.0 | 0.01 |
| FP16 input, FP16 accum | 16.0 | 0.05 |
| FP8 E4M3 input, FP16 accum | 0.5 | 0.01 |

**NEVER print "Test Passed" without actual numerical comparison.**
Timing via CUDA events. Default: 10 warmup + 50 timed.

## BASELINE (iter000)

cuBLAS harness (`.cu`) — calls `cublasGemmEx`. Library calls allowed only in iter000.
