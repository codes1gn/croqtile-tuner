---
name: croq-baseline
description: One-time baseline preparation contract for croq-tune, with persistent workspace path, naming, schema, and first kernel draft.
---

# Croq-Baseline

Use this skill only in `PREPARATION_ONCE`, before round counting starts for a new shape/session.

## Preparation Contract

1. **MANDATORY: Environment Validation** — see "Environment Validation" section below
2. Detect environment and baseline library readiness using:
   - `.claude/skills/croq-baseline/tools/prepare_baseline_env.py`
3. Check baseline readout in the persistent baseline workspace
4. If readout exists, reuse it directly
5. If missing, run baseline readout and persist it before first tuning round
6. **Draft the first kernel source** — see "First Kernel Draft" section below

This preparation is outside tuning round count.

---

## Environment Validation (MANDATORY, BLOCKING)

**BEFORE any tuning work begins**, validate these prerequisites.

**If ANY check fails, auto-fix first before escalating:**

```bash
bash .cursor/skills/croq-env-fix/fix-env.sh
```

Load the `croq-env-fix` skill for details. The fix script handles `perf_event_paranoid`,
CUDA PATH, and nvidia profiling restrictions automatically (may prompt for sudo once).
After `fix-env.sh` completes, re-run validation. Only escalate to user if auto-fix fails.

### 1. ncu Profiling Capability

```bash
# Check ncu is available
which ncu || ls /usr/local/cuda*/bin/ncu

# Check perf_event_paranoid (MUST be <= 2)
cat /proc/sys/kernel/perf_event_paranoid

# Check nvidia driver allows non-root profiling (MUST be 0)
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly

# Test ncu actually works (run on a simple kernel)
ncu --version
```

**If `perf_event_paranoid > 2`:**
- Run `bash .cursor/skills/croq-env-fix/fix-env.sh` (auto-fixes via sudo)
- If auto-fix fails: STOP and tell user to run `sudo sysctl -w kernel.perf_event_paranoid=2`
- DO NOT proceed until fixed

**If `RmProfilingAdminOnly != 0`:**
- Run `bash .cursor/skills/croq-env-fix/fix-env.sh` (installs modprobe config)
- If auto-fix fails: STOP and tell user to reload nvidia modules
- DO NOT proceed until fixed

### 2. CUDA Compiler

```bash
# Check nvcc
which nvcc || ls /usr/local/cuda*/bin/nvcc

# Test compilation
echo 'int main(){}' > /tmp/test.cu && nvcc /tmp/test.cu -o /tmp/test && rm /tmp/test.cu /tmp/test
```

**If nvcc not found:**
- Run `bash .cursor/skills/croq-env-fix/fix-env.sh` (finds and adds CUDA to PATH)
- If auto-fix fails: STOP and tell user "CUDA toolkit not available or misconfigured"
- DO NOT proceed until fixed

### 3. GPU Availability

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

**If nvidia-smi fails:**
- STOP immediately
- Tell user: "No GPU available or driver not loaded"
- DO NOT proceed until fixed (no auto-fix for missing GPU)

### Validation Output

After validation passes, create a validation record:

```json
{
  "validation": "passed",
  "ncu_version": "...",
  "perf_event_paranoid": 2,
  "nvcc_version": "...",
  "gpu_name": "...",
  "validated_at": "ISO-8601"
}
```

Store in `baseline-workspace/_env/validation.json`.

**CRITICAL**: Tuning CANNOT start if validation fails. No fallbacks. No workarounds.

---

## First Kernel Draft (MANDATORY)

Before `croq-tune` enters the round loop, this skill MUST produce the first compilable kernel source:

1. Create `tuning/<gpu>/<dsl>/srcs/<shape_key>/iter001_draft.<cu|co>`
2. The draft must be a **real implementation**, not a library wrapper
3. For `dsl=cuda`: write pure CUDA C++ using CUDA intrinsics, PTX inline asm, or raw kernel code
4. For `dsl=croqtile`: write pure `.co` using Choreo DSL primitives

### What the Draft MUST Contain

- A complete, compilable kernel function for the target operator (e.g., matmul)
- Proper thread/block indexing
- Memory access patterns appropriate for the shape
- Basic tiling strategy (even if naive)

### What the Draft MUST NOT Contain

- Calls to external libraries (cuBLAS, cuTLASS, cuSPARSELt, etc.)
- Wrapper code that delegates compute to library functions
- Framework calls (PyTorch, TensorFlow ops)

Library calls are **only** permitted in baseline measurement (iter000), never in tuning iterations.

### Draft Output Contract

After this skill completes, the following must exist:

- `tuning/<gpu>/<dsl>/srcs/<shape_key>/iter001_draft.<cu|co>` — compilable source
- `tuning/<gpu>/<dsl>/cmd/<shape_key>/build_iter001.sh` — build script
- `tuning/<gpu>/<dsl>/cmd/<shape_key>/run_iter001.sh` — run script
- Checkpoint updated with `next_state: "PROFILE"` pointing to iter001

## Trigger Rules

Run this skill when:

1. entering `PREPARATION_ONCE` for a new `(dsl, operator, dtype, shape_key)`
2. baseline env report is missing or stale
3. user explicitly asks for a dependency setup check

Dependency selection policy:

- default mode is `--libs auto` (agent decides from `operator` and `kernel`)
- user can override with explicit `--libs cublas,torch-cuda,cusparselt`

## CLI Usage

Auto-select libs from target op/kernel:

```bash
python3 .claude/skills/croq-baseline/tools/prepare_baseline_env.py \
  --dsl cuda \
  --operator spmm \
  --kernel sparse_gemm \
  --shape-key f16_4096x8192x8192 \
  --libs auto
```

Explicit library set:

```bash
bash .claude/skills/croq-baseline/tools/prepare_baseline_env.sh \
  --dsl cuda \
  --operator spmm \
  --kernel sparse_gemm \
  --shape-key f16_4096x8192x8192 \
  --libs cublas,torch-cuda,cusparselt
```

Output:

- report file: `baseline-workspace/_env/prep_<timestamp>.json`
- latest symlink-style copy: `baseline-workspace/_env/latest.json`

## Baseline Workspace Contract

Persistent root:

- `baseline-workspace/<dsl>/<operator>/<dtype>/`

Naming:

- `shape_key`: `<operator>_<dtype>_<dimensions>` (for example `matmul_f16_4096x8192x8192`)
- readout file: `readouts/<shape_key>.json`
- optional profile file: `profiles/<shape_key>/ncu_baseline.txt`
- summary table: `tables/readouts.tsv`

### Form Schema (`forms/readout_request.json`)

```json
{
  "schema": "baseline-readout-request-v1",
  "dsl": "cuda",
  "operator": "spmm",
  "dtype": "f16",
  "shape_key": "f16_4096x8192x8192",
  "shape": { "m": 4096, "n": 8192, "k": 8192 },
  "warmup": 10,
  "iters": 50,
  "samples": 5
}
```

### Data Schema (`readouts/<shape_key>.json`)

```json
{
  "schema": "baseline-readout-v1",
  "dsl": "cuda",
  "operator": "spmm",
  "dtype": "f16",
  "shape_key": "f16_4096x8192x8192",
  "shape": { "m": 4096, "n": 8192, "k": 8192 },
  "kernel": "iter000_baseline",
  "verification": { "passed": true, "max_abs_err": 0.0, "max_rel_err": 0.0 },
  "samples": [{ "idx": 1, "time_ms": 0.0, "tflops": 0.0 }],
  "median": { "time_ms": 0.0, "tflops": 0.0 },
  "commands": { "build": "", "run": "", "verify": "" },
  "artifacts": { "stdout": "", "stderr": "", "profile": "" },
  "recorded_at": "ISO-8601"
}
```
