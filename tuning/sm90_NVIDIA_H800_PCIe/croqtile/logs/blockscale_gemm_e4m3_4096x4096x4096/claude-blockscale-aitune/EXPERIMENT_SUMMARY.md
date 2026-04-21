# Blockscale GEMM v2 — AI-Tune Experiment Summary

**Branch:** `ai-tune/2026-03-22/blockscale_gemm_v2`
**Target:** NVIDIA H100 SXM (SM90a), CUDA 12.8
**Kernel:** FP8 E4M3 block-scaled GEMM with FP16 accumulation
**Problem sizes:** 2048³ and 4096³

---

## Current Best Kernels

| Problem Size | Iteration | WGMMA Config | TFLOPS | HW Eff% | File |
|---|---|---|---|---|---|
| **2048³** | iter049 | M64N128K32 | **380** | 12.5% | `blockscale_gemm_e4m3_iter049_overlap_tma_scale.cu` |
| **4096³** | iter066 | M64N256K32 | **621** | 20.5% | `blockscale_gemm_e4m3_iter066_n256_prefetch.cu` |

---

## Compilation Commands

### Standard Build (all iterations from iter043 onward)
```bash
nvcc -arch sm_90a -O2 \
  -D__CHOREO_TARGET_CUTE__ -D__USE_CUDA_TYPE__ \
  -I runtime -I extern/cutlass/include \
  -lcuda \
  -o <output_binary> \
  <source_file>.cu
```

### Variant: O3 optimization (iter059)
```bash
nvcc -arch sm_90a -O3 \
  -D__CHOREO_TARGET_CUTE__ -D__USE_CUDA_TYPE__ \
  -I runtime -I extern/cutlass/include \
  -lcuda \
  -o <output_binary> <source_file>.cu
```

### Variant: Register limit (iter039)
```bash
nvcc -arch sm_90a -O2 --maxrregcount 64 \
  -D__CHOREO_TARGET_CUTE__ -D__USE_CUDA_TYPE__ \
  -I runtime -I extern/cutlass/include \
  -lcuda \
  -o <output_binary> <source_file>.cu
```

### Earlier iterations (before iter043, no CUDA type defines)
```bash
nvcc -arch sm_90a -O2 \
  -I runtime -I extern/cutlass/include \
  -lcuda \
  -o <output_binary> <source_file>.cu
```

### Choreo .co files (iter003, iter004, iter007, etc.)
```bash
./build/choreo -t cute -arch=sm_90a <source_file>.co -o <output>
```

### Execution (verification + benchmark)
```bash
./<binary> --verify               # correctness check + timing
./<binary> --skip-verify           # timing only
./<binary> --disable-timing        # verification only
```

### Suppress common warnings
```bash
-diag-suppress 2361,177,20054
```

---

## Epoch 1: Baselines & Early Exploration (baseline – iter012)

| Iter | Name | TFLOPS@2k | TFLOPS@4k | Status | Key Insight |
|---|---|---|---|---|---|
| baseline | 1p1c warpspec | 314 | — | — | Starting point: 1-warpgroup warpspec N128 |
| baseline | non-warpspec | 314 | — | — | Same perf without warp specialization |
| baseline | persistent 1p1c | 135 | — | — | Persistent kills perf via WGMMA divergence |
| iter003 | 4-stage pipeline | 242 | — | DISCARD | Extra SMEM reduces occupancy |
| iter004 | N256 WGMMA | 225 | — | DISCARD | 114KB SMEM kills occupancy |
| iter005 | SW pipeline | 199 | — | DISCARD | Register overhead from double-buffer accum |
| iter006 | Deferred commit | 311 | — | DISCARD | Same as baseline |
| iter007 | Large problem | — | 398 | KEEP | 4096³ shows +27% from better SM utilization |
| iter009 | N64 WGMMA | 291 | — | DISCARD | Worse compute-to-memory ratio |
| iter010 | F32 accumulation | 220 | — | DISCARD | 2x register usage kills occupancy |
| iter011 | Single arrive | 318 | — | KEEP? | Marginal +1.1% |
| iter012 | Async TMA double-buf | 231 | — | DISCARD | 49KB SMEM kills occupancy |

**Key learnings:** SMEM > 24KB kills occupancy. WGMMA throughput is the bottleneck.

---

## Epoch 2: TMA & Code Structure (iter013 – iter025)

| Iter | Name | TFLOPS@2k | TFLOPS@4k | Status | Key Insight |
|---|---|---|---|---|---|
| iter013 | Precompute scale | 318 | — | KEEP? | Marginal +1.1% |
| iter019 | TMA overlap | 358 | 424 | KEEP | +14%@2k — concurrent LHS+RHS TMA loads |
| batch-tma | Compiler --batch-tma | 363 | 419 | KEEP | Compiler-level TMA batching |

**Key learnings:** Batching/overlapping TMA loads is the single biggest optimization so far.

---

## Epoch 3: CTA Swizzle & Memory Hierarchy (iter026 – iter036)

| Iter | Name | TFLOPS@2k | TFLOPS@4k | Status | Key Insight |
|---|---|---|---|---|---|
| iter026 | __restrict__ | 356 | — | DISCARD | No effect |
| iter027 | SW pipeline | 274 | — | DISCARD | Register+SMEM pressure |
| iter028 | Explicit sync | 352 | — | DISCARD | Compiler barriers better |
| iter029 | CTA swizzle W=4 | 353 | 465 | KEEP@4k | +10.5%@4k from L2 locality |
| iter030 | Bitwise swizzle | 351 | 405 | DISCARD | Somehow slower |
| iter031 | Combined cleanup | 367 | 408 | KEEP@2k | Best@2k at this point |
| iter032 | Persistent | 363 | 417 | DISCARD | Loop overhead |
| iter033 | Single barrier | 361 | 475 | KEEP@4k | +13.5%@4k, single combined barrier |
| iter034 | Scale prefetch | 364 | — | KEEP? | Marginal +0.8% |
| iter035 | Swizzle8 + L2 promo | — | 462 | DISCARD | W=8 swizzle worse than W=4 |
| iter036 | Combined best | 363 | — | DISCARD | Marginal |

**Key learnings:** CTA swizzle gives big gains at 4096³. Single barrier reduces overhead.

---

## Epoch 4: WGMMA Scheduling & C7520 Battle (iter037 – iter048)

| Iter | Name | TFLOPS@2k | TFLOPS@4k | Status | Key Insight |
|---|---|---|---|---|---|
| iter037 | stmatrix store | 362 | — | DISCARD | Output store is <2% of time |
| iter038 | Scale via SMEM | 339 | — | DISCARD | Extra sync > strided access savings |
| iter039 | maxrregcount=64 | 218 | — | DISCARD | Massive register spilling |
| iter040 | Split WGMMA | 369 | 486 | KEEP | New best both sizes |
| iter041 | Interleave scale | 368 | 486 | DISCARD | Same as iter040 |
| iter042 | Persistent v2 | 145 | — | DISCARD | -60.7%, sequential worse than HW scheduling |
| iter043 | Static double-buf | 211 | — | DISCARD | C7520 WGMMA serialization |
| iter044 | Predicated PTX TMA | 362 | 473 | DISCARD | C7520 fixed but no speedup |
| iter045 | Swizzle W=1 | 375 | 497 | KEEP | New best! Column-first ordering best |
| iter046 | Split commit | 373 | 496 | DISCARD | Per-WGMMA commit no help |
| iter047 | W=1 pipeline | 270 | — | DISCARD | 2x SMEM halves occupancy |
| iter048 | Prefetch scale_a | 368 | — | DISCARD | L1 cache already handles it |

**Key learnings:** C7520 (WGMMA serialization from divergent `__CHOREO_BLOCK_SINGLE__`) blocks all pipelining approaches. Swizzle W=1 best for these sizes.

---

## Epoch 5: TMA-Scale Overlap & N256 Breakthrough (iter049 – iter057)

| Iter | Name | TFLOPS@2k | TFLOPS@4k | Status | Key Insight |
|---|---|---|---|---|---|
| iter049 | TMA-scale overlap | 380 | 504 | **KEEP** | **Best@2k!** TMA for next block overlaps with scale_accumulator |
| iter050 | ScaleOut::Zero | 382 | — | DISCARD | Same as iter049 |
| iter051 | N256 WGMMA | 372 | 602 | KEEP@4k | +19.4%@4k! Doubled compute density |
| iter052 | TMA L2 prefetch | 301 | — | DISCARD | L2 contention |
| iter053 | N256 + L2 RHS | — | 610 | KEEP@4k | +1.3% from L2_256B on RHS |
| iter054 | K-loop unroll 2 | 381 | 612 | DISCARD | Compiler already optimizes |
| iter055 | Register pipeline | 167 | — | DISCARD | -56%, stack spilling catastrophe |
| iter056 | N64 tile | 350 | — | DISCARD | Halved compute density |
| iter057 | Direct global store | 315 | — | DISCARD | Uncoalesced stores |

**Key learnings:** N256 WGMMA is transformative for 4096³ (+19%). TMA-scale overlap gives the final +1.3% for N128.

---

## Epoch 6: Diminishing Returns (iter058 – iter071)

| Iter | Name | TFLOPS@2k | TFLOPS@4k | Status | Key Insight |
|---|---|---|---|---|---|
| iter058 | K-tile 256 | 300 | — | DISCARD | 49KB SMEM kills occupancy |
| iter059 | -O3 flag | 377 | — | DISCARD | O3 no better than O2 |
| iter060 | L2 both tensors | 378 | — | DISCARD | No benefit at 2048³ |
| iter061 | Delayed scale | 372 | — | DISCARD | Register pressure from 2x fragments |
| iter062 | 2-warpgroup M128 | 244 | — | DISCARD | C7520 serialization |
| iter063 | Scale prefetch | 380 | 507 | DISCARD | Marginal |
| iter064 | Persistent N128 | 362 | 513 | DISCARD | Tile loop overhead |
| iter065 | N256 persistent | — | 561 | DISCARD | Breaks L2 locality |
| iter066 | N256 + prefetch | — | **621** | **KEEP@4k** | **Best@4k!** Scale prefetch +2.8% |
| iter067 | N256 L2 both | — | 610 | DISCARD | L2 on LHS hurts |
| iter068 | Shfl scale broadcast | 357 | — | DISCARD | Shfl latency + divergence |
| iter069 | Precomputed descs | 380 | — | DISCARD | Compiler already hoists |
| iter070 | Combined micro-opts | 380 | — | DISCARD | Compute-bound ceiling |
| iter071 | Split-K atomic | 197 | — | DISCARD | FP16 atomicAdd too slow |

**Key learnings:** N128@2k is firmly compute-bound at WGMMA throughput (~380 TFLOPS ceiling). N256 prefetch gives the last +2.8%@4k.

---

## Performance Progression

```
2048³ (N128):  314 → 358 → 369 → 375 → 380 TFLOPS  (+21% total)
4096³ (N128):        424 → 486 → 497 → 504 TFLOPS
4096³ (N256):                          602 → 610 → 621 TFLOPS  (+47% total vs initial 4k)
```

## Blockers for Further Improvement

1. **C7520 WGMMA serialization** — Any multi-warpgroup approach with `__CHOREO_BLOCK_SINGLE__` TMA divergence causes WGMMA serialization. Blocks pipelining and M128 tiles.
2. **SMEM occupancy cliff** — Beyond 24KB (N128) or 40KB (N256) SMEM, occupancy drops catastrophically.
3. **WGMMA compute bound** — At 380 TFLOPS@2k, the kernel is at ~38% of FP8 peak, limited by WGMMA instruction throughput and blockscale overhead.
4. **FP16 atomic penalty** — Split-K not viable due to expensive FP16 atomicAdd.

---

## File Manifest

- `results.tsv` — Full iteration tracking (TSV format)
- `EXPERIMENT_SUMMARY.md` — This document
- `blockscale_gemm_e4m3_iter*.cu` / `*.co` — All kernel source artifacts (71 iterations)
- `iter*` (binaries) — Compiled benchmark binaries (staged, not committed as binaries)
