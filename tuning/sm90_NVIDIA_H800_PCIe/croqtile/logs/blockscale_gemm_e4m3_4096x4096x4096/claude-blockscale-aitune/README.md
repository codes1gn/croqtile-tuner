# Blockscale GEMM E4M3 AI-Tune Full History (2026-03-22)

**Full migration** of all 71 iterations from `ai-tune/2026-03-22/blockscale_gemm_v2`.
All kernels target SM90a (H800 PCIe).

## Summary Table

| Iter | TFLOPS @2k | TFLOPS @4k | Key Optimization |
|------|-----------|-----------|------------------|
| baseline | 314 | 398 | 1-warpgroup warpspec N128 |
| **iter049** | **380** | 504 | TMA-scale overlap (BEST @2k) |
| iter051 | 372 | 602 | N256 WGMMA (+19.4% @4k) |
| iter053 | — | 610 | N256 + L2 RHS |
| **iter066** | — | **621** | N256 + prefetch (BEST @4k) |

**Best @2048³**: iter049 (380 TFLOPS, +21.0% over baseline)
**Best @4096³**: iter066 (621 TFLOPS, +56.1% over baseline)

## Migration Info

- **Source branch**: `origin/ai-tune/2026-03-22/blockscale_gemm_v2`
- **Migration date**: 2026-04-21
- **Strategy**: Full migration (all iterations, not winners-only)

### File Counts

| Type | Count |
|------|-------|
| .cu files | 69 |
| .co files | 17 |
| .md files | 1 |
| **Total** | **87** |

### Key File Mapping

Previous winners-only migration used simplified names. Full migration uses source names:

| Old Name (winners-only) | New Name (full migration) |
|------------------------|---------------------------|
| `iter049_blockscale.cu` | `blockscale_gemm_e4m3_iter049_overlap_tma_scale.cu` |
| `iter051_blockscale.cu` | `blockscale_gemm_e4m3_iter051_n256.cu` |
| `iter053_blockscale.cu` | `blockscale_gemm_e4m3_iter053_n256_l2rhs.cu` |
| `iter066_blockscale.cu` | `blockscale_gemm_e4m3_iter066_n256_prefetch.cu` |

## Build & Run

Standard build for iterations from iter043 onward:

```bash
nvcc -arch sm_90a -O2 \
  -D__CHOREO_TARGET_CUTE__ -D__USE_CUDA_TYPE__ \
  -I runtime -I extern/cutlass/include \
  -lcuda \
  -o <output_binary> \
  <source_file>.cu
```

For Choreo .co files:

```bash
./build/choreo -t cute -arch=sm_90a <source_file>.co -o <output>
```

Execution:

```bash
./<binary> --verify           # correctness + timing
./<binary> --skip-verify       # timing only
./<binary> --disable-timing    # verification only
```

## Detailed Optimization History

See `EXPERIMENT_SUMMARY.md` in `srcs/` directory for complete epoch-by-epoch analysis.
