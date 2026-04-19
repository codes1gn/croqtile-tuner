---
name: choreo-kernel-examples
description: Reference collection of Choreo GPU kernel implementations. Use when studying kernel patterns, tiling strategies, warp specialization, TMA/DMA forms, or pipeline staging before editing .co files.
---

# Choreo Kernel Examples

This skill provides reference GPU kernel implementations in Choreo DSL (`.co` files). Study these before writing or editing kernels.

## Required Environment Variables

```bash
export CHOREO_HOME=/home/albert/workspace/croqtile   # choreo compiler repo
export CUTE_HOME=$CHOREO_HOME/extern/cutlass         # CUTLASS headers
export CUDA_HOME=/usr/local/cuda                     # CUDA toolkit
```

**Note**: SM90 kernels with TMA require CUDA 12.9+. SM86/SM80 kernels work with CUDA 12.2+.

## Directory Structure

```
choreo-kernel-examples/
├── matmul/          # Dense FP16/FP32 GEMM (SM90, production-tuned)
├── matmul_bf16/     # BF16 matrix multiplication
├── gemm_sp/         # Sparse GEMM with 2:4 sparsity (SM90, production-tuned)
├── blockscale_gemm/ # FP8 block-scaled GEMM
├── blockscale_gemm_v2/ # FP8 block-scaled GEMM v2
├── bmm/             # Batched matrix multiplication
├── gemv/            # Matrix-vector multiplication
├── hgemm_rc/        # Half-precision GEMM (row-col)
├── hgemm_rr/        # Half-precision GEMM (row-row)
├── fused_moe/       # Mixture-of-experts fused kernel
├── moe_gemm/        # MoE GEMM
├── topk/            # TopK kernel
├── gemm-tests/      # GEMM test kernels (SM86, SM90)
├── gemm-sp-tests/   # Sparse GEMM test kernels (SM80, SM90)
├── matmul-tests/    # MatMul test kernels (various configs)
├── mma-tests/       # MMA/WMMA operation tests
├── copy-tests/      # Copy/TMA operation tests
├── type-tests/      # Data type operation tests
└── misc-tests/      # Misc utility tests
```

## Key Reference Kernels

### Dense GEMM (matmul/, matmul-tests/)

| File | SM | Description |
|------|-----|-------------|
| `matmul/matmul_f16_dyn_sm90.co` | 90 | Baseline FP16 GEMM with TMA+swizzle |
| `matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co` | 90 | 1-producer/1-consumer warp-specialized |
| `matmul/matmul_f16_dyn_sm90_warpspec_1p2c.co` | 90 | 1-producer/2-consumer |
| `matmul/matmul_f16_dyn_sm90_warpspec_1p3c.co` | 90 | 1-producer/3-consumer (wider CTA) |
| `matmul/matmul_f16_dyn_sm86.co` | 86 | **SM86-compatible** (RTX 3070/3080/3090) |
| `matmul-tests/matmul_f16_basic.co` | 90 | Basic FP16 matmul test |
| `matmul-tests/matmul_warpspec.co` | 90 | Warp specialization test |
| `gemm-tests/gemm_sm86.co` | 86 | **SM86-compatible** GEMM test |

### Sparse GEMM (gemm_sp/, gemm-sp-tests/)

| File | SM | Description |
|------|-----|-------------|
| `gemm_sp/gemm_sp_f16_dyn_sm90_warpspec_1p2c_swizzle128_128_prepack.co` | 90 | Production sparse GEMM |
| `gemm_sp/gemm_sp_e4m3_*.co` | 90 | FP8 E4M3 sparse GEMM variants |
| `gemm-sp-tests/gemm_sp_sm80.co` | 80 | **SM80-compatible** (A100) |
| `gemm-sp-tests/gemm_rc_wgmma_sp_wgmma.co` | 90 | WGMMA sparse test |

### MMA/WMMA Operations (mma-tests/)

| File | SM | Description |
|------|-----|-------------|
| `mma_f16.co` | 80+ | FP16 MMA operations |
| `wmma.co`, `wmma_f16.co` | 80+ | WMMA operations |
| `ptx_mma.co` | 80+ | PTX MMA intrinsics |
| `mma_fp8.co` | 90 | FP8 MMA operations (Hopper) |

### Copy/TMA Operations (copy-tests/)

| File | SM | Description |
|------|-----|-------------|
| `copy.co`, `copy_if_g2s.co` | 80+ | Global-to-shared copy |
| `make_tiled_copy.co` | 80+ | Tiled copy patterns |
| `tma_f32.co`, `tma_v2.co` | 90 | TMA operations (Hopper) |

## Common Patterns to Study

### 1. Warp Specialization (1p1c, 1p2c, 1p3c)

```
parallel p1 by 2 : group-4 {
  if (p1 == 0) {
    // Producer: TMA loads
  } else {
    // Consumer: MMA compute
  }
}
```

### 2. Staged Pipeline Events

```
shared event full[STAGES], empty[STAGES];
// Producer
tma.copy.async<full[stage]>.swiz<128> src => dst;
wait empty[stage];
trigger full[stage];
// Consumer
wait full[stage];
mma.op ... ;
trigger empty[stage];
```

### 3. TMA Copy with Swizzle

```
tma.copy.swiz<128> global_tile => shared_tile;
tma.copy.async<event>.swiz<64> global_tile => shared_tile;
```

### 4. Subspan Tile Iteration

```
subspan(buffer, [TILE_M, TILE_K]).step([STRIDE_M, STRIDE_K]).at([i, k])
```

### 5. MMA Operations

```
mma.fill.f16 0.0f;
mma.load.swiz<128> shared_tile;
mma.op <64, 256, 16> accum, a, b;
mma.commit;
mma.store accum, output;
```

## SM Compatibility Guide

| GPU | SM | Compatible folders |
|-----|----|--------------------|
| RTX 3070/3080/3090 | 86 | `matmul/matmul_f16_dyn_sm86.co`, `gemm-tests/gemm_sm86.co`, `mma-tests/`, `copy-tests/` (non-TMA) |
| A100 | 80 | `gemm-sp-tests/gemm_sp_sm80.co`, `mma-tests/`, most `*-tests/` |
| H100/H800 | 90 | All folders |

## Usage

Before writing or editing a `.co` file:

1. Find 2-3 similar kernels in this collection
2. Study their tiling, staging, and warp-specialization patterns
3. Match their syntax and conventions
4. Use `choreo-syntax` skill for DSL grammar details

## Compiling Kernels

```bash
# Set environment
export CHOREO_HOME=/path/to/croqtile  # or choreo repo

# Generate script and run
$CHOREO_HOME/choreo -gs -t cute -arch=sm_86 kernel.co -o output.cute.result
bash output.cute.result --execute

# For SM90 (Hopper):
$CHOREO_HOME/choreo -gs -t cute -arch=sm_90a --use-warpspec --use-prepack \
    kernel.co -o output.cute.result
```

## Related Skills

- `choreo-syntax` — DSL grammar and primitive reference
- `base-tune` — Kernel optimization loop protocol (simple)
- `croq-tune` — Kernel optimization loop protocol (with harness scripts)
