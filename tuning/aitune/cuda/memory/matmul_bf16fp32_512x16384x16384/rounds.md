# Tuning Rounds: matmul_bf16fp32_512x16384x16384

## iter001 - 2026-04-14T10:04:00+00:00
- kernel: `iter001_draft_wmma`
- tflops: `9.91`
- decision: **DISCARD**
- bottleneck: `latency_bound`
- idea: First WMMA bf16 kernel draft. 5-way bank conflict (79.56%).

## iter002 - 2026-04-14T10:05:00+00:00
- kernel: `iter002_no_bank_conflict`
- tflops: `18.90`
- decision: **KEEP**
- bottleneck: `latency_bound`
- idea: Fix shared memory bank conflicts with padding (+8 bf16 per row). +90.7% gain.

## iter003 - 2026-04-14T10:06:00+00:00
- kernel: `iter003_low_register`
- tflops: `12.96`
- decision: **DISCARD**
- bottleneck: `compute_bound`
- idea: Reduce register pressure (126->64), larger block (512 threads). -31.4%.

## iter004 - 2026-04-14T10:07:00+00:00
- kernel: `iter004_vectorized_ldg`
- tflops: `14.40`
- decision: **DISCARD**
- bottleneck: `compute_bound`
- idea: Vectorized bf162 loads with __ldg. Complex bounds checking hurts perf. -23.8%.

## iter005 - 2026-04-14T10:08:00+00:00
- kernel: `iter005_double_buffer`
- tflops: `15.73`
- decision: **DISCARD**
- bottleneck: `compute_bound`
- idea: Double buffering to hide latency. Extra register/smem pressure. -16.8%.

## iter006 - 2026-04-14T10:09:00+00:00
- kernel: `iter006_large_tile`
- tflops: `2.73`
- decision: **DISCARD**
- bottleneck: `compute_bound`
- idea: 4x4 WMMA tiles per warp. 252 registers -> spill to local mem. -85.6%.

## iter007 - 2026-04-14T10:10:00+00:00
- kernel: `iter007_k32`
- tflops: `9.88`
- decision: **DISCARD**
- bottleneck: `memory_bound`
- idea: K-tile 32. More loads per compute. -47.7%.

## attempt0005 - 2026-04-14T10:11:00+00:00
- kernel: `iter008_async`
- tflops: `0.00`
- decision: **SEGFAULT**
- bottleneck: `n/a`
- idea: cp.async single bf16 load - alignment issue.

## iter009 - 2026-04-14T11:12:00+00:00
- kernel: `iter009_coalesced_load`
- tflops: `6.17`
- decision: **DISCARD**
- bottleneck: `compute_bound`
- idea: Complex load pattern with branches. -67.4%.

## iter010 - 2026-04-14T11:13:00+00:00
- kernel: `iter010_2warp_3x2`
- tflops: `5.73`
- decision: **DISCARD**
- bottleneck: `compute_bound`
- idea: 2 warps, 3x2 tiles each. 162 regs, small grid. -69.7%.

## iter011 - 2026-04-14T11:14:00+00:00
- kernel: `iter011_swizzle`
- tflops: `15.20`
- decision: **DISCARD**
- bottleneck: `compute_bound`
- idea: XOR swizzle for bank conflict. WMMA needs contiguous mem. -19.6%.

## iter012 - 2026-04-14T11:15:00+00:00
- kernel: `iter012_simple_opt`
- tflops: `11.14`
- decision: **DISCARD**
- bottleneck: `compute_bound`
- idea: launch_bounds(128,4) forces 104 regs. Register spill. -41.1%.

## attempt0006 - 2026-04-14T11:16:00+00:00
- kernel: `iter013_warp_spec`
- tflops: `0.00`
- decision: **HANG**
- bottleneck: `n/a`
- idea: Warp specialization with atomicExch sync. Deadlock.

## iter014 - 2026-04-14T11:17:00+00:00
- kernel: `iter014_larger_output`
- tflops: `11.17`
- decision: **DISCARD**
- bottleneck: `compute_bound`
- idea: 128x64 tile, 8 warps. More warps but worse perf. -40.9%.

## iter015 - 2026-04-14T11:18:00+00:00
- kernel: `iter015_tall_shape`
- tflops: `9.61`
- decision: **DISCARD**
- bottleneck: `compute_bound`
- idea: 32x128 tile for tall shape. A broadcast inefficient. -49.2%.

## iter016 - 2026-04-14T11:19:00+00:00
- kernel: `iter016_pad4`
- tflops: `18.72`
- decision: **DISCARD**
- bottleneck: `latency_bound`
- idea: PAD=4 (vs PAD=8). Slightly worse. -1.0%.

## iter017 - 2026-04-14T11:20:00+00:00
- kernel: `iter017_warp_tile_64`
- tflops: `4.54`
- decision: **DISCARD**
- bottleneck: `compute_bound`
- idea: 64x64 per warp. 255 regs + 1 warp/block. -76.0%.

## iter018 - 2026-04-14T11:21:00+00:00
- kernel: `iter018_ldmatrix`
- tflops: `18.62`
- decision: **DISCARD**
- bottleneck: `latency_bound`
- idea: __align__(16) shared mem. No improvement. -1.5%.

## iter019 - 2026-04-14T11:22:00+00:00
- kernel: `iter019_ptx_mma`
- tflops: `11.62`
- decision: **DISCARD**
- bottleneck: `compute_bound`
- idea: Direct PTX mma.sync. K-loop incomplete. -38.5%.

## iter020 - 2026-04-14T11:23:00+00:00
- kernel: `iter020_large_tile`
- tflops: `9.87`
- decision: **DISCARD**
- bottleneck: `register_spill`
- idea: 128x128 tile too large. Register pressure. -47.8%.

## iter021 - 2026-04-14T12:24:00+00:00
- kernel: `iter021_k32`
- tflops: `10.24`
- decision: **DISCARD**
- bottleneck: `smem_bound`
- idea: K=32 increases smem. No compute benefit. -45.8%.

## iter022 - 2026-04-14T12:25:00+00:00
- kernel: `iter022_bcol`
- tflops: `19.51`
- decision: **KEEP**
- bottleneck: `latency_bound`
- idea: B col-major in smem. +3.2% from iter002.

## iter023 - 2026-04-14T12:26:00+00:00
- kernel: `iter023_bcol_vec`
- tflops: `19.35`
- decision: **DISCARD**
- bottleneck: `latency_bound`
- idea: Added __ldg. No improvement. -0.8%.

## iter024 - 2026-04-14T12:27:00+00:00
- kernel: `iter024_8warps`
- tflops: `17.65`
- decision: **DISCARD**
- bottleneck: `register_pressure`
- idea: 8 warps. Occupancy drop. -9.5%.

## iter025 - 2026-04-14T12:28:00+00:00
- kernel: `iter025_prefetch`
- tflops: `17.84`
- decision: **DISCARD**
- bottleneck: `smem_pressure`
- idea: Double buffer prefetch. smem 2x. -8.6%.

## iter026 - 2026-04-14T12:29:00+00:00
- kernel: `iter026_32x64`
- tflops: `13.28`
- decision: **DISCARD**
- bottleneck: `coverage_issue`
- idea: 32x64 tile. Wrong warp coverage. -31.9%.

## iter027 - 2026-04-14T12:30:00+00:00
- kernel: `iter027_unroll`
- tflops: `20.81`
- decision: **KEEP**
- bottleneck: `latency_bound`
- idea: No bounds check. Direct store. +6.7%.

## iter028 - 2026-04-14T12:31:00+00:00
- kernel: `iter028_bcol_unroll`
- tflops: `20.97`
- decision: **KEEP**
- bottleneck: `latency_bound`
- idea: B col-major + no bounds. +0.8%.

## iter029 - 2026-04-14T12:32:00+00:00
- kernel: `iter029_pad4`
- tflops: `21.62`
- decision: **KEEP**
- bottleneck: `latency_bound`
- idea: SMEM_PAD=4 (from 8). +3.1%.

## iter030 - 2026-04-14T12:33:00+00:00
- kernel: `iter030_pad2`
- tflops: `21.11`
- decision: **DISCARD**
- bottleneck: `bank_conflict`
- idea: SMEM_PAD=2 too small. -2.4%.

## iter031 - 2026-04-14T12:34:00+00:00
- kernel: `iter031_occ8`
- tflops: `2.39`
- decision: **DISCARD**
- bottleneck: `register_spill`
- idea: launch_bounds 8. Catastrophic spill. -88.9%.

## iter032 - 2026-04-14T12:35:00+00:00
- kernel: `iter032_48x64`
- tflops: `SEGFAULT`
- decision: **DISCARD**
- bottleneck: `bounds_error`
- idea: 48x64 block. M=512 not divisible by 48.

## iter033 - 2026-04-14T13:36:00+00:00
- kernel: `iter033_grid_swap`
- tflops: `21.65`
- decision: **KEEP**
- bottleneck: `latency_bound`
- idea: Grid (M,N) swap. +0.1%.

## iter034 - 2026-04-14T13:37:00+00:00
- kernel: `iter034_nosync`
- tflops: `21.47`
- decision: **DISCARD**
- bottleneck: `overhead`
- idea: Restructured K-loop. More code. -0.8%.

## iter035 - 2026-04-14T13:38:00+00:00
- kernel: `iter035_cg_sync`
- tflops: `21.38`
- decision: **DISCARD**
- bottleneck: `overhead`
- idea: cg::sync slower than __syncthreads. -1.2%.

## iter036 - 2026-04-14T13:39:00+00:00
- kernel: `iter036_restrict`
- tflops: `22.10`
- decision: **KEEP**
- bottleneck: `latency_bound`
- idea: extern smem + __restrict__. +2.1%.

## iter037 - 2026-04-14T13:40:00+00:00
- kernel: `iter037_nopad`
- tflops: `18.33`
- decision: **DISCARD**
- bottleneck: `bank_conflict`
- idea: SMEM_PAD=0 severe conflicts. -17.1%.

## iter038 - 2026-04-14T13:41:00+00:00
- kernel: `iter038_maxreg`
- tflops: `21.73`
- decision: **DISCARD**
- bottleneck: `reg_limit`
- idea: --maxrregcount=64. -1.7%.

## iter039 - 2026-04-14T13:42:00+00:00
- kernel: `iter039_vec_store`
- tflops: `21.65`
- decision: **DISCARD**
- bottleneck: `overhead`
- idea: smem staging for C. -2.0%.

## iter040 - 2026-04-14T13:43:00+00:00
- kernel: `iter040_unroll2`
- tflops: `21.17`
- decision: **DISCARD**
- bottleneck: `code_size`
- idea: Manual K-loop unroll x2. -4.2%.

## iter041 - 2026-04-14T13:44:00+00:00
- kernel: `iter041_swizzle`
- tflops: `20.88`
- decision: **DISCARD**
- bottleneck: `correctness`
- idea: XOR swizzle breaks WMMA layout. -5.5%.

## iter042 - 2026-04-14T13:45:00+00:00
- kernel: `iter042_min_smem`
- tflops: `1.57`
- decision: **DISCARD**
- bottleneck: `register_spill`
- idea: 32x32 block + high occ. Severe spill. -92.9%.

## iter043 - 2026-04-14T13:46:00+00:00
- kernel: `iter043_splitk`
- tflops: `20.99`
- decision: **DISCARD**
- bottleneck: `atomic_overhead`
- idea: Split-K=4. Atomic add overhead. -5.0%.

## iter044 - 2026-04-14T13:47:00+00:00
- kernel: `iter044_maxreg128`
- tflops: `22.42`
- decision: **KEEP**
- bottleneck: `latency_bound`
- idea: iter036 + --maxrregcount=128. +1.4%.

## iter045 - 2026-04-14T14:48:00+00:00
- kernel: `iter045_maxreg96`
- tflops: `22.34`
- decision: **DISCARD**
- bottleneck: `reg_limit`
- idea: maxrregcount=96. -0.4%.

## iter046 - 2026-04-14T14:49:00+00:00
- kernel: `iter046_maxreg160`
- tflops: `21.38`
- decision: **DISCARD**
- bottleneck: `low_occ`
- idea: maxrregcount=160. -4.6%.

## iter047 - 2026-04-14T14:50:00+00:00
- kernel: `iter047_maxreg112`
- tflops: `22.65`
- decision: **KEEP**
- bottleneck: `latency_bound`
- idea: maxrregcount=112. +1.0%.

## iter048 - 2026-04-14T14:51:00+00:00
- kernel: `iter048_maxreg104`
- tflops: `21.73`
- decision: **DISCARD**
- bottleneck: `reg_limit`
- idea: maxrregcount=104. -4.1%.

## iter049 - 2026-04-14T14:52:00+00:00
- kernel: `iter049_maxreg120`
- tflops: `21.86`
- decision: **DISCARD**
- bottleneck: `reg_limit`
- idea: maxrregcount=120. -3.5%.

## iter050 - 2026-04-14T14:53:00+00:00
- kernel: `iter050_maxreg108`
- tflops: `21.71`
- decision: **DISCARD**
- bottleneck: `reg_limit`
- idea: maxrregcount=108. -4.2%.

## iter051 - 2026-04-14T14:54:00+00:00
- kernel: `iter051_maxreg116`
- tflops: `22.30`
- decision: **DISCARD**
- bottleneck: `reg_limit`
- idea: maxrregcount=116. -1.5%.

## iter052 - 2026-04-14T14:55:00+00:00
- kernel: `iter052_fastmath`
- tflops: `21.78`
- decision: **DISCARD**
- bottleneck: `precision`
- idea: --use_fast_math. -3.8%.

## iter053 - 2026-04-14T14:56:00+00:00
- kernel: `iter053_lb3`
- tflops: `22.45`
- decision: **DISCARD**
- bottleneck: `launch_bounds`
- idea: launch_bounds(128,3). -0.9%.

## iter054 - 2026-04-14T14:57:00+00:00
- kernel: `iter054_persistent`
- tflops: `16.39`
- decision: **DISCARD**
- bottleneck: `loop_overhead`
- idea: Persistent kernel. Grid-stride overhead. -27.6%.

## iter055 - 2026-04-14T14:58:00+00:00
- kernel: `iter055_6warps`
- tflops: `5.61`
- decision: **DISCARD**
- bottleneck: `reg_pressure`
- idea: 64x96 with 6 warps. Severe issues. -75.2%.

## iter056 - 2026-04-14T14:59:00+00:00
- kernel: `iter056_funroll`
- tflops: `22.46`
- decision: **DISCARD**
- bottleneck: `no_effect`
- idea: -Xcompiler -funroll-loops. -0.8%.

## iter057 - 2026-04-14T15:00:00+00:00
- kernel: `iter057_vec`
- tflops: `21.77`
- decision: **DISCARD**
- bottleneck: `degradation`
- idea: -extra-device-vectorization. -3.9%.

## iter058 - 2026-04-14T15:01:00+00:00
- kernel: `iter058_cpasync`
- tflops: `-`
- decision: **SEGFAULT**
- bottleneck: `mem_error`
- idea: cp.async single element. SEGFAULT.

## iter059 - 2026-04-14T15:02:00+00:00
- kernel: `iter059_cpasync16`
- tflops: `20.42`
- decision: **DISCARD**
- bottleneck: `bank_conflict`
- idea: cp.async 16B + no pad. -9.8%.

## iter060 - 2026-04-14T15:03:00+00:00
- kernel: `iter060_4warp_4x4`
- tflops: `20.61`
- decision: **DISCARD**
- bottleneck: `reg_pressure`
- idea: 128x64 block, 8 tiles/warp. -9.0%.

## iter061 - 2026-04-14T15:04:00+00:00
- kernel: `iter061_32x64_occ`
- tflops: `16.23`
- decision: **DISCARD**
- bottleneck: `reg_pressure`
- idea: 32x64 2 warps, 163 regs. -28.3%.

## iter062 - 2026-04-14T15:05:00+00:00
- kernel: `iter062_1tile_warp`
- tflops: `5.60`
- decision: **DISCARD**
- bottleneck: `grid_overhead`
- idea: 32x32 1 tile/warp, too many blocks. -75.3%.

## iter063 - 2026-04-14T15:06:00+00:00
- kernel: `iter063_prefetch_reg`
- tflops: `21.63`
- decision: **DISCARD**
- bottleneck: `smem_overhead`
- idea: Double buffer prefetch, 2x smem. -4.5%.

## iter064 - 2026-04-14T15:07:00+00:00
- kernel: `iter064_pragma`
- tflops: `21.15`
- decision: **DISCARD**
- bottleneck: `code_size`
- idea: #pragma unroll 4 on K-loop. -6.6%.

## iter065 - 2026-04-14T15:08:00+00:00
- kernel: `iter065_cache_config`
- tflops: `22.23`
- decision: **DISCARD**
- bottleneck: `cache_config`
- idea: cudaFuncCachePreferShared. -1.9%.

## iter066 - 2026-04-14T15:09:00+00:00
- kernel: `iter066_prefer_l1`
- tflops: `8.94`
- decision: **DISCARD**
- bottleneck: `cache_config`
- idea: cudaFuncCachePreferL1. -60.5%.

## iter067 - 2026-04-14T15:10:00+00:00
- kernel: `iter067_aligned`
- tflops: `21.89`
- decision: **DISCARD**
- bottleneck: `alignment`
- idea: Static smem + 128B align. -3.4%.

## iter068 - 2026-04-14T15:11:00+00:00
- kernel: `iter068_warp_sync`
- tflops: `21.94`
- decision: **DISCARD**
- bottleneck: `sync_overhead`
- idea: __syncwarp() added. -3.1%.

## iter069 - 2026-04-14T16:12:00+00:00
- kernel: `iter069_ldg`
- tflops: `21.34`
- decision: **DISCARD**
- bottleneck: `no_benefit`
- idea: __ldg() for global loads. -5.8%.

## iter070 - 2026-04-14T16:13:00+00:00
- kernel: `iter070_k32`
- tflops: `17.04`
- decision: **DISCARD**
- bottleneck: `smem_reg_pressure`
- idea: BLOCK_K=32. -24.8%.

## iter071 - 2026-04-14T16:14:00+00:00
- kernel: `iter071_stream_k`
- tflops: `22.12`
- decision: **DISCARD**
- bottleneck: `memset_overhead`
- idea: Stream-K 2-split. Avg 22.12 vs 22.65.

## iter072 - 2026-04-14T16:15:00+00:00
- kernel: `iter072_2x1_tiles`
- tflops: `14.92`
- decision: **DISCARD**
- bottleneck: `grid_overhead`
- idea: 64x32 block, 2x1 tiles. -34.1%.

## iter073 - 2026-04-14T16:16:00+00:00
- kernel: `iter073_asm_ld`
- tflops: `-`
- decision: **SEGFAULT**
- bottleneck: `alignment`
- idea: PTX 128-bit global load. SEGFAULT.
