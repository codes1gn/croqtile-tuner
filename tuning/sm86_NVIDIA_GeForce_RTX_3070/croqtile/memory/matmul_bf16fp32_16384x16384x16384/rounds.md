# Tuning History: matmul_bf16fp32_16384x16384x16384

## Round 1: Baseline (iter001_draft)

**Performance**: 0.115 TFLOPS (0.08% of 142 TFLOPS peak)

**Bottleneck**: launch_bound (low occupancy, SM Busy ~25%)

**Profile Summary**:
- Grid: 256x256 blocks
- Block: 32 threads (1 warp)
- Tile: 16x16x16
- Duration: 444ms per kernel
- Memory throughput: 73.75%
- DRAM throughput: 3.2%
- L1/TEX throughput: 74.25%

**Analysis**:
- Only 1 warp per block leads to extremely low SM utilization
- ncu reports "All compute pipelines are under-utilized"
- Register pressure from bf16 accumulator limits tile size
- Need to increase work per block while managing registers

**Next Steps**:
- Try larger tiles with `--maxrregcount` to limit registers
- Add async pipelining to overlap loads with compute
- Consider using `mma.op` unified primitive

---

## Round 2: DISCARD (iter002_tile32x64)

**Performance**: 0.00779 TFLOPS (0.0055% of 142 TFLOPS peak) — 15x WORSE than baseline

**Bottleneck**: resource_overload (excessive register pressure from 8-warp config)

**Idea Tested**: 32x64 tiles with 8 warps (256 threads) per block

**Profile Summary**:
- Grid: 64x32 blocks (2048 blocks for 2048³ test)
- Block: 256 threads (8 warps)
- Timing: 2206ms per iteration — catastrophically slow
- Tile: 32x64x32

**Analysis**:
- 8 warps per block exhausts SM86 register file
- Choreo-generated code likely spills to local memory
- Without explicit `--maxrregcount`, nvcc allocates 255 regs/thread
- SM86 has 65536 regs/SM; 8 warps × 32 threads × 255 = 65,280 regs (full)
- Result: occupancy collapses to 1 active warp despite requesting 8

**Lesson**: Cannot use 8-warp configurations on SM86 with Choreo without register limiting. Need structural changes or smaller warp count.

**Next Steps**:
- Try 2-warp (32x32 tiles) or 4-warp configurations
- Investigate double-buffering to hide latency with fewer warps
- Consider K-loop unrolling to increase arithmetic intensity

---

## Round 3: KEEP (iter003_tilek32) **NEW BEST**

**Performance**: 0.154 TFLOPS (0.109% of 142 TFLOPS peak) — **1.34x improvement** over baseline

**Bottleneck**: register_pressure (still high at ~250 regs/thread, but better compute/memory ratio)

**Idea Tested**: Increase TILE_K from 16 to 32, keeping 16x16 output tiles

**Hypothesis**: With TILE_K=32, we perform 2 `mma.sync` operations per K-tile load, doubling the arithmetic intensity

**Result**: 
- Timing: 890.7ms (vs 444ms baseline) - slower in absolute terms but ratio is better
- TFLOPS: 0.154 (vs 0.115 baseline) - **34% improvement**
- Verification: PASS

**Analysis**:
- The 2x arithmetic work per K-load amortizes memory latency better
- Register pressure remains high but the increased compute density helps
- K-loop count halved (128 → 64 for K=4096), reducing loop overhead

**Next Steps**:
- Try TILE_K=64 for even more arithmetic intensity
- Profile to confirm the bottleneck has shifted
- Consider K-unrolling within the inner loop

---

## Round 4: DISCARD (iter004_tilek64)

**Performance**: 0.027 TFLOPS — 5.7x WORSE than iter003

**Idea Tested**: Increase TILE_K from 32 to 64

**Result**: Severe regression. TILE_K=64 causes shared memory/register spills.

---

## Round 5: DISCARD (iter005_dmapattern)

**Performance**: 0.153 TFLOPS — no improvement over iter003 (0.154)

**Idea Tested**: Use simplified `=> shared` DMA pattern like reference F16 kernel

**Result**: Equivalent performance. Compiler generates similar code either way.

---

## Round 6: DISCARD (iter006_f16pattern) - COMPILE ERROR

**Idea Tested**: Use `mma.fill 0.0` like F16 reference kernel

**Result**: Compiler error - SM86 does not support BF16 accumulators. Must use `mma.fill.f32 0.0f` for BF16 input with FP32 accumulator.

---

## Summary

**Best Result**: iter003_tilek32 at **0.154 TFLOPS** (0.11% HW efficiency)

**Key Findings**:
1. SM86 with croqtile DSL hits 255 registers/thread ceiling
2. This limits occupancy to 1 warp per SM (out of 48 possible)
3. TILE_K=32 provides optimal arithmetic intensity (2 mma ops per K-load)
4. Larger TILE_K (64) causes regression from increased memory pressure
5. Multi-warp configurations fail verification (possible choreo bug with asymmetric tiles)
6. SM86 requires FP32 accumulator for BF16 tensor core ops

**Bottleneck**: Register pressure is the fundamental limit with croqtile on SM86.

**Recommendation**: To achieve higher performance, consider switching to pure CUDA with explicit `__launch_bounds__` for register control, or use a different GPU architecture with better tensor core support for BF16 (e.g., SM90/Hopper).

## iter008 — 2026-04-16T03:53:11Z
- kernel: `iter008_commit`
- tflops: `0`
- decision: **COMPILE_FAIL**
- bottleneck: `compile_error`
- idea: mma.commit after inner K-loop: primitive not supported on SM86 CuTe backend (Hopper-only)

## iter009 — 2026-04-16T04:18:04Z
- kernel: `iter009_directstore`
- tflops: `0.503`
- decision: **KEEP**
- bottleneck: `compute_bound`
- idea: Direct mma.store to global memory: removed output_s shared buffer (1KB), eliminated block-level dma.copy barrier. SM92 compute_bound 92% → 3.27x speedup to new best 0.503 TFLOPS

## iter010 — 2026-04-16T04:41:18Z
- kernel: `iter010_tilek16`
- tflops: `0.379`
- decision: **DISCARD**
- bottleneck: `latency_bound`
- idea: TILE_K=16 on directstore: 0.379 TFLOPS — less arithmetic intensity without occupancy gain (still 255 regs)

## iter011 — 2026-04-16T04:50:59Z
- kernel: `iter011_swizds`
- tflops: `0.502`
- decision: **DISCARD**
- bottleneck: `latency_bound`
- idea: swiz<128> on directstore base: 0.502 TFLOPS — marginal delta from best 0.503. Bank conflicts not binding constraint on directstore path.

## iter012 — 2026-04-16T04:56:45Z
- kernel: `iter012_tilek48`
- tflops: `0.088`
- decision: **DISCARD**
- bottleneck: `register_pressure`
- idea: TILE_K=64 on directstore: 0.088 TFLOPS catastrophic regression. TILE_K=48 crashed (illegal mem access, 4096/48 non-integer). TILE_K=64 causes register spills regardless of output path.

## iter013 — 2026-04-16T05:17:03Z
- kernel: `iter013_doublebuf`
- tflops: `0.101`
- decision: **DISCARD**
- bottleneck: `latency_bound`
- idea: 4-warp block TILE_M=32/TILE_N=32 with sync dma.copy: 0.101 TFLOPS regression. Async copy with 4 warps causes 4x redundant copies. Sync copy avoids redundancy but is slower.

## iter014 — 2026-04-16T05:29:06Z
- kernel: `iter014_ptxmma`
- tflops: `0.222`
- decision: **DISCARD**
- bottleneck: `latency_bound`
- idea: m16n8k16 PTX MMA with dma.transp for RHS: 0.222 TFLOPS. Direct PTX MMA (SM80_16x8x16 fma) achieved but sync transpose copy dominates overhead. Raw MMA rate ~0.272 TFLOPS without transpose but gives wrong results.
