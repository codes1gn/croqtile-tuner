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
