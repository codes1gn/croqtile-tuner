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
