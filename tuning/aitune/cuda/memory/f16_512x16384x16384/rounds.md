# Tuning Rounds: f16_512x16384x16384

## iter000 Baseline - 2026-04-14

- **shape**: `f16_512x16384x16384` (M=512, N=16384, K=16384)
- **baseline**: torch.mm FP16
- **TFLOPS**: 34.26
- **device**: NVIDIA GeForce RTX 3070 (Ampere, sm_86)

## PREPARATION - iter001_draft

- **dsl**: cuda
- **status**: First kernel draft ready
- **artifacts**:
  - source: `tuning/aitune/cuda/srcs/f16_512x16384x16384/iter001_draft.cu`
  - build: `tuning/aitune/cuda/cmd/f16_512x16384x16384/build_iter001.sh`
  - run: `tuning/aitune/cuda/cmd/f16_512x16384x16384/run_iter001.sh`

### Kernel Design (iter001_draft)

- **Algorithm**: Naive tiled matrix multiplication with shared memory blocking
- **Tile size**: 32x32x32
- **Block size**: 16x16 threads
- **Implementation**: Pure CUDA (no library calls)
- **Data type**: FP16 (half) with FP32 accumulation

### Next State: PROFILE

The draft kernel is ready for profiling. Next steps:
1. Compile: `bash tuning/aitune/cuda/cmd/f16_512x16384x16384/build_iter001.sh`
2. Run: `bash tuning/aitune/cuda/cmd/f16_512x16384x16384/run_iter001.sh`
3. Profile with ncu to identify bottlenecks
4. Begin IDEA -> IMPLEMENT -> MEASURE -> DECIDE loop
