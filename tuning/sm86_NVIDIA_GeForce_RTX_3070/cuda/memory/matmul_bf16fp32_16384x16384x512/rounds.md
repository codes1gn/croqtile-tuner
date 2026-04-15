# Tuning Rounds — matmul_bf16fp32_16384x16384x512

## Round 0 — iter000_baseline (KEEP)
- **TFLOPS**: 38.73
- **Bottleneck**: baseline (cuBLAS)
- **Idea**: cuBLAS GemmEx reference

## Round 1 — iter001_draft (KEEP)
- **TFLOPS**: 17.17
- **Bottleneck**: initial draft — needs profiling
- **Idea**: First WMMA kernel, 128x128 tiles, BK=16, 2-stage double-buffer
