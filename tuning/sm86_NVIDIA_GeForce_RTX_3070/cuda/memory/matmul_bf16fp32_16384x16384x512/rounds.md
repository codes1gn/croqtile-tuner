## iter000 — 2026-04-15T09:40:03Z
- kernel: `iter000_baseline`
- tflops: `38.73`
- decision: **KEEP**
- bottleneck: `baseline`
- idea: cuBLAS reference 38.73 TFLOPS

## iter001 — 2026-04-15T09:40:03Z
- kernel: `iter001_draft`
- tflops: `17.17`
- decision: **KEEP**
- bottleneck: `draft`
- idea: First WMMA 128x128 BK=16 double-buffer draft

## iter002 — 2026-04-15T09:56:15Z
- kernel: `iter002_warp2x4`
- tflops: `16.81`
- decision: **DISCARD**
- bottleneck: `compute_bound`
- idea: Increase per-warp WMMA work by switching warp grid from 4x4 to 4x2, so each warp computes 2x4 tiles to improve tensor-core issue utilization.

## iter003 — 2026-04-15T09:59:27Z
- kernel: `iter003_maxr60`
- tflops: `17.2`
- decision: **KEEP**
- bottleneck: `compute_bound`
- idea: Constrain register usage with --maxrregcount=60 to improve active warps and issue efficiency.

## iter004 — 2026-04-15T09:59:55Z
- kernel: `iter004_maxr56`
- tflops: `17.29`
- decision: **KEEP**
- bottleneck: `compute_bound`
- idea: Constrain register usage with --maxrregcount=56 to improve active warps and issue efficiency.

## iter005 — 2026-04-15T11:02:38Z
- kernel: `iter005_bk32`
- tflops: `19.56`
- decision: **KEEP**
- bottleneck: `compute_bound`
- idea: Increase BK from 16 to 32 and execute two WMMA K-slices per stage to reduce loop/sync overhead and improve tensor-core issue utilization.
