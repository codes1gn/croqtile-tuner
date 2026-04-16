
## iter002 — 2026-04-16T05:11:59Z
- kernel: `iter002_vecload`
- tflops: `5.36`
- decision: **DISCARD**
- bottleneck: `launch_bound`
- idea: Vectorize global-to-smem loads with float4, remove launch_bounds, BK=64
