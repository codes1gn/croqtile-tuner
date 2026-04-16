
## iter002 — 2026-04-16T05:11:59Z
- kernel: `iter002_vecload`
- tflops: `5.36`
- decision: **DISCARD**
- bottleneck: `launch_bound`
- idea: Vectorize global-to-smem loads with float4, remove launch_bounds, BK=64

## iter003 — 2026-04-16T05:20:49Z
- kernel: `iter003_dbuf`
- tflops: `2.60`
- decision: **DISCARD**
- bottleneck: `launch_bound`
- idea: Double-buffered smem with register-cached loads, BK=32
