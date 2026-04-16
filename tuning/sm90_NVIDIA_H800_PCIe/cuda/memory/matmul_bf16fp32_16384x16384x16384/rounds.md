
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

## iter004 — 2026-04-16T05:30:31Z
- kernel: `iter004_smalltile`
- tflops: `1.38`
- decision: **DISCARD**
- bottleneck: `launch_bound`
- idea: Reduce tile to BM=64 BN=64 BK=32 for higher occupancy

## iter005 — 2026-04-16T05:33:49Z
- kernel: `iter005_mmasync`
- tflops: `0.0`
- decision: **DISCARD**
- bottleneck: `launch_bound`
- idea: mma.sync PTX + manual B fragment loading - incorrect register layout

## iter006 — 2026-04-16T05:37:36Z
- kernel: `iter006_coalesced`
- tflops: `8.60`
- decision: **DISCARD**
- bottleneck: `launch_bound`
- idea: 128-bit coalesced loads into padded smem - padding broke alignment

## iter007 — 2026-04-16T05:46:01Z
- kernel: `iter007_nopad`
- tflops: `1.78`
- decision: **DISCARD**
- bottleneck: `launch_bound`
- idea: Remove smem padding + maxrregcount=128 for higher occupancy
