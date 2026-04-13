## Round bootstrap - 2026-04-13T17:06:33.681145+00:00
- shape: `f16_512x16384x16384`
- baseline_tflops: `473.79`
- result: baseline stored; next state returns to IDEA due to missing custom kernel source.

## Round iter001 - 2026-04-13T17:08:00.000000+00:00
- shape: `f16_512x16384x16384`
- candidate: `iter001_cublas_scaffold.cu`
- measured_tflops: `594.13`
- decision: `KEEP`
- next_state: `PROFILE`
