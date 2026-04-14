---
name: croq-profile
description: Mandatory ncu profiling contract for croq-tune. Every PROFILE step MUST run ncu.
---

# Croq-Profile

## The One Rule

**Every PROFILE step MUST run `ncu` on the current best kernel.**

No exceptions. No "lightweight alternatives". No skipping.

## Why This Is Mandatory

Without ncu data, optimization ideas are guesses. Guesses waste iterations.
ncu takes ~30 seconds. A wrong optimization idea wastes 5+ minutes.
Always profile. Always use ncu.

## PROFILE Step Execution

```bash
# MANDATORY: Run ncu on current best kernel
ncu --set full \
    --export tuning/aitune/<dsl>/perf/<shape_key>/ncu_iter<NNN>.ncu-rep \
    --force-overwrite \
    <kernel_binary> [args]

# Extract key metrics
ncu --import tuning/aitune/<dsl>/perf/<shape_key>/ncu_iter<NNN>.ncu-rep \
    --csv --page raw > tuning/aitune/<dsl>/perf/<shape_key>/ncu_iter<NNN>.csv
```

## Required Output

After PROFILE, you MUST have:

1. `ncu_iter<NNN>.ncu-rep` - full ncu report file
2. `ncu_iter<NNN>.csv` - extracted metrics in CSV
3. Bottleneck identification from ncu data:
   - memory_bound: DRAM throughput > 80%, compute < 50%
   - compute_bound: SM occupancy issues, low IPC
   - latency_bound: high stall cycles, poor instruction mix
   - launch_bound: kernel launch overhead > kernel time

## Handoff to IDEA

Provide this JSON to IDEA step:

```json
{
  "bottleneck": "<memory_bound|compute_bound|latency_bound|launch_bound>",
  "confidence": "high",
  "evidence": {
    "ncu_report": "tuning/aitune/<dsl>/perf/<shape_key>/ncu_iter<NNN>.ncu-rep",
    "key_metrics": {
      "dram_throughput_pct": <value>,
      "sm_occupancy_pct": <value>,
      "compute_throughput_pct": <value>,
      "achieved_bandwidth_gb_s": <value>
    }
  }
}
```

## What Happens If ncu Fails

1. Check GPU availability: `nvidia-smi`
2. Check ncu path: `which ncu` or `/usr/local/cuda/bin/ncu`
3. Retry with reduced metrics: `ncu --set basic`
4. If still fails: log error, skip this PROFILE, continue with last known bottleneck

## FORBIDDEN

- Skipping ncu "because timing is enough"
- Guessing bottleneck without ncu data
- Using only compiler output as "profiling"
- Proceeding to IDEA without ncu evidence
