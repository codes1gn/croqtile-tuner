---
name: croq-profile
description: Mandatory ncu profiling contract for croq-tune. Every PROFILE step MUST run ncu.
strict: true
enforcement: mandatory
skip-penalty: protocol-violation-stop
---

# Croq-Profile

## The One Rule (INVIOLABLE)

**Every PROFILE step MUST run `ncu` on the current best kernel.**

No exceptions. No "lightweight alternatives". No skipping.

**SELF-ENFORCEMENT CHECKPOINT:**
Before proceeding from PROFILE to IDEA, verify:
- [ ] ncu command was executed (not just planned)
- [ ] ncu output was parsed for bottleneck identification
- [ ] Bottleneck is one of: memory_bound, compute_bound, latency_bound, launch_bound
- [ ] Evidence JSON is prepared for IDEA step

If ANY checkbox is not met, DO NOT proceed to IDEA.

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

## ncu Failure Handling

**Note**: ncu availability is validated at baseline (croq-baseline). If you're in the tuning loop, ncu SHOULD work.

If ncu fails unexpectedly during tuning:

1. **Transient errors** (GPU busy, timeout):
   - Wait 10 seconds and retry once
   - If retry succeeds, continue
   
2. **Permission errors** (should not happen if baseline passed):
   - STOP immediately — this indicates baseline validation was bypassed
   - Escalate to user with full error message
   - DO NOT continue

3. **Any persistent failure**:
   - STOP the tuning loop
   - Log the error
   - Escalate to user

**No fallback strategies. No guessing. No skipping.**

## FORBIDDEN

- Skipping ncu "because timing is enough"
- Guessing bottleneck without ncu data
- Using only compiler output as "profiling"
- Proceeding to IDEA without ncu evidence
- Generating fake iteration data to meet stop conditions
- Continuing tuning after ncu failure
