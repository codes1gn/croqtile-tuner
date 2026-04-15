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
    --export tuning/<gpu>/<dsl>/perf/<shape_key>/ncu_iter<NNN>.ncu-rep \
    --force-overwrite \
    <kernel_binary> [args]

# Extract key metrics
ncu --import tuning/<gpu>/<dsl>/perf/<shape_key>/ncu_iter<NNN>.ncu-rep \
    --csv --page raw > tuning/<gpu>/<dsl>/perf/<shape_key>/ncu_iter<NNN>.csv
```

## Required Output

After PROFILE, you MUST have:

1. `ncu_iter<NNN>_<tag>.ncu-rep` — full ncu report file
2. `ncu_iter<NNN>_<tag>.csv` — extracted metrics in CSV

## Bottleneck Classification — Use the Harness (MANDATORY)

**DO NOT manually read the CSV and guess a bottleneck.** Use `profile_extract.sh`:

```bash
bash .claude/skills/croq-profile/profile_extract.sh \
    --csv  tuning/<gpu>/<dsl>/perf/<shape_key>/ncu_iter<NNN>_<tag>.csv \
    --iter iter<NNN>_<tag>
```

The script emits one-line JSON to stdout:

```json
{
  "iter": "iter012_myname",
  "bottleneck": "memory_bound",
  "confidence": "high",
  "evidence": {
    "ncu_csv": "tuning/<gpu>/.../ncu_iter012_myname.csv",
    "key_metrics": {
      "dram_throughput_pct": 91.3,
      "compute_throughput_pct": 38.2,
      "sm_occupancy_pct": 62.5,
      "achieved_bandwidth_gb_s": 2890.1
    }
  }
}
```

Capture the output and pass the JSON verbatim to the IDEA step:

```bash
PROFILE_JSON=$(bash .claude/skills/croq-profile/profile_extract.sh \
    --csv  tuning/<gpu>/<dsl>/perf/<shape_key>/ncu_iter<NNN>_<tag>.csv \
    --iter iter<NNN>_<tag>)
echo "PROFILE: $PROFILE_JSON"
```

**Classification thresholds (fixed — do not override):**

| Bottleneck | Condition |
|---|---|
| `memory_bound` | DRAM% > 80 AND compute% < 50 |
| `compute_bound` | compute% > 70 AND DRAM% < 60 |
| `launch_bound` | occupancy% < 15 AND DRAM% < 40 AND compute% < 40 |
| `latency_bound` | stall% > 40 AND DRAM% < 60 AND compute% < 60 |
| `mixed` | none of the above (highest% wins) |

**Exit codes:** 0=ok, 1=usage error, 2=CSV missing, 3=metrics not found.  
If exit code ≠ 0: fix the CSV path or re-run ncu. Do NOT proceed to IDEA without valid output.

## Handoff to IDEA

Use the JSON object from `profile_extract.sh` verbatim. No manual re-interpretation.

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
