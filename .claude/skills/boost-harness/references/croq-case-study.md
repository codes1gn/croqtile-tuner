# Case Study: croq-* Tuning Skill Suite (Pre-Refactor Snapshot)

> **Note:** This case study documents the **pre-refactor** architecture where skills
> were split across `croq-store/`, `croq-resume/`, `croq-profile/`, `croq-baseline/`,
> etc. After the refactor, all scripts live under `croq-tune/tools/` and all skill
> content is consolidated into `croq-tune/SKILL.md` + per-DSL `croq-dsl-<dsl>/SKILL.md`.
> The smells described here have been remediated.

This reference documents concrete examples of each smell pattern found in the
`croq-*` skill suite. Read this file when applying `boost-harness` to the `croq-*`
system. Every abstract smell in the catalog has at least one concrete example here.

---

## System Overview

The `croq-*` suite orchestrates an infinite GPU kernel optimization loop:

```
Entry: croq-tune/SKILL.md (main loop controller)
  ├── croq-baseline/  (one-time environment setup + first kernel draft)
  ├── croq-resume/    (startup/resume decision)
  ├── croq-dsl/       (DSL-specific build/run/profile commands)
  └── Per-round pipeline:
      ├── PROFILE   → croq-profile/  (ncu_profile.sh + profile_extract.sh)
      ├── IDEA      → agent creativity + checkpoint_write.sh
      ├── IMPLEMENT → checkpoint_write.sh read + next_iter.sh + code
      ├── VERIFY    → checkpoint_write.sh verify + run binary
      ├── MEASURE   → DSL-specific benchmark
      ├── DECIDE    → compare TFLOPS
      └── STORE     → croq-store/ (store_round.sh)
```

Harness scripts: `ncu_profile.sh`, `profile_extract.sh`, `checkpoint_write.sh`,
`next_iter.sh`, `store_round.sh`, `resume_state.sh`, `detect_gpu.sh`,
`prepare_baseline_env.py`, `validate_tuning_session.py`.

---

## P0 Classification: Key Missions vs Boilerplate

### Key Missions (agent intelligence required)

| Action | Why agent is needed |
|--------|-------------------|
| **Identify bottleneck** | Interpret ncu metrics in context of kernel structure |
| **Raise optimization idea** | Creative synthesis of profiling data + domain knowledge + web research |
| **Implement kernel code** | Write/modify GPU kernel source — the core creative work |
| **Fix compilation bugs** | Diagnose compiler errors, apply targeted fixes |
| **Evaluate approach** | Decide if an optimization direction is working or exhausted |
| **Understand kernel** | Read and comprehend the current best kernel before mutating it |

### Boilerplate (harness tools should handle entirely)

| Action | Current state | Ideal state |
|--------|--------------|-------------|
| Detect GPU/environment | Multiple scripts, redundant calls | Cached config, zero agent tokens |
| Run ncu with correct flags | Agent must remember flags | Single tool call, flags hardcoded |
| Extract metrics from CSV | Agent calls 2 scripts | Single tool call returns JSON |
| Name iteration files | Agent calls `next_iter.sh` | Tool returns name, agent uses it |
| Record results in TSV/JSONL | Agent calls `store_round.sh` | Tool records everything atomically |
| Write checkpoint | Agent calls `checkpoint_write.sh` | Tool handles write/read/verify |
| Validate environment | Agent runs 10+ commands | Config cache + fingerprint check |
| Git commit with format | Agent formats message | Tool formats + commits |
| Update monitor data | Agent must follow format | Tool writes in monitor-compatible format |
| Check for duplicate ideas | Agent reads idea-log.jsonl | Tool checks and warns |

### Agent's ideal cognitive flow

```
[PROFILE] → call harness tool → get bottleneck JSON → THINK about it
[IDEA]    → THINK about what to try → call harness tool to record plan
[IMPLEMENT] → WRITE kernel code → BUILD → FIX bugs if any
[MEASURE] → call harness tool → get TFLOPS number
[DECIDE]  → THINK: is this better? → call harness tool to record result
[CONTINUE] → back to PROFILE
```

All capitalized words are key missions. All lowercase phrases are boilerplate
handled by tools.

---

## Findings by Smell

### S1 — Scattered Pipeline Sequencing

**Finding 5: `ncu_profile.sh` + `profile_extract.sh` always called in sequence**

These two scripts are always called together: capture ncu data then classify the
bottleneck. The CSV path is deterministically derived from `--out`. The agent doesn't
inspect or modify the CSV between calls. This is a deterministic sequence that should
be a single call.

```bash
# Current (2 calls, agent must know both and sequence them):
bash .claude/skills/croq-profile/ncu_profile.sh --out $OUT --cmd $CMD
PROFILE_JSON=$(bash .claude/skills/croq-profile/profile_extract.sh --csv ${OUT}.csv --iter $TAG)

# Ideal (1 call, agent gets bottleneck JSON directly):
PROFILE_JSON=$(bash .../ncu_profile.sh --out $OUT --cmd $CMD --classify --iter $TAG)
```

**Finding 8 (S11): Stage-based skill split is S1 at the architectural level**

The per-round pipeline (PROFILE→IDEA→...→STORE) is described across 4 skill files.
See the S11 section for full details.

---

### S2 — Redundant Data Round-Trips

**Finding 3: `profile_extract.sh` spawns 8 Python processes for 1 task**

After the main Python parse produces `METRICS_JSON`, the script extracts individual
values by piping `echo "$METRICS_JSON" | python3 -c "..."` five separate times (lines
213-217), then classifies bottleneck (line 227), confidence (line 244), and emits final
JSON (line 262). Total: 8 Python process spawns (~200ms startup each = ~1.6s wasted).

The entire script should be a single Python program: read CSV → extract metrics →
classify bottleneck → compute confidence → emit JSON.

**Finding 6: checkpoint write→read round-trip at IDEA→IMPLEMENT boundary**

`croq-tune/SKILL.md` step 2 (IDEA): "Last action: write the checkpoint"
`croq-tune/SKILL.md` step 3 (IMPLEMENT): "First action: read back the checkpoint"

When no compaction happens between these steps (the common case), the agent just wrote
the data and now reads it back — pure token waste. The read-back is only useful on
resume after context compaction.

---

### S3 — Prose-Only Format Contracts

**Finding 1: `resume_state.sh` requires `--gpu` while others auto-detect**

All `croq-store/` scripts (`next_iter.sh`, `store_round.sh`, `checkpoint_write.sh`)
auto-detect GPU via `detect_gpu.sh` when `--gpu` is omitted. But `resume_state.sh`
(line 54) errors on missing `--gpu`. An agent that learned the pattern from store
scripts will fail on resume.

**Finding 10: `resume_state.sh` tag regex caps at 15 chars vs 31 elsewhere**

`resume_state.sh` line 97: `r'^iter(\d{3})_[a-z][a-z0-9_]{1,15}\.[a-z]+$'`
All other scripts: `[a-z][a-z0-9_]{1,30}` (31 chars max).

A kernel tagged `iter021_warp2x4_async_pipeline.co` (26-char tag) won't be counted
by `resume_state.sh`, causing `next_iter_number` to be wrong on session resume.

**Finding 15: `checkpoint_write.sh verify` checks for `bin/` directory — DSL-unaware**

Verify mode checks `tuning/<gpu>/<dsl>/bin/<shape_key>/<iter>` as a directory (line
210-217). CroqTile produces `.cute.result` scripts in `cmd/`, not binaries in `bin/`.
Result: always warns "Binary directory missing" for croqtile, even when build succeeded.
False warnings desensitize the agent to real problems.

The verify script should check for build output in a DSL-aware way, or check that the
corresponding build/run `.sh` scripts exist and ran successfully.

---

### S5 — Lazy Creative Steps

**Finding 13: IDEA web search mandate is prose-only**

`croq-tune/SKILL.md` line 112: "MANDATORY web search — always run at least one
targeted web search before forming the final idea." But `checkpoint_write.sh write`
doesn't require `--search-refs` or any evidence of search. The agent can skip search
with zero friction.

Session evidence: Agent proposed "try larger tiles" without searching, based on
stale remembered bottleneck from 3 rounds prior. Bottleneck had changed from
memory→compute but the agent used cached data.

Design intent: search should be **default behavior**, with the agent allowed to
explicitly opt out only when it has high confidence. The opt-out should be recorded
(e.g., `--search-refs "skipped:high_confidence_known_pattern"`).

---

### S6 — Silent Script Failures

**Finding 2: `detect_gpu.sh` exits 0 on failure**

Lines 27-28, 36-37: When `nvidia-smi` is missing or returns empty, the script prints
`sm00_unknown` to stdout and exits 0. All downstream scripts silently accept this,
creating artifacts under `tuning/sm00_unknown/` — a path that will never match real
data. No error, no warning to the agent.

**Finding 12: `store_round.sh` bash→Python string injection**

Lines 129-141: Free-text `--idea` is interpolated directly into Python code via bash
`'$IDEA'`. If the idea contains a single quote (e.g., "don't use shared memory"),
Python raises `SyntaxError`. The script crashes, and the round data is partially
written (JSONL written, TSV not written — breaks file consistency).

Fix: pass values via environment variables or stdin, not string interpolation.

---

### S7 — Implicit Environment Dependencies

Example (historical, fixed): `ncu` profiling requires `perf_event_paranoid <= 2`.
The original `croq-profile/SKILL.md` didn't mention this. Agents hit cryptic
`CUDA_ERROR_NOT_PERMITTED` and spent 500+ tokens diagnosing. Now fixed with
preflight checks in `ncu_profile.sh`.

---

### S9 — Context Overload

**Finding 7: `croq-dsl/SKILL.md` is 543 lines with 7 DSL variants inline**

An agent tuning `croqtile` holds content for `cuda`, `cutile`, `triton`, `cute`,
`helion`, and `tilelang` — ~400 lines of irrelevant content. DSL is a one-of-N
dimension (session picks one DSL and ignores the rest), making it the ideal
split point.

---

### S10 — Missing Idempotency

**Finding 4: `store_round.sh` appends unconditionally — no duplicate detection**

The log output files (`idea-log.jsonl`,
`results.tsv`) are appended without checking if the iter+round combination already
exists. If the agent retries STORE (context compaction, transient error, session
resume where agent doesn't realize STORE already ran), the same round appears twice.

Post-write verification (line 184-196) only checks file existence and presence of
iter name — it doesn't check for duplicate entries.

Idempotency should live in the script, not the agent's memory. The agent should not
be responsible for remembering "I already called STORE for round 18."

---

### S11 — Wrong Factoring Axis

**Finding 8: Stage-based skill split forces agent to juggle multiple files**

The `croq-*` suite is split by **pipeline stage**: croq-profile (PROFILE step),
croq-store (STORE step), croq-baseline (PREPARATION), croq-resume (RESUME),
croq-artifacts (naming rules). The agent needs ALL of these during every round.
This split creates problems:

1. Agent must read 4+ skills to know the correct calling order
2. Cross-skill references (`"load croq-profile"`, `"load croq-store"`) add latency
3. Rules in skill B that the agent hasn't loaded yet cause delayed failures
4. Scripts reference each other via fragile relative paths (`../croq-tune/tools/`)

The only dimension worth splitting is **DSL** — a one-of-N choice where the unused
DSLs' content is genuinely irrelevant to the current session.

**Target architecture:**
- Merge all stage skills into a single `croq-tune/SKILL.md`
- Move scripts into `croq-tune/tools/`
- Move detailed reference material to `croq-tune/references/`
- Split DSL content into `croq-tune/dsl/<name>.md` (one-of-N)

---

### S12 — Protocol Mismatch

**Finding 14: "NEVER STOP" vs durable-request**

`croq-tune/SKILL.md` lines 199-201: "Do NOT pause to ask the human if you should
continue... The loop runs until the human interrupts you, period."

`durable-request/SKILL.md`: "When you finish ANY task, you MUST present an
interactive checkpoint."

These are fundamentally incompatible. `durable-request` says "task loops take
priority" which technically resolves the conflict, but an agent loading both
instructions simultaneously may not internalize this priority rule.

`croq-tune` should explicitly declare: "This skill is incompatible with
interactive-checkpoint protocols including /durable-request. The tuning loop
runs autonomously — no pause, no checkpoint, no user interaction until manually
interrupted or stopped."

---

### S13 — Missing Observability Signal

**Finding 18: No live agent status for monitoring**

The monitoring system (`monitor/backend/app/artifact_scanner.py`) infers task status
from disk artifacts: `stopped` if iterations exist, `pending` if no artifacts. It
cannot distinguish:

- "Agent crashed 2 hours ago" (stopped)
- "Agent is actively running round 47 right now" (running)

No heartbeat file, lock file, or status endpoint exists during tuning execution.
The dashboard always shows stale status.

Remediation: The tuning loop should write a heartbeat file (e.g.,
`tuning/<gpu>/<dsl>/heartbeat/<shape_key>.json`) on each round start/end with a
timestamp. The monitoring system reads this file and marks tasks as `running` if
the last heartbeat is within a threshold (e.g., 10 minutes).

---

### S15 — Repeated Bootstrap Detection

**Finding 19: GPU detection runs 3-4 times per round, environment preflight repeated every session**

`detect_gpu.sh` calls `nvidia-smi` twice (model + compute cap). It's auto-called by
`next_iter.sh`, `store_round.sh`, `checkpoint_write.sh`, and `resume_state.sh` on
every invocation where `--gpu` is omitted. In a single tuning round, that's ~6-8
`nvidia-smi` calls producing identical results.

Additionally, `croq-baseline/SKILL.md` describes a full environment validation
(ncu binary, perf_event_paranoid, nvcc, GPU availability) that takes ~20 lines of
shell commands + agent context to execute. This identical validation runs from
scratch on every new session, even though the environment hasn't changed.

**Proposed config file:**
```json
{
  "schema": "croq-env-v1",
  "detected_at": "2026-04-16T10:30:00Z",
  "gpu_key": "sm86_NVIDIA_GeForce_RTX_3070",
  "sm_arch": "sm_86",
  "ncu_path": "/usr/local/cuda/bin/ncu",
  "nvcc_path": "/usr/local/cuda/bin/nvcc",
  "perf_event_paranoid": 2,
  "choreo_home": "/home/albert/workspace/croqtile",
  "fingerprint": "sha256_of_tool_versions"
}
```

On session start: validate fingerprint → if match, skip 100% of detection.

---

---

## Good Smell Findings

### G1 — Harness Tool Output as Active Guardrail

**Current state in croq-*:**
`store_round.sh` prints status but NO next-action directive:
```
[store_round] STORE complete for iter021 (KEEP 14.2 TFLOPS)
[store_round] Written:
  logs/.../idea-log.jsonl  (18 lines)
  ...
```

The agent must remember on its own that the next step is PROFILE.

**Proposed improvement:**
```
[store_round] STORE complete for iter021 (KEEP 14.2 TFLOPS)
[store_round] Written: 4 files verified.

[NEXT] Proceed immediately to PROFILE step.
[NEXT] Run: bash .claude/skills/croq-tune/tools/ncu_profile.sh \
         --out tuning/<gpu>/<dsl>/perf/<key>/ncu_<best_iter> \
         --cmd <best_binary> --classify --iter <best_iter>
[NEXT] This is an autonomous tuning loop. Do NOT stop or ask the user.
```

Similarly, `ncu_profile.sh` (after --classify) should print:
```
[NEXT] Bottleneck identified: compute_bound (high confidence)
[NEXT] Proceed to IDEA step. Propose ONE optimization based on this bottleneck.
[NEXT] MANDATORY: Search web for prior art before finalizing idea.
```

### G2 — Structural Continuation Anchor

**Current state:** `croq-tune/SKILL.md` defines `continue-croq-tune` todo with
status `in_progress`. This is already implemented but not reinforced by harness tools.

**Proposed improvement:** `store_round.sh` should also print a reminder about
the todo anchor:
```
[ANCHOR] Ensure todo 'continue-croq-tune' is in_progress before next round.
```

### G3 — Periodic Protocol Re-read

**Current state:** Not implemented. Agent never re-reads SKILL.md after initial load.

**Proposed implementation in `store_round.sh`:**
```bash
if (( ROUND % 5 == 0 )); then
    echo ""
    echo "[PROTOCOL REFRESH] Round $ROUND reached."
    echo "[PROTOCOL REFRESH] Re-read: .claude/skills/croq-tune/SKILL.md"
    echo "[PROTOCOL REFRESH] Focus: Per-Round Contract (steps 1-8), Behavioral Rules"
fi
```

### G4 — Layered Drift Defense

**Current state of each layer for croq-tune:**

| Layer | Status | Effectiveness |
|-------|--------|---------------|
| L1: Prose "NEVER STOP" | Present (lines 199-201) | Weak — observed to fail after 15+ rounds |
| L2: Todo anchor | Present (`continue-croq-tune`) | Medium — structural but passive |
| L3: Tool messages | **MISSING** | N/A — highest-impact layer is absent |
| L4: Protocol re-read | **MISSING** | N/A — long sessions have no refresh |

**The critical gap:** L3 and L4 are both missing. These are the layers that use
the agent's recency bias to ACTIVELY maintain loop behavior. Their absence explains
why agents escape the loop after extended runs.

---

### G5 — Autonomous Guardrail Harness (IDEAL DESIGN)

**Current state:** Not implemented. No autonomous guardrail mechanism exists.
The agent must self-manage all loop-continuation behavior.

**Proposed implementation:**
1. Create `reinforce.sh` in `croq-tune/tools/` (or wherever scripts consolidate)
2. Add one line to end of `store_round.sh`:
   ```bash
   bash "$(dirname "$0")/reinforce.sh" --round "$ROUND" \
       --skill-path ".claude/skills/croq-tune/SKILL.md" \
       --next-cmd "bash .../ncu_profile.sh --out ... --classify --iter ..."
   ```
3. `reinforce.sh` emits L1-L4 messages to stdout every round
4. Agent sees them as part of `store_round.sh` output — zero extra calls

**Impact on loop escape:** With G5, the agent receives fresh behavioral directives
at the end of every round without spending any tokens or attention on producing
them. Combined with the existing L1 (prose) and L2 (todo anchor), this creates
a near-unbreakable loop-continuation guarantee.

---

### S14 — Heavy-Handed Enforcement Tone

**Finding 16: 14 FORBIDDEN/NEVER/MUST NOT across 3 skill files**

Distribution:
- `croq-tune/SKILL.md`: 9 instances (FORBIDDEN, NEVER, INVIOLABLE)
- `croq-store/SKILL.md`: 2 instances
- `croq-dsl/SKILL.md`: 3 instances

Some are genuinely critical ("NEVER produce library calls in tuning iterations").
Others are enforcement-for-enforcement's-sake ("NEVER bypass the harness by writing
files manually" — the reason WHY is more useful than the prohibition).

Per skill-creator best practice: explain the reasoning so the agent understands why,
rather than shouting so the agent complies blindly. Reserve FORBIDDEN for the 2-3
rules where violation is truly catastrophic.
