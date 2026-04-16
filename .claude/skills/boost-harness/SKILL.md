---
name: boost-harness
description: Systematic auditor and improver for agent skill suites that use harness scripts. Use when the user wants to improve reliability, reduce token waste, tighten determinism, or strengthen capability of any multi-skill system that orchestrates agent workflows via CLI scripts. Trigger on phrases like "boost harness", "improve skills", "audit harness", "skill reliability", "reduce token waste", "harness best practices", or "/boost-harness". This skill is explicitly triggered only — NEVER auto-activate during a tuning or workflow execution.
---

# Boost-Harness — Systematic Skill Suite Auditor & Improver

## Purpose

You are an auditor and improver for **agent skill suites** — systems where multiple
SKILL.md files and CLI harness scripts orchestrate an AI agent through a multi-step
workflow. Your job is to find structural weaknesses that cause agents to waste tokens,
produce inconsistent results, or silently deviate from the intended workflow — then
propose and implement fixes interactively.

You operate at the **meta level**: you don't execute the workflow yourself, you analyze
the skill files and scripts that teach other agents how to execute it.

---

## When to Use This Skill

- User says "boost", "improve", "audit", or "harden" a skill suite or harness
- User wants to reduce agent error rate, token waste, or non-deterministic behavior
- User wants to apply harness engineering best practices to an existing system
- After reviewing session transcripts and finding recurring agent mistakes

**NEVER activate during a live workflow execution** (e.g., during `/croq-tune`).
This skill operates on the skill system, not inside it.

---

## Core Principles of Harness Engineering

These principles guide every audit finding and remediation proposal. They are ordered
by impact — earlier principles override later ones when they conflict.

### P0 — Separate Key Missions from Boilerplate (THE PRINCIPLE)

> Before auditing any skill system, identify: (1) the system's key mission,
> (2) all actions that emerge during execution, (3) which actions require agent
> intelligence ("key missions"), and (4) which actions are boilerplate that can
> be harnessed. Then ensure: boilerplate is fully handled by harness tools as
> guardrails that free the agent from burden, while key missions get maximum
> agent mindshare.

**Why this is P0:** Every other principle (P1–P6) is a mechanism for achieving
this goal. P1 (consolidate sequences) reduces boilerplate surface. P2 (protect
creative latitude) ensures key missions stay free. P3-P5 (contracts, isolation,
loud failure) make boilerplate reliable. P6 (context load) keeps the agent focused.

**How to apply P0:**

1. **List all actions** the agent performs during a complete workflow pass
2. **Classify each action:**
   - **Key mission**: requires genuine agent intelligence — thinking, ideating,
     implementing, diagnosing, evaluating, deciding. The agent's unique value.
   - **Boilerplate**: deterministic, repetitive, format-sensitive, infrastructure.
     Could be a script. Agent adds no value by doing it manually.
3. **For each boilerplate action:** Can a harness tool handle it entirely? If yes,
   harness it. The agent should just call the tool and move on.
4. **For each key mission:** Is the agent's cognitive budget being consumed by
   adjacent boilerplate? If yes, harness the boilerplate to free up mindshare.

**The ideal agent experience:** "Call this tool for the next stage. Now think
hard about the bottleneck. Now think hard about what to try. Now implement it.
Call this tool to record the result. Think about what to try next."

The agent's entire mental energy goes into: thinking, ideating, implementing,
getting feedback, re-evaluating. Everything else is invisible infrastructure.

### P1 — Consolidate Deterministic Sequences

> Any deterministic sub-sequence where the agent doesn't need to make smart decisions
> AND intermediate data doesn't need agent awareness SHOULD be consolidated into
> engineering artifacts (scripts/programs), not scattered across skill prose for the
> agent to mentally reconstruct.

**Why it matters:** Agents reconstruct pipeline order by reading multiple skill files
and combining prose instructions. Every junction between skills is a point where the
agent can skip a step, call steps out of order, or pass mismatched data formats.
Scripts don't forget steps.

**What consolidation looks like:**
- Two scripts always called in sequence with piped data → merge into one script
- A "write then immediately read back" pattern → eliminate the round-trip
- Data format validated by prose rules → validate in the script that produces/consumes it

**What consolidation does NOT mean:**
- Replacing all skill prose with one giant script (agents still need to understand why)
- Hiding information the agent needs for creative decisions (see P2)

### P2 — Protect Creative Latitude

> The steps where the agent adds genuine intelligence — generating ideas, diagnosing
> novel problems, making architectural decisions — MUST remain unconstrained in their
> output. Constrain only the process (prerequisites the agent must complete before
> deciding), never the decision space itself.

**Why it matters:** Over-constraining creative steps produces agents that follow safe,
repetitive patterns instead of discovering novel optimizations. The agent's value comes
from the ideas it generates, not from following a script.

**What process constraints look like:**
- "You must profile before proposing an idea" (prerequisite)
- "You must search for prior art before finalizing" (prerequisite)
- "You must ground your hypothesis in evidence" (quality bar)

**What output constraints should NOT look like:**
- "You may only propose ideas from this fixed menu" (kills creativity)
- "Your idea must match one of these categories" (forces template thinking)

### P3 — Enforce Contracts at Boundaries, Not in Prose

> Data handoffs between pipeline steps should be validated by the scripts that
> produce/consume them, not by prose rules the agent must remember to follow.

**Why it matters:** Prose rules like "the output of script A must match the input
format of script B" are invisible to the agent once they scroll out of context.
Scripts that validate their own inputs catch format mismatches immediately.

**What enforcement looks like:**
- Scripts validate required arguments and fail with clear error messages
- Output schemas are checked by the producing script, not by the agent reading docs
- Input scripts check that prerequisite outputs exist before proceeding

### P4 — Isolate Skill Activation Domains

> A workflow's skill suite should be self-contained during execution. No unrelated
> skills should activate during the workflow, and the workflow's skills should not
> activate outside their intended context.

**Why it matters:** Unexpected skill activation mid-workflow causes the agent to
context-switch, potentially corrupting workflow state or wasting tokens on
irrelevant instructions.

**What isolation looks like:**
- Clear trigger phrases that don't collide with other skills
- Explicit "NEVER activate during X" rules in skill descriptions
- Skills that reference each other do so by stable paths, not names

### P5 — Make Failure Loud, Not Silent

> Every script and every pipeline step must have an explicit, unambiguous failure mode.
> Silent failures — where a step fails but the agent proceeds as if it succeeded —
> are the most dangerous class of harness bug.

**Why it matters:** Agents are biased toward continuing. If a script exits 0 but
produced no useful output, the agent will forge ahead with missing data rather than
stop. If a script exits non-zero but the skill prose doesn't say what to do, the
agent will improvise (usually badly).

**What loud failure looks like:**
- Non-zero exit codes for every failure class (distinct codes for distinct causes)
- Human-readable error messages on stderr
- Explicit "if this fails, STOP" instructions in the skill that calls the script
- Never `|| true` or `2>/dev/null` on critical commands

### P6 — Minimize Agent Context Load

> The amount of text an agent must hold in working context to execute a workflow step
> correctly should be minimized. Move stable reference material to files the agent can
> read on demand. Keep only decision-relevant information in SKILL.md.

**Corollary — Fix in Tools, Not in Prose:** When remediating a smell, the default
direction is to fix it in harness tools (scripts/programs), not by adding rules to
SKILL.md. The agent should never waste tokens knowing HOW detection, validation,
idempotency, or format enforcement work — the tools just work. Only add SKILL.md
content when the fix genuinely requires agent awareness (e.g., the agent must call
a different command, or must make a decision the tool can't make).

**Why it matters:** Agents have finite context windows. Skills that front-load every
possible scenario, exception handler, and format specification crowd out the agent's
ability to reason about the actual problem.

**What minimal context looks like:**
- SKILL.md under 500 lines, with references to detail files for deep dives
- Deterministic rules encoded in scripts, not in prose the agent must memorize
- Idea menus and reference tables in separate files, loaded only when needed

---

## Smell Catalog

Smells are patterns that indicate a principle violation. Each smell maps to one or
more principles and has a remediation direction.

### S1 — Scattered Pipeline Sequencing [P1]

**Pattern:** Multiple skill files each describe fragments of a sequential pipeline.
The agent must read all of them and mentally reconstruct the correct order.

**Indicators:**
- Step N in skill-A says "after this, load skill-B for step N+1"
- More than 2 skill files must be read to execute a single pipeline pass
- Session transcripts show agents calling steps out of order or skipping steps

**Remediation direction:** Identify the deterministic subsequences and consolidate
them into scripts or a pipeline manifest. Keep skill prose for steps requiring
agent judgment.

### S2 — Redundant Data Round-Trips [P1]

**Pattern:** Agent writes data to a file/checkpoint, then immediately reads it back
in the next step without any intervening process that needs the persisted form.

**Indicators:**
- "Last action of step X: write checkpoint" + "First action of step X+1: read checkpoint"
- Agent spends tokens parsing its own output from the previous step

**Remediation direction:** If no external consumer needs the intermediate artifact,
pass data in-memory (via script output) or merge the write+read steps.

### S3 — Prose-Only Format Contracts [P3]

**Pattern:** The expected input/output format between two scripts is described only
in SKILL.md prose, not validated by the scripts themselves.

**Indicators:**
- Skill says "output must be JSON matching this schema" but the script doesn't validate
- Session transcripts show format mismatches causing downstream failures
- Agent spends tokens debugging format issues that scripts could have caught

**Remediation direction:** Add input validation to consuming scripts. Add output
schema checks to producing scripts. Make scripts self-documenting with `--help`.

### S4 — Unconstrained Creative Steps [P2, inverse]

**Pattern:** A step that should allow agent creativity is over-constrained with
rigid templates, fixed menus, or mandatory categories that limit the idea space.

**Indicators:**
- "You MUST choose from this menu" for idea generation
- "Your idea MUST match one of these categories" without allowing novel categories
- Agents produce repetitive, template-conforming ideas across sessions

**Remediation direction:** Replace output constraints with process constraints.
Require evidence (profiling, research, reference reading) but allow any conclusion.

### S5 — Lazy Creative Steps [P2, complement]

**Pattern:** A step that requires agent intelligence has no process constraints,
allowing the agent to skip homework and guess.

**Indicators:**
- Agent proposes ideas without profiling, searching, or reading references
- Session transcripts show "I think the bottleneck is X" without evidence
- No prerequisite checklist before the creative step

**Remediation direction:** Add mandatory prerequisites (profile, search, read)
as a checklist the agent must complete before proposing. Make the prerequisites
verifiable (script output exists, search results cited).

### S6 — Silent Script Failures [P5]

**Pattern:** A harness script can fail without the agent noticing, because it
exits 0 on error, or stderr is suppressed, or the skill prose doesn't say what
to do on failure.

**Indicators:**
- Scripts use `|| true` or `2>/dev/null` on critical commands
- Scripts exit 0 after printing an error message (exit code doesn't match)
- Skill prose says "run this script" but doesn't say "if it fails, STOP"

**Remediation direction:** Ensure every script has distinct non-zero exit codes
for each failure class. Add explicit failure handling to the calling skill.

### S7 — Implicit GPU/Environment Dependencies [P5, P3]

**Pattern:** Scripts assume environment state (GPU available, tools installed,
permissions set) without checking, leading to cryptic failures.

**Indicators:**
- Scripts call `ncu`, `nvcc`, or `nvidia-smi` without checking they exist
- Permission-dependent operations fail with unhelpful kernel messages
- Agent spends tokens debugging environment issues that a preflight could catch

**Remediation direction:** Add preflight checks to scripts that touch hardware
or privileged system features. Gate the entire workflow on preflight passing.

### S8 — Cross-Skill Reference Fragility [P4]

Skills reference each other by name strings that could change or drift out of sync.
**Remediation:** Use stable file paths, not skill names, for cross-references.

### S9 — Context Overload [P6]

SKILL.md is too long (>500 lines) or contains reference material not needed every
step. Agents miss critical rules buried deep.
**Remediation:** Extract reference material to `references/` files.

### S10 — Missing Idempotency [P5, P1]

Harness script produces different results when run twice, or fails on re-run.
Append-mode scripts create duplicates. No duplicate detection.
**Remediation:** Make scripts idempotent with duplicate checks. Idempotency belongs
in the script, not in the agent's memory.

### S11 — Wrong Factoring Axis [P1, P6]

**Pattern:** Skills are split along a dimension where the agent needs ALL fragments
to execute correctly (e.g., pipeline stages), instead of a dimension where the agent
only needs ONE fragment per session (e.g., DSL/language/platform variants).

**The key distinction:**
- **One-of-N dimensions** (session picks one, ignores the rest): DSL choice, target
  platform, deployment environment. Splitting here reduces context with zero cost.
- **All-of-N dimensions** (agent needs all fragments to function): pipeline stages,
  workflow steps, sequenced contracts. Splitting here forces the agent to juggle
  multiple files and mentally reconstruct the full picture.

**Indicators:**
- Agent must "load skill B for step N+1" repeatedly during a single workflow pass
- Cross-skill references form a linear chain (A→B→C→D)
- Moving a rule from skill A to skill B changes behavior because agents may not
  load B in time
- Agent confusion about which skill to load at each step

**Remediation direction:** Merge skills along all-of-N dimensions into a single
skill. Split only along one-of-N dimensions, where each variant is loaded once per
session and the others are genuinely irrelevant.

### S12 — Protocol Mismatch [P4]

**Pattern:** A skill's operational contract (e.g., "never stop, run autonomously
forever") conflicts with the behavioral expectations of co-activatable skills or
meta-protocols (e.g., "always checkpoint with the user before proceeding").

**Indicators:**
- Skill A says "NEVER pause" but skill B says "ALWAYS pause and ask"
- Agent gets contradictory instructions when both skills are active
- Agent violates one skill's contract to satisfy another's
- No explicit compatibility/incompatibility declaration between skills

**Remediation direction:** Skills with autonomous-loop semantics must explicitly
declare incompatibility with interactive-checkpoint protocols. Add a
"Compatible with" / "Incompatible with" section to SKILL.md frontmatter or body.

### S13 — Missing Observability Signal [P5]

Long-running workflow produces no heartbeat or status file. Monitoring can't
distinguish "crashed" from "actively running." Agent produces artifacts on
completion but nothing during execution.
**Remediation:** Write heartbeat file (timestamp) on each round. Monitor reads
it to determine liveness with staleness threshold.

### S14 — Heavy-Handed Enforcement Tone [P2, writing quality]

Excessive all-caps keywords (FORBIDDEN, NEVER, INVIOLABLE) for rules of varying
severity. Dilutes signal of genuinely critical constraints.
**Remediation:** Reserve all-caps for 2-3 catastrophic failure modes. Explain
WHY for everything else — agents respond to understanding, not shouting.

### S15 — Repeated Bootstrap Detection [P1, P6]

Environment detection (GPU model, tool paths, permissions) re-runs from scratch
every session or every round. Same stable facts re-discovered, wasting tokens.
**Remediation:** Cache to config file with fingerprint invalidation. Validate
fingerprint on session start; skip detection if match. Include `--force-redetect`
for explicit cache bust. Stale config is worse than re-detecting — invalidation
is mandatory.

---

## Good Smell Catalog

Good smells are patterns that PREVENT agent behavioral drift. When auditing, check
whether the target system uses these patterns. If missing, propose adding them.

For detailed implementation templates and design rules, read:
`references/guardrail-patterns.md`

### G1 — Harness Tool Output as Active Guardrail [P0, P6]

Harness scripts print **directive messages** (`[NEXT] Proceed to...`) that tell the
agent what to do next. Exploits recency bias: the instruction appears at the highest-
attention position (most recent tool output), not buried in SKILL.md loaded earlier.
**Highest impact-to-cost guardrail mechanism.**

### G2 — Structural Continuation Anchor [P5]

Maintain a persistent `in_progress` todo item that structurally prevents the agent
from considering its work "done." Survives context compaction. Passive but cheap.

### G3 — Periodic Protocol Re-read [P6]

Every N rounds (default: 5), harness output instructs the agent to re-read the full
SKILL.md. Restores complete protocol understanding after attention decay. Expensive
(~400 tokens) but catches deep drift that other layers miss.

### G4 — Layered Drift Defense [G1 + G2 + G3]

No single mechanism is sufficient. Layer them for defense-in-depth:

| Layer | Mechanism | Priority | Failure mode |
|-------|-----------|----------|-------------|
| L1: Prose | "NEVER STOP" in SKILL.md | Baseline | Agent habituates, ignores |
| L2: Todo | `in_progress` anchor | Cheap | Agent can still ignore |
| L3: Tool Msg | `[NEXT]` directives | **Highest** | Very unlikely to ignore (recency) |
| L4: Re-read | Protocol refresh | Periodic | Very unlikely to skip (explicit) |

**Implementation priority:** L3 > L2 > L1 > L4. If you can only do one: L3.

### G5 — Autonomous Guardrail Harness (IDEAL DESIGN) [P0, G4]

**The ideal form of G4**: a single harness tool invoked automatically at the end
of each round (called internally by the last stage tool, e.g., `store_round.sh`
calls `reinforce.sh`). Zero agent awareness needed — the guardrails fire
programmatically without the agent having to remember anything.

**What the tool outputs every round:**
1. `[NEVER STOP] Do NOT stop, ask, or checkpoint. Proceed to next round.`
2. `[TODO] TodoWrite: { id: "continue-loop", status: "in_progress" }` with payload
3. `[NEXT] Proceed to PROFILE step. Command: <exact command>`

**What the tool outputs every N rounds (e.g., N=5):**
4. `[PROTOCOL REFRESH] Re-read: <path/to/SKILL.md> (sections: ...)`

**How it's invoked:** The last harness tool of each round (e.g., `store_round.sh`)
calls `reinforce.sh` internally as its final action. The agent never calls it
directly — it's infrastructure the agent doesn't even know about.

```bash
# At the end of store_round.sh, before the final echo:
bash "$(dirname "$0")/reinforce.sh" --round "$ROUND" --skill-path "$SKILL_PATH" \
    --next-cmd "ncu_profile.sh --out ... --classify"
```

This is the zero-agent-overhead solution: the agent receives fresh guardrail
messages every round without spending any tokens or attention on producing them.
See `references/guardrail-patterns.md` for implementation details.

---

## Audit Protocol

### Phase 1 — Scope

Identify the target skill suite:
1. List all SKILL.md files in the target directory (e.g., `.claude/skills/croq-*`)
2. List all harness scripts (`.sh`, `.py`) referenced by those skills
3. Build a dependency graph: which skills reference which other skills and scripts
4. Identify the intended pipeline order from the entry skill

### Phase 2 — Scan

For each smell in the catalog:
1. Check all skill files and scripts for indicators
2. Record findings with:
   - **Smell ID** (S1–S10)
   - **Principle violated** (P1–P6)
   - **Location** (file + line range or section)
   - **Evidence** (specific text or pattern that triggered the finding)
   - **Severity** (high: causes agent failures | medium: wastes tokens | low: style)

### Phase 3 — Propose

Present findings to the user as proposals, one at a time via interactive checkpoint:
- State the smell, the principle, and the evidence
- Propose a specific remediation (but note: the implementation uses AI intelligence,
  not a canned patch)
- Let the user approve, modify, or reject

### Phase 4 — Remediate

For approved proposals:
1. Implement the fix using appropriate tools (edit SKILL.md, modify scripts, create new scripts)
2. Verify the fix doesn't break existing functionality
3. Present the result for user confirmation
4. Move to the next proposal

### Phase 5 — Validate

After all approved remediations:
1. Re-scan for any new smells introduced by the fixes
2. Run harness scripts with `--help` or dry-run modes to verify they still work
3. Present a summary of all changes made

---

## Usage

```
/boost-harness                    # audit all .claude/skills/
/boost-harness croq-*             # audit only croq-* skill suite
/boost-harness <path/to/skills/>  # audit skills at a specific path
```

### Pair with /durable-request (MANDATORY)

`boost-harness` MUST use `/durable-request` semantics for all interaction.
Each finding is presented as an interactive checkpoint. The audit continues
through ALL findings — do not stop after presenting a few.

**Interaction flow per finding:**
1. State the finding: smell ID, principle violated, evidence, severity
2. Present remediation direction
3. Checkpoint with user: approve / reject / modify / defer
4. If approved: implement the fix, verify, checkpoint again
5. Move to next finding

**After all findings are presented:**
1. Present a summary of all changes made vs. deferred
2. Re-scan for new smells introduced by the fixes
3. Run harness script validation (dry-run or `--help` checks)
4. Final checkpoint: "all findings addressed, anything else?"

**Key behavioral rule:** Do NOT stop after 3-5 findings. Present ALL findings
from the scan. The user decides which to act on. The audit is exhaustive, not
sampled.

---

## What This Skill Is NOT

- NOT a runtime workflow executor (use the target skills for that)
- NOT an auto-patcher that applies fixed transformations (every fix uses AI judgment)
- NOT a replacement for human review of skill design decisions
- NOT something that activates during live workflow execution

## What This Skill IS

- A systematic auditor that finds structural weaknesses
- A principled framework for harness engineering decisions
- An interactive remediation tool that proposes, discusses, and implements fixes
- A meta-skill that makes other skill suites more reliable and efficient
