# Guardrail Patterns — Preventing Agent Behavioral Drift

Reference file for Good Smell patterns G1-G4. Read when auditing a skill system
for missing drift-prevention mechanisms, or when implementing guardrails.

---

## The Problem: Agent Loop Escape

Agents in long-running loops (20+ rounds) gradually "forget" they should keep going.
Root cause: as context grows, early instructions decay in the agent's effective
attention. The agent's recency bias causes it to focus on recent tool output and
forget meta-protocol loaded hundreds of lines ago.

**Observed failure modes:**
- Agent asks user "should I continue?" despite NEVER STOP rule
- Agent presents a checkpoint/summary as if the task is complete
- Agent proposes stopping "at a good point" after N rounds
- Agent starts treating each round as a standalone task

---

## G1 — Harness Tool Output as Active Guardrail

### Mechanism

Harness scripts print **directive messages** that tell the agent exactly what to do
next. These messages appear in the agent's most recent tool output — the highest-
attention position in the context window.

### Why It Works

Agents pay disproportionate attention to their most recent tool output (recency bias).
By injecting next-action directives at this position, the harness exploits the bias
instead of fighting it. The instruction appears at the exact moment the agent decides
what to do next.

### Implementation Template

```bash
# At the end of any stage-boundary harness script:

# 1. Primary output (what the tool did)
echo "[tool_name] ACTION complete: $SUMMARY"

# 2. Next-action directive (what the agent should do now)
echo ""
echo "[NEXT] Proceed to <NEXT_STAGE> step."
echo "[NEXT] Run: <exact command or tool to invoke>"

# 3. Behavioral reminder (at high-risk decision points only)
echo "[NEXT] This is an autonomous loop. Do NOT stop or ask the user."
```

### Design Rules

1. **Place directives AFTER primary output** — the agent reads status first, then
   gets told what to do. Don't mix status and directives.

2. **Use consistent prefix** (`[NEXT]`) — the agent can pattern-match on this,
   making the directive structurally distinct from status output.

3. **Include both WHAT and HOW** — don't just say "proceed to PROFILE." Include
   the actual command to run. This eliminates the agent's need to look up the
   command in SKILL.md.

4. **Behavioral reminders only at high-risk points** — don't put "do NOT stop"
   on every script. Put it on the scripts at loop-continuation decision points
   (typically the last script of each round, e.g., STORE).

5. **Keep messages short (3-5 lines)** — verbose messages become noise. The agent
   will start ignoring them if they're always a wall of text.

### Which Tools Should Have Directives

In a typical pipeline (PROFILE → IDEA → IMPLEMENT → VERIFY → MEASURE → STORE):

| Tool | Directive content | Behavioral reminder? |
|------|------------------|---------------------|
| ncu_profile.sh (with --classify) | "Proceed to IDEA. Bottleneck: X" | No — mid-pipeline |
| checkpoint_write.sh write | "Proceed to IMPLEMENT. Build exactly: Y" | No |
| store_round.sh | "Proceed to PROFILE. Run ncu_profile.sh" | YES — loop boundary |

Only `store_round.sh` (the loop boundary) needs the "do NOT stop" reminder.

---

## G2 — Structural Continuation Anchor

### Mechanism

Use the agent framework's built-in todo/task system to maintain a persistent
`in_progress` item. The agent's instruction-following behavior treats unfinished
todos as incomplete obligations.

### Why It Works

TodoWrite is a first-class agent tool. Most agents are trained to not end their
turn with `in_progress` items. The anchor is a structural constraint — it doesn't
rely on the agent remembering a prose rule, it relies on the agent's built-in
behavior around task completion.

### Implementation Template

```
# At loop entry:
TodoWrite([
  { id: "continue-loop", content: "Continue <workflow> loop", status: "in_progress" }
])

# At each round boundary:
TodoWrite([
  { id: "round-step", content: "Round N complete", status: "completed" },
  { id: "continue-loop", status: "in_progress" }  # keep anchor alive
])
```

### Design Rules

1. **Create at loop entry, never mark completed** — the anchor lives as long as
   the loop runs. Only mark it completed when the loop should genuinely stop.

2. **Clean up step todos periodically** — after 20+ rounds, the todo list gets
   polluted with completed step items. Run cleanup when count > 20.

3. **Combine with G1** — the todo anchor is passive (agent must notice it). G1's
   tool message is active (appears in recent output). Together they cover both
   attention modes.

### Limitation

This is a hint, not a hard gate. An agent CAN ignore an in_progress todo and
end its turn. Observed to fail in ~10% of long sessions. Use as part of layered
defense, not as the sole mechanism.

---

## G3 — Periodic Protocol Re-read

### Mechanism

At regular intervals (every N rounds), harness tool output instructs the agent to
re-read the complete skill protocol, refreshing its understanding of the workflow.

### Why It Works

Context compaction and natural attention decay degrade the agent's grip on early
instructions. A periodic re-read fully restores the protocol in recent context.
The agent re-absorbs ALL rules, not just the ones it remembered.

### Implementation Template

```bash
# In the last harness tool of each round (e.g., store_round.sh):
if (( ROUND % 5 == 0 )); then
    echo ""
    echo "[PROTOCOL REFRESH] Round $ROUND reached."
    echo "[PROTOCOL REFRESH] Re-read the protocol before next round:"
    echo "[PROTOCOL REFRESH]   Read file: <path/to/SKILL.md>"
    echo "[PROTOCOL REFRESH]   Focus: Per-Round Contract, Behavioral Rules"
fi
```

### Design Rules

1. **Every 5-10 rounds** — too frequent wastes tokens (~400 per refresh), too
   infrequent lets drift accumulate. 5 is aggressive, 10 is relaxed.

2. **Name specific sections** — don't tell the agent to "re-read everything."
   Name the sections that contain the most critical rules (round contract,
   behavioral rules, stop conditions).

3. **Place in the last tool of each round** — so the re-read happens BEFORE
   the next round starts, not in the middle of one.

4. **Consider context usage** — if the agent framework exposes context usage
   (e.g., >60% used), trigger a re-read regardless of round number.

### Token Cost

- ~400 tokens per re-read (SKILL.md is ~300-500 lines)
- At every 5 rounds: ~80 tokens/round amortized
- At every 10 rounds: ~40 tokens/round amortized
- Acceptable for the stability gain — a single wasted round from drift costs
  ~2000+ tokens (failed idea + recovery)

---

## G4 — Layered Drift Defense

### The Combined Pattern

No single guardrail mechanism is reliable alone. Layer them:

```
┌─────────────────────────────────────────────────────────────┐
│ L1: Prose Rule (SKILL.md)         ← loaded once, decays     │
│   ├─ "NEVER STOP"                                           │
│   └─ Sets baseline intent                                   │
│                                                              │
│ L2: Todo Anchor (TodoWrite)       ← persistent, passive     │
│   ├─ "continue-loop: in_progress"                           │
│   └─ Structural "not done" signal                           │
│                                                              │
│ L3: Tool Message (every round)    ← active, high-attention  │
│   ├─ "[NEXT] Proceed to PROFILE"                            │
│   └─ Appears in most-recent output                          │
│                                                              │
│ L4: Protocol Re-read (every N)    ← expensive, comprehensive│
│   ├─ "[PROTOCOL REFRESH] Re-read SKILL.md"                  │
│   └─ Full instruction restoration                           │
└─────────────────────────────────────────────────────────────┘
```

### Failure Mode Analysis

| Layer | How it fails | What catches the failure |
|-------|-------------|------------------------|
| L1 | Agent habituates to all-caps rules | L3 provides fresh recency-based reminder |
| L2 | Agent ignores in_progress todo | L3's explicit "do NOT stop" overrides |
| L3 | Agent treats repeated messages as noise | L4 fully refreshes protocol |
| L4 | Agent skips the re-read | L3 still provides per-round directive |

For the agent to escape the loop, it would need to simultaneously:
1. Ignore the prose rule (L1) — common after 20 rounds
2. Ignore the in_progress todo (L2) — possible but uncommon
3. Ignore the tool output directive (L3) — very unlikely (recency bias)
4. Skip the protocol re-read (L4) — very unlikely (explicit instruction)

**Combined failure probability is near zero.** This is why layered defense works.

### Implementation Priority

If you can only implement one: **L3 (tool messages)** — highest impact, lowest cost.
If you can implement two: **L3 + L2** — active reminder + structural anchor.
All four is ideal for critical autonomous loops.

---

## G5 — Autonomous Guardrail Harness (IDEAL DESIGN)

### The Key Insight

G1-G4 still require someone to wire the guardrails into the right places. G5
eliminates this entirely: a single `reinforce.sh` script is called internally by
the last harness tool of each round. The agent never calls it, never knows about
it, never spends tokens on it. The guardrails fire programmatically.

### Architecture

```
Agent calls store_round.sh (normal workflow)
  │
  └─ store_round.sh internally calls reinforce.sh (agent unaware)
       │
       ├─ Every round:
       │   ├─ Print [NEVER STOP] behavioral reminder
       │   ├─ Print [TODO] continuation anchor payload
       │   └─ Print [NEXT] exact command for next stage
       │
       └─ Every N rounds:
           └─ Print [PROTOCOL REFRESH] re-read instruction
```

### Implementation: `reinforce.sh`

```bash
#!/usr/bin/env bash
# reinforce.sh — Autonomous guardrail message emitter.
# Called internally by stage-boundary harness tools. NOT called by agents.
#
# USAGE (from store_round.sh):
#   bash "$(dirname "$0")/reinforce.sh" \
#       --round 18 \
#       --skill-path ".claude/skills/croq-tune/SKILL.md" \
#       --next-cmd "bash .../ncu_profile.sh --out ... --classify --iter ..."
#
# STDOUT: guardrail messages for the agent to see
# All messages go to stdout (mixed with the calling tool's output)

set -euo pipefail

ROUND=0
SKILL_PATH=""
NEXT_CMD=""
REFRESH_INTERVAL=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --round)            ROUND="$2"; shift 2 ;;
        --skill-path)       SKILL_PATH="$2"; shift 2 ;;
        --next-cmd)         NEXT_CMD="$2"; shift 2 ;;
        --refresh-interval) REFRESH_INTERVAL="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# ── L1: Behavioral reminder (every round) ─────────────────────
echo ""
echo "[NEVER STOP] This is an autonomous tuning loop."
echo "[NEVER STOP] Do NOT stop, ask questions, or checkpoint with user."
echo "[NEVER STOP] Proceed immediately to next round."

# ── L2: Continuation anchor payload (every round) ─────────────
echo ""
echo "[TODO] Update todos for next round:"
echo "[TODO]   TodoWrite(["
echo "[TODO]     { id: \"continue-croq-tune\", content: \"Continue /croq-tune loop\", status: \"in_progress\" },"
echo "[TODO]     { id: \"round-step\", content: \"Round $ROUND complete\", status: \"completed\" }"
echo "[TODO]   ])"

# ── L3: Next-action directive (every round) ────────────────────
if [[ -n "$NEXT_CMD" ]]; then
    echo ""
    echo "[NEXT] Proceed to PROFILE step for next round."
    echo "[NEXT] Run: $NEXT_CMD"
fi

# ── L4: Protocol re-read (every N rounds) ─────────────────────
if [[ -n "$SKILL_PATH" ]] && (( ROUND % REFRESH_INTERVAL == 0 )) && (( ROUND > 0 )); then
    echo ""
    echo "[PROTOCOL REFRESH] Round $ROUND reached (refresh interval: $REFRESH_INTERVAL)."
    echo "[PROTOCOL REFRESH] Before next round, re-read the protocol:"
    echo "[PROTOCOL REFRESH]   Read file: $SKILL_PATH"
    echo "[PROTOCOL REFRESH]   Focus: Per-Round Contract (steps 1-8), Behavioral Rules"
fi
```

### Integration Point

The calling tool (e.g., `store_round.sh`) adds one line at the end:

```bash
# At the very end of store_round.sh, after success message:
bash "$(dirname "$0")/reinforce.sh" \
    --round "$ROUND" \
    --skill-path ".claude/skills/croq-tune/SKILL.md" \
    --next-cmd "bash .claude/skills/croq-tune/tools/ncu_profile.sh --out ${NCU_BASE}/ncu_${BEST_ITER} --cmd ${BEST_BIN} --classify --iter ${BEST_ITER}"
```

### What the Agent Sees (example, round 18)

```
[store_round] STORE complete for iter018 (KEEP 14.2 TFLOPS)
[store_round] Written: 4 files verified.

[NEVER STOP] This is an autonomous tuning loop.
[NEVER STOP] Do NOT stop, ask questions, or checkpoint with user.
[NEVER STOP] Proceed immediately to next round.

[TODO] Update todos for next round:
[TODO]   TodoWrite([
[TODO]     { id: "continue-croq-tune", content: "Continue /croq-tune loop", status: "in_progress" },
[TODO]     { id: "round-step", content: "Round 18 complete", status: "completed" }
[TODO]   ])

[NEXT] Proceed to PROFILE step for next round.
[NEXT] Run: bash .claude/skills/croq-tune/tools/ncu_profile.sh --out ...
```

### What the Agent Sees (round 20, with protocol refresh)

Same as above, plus:
```
[PROTOCOL REFRESH] Round 20 reached (refresh interval: 5).
[PROTOCOL REFRESH] Before next round, re-read the protocol:
[PROTOCOL REFRESH]   Read file: .claude/skills/croq-tune/SKILL.md
[PROTOCOL REFRESH]   Focus: Per-Round Contract (steps 1-8), Behavioral Rules
```

### Why This Is the Ideal Design

1. **Zero agent tokens spent on guardrails** — the messages appear automatically
2. **Zero agent decisions about guardrails** — it doesn't choose to call reinforce.sh
3. **Impossible to forget** — it's called by the harness, not by the agent
4. **All 4 layers in one script** — no scattered guardrail logic
5. **Configurable** — refresh interval, next command, skill path are parameters
6. **Non-breaking** — if reinforce.sh fails, store_round.sh still completed
   successfully (the store is already done before reinforce runs)
