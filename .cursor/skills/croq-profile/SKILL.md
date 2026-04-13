---
name: croq-profile
description: On-demand profiling contract for croq-tune. Use only when profiling the current best or deciding whether profiling evidence is sufficient for the next idea.
---

# Croq-Profile

Use this skill only when profiling is needed.

## Core Rule

Profile before every new idea.

Choose the lightest profiling signal that still answers the current question:

1. compiler diagnostics or build output
2. runtime timing deltas
3. targeted metrics
4. full `ncu`

## Mandatory Full-Profile Triggers

Run a full `ncu` pass when any of these is true:

1. baseline characterization or first real iteration
2. the previous round found a new best
3. consecutive discards indicate the search is stalled
4. the current evidence is too weak to justify the next idea
5. the optimization category is changing because the search is stuck
6. PTX, SASS, inline PTX, or inline ASM work is being considered

## Output Contract

- save profile artifacts under `tuning/aitune/<dsl>/perf/<key>/`
- identify the bottleneck category from the best available evidence
- feed that bottleneck into the next ideation step
- provide a concise handoff object for `IDEA`: `{ bottleneck, confidence, evidence_paths }`
- if confidence is weak, explicitly request targeted web-search input before finalizing the idea

This skill owns detailed profiling behavior so the entry skill does not need to carry it in hot context all the time.
