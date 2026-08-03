# croqtile-tuner v2 — standalone tuning binary

AI-driven GPU kernel tuning loop (TypeScript + Pi SDK). This is the v2
migration from the skill-driven system (SKILL.md + IDE) to a programmable,
monitorable binary.

## Quick start

```bash
npm install
cp config.example.yaml config.yaml   # edit task fields
npm start -- --config config.yaml
```

Or fully CLI-driven:

```bash
npm start -- --kernel kernel.co \
  --build "bash build_iter.sh" \
  --profile "./iter000_swizzle 2048 2048 2048" \
  --dsl croqtile --rounds 5 --store
```

Requires either an LLM API key (`<PROVIDER>_API_KEY` env or `model.api_key`
in config) or a local ollama server (`model.provider: ollama`).

## How it works

Each round: the agent (Pi SDK, read/write/bash tools) makes **one** targeted
change → the orchestrator runs a deterministic build gate (compile must
pass) → benchmarks → parses TFLOPS → compares vs baseline and previous
round → **KEEP** or **REJECT** (regressions are reverted to the best-known
iteration). The agent's self-reported numbers are never trusted — all
measurement is orchestrator-side.

Artifacts in `<cwd>/iters/`:

| File | Content |
|---|---|
| `iter000.*` … `iterNNN.*` | per-round kernel snapshots (iter000 = baseline) |
| `trajectory.jsonl` | full agent trajectory — every tool call + response, per round |

With `--store`, each measured round also persists via the skill system's
`store_round.sh` into `tuning/<gpu>/<dsl>/logs/<shape_key>/<model>/`
(identical format to the old skill system, so data stays compatible).

## Config

All behavior is controlled by one YAML file (see `config.example.yaml`):
model endpoint, task (kernel/build/profile/dsl/gpu/shape-key), and
orchestrator (rounds, store, per-round timeout). CLI flags override config
values.

## DSL knowledge

`--dsl <name>` injects `.claude/skills/croq-dsl-<name>/SKILL.md` (syntax,
build pipeline, IDEA menu) into the agent's system prompt — knowledge
transfer, not rewrite.

## Tests

```bash
npm test             # unit + integration (faux sessions, no API needed)
npm run test:ollama  # local-model provider tests
npm run test:live    # live API smoke test (needs an API key)
```
