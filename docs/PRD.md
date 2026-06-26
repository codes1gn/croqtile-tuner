# CroqTile-Tuner v2 — PRD

> 从 skill-driven (SKILL.md + IDE) 迁移到 standalone binary (TS + Pi SDK + Zod)

## Problem

当前系统依赖 Cursor/Claude Code/OpenCode 读取 SKILL.md 来执行 tuning loop。问题:

1. **不可编程** — 不能程序化控制 loop (retry, gate, routing)
2. **不可监控** — agent 行为是黑盒，无法实时观察
3. **模型锁定** — 绑定 IDE 支持的模型
4. **不可分发** — 用户必须装 IDE + 配技能文件才能用
5. **无法做 grey testing / RL** — 没有结构化轨迹数据

## Target Users

- 我自己 (kernel 优化研究者 / 工具开发者)
- 未来: 任何有 GPU + LLM API key 的人

## Solution

一个独立 binary，两种使用模式:

```bash
# Mode 1: Non-interactive (config 全定义好)
$ ./croqtile-tuner --config config.yaml

# Mode 2: Interactive (chat-like, 问答对齐 task)
$ ./croqtile-tuner --interactive
> optimize my CroqTile matmul
? Dtype? fp16
? Shape (MxNxK)? 16384x16384x16384
? Target? beat cuBLAS
→ Starting tuning...
```

Task 定义需要: DSL + Operation + Dtype + Shape + Target + Baseline。
Schema 从实践中 emerge，不预先过度设计。

## Architecture

```
croqtile-tuner (standalone TS binary)
├── Config (YAML → Zod validated)
├── Orchestrator (macro loop, deterministic gates)
│   ├── PROFILE → IDEA → IMPLEMENT → VERIFY → MEASURE → DECIDE → STORE
│   ├── Model router (per-step model mapping)
│   └── Retry logic (max_retries, timeout, fallback)
├── Agent Runtime (Pi SDK, embedded)
│   ├── Tools: bash, read, write, edit (LLM familiar env)
│   ├── Custom tools: compile, benchmark (optional inject)
│   ├── Session management (Pi built-in)
│   └── max_turns control
├── Verification (Zod schemas)
│   ├── Output validation gates
│   └── Correctness tests
├── Knowledge Modules (Pi skills / system prompts)
│   ├── DSL-specific: croqtile, triton, helion, cuda, ...
│   ├── Hardware specs: sm90, H800, ...
│   └── Playbooks: optimization patterns
├── Trajectory Recorder (JSONL)
│   └── Every tool call + LLM response → future RL/grey-test data
└── Monitor Interface (event emitter, SSE-ready)
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | TypeScript | 与 Pi SDK 同语言, 无桥接 |
| Runtime | Bun | 原生 TS, `bun build --compile` 出 binary |
| Agent Runtime | Pi SDK (embedded) | 真实 coding env → LLM 最佳质量输出 |
| Validation | Zod | TS 的 Pydantic, schema + type inference |
| Robustness | 双层: agent tools + orchestrator gates | Belt AND suspenders |
| LLM env | bash/read/write/edit | 与 Claude Code 训练一致 |
| Model support | Pi 内置 20+ providers | Model-agnostic, config 切换 |
| Data format | 兼容现有 tuning/ 目录结构 | 渐进迁移不丢数据 |

## Dual-Layer Robustness

```
Orchestrator (deterministic lock)
  └── Agent (has tools, can self-fix — low-hanging fruit)
       └── write → compile → error → fix → retry (agent does this)
  └── Gate check (orchestrator ensures correctness)
       └── compile passed? max_turns? quality OK?
```

- Agent 有 tool 能力 → 发挥 LLM 训练优势 (cheap bonus)
- Orchestrator 有确定性检查 → 不依赖 agent 可靠性

## Scope

### v1 (MVP)

- [ ] 单任务 tuning (1 shape, 1 op, 1 DSL)
- [ ] Pi SDK embedded agent (bash/read/write/edit)
- [ ] Config file: model endpoint + API key + task params
- [ ] Per-step model mapping (config 指定)
- [ ] Compile + benchmark + profile 自动化
- [ ] Trajectory JSONL recording
- [ ] Zod verification gates
- [ ] 复用现有 .claude/skills/croq-tune/tools/*.sh

### v1 预留接口 (不实现)

- [ ] Multi-model grey testing
- [ ] Custom RL-tuned model 接入
- [ ] Monitor web UI (事件格式先定好)
- [ ] Pi extensions for domain-specific behavior

### 明确不做

- RL training pipeline (只收集数据)
- Web UI / Monitor 实现
- 跨任务学习
- 分布式多机
- Self-training flywheel

## Migration Path

```
Phase 1: Coexist
  新 TS 系统 alongside 旧 skill 系统。旧系统照常用。
  先让 TS 系统对一个 DSL (croqtile) 跑通。

Phase 2: Parity
  TS 系统功能 >= 旧系统 for croqtile DSL。
  验证 tuning 质量不退步。

Phase 3: Expand
  逐 DSL 迁移 (triton, cuda, helion, ...)。
  知识从 SKILL.md → Pi skills/system prompt。

Phase 4: Deprecate
  旧 skill 系统标记 deprecated。
  Monitor 切到 TS 实现。
```

## Config Schema (draft)

```yaml
model:
  provider: anthropic              # or openai, google, local, ...
  model: claude-sonnet-4-6
  api_key: ${ANTHROPIC_API_KEY}    # env var expansion

task:
  dsl: croqtile
  op: matmul
  dtype: fp16
  shape: [16384, 16384, 16384]

orchestrator:
  max_rounds: 50
  max_agent_turns: 30              # per-step agent turn limit
  timeout_per_step_s: 300

steps:
  idea:
    model: claude-sonnet-4-6       # can override per-step
  implement:
    model: claude-sonnet-4-6
  # future: different models per step for grey testing

output:
  dir: ./tuning/results
  trajectory: true                 # record all tool calls to JSONL
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | TypeScript (ESM, strict) |
| Runtime | Bun |
| Agent | Pi SDK (`@earendil-works/pi-coding-agent`) |
| Schema | Zod |
| Config | YAML → Zod parse |
| Binary | `bun build --compile` |
| Shell scripts | Reuse existing bash (build_iter.sh, ncu_profile.sh, ...) |
| Future Monitor | Fastify + SSE |
| Future Frontend | TBD |

## References

- Design research: `docs/ai-kernel-tuning-landscape.md` (§11 for runtime selection rationale)
- Existing skills: `.claude/skills/croq-tune/SKILL.md` (domain knowledge source)
- Pi SDK integration example: OpenClaw (`github.com/openclaw/openclaw`)
