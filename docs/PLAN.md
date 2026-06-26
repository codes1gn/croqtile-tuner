# CroqTile-Tuner v2 — Development Plan

## Philosophy

递归进化。每一步可跑。Schema 从实践 emerge。
每个 task 是一个 commit-size 的 goal，不是一个 component。

---

## Iteration 0: Proof-of-life

**Goal**: Pi SDK 跑起来，agent 能做一件事。

| # | Goal | Done when |
|---|------|-----------|
| 0.1 | bun project + tsconfig strict ESM | `bun run src/main.ts` 打印 "hello" |
| 0.2 | 装 pi-coding-agent, import 不报错 | 编译通过 |
| 0.3 | 用 hardcoded API key 创建 agent session | session 对象存在，无 crash |
| 0.4 | agent 执行 "write hello.cu" | workspace 里出现 hello.cu |
| 0.5 | 观察 agent 行为 — 用了哪些 tools, 几轮完成 | 有 mental model |

---

## Iteration 1: Agent 写+编译 kernel (自循环)

**Goal**: Agent 能自主 write→compile→fix→compile 直到通过。

| # | Goal | Done when |
|---|------|-----------|
| 1.1 | 改 prompt: "write a simple matmul, compile with nvcc" | agent 尝试 bash nvcc |
| 1.2 | 确认 agent 看到 compile error 后会 edit 代码 | 至少自修一次 |
| 1.3 | 加 max_turns 限制防止跑飞 | 超限时 graceful 停止 |
| 1.4 | 产出可编译 kernel | nvcc 编译通过 (exit code 0) |
| 1.5 | 如果失败: 调整 prompt/model/max_turns 直到成功 | 稳定复现成功 |

---

## Iteration 2: 外层循环

**Goal**: 多轮优化，每轮基于上一轮结果改进。

| # | Goal | Done when |
|---|------|-----------|
| 2.1 | 提取 model/workspace 到变量 (最小 config) | 不再 hardcode |
| 2.2 | 外层 for loop 跑 N 轮 | 3 轮完成不崩 |
| 2.3 | 每轮把上一轮 kernel + 结果传给 agent | agent prompt 包含 prev context |
| 2.4 | 每轮输出存为 iter001.cu, iter002.cu | 文件存在且不同 |
| 2.5 | 简单 stdout log: "Round 1/3 complete" | 知道进度 |

---

## Iteration 3: Profile + Measure

**Goal**: 用真实数据驱动优化，知道每轮好不好。

| # | Goal | Done when |
|---|------|-----------|
| 3.1 | 调 ncu_profile.sh 获取 baseline 性能 | 有 TFLOPS 数字 |
| 3.2 | 把 profile 数据注入 agent prompt | agent 看到 "bottleneck: memory bandwidth" |
| 3.3 | 每轮跑 benchmark 拿 TFLOPS | 每轮有数值 |
| 3.4 | 比较: 这轮 vs baseline, 这轮 vs 上轮 | 打印 "+3.2% vs baseline" |
| 3.5 | 用性能变化决定 "accept/reject 这轮 kernel" | 有 decision 逻辑 |

---

## Iteration 4: 真实 DSL — CroqTile

**Goal**: 对真实 CroqTile kernel 跑完整 tuning loop。

| # | Goal | Done when |
|---|------|-----------|
| 4.1 | 从 SKILL.md 提取 CroqTile 知识，注入 prompt | agent 知道 .co 语法 |
| 4.2 | 用 build_iter.sh 替代 nvcc | CroqTile kernel 能编译 |
| 4.3 | 选一个真实 task (matmul fp16 16384) | 有 baseline 数据 |
| 4.4 | 跑 5+ 轮 | 不崩 |
| 4.5 | 至少一轮有改进 (或正确判断无改进) | 系统 make sense |
| 4.6 | 用 store_round.sh 存结果 | 结果目录结构正确 |

---

## Iteration 5: 加固

**Goal**: 系统在 failure 下稳定，有可回溯的数据。

| # | Goal | Done when |
|---|------|-----------|
| 5.1 | Agent 超时 → orchestrator 杀掉并 retry | 不 hang |
| 5.2 | Agent 产出垃圾 → 被检测到并跳过 | 不把坏 kernel 存为"改进" |
| 5.3 | API rate limit → 等待并重试 | 不崩退 |
| 5.4 | 记录 trajectory (每个 tool call + response) → JSONL | 文件可 parse |
| 5.5 | 从 trajectory 能回答 "agent 为什么做了这个决定" | 可调试 |
| 5.6 | 现在 config 该有哪些字段已经清楚 → 写 Zod schema | config 验证有意义 |

---

## Iteration 6: 打包

**Goal**: 别人能用。

| # | Goal | Done when |
|---|------|-----------|
| 6.1 | config.yaml 从所有硬编码中 emerge | 一个文件控制一切 |
| 6.2 | `bun build --compile` 出 binary | binary 存在 |
| 6.3 | binary 在无 bun 环境跑通 | 新机器上跑 |
| 6.4 | `--help`, `--version` 工作 | 基本 CLI UX |
| 6.5 | SIGINT graceful shutdown | Ctrl+C 保存当前状态 |
| 6.6 | config.example.yaml + 简短使用说明 | 别人能上手 |

---

---

## Open Design Thread: Task Definition & Interactive Mode

**问题**: 用户如何告诉 tuner "要做什么"？

**两种模式 (都需要支持):**
1. **Config (non-interactive)**: task 全在 YAML 定义，binary 直接跑
2. **Interactive (chat-like)**: 用户说 "optimize my matmul"，系统问答对齐

**Task 至少需要的信息 (从实践 emerge，不预设 schema):**
- DSL (croqtile / triton / cuda)
- Operation (matmul / conv / attention)
- Dtype (fp16 / bf16 / e4m3)
- Shape (MxNxK or custom)
- Target (beat cuBLAS? maximize TFLOPS? reach X% peak?)
- Baseline reference

**何时解决**: Iter 2-3 过程中自然发现需要什么字段。Interactive 模式在 Iter 6 时考虑。

---

## 跨 iteration 规则

- 每步**实际运行**验证 — 不是代码写完
- 允许后退: 如果 iter N 发现 iter N-1 的东西不对，先修再前进
- 不预设 schema — 等 pattern 稳定后再加 type/validation
- Commit message 粒度 = 上表一行

## Future Iterations

- Iter 7: 多模型 + per-step routing
- Iter 8: Grey testing
- Iter 9: Monitor (TS)
- Iter 10+: 其他 DSL, RL model, self-training
