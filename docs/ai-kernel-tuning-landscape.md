# AI Kernel Tuning 行业调研与设计参考

> 调研日期: 2026-06-25
> 目的: 为 croqtile-tuner 架构转型提供行业设计参考

---

## 一、行业全景分类

AI 内核调优系统可按驱动模式分为五大流派：

| 流派 | 代表系统 | 搜索方式 | 代码修改层 |
|------|---------|----------|-----------|
| 编译器驱动 | TVM/Ansor, XLA, Triton autotuner | 参数空间搜索 | Schedule/配置参数 |
| 进化算法驱动 | NVIDIA CompileIQ | 遗传/进化算法 | 编译器内部控制参数 |
| 单 Agent 循环 | AutoKernel, TritonForge | LLM hill-climb | 内核源代码 |
| 多 Agent 协作 | Astra, KernelEvolve, AKG Agent, Two-Stage Tuner | 多角色协作 | 内核源代码+参数 |
| 分层 RL + LLM | MTMC/QiMeng-Kernel | RL策略+LLM实现 | 优化动作序列→代码 |

---

## 二、编译器驱动系统

### 2.1 Apache TVM — AutoTVM & Ansor

**架构:**
- **AutoTVM (第一代)**: 模板驱动 → 人类专家定义 schedule 模板 + 可调旋钮 → XGBoost 代价模型 + 搜索算法探索参数空间
- **Ansor (第二代)**: 无模板 → 从 DAG 自动推导搜索空间 → 分层 (Sketch + Annotation) → 进化搜索 + 学习代价模型

**核心设计:**
```
输入: Tensor Expression (TE)
     ↓
搜索空间构造 (自动/模板)
     ↓
┌─────── 搜索循环 ───────┐
│ Tuner 提议配置           │
│ Runner 编译+设备上测量    │
│ 代价模型更新             │
└─────────────────────────┘
     ↓
输出: 最优 Schedule → TIR → 机器码
```

**关键设计决策:**
- Compute 与 Schedule 分离 — 同一计算可有多种调度
- 代价模型（XGBoost/Random Forest）减少设备测量次数
- Task Scheduler 在子图间分配调优预算（基于 Amdahl 定律启发）
- 持久化缓存 JSON 格式调优结果

**局限:** 搜索空间受限于预定义规则，无法做结构性代码变换。

---

### 2.2 Google XLA Autotuning

**架构:**
```
HLO Module → Fusion Pass → Autotuner Pass
                              ↓
              ┌──────────────────────────────┐
              │ 后端: Triton / cuBLAS-LT     │
              │ 生成候选配置                   │
              │ GpuIndexingPerformanceModel    │
              │   → 过滤候选 (分析模型)        │
              │ 设备上 Profile                 │
              │   → 选择最优                   │
              │ 缓存到磁盘                     │
              └──────────────────────────────┘
```

**关键设计决策:**
- 编译 pipeline 中的 Pass（不是独立工具）
- 多后端支持: Triton codegen + cuBLAS-LT 同时作为候选
- 分析模型做初筛 + 实际 profiling 做最终决定
- 分布式 autotuning: 跨主机分片 HLO + KV store 同步结果
- 持久化: `--xla_gpu_dump_autotune_results_to` / `--xla_gpu_load_autotune_results_from`
- 粒度: Per-fusion，每个 fusion 独立 autotune
- Level 控制: `--xla_gpu_autotune_level` (0-4) 控制激进程度

**与传统 autotuner 的区别:** XLA 不直接修改内核代码，而是选择最优的 codegen 后端 + 配置组合。

---

### 2.3 Triton Autotuner

**架构:**
```python
@triton.autotune(
    configs=[Config(BLOCK_SIZE=128, num_warps=4), ...],
    key=['x_size']
)
@triton.jit
def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
    ...
```

**核心设计:**
- 用户显式列举候选配置（非自动搜索空间生成）
- 运行时 benchmark 选择最优配置
- `key` 参数决定何时重新 autotune（按输入特征分桶）
- 支持 `prune_configs_by` 用性能模型预筛选
- 结果缓存到磁盘避免重复 tuning

**最新进展 (2025-2026):**
- **tritonBLAS**: 分析模型完全替代运行时搜索（零调优开销）
- **triton-dejavu**: 跨部署生命周期复用 autotune 结果
- **并行编译**: 多进程并发编译 kernel 变体
- **决策树启发式**: 将 autotune 结果导出为简单 if-else 规则

**局限:** 搜索空间由人工定义，只搜索参数不改代码结构。

---

### 2.4 Helion Autotuner (PyTorch/Meta)

**架构:**
```
@helion.kernel → 隐式搜索空间构造
    ↓
┌─── Autotuning Loop ───┐
│ LFBO (贝叶斯优化)      │
│  → 轻量 RF 分类器      │
│  → 预测有前景的配置    │
│  → 编译+基准测试       │
│  → 更新模型            │
└────────────────────────┘
    ↓
最优 Config → 锁定到生产
```

**核心创新:**
- **隐式搜索空间**: 单个 `hl.tile` 调用自动展开为数千种 Triton 变体
- **LFBO (Likelihood-Free Bayesian Optimization)**: 默认搜索策略
- **LLM-Guided Autotuning (2026)**: LLM 推理内核结构 → 提议配置 → 10x 减少 compile/bench 次数
- **混合搜索**: LLM seeding (快速到达好配置) + LFBO (精细搜索)
- 最终输出 Triton 代码 → 用户可审查

**设计哲学:** 提高抽象层级，让 autotuner 搜索实现细节（block size, loop order, indexing, warp count 等）。

---

### 2.5 TileLang Autotuner (Microsoft)

**架构:**
```
用户: 定义 dataflow (T.gemm, T.copy)
  + 保留优化参数 (block_M, block_N, ...)
      ↓
Carver: 生成候选配置 (分析模型排序 top-k)
      ↓
AutoTuner: 并行编译+benchmark → 选最优
      ↓
编译器: Layout推理 + Pipeline推导 + Warp特化
```

**关键设计决策:**
- Dataflow 与 Scheduling 解耦
- Carver 框架: 用硬件模型预排序候选，减少搜索量
- 编译器自动推导: 线程绑定、内存布局、软件流水线
- 支持用户手动 override 编译器决策

---

### 2.6 NVIDIA CompileIQ

**架构:**
```
输入: 已有内核源码 (CUDA/Triton/Helion)
      ↓
定义: 目标函数 (runtime/power/compile_time)
      ↓
┌─── 进化搜索 ───┐
│ 编译器搜索空间  │  ← PtxasSearchSpace / NvccSearchSpace
│ 候选 ACF 生成   │
│ 编译+测量       │
│ 进化引擎迭代    │
│ Pareto 前沿     │
└─────────────────┘
      ↓
输出: Advanced Control File (ACF)
      → --apply-controls 编译时使用
```

**核心创新:**
- **层级不同**: 不改源码，只调编译器内部参数（寄存器分配策略、指令调度策略、循环变换等）
- **多目标优化**: Pareto 前沿 (runtime vs compile_time vs power)
- **可复现**: ACF 文件可版本控制，同一 ACF 产生同一二进制
- **补充性**: 在源码级优化已经到瓶颈后的额外 5-15% 提升
- **分布式**: RayWorker / MultiProcessWorker 并行搜索

**适用场景:** 源码级调优后的"最后一英里"优化。Meta 在 TritonBench 和 Helion 上获得 15% 性能提升。

---

## 三、LLM Agent 驱动系统

### 3.1 AutoKernel (RightNow AI)

**架构 — 极简单 Agent 循环:**
```
Phase A: 模型 Profiling → 提取瓶颈内核 (Amdahl 排序)
Phase B: ┌── Agent Loop ──┐
         │ 编辑 kernel.py  │
         │ bench.py 验证    │  ← 5 阶段正确性检查
         │ KEEP/REVERT     │
         │ git commit/reset│
         └─────────────────┘
Phase C: 端到端验证 + 总加速比
```

**核心设计决策:**
- **单文件修改**: Agent 只碰 `kernel.py`，scope 最小化
- **909 行 program.md**: 6 层优化 playbook 编码专家知识
- **Git 作为状态管理**: KEEP = commit, REVERT = git reset
- **results.tsv**: 纯文本日志，无依赖，Agent 可读
- **Amdahl 编排**: orchestrate.py 基于影响排序决定下一个优化目标
- **~90 秒/迭代**: 一夜 300-400 个实验
- **双后端**: Triton (快速迭代) + CUDA C++ (极致性能)

**六层 Playbook:**
1. Block size tuning (tile 维度, num_warps, num_stages)
2. Memory access (coalesced loads, prefetch, L2 swizzle)
3. Compute (TF32, epilogue fusion)
4. Advanced (split-K, persistent kernel, warp specialization)
5. Architecture-specific (TMA/Hopper, cp.async/Ampere)
6. Kernel-specific algorithms (online softmax, Welford's)

**与 croqtile-tuner 对比:**
| 维度 | AutoKernel | CroqTile-Tuner |
|------|-----------|----------------|
| 复杂度 | 9,200 行 Python | ~5,000 行 skill/harness scripts |
| Agent 指令 | 909 行 program.md | ~750 行 SKILL.md |
| 状态管理 | Git commit/reset | results.tsv + idea-log.jsonl + checkpoints |
| 编排 | Amdahl orchestrator | 手动指定 shape_key |
| 多 DSL | Triton + CUDA C++ | 7 DSL (croqtile, cuda, cute-dsl, cute-cpp, triton, tilelang, helion) |
| Profiling | 无 ncu (只用 bench.py) | ncu 深度 profiling + 瓶颈分类 |
| 搜索策略 | LLM hill-climb | LLM hill-climb + 强制结构多样性 |

---

### 3.2 KernelEvolve (Meta, ISCA 2026)

**架构 — 生产级图搜索:**
```
输入: Kernel Specification
      ↓
Retrieval-Augmented Prompt Synthesis
  → 硬件约束 + 历史信号 + 运行时诊断
      ↓
┌──── Graph-Based Search ────┐
│ Selection Policy            │  ← UCB/Thompson Sampling
│ Universal Operator (LLM)    │
│ Fitness Function (Profiling)│
│ Termination Rule            │
└─────────────────────────────┘
      ↓
多后端输出: Triton / CuTe DSL / FlyDSL / CUDA / HIP / MTIA C++
```

**核心设计决策:**
- **图搜索而非线性循环**: 优化过程建模为树/图，支持分支探索
- **RAG 知识库**: 持久化硬件约束 + 历史优化记录 → 动态注入 prompt
- **统一适配 prompt**: 不用多个模板，单一自适应接口统一 debug/tune/verify
- **多层 profiling 堆栈**:
  - TritonBench: 正确性 + 加速比
  - PyTorch Profiler: 系统级执行时序
  - NCU: 内核级硬件指标 (occupancy, throughput, instruction mix)
  - Proton: 指令级延迟和流水线行为
  - MTIA Insight: 自研加速器指标
- **跨平台**: NVIDIA GPU + AMD GPU + CPU + MTIA
- **100% KernelBench 通过率** (250 题全通)
- **开发周期**: 从周缩短到小时

**独特之处:** 将优化视为图搜索问题 + 知识库驱动的 RAG prompt，是目前工业界最成熟的系统。

---

### 3.3 TritonForge (学术)

**架构:**
```
┌── Pipeline ──┐
│ Test Generator (A): 理解代码 → 生成性能测试   │
│ Profiling Module: NCU 采集指标 → 识别瓶颈      │
│ Kernel Optimizer (B): 读取指标 → 提出代码修改  │
│ Fault-Aware Remediation: 处理编译/运行时错误   │
└──── 迭代循环 ────┘
```

**设计特点:**
- Profiling 数据直接反馈到代码生成过程
- 模块化，模型无关 (可换 LLM)
- 平均 1.76x 加速，最高 5x

---

### 3.4 Astra (Stanford, NeurIPS 2025)

**架构 — 四 Agent 协作:**
```
┌─────────────────────────────────────────┐
│           Iterative Loop (R=5 rounds)   │
│                                         │
│  Testing Agent  → 正确性验证             │
│  Profiling Agent → ncu 性能分析          │
│  Planning Agent  → 分析瓶颈+提出策略     │
│  Coding Agent    → 实施代码修改          │
│                                         │
└─────────────────────────────────────────┘
```

**设计特点:**
- 从已有 CUDA 实现出发（SGLang 内核），非从 PyTorch 翻译
- 4 个专门化 Agent 各司其职
- 使用 OpenAI Agents SDK 实现
- 零 shot prompting + o4-mini 即可工作
- 平均 1.32x 加速

---

## 四、分层/混合架构

### 4.1 MTMC / QiMeng-Kernel (AAAI 2026)

**架构 — 策略与实现解耦:**
```
┌── Macro Thinking (RL-trained lightweight LLM) ──┐
│ DeepSeek-Coder-1.3B + PPO                       │
│ 输出: 语义优化动作 (tiling, loop fusion, ...)    │
│ 状态-动作-奖励 循环                              │
└────────────────────────────────────────────────┘
         ↓ 优化动作 (type + region)
┌── Micro Coding (General-purpose LLM) ──┐
│ Gemini 2.5 Pro                          │
│ 输入: 当前代码 + 动作 + in-context 示例 │
│ 输出: 增量代码修改                       │
│ 每步只做一个原子优化                     │
└──────────────────────────────────────────┘
         ↓
验证 → 反馈 → 下一轮 Macro Thinking
```

**核心创新:**
- **RL 训练的策略网络**: 轻量 LLM 学会优化策略排序
- **增量实现**: 每步只做一个原子修改，避免全内核生成错误
- **准确率飞跃**: KernelBench L1-2 近 100%, L3 70% (vs LLM 直接生成 <50%)
- **泛化**: 在 TritonBench 上不退化（vs KernelLLM 从 40% 崩到 2-4%）

**对 croqtile-tuner 的启示:** 考虑将"决定做什么优化"与"如何实现优化"分离为两个模块。

---

### 4.2 Two-Stage GPU Kernel Tuner (2026)

**架构 — 语义重构 + 参数搜索:**
```
Level 1 (Semantic): LLM Agent 做保持语义的结构重写
  → 改变 tiling 策略、内存层次、向量化、循环变换
  → 输出: 可参数化的 template kernel

Level 2 (Parameter): 搜索器在硬件约束下探索参数空间
  → BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES...
  → 约束: shared_mem <= max, threads <= max, registers <= max
  → 输出: 通用配置 + per-shape 最优配置表
```

**四 Agent 协作:**
- **Planning Agent**: 编排整体流程
- **Generation Agent**: L1 结构重写 + L2 模板化
- **Tuning Agent**: 推导可行域 + 搜索
- **Testing Agent**: 正确性 + 性能测量

**核心创新:**
- **解决 LLM 直接重写的不稳定性**: 将自由形式重写约束为模板化
- **可复现**: 参数化模板 + 确定性搜索 = 可解释、可重现的优化
- **SGLang 实测**: 超过 3x 加速
- **可扩展**: 框架可扩展到 OpenCL, HIP

**对 croqtile-tuner 的启示:** 当前设计的"IDEA → IMPLEMENT"可以细化为"结构重写 → 模板化 → 参数搜索"三步。

---

### 4.3 AKG Agent (Huawei/MindSpore)

**架构 — 通用多 Agent 平台:**
```
┌── 工作流编排 (LangGraph) ──┐
│                             │
│  Designer → Unified Sketch  │
│  Coder    → DSL 代码        │
│  Verifier → 编译+验证+Profile │
│  Conductor → 错误路由+迭代   │
│                             │
└─────────────────────────────┘
```

**核心设计决策:**
- **Document-Driven Integration (DDI)**: 新 DSL/硬件只需写文档，不改 Agent 代码
- **LangGraph 工作流**: 确定性流程 + ReAct Agent 开放探索
- **树状 Trace 系统**: 支持非线性探索和断点续跑
- **Skill 系统**: 可扩展的技能单元 + 动态知识注入
- **多后端**: Triton (Ascend/CUDA), TileLang, AscendC, CUDA-C, CPP
- **服务化架构**: Client-Server-Worker 分离
- **UCB 搜索策略**: 自适应搜索 + 进化算法

**对 croqtile-tuner 的启示:**
- DDI 模式: 当前 `croq-dsl-<dsl>` 已经是类似思路
- LangGraph 工作流: 比纯 skill 文本更有结构
- Trace 系统: 支持探索分支回溯

---

## 五、评测框架

### 5.1 KernelBench (Stanford)

- 250 个标准化 PyTorch ML 任务 (L1: 单算子, L2: 简单融合, L3: 完整架构)
- 指标: `fast_p` = 正确且加速 > p 的比例
- 支持迭代优化 + profiling 反馈
- 语言无关 (CUDA/Triton/ThunderKittens/CUTLASS 均可)
- 开源: github.com/ScalingIntelligence/KernelBench

### 5.2 MultiKernelBench

- 扩展到多平台: NVIDIA (CUDA) + Huawei NPU (AscendC) + Google TPU (Pallas)
- 285 个任务，14 个功能类别
- 模块化后端抽象层

---

## 六、设计模式总结与对比

### 6.1 核心设计维度

| 维度 | 选项 |
|------|------|
| 搜索方式 | 参数搜索 / LLM hill-climb / RL策略 / 进化算法 / 图搜索 |
| 代码修改粒度 | 编译器参数 / 配置参数 / 源码结构 / 全内核重写 |
| Agent 架构 | 单 Agent 循环 / 多 Agent 分工 / 分层 (策略+实现) |
| 状态管理 | Git / JSON checkpoint / 数据库 / 内存 |
| 知识注入 | 硬编码 playbook / RAG 知识库 / 文档驱动 / RL 学习 |
| 正确性保证 | 单测 / 多阶段验证 / 数值稳定性测试 / 确定性检查 |
| Profiling 深度 | 无 / timing-only / ncu 基础指标 / SASS 分析 / 全栈 |
| 多硬件支持 | 单平台 / 多 NVIDIA arch / 跨厂商 |
| 持久化/复现 | 无 / 缓存 / 版本控制 / 完整实验记录 |

### 6.2 CroqTile-Tuner 当前设计定位

```
搜索方式:    LLM hill-climb (+ 强制结构多样性规则)
代码修改:    源码结构级 (全内核变体)
Agent 架构:  单 Agent (SKILL.md 指令驱动)
状态管理:    文件系统 (results.tsv, idea-log.jsonl, checkpoints)
知识注入:    硬编码 playbook (SKILL.md 6+规则) + web search
正确性:      编译+运行验证 (单阶段)
Profiling:   ncu 深度 profiling + 瓶颈分类 + SASS 逆向分析
多硬件:      多 NVIDIA arch (sm86, sm90)
持久化:      Git commit + TSV/JSONL + raw session transcript
多 DSL:      7 个 DSL 独立 skill
```

### 6.3 行业最佳实践汇总

从各系统中提取的关键设计最佳实践：

**1. 搜索策略分层 (Two-Stage / MTMC)**
- L1 结构性变换 (LLM/RL): 改变搜索空间本身
- L2 参数调优 (搜索器): 在固定结构内找最优点
- 好处: 稳定、可复现、可解释

**2. 知识库驱动 (KernelEvolve / AKG)**
- 持久化硬件约束 + 历史优化 + 外部知识
- RAG 动态注入 prompt，而非硬编码
- 支持新硬件无需改代码

**3. 多层 Profiling (KernelEvolve / croqtile-tuner)**
- Timing → NCU metrics → SASS 分析 → Instruction-level
- 瓶颈分类自动化 (compute/memory/latency bound)
- Profiling 信号直接驱动下一步决策

**4. 正确性多阶段验证 (AutoKernel)**
- Smoke test → Shape sweep → Numerical stability → Determinism → Edge cases
- 在性能测量前确保正确性

**5. Amdahl 编排 (AutoKernel / KernelEvolve)**
- 基于影响力自动排序优化目标
- 避免在低影响内核上浪费时间

**6. 文档驱动集成 (AKG DDI)**
- DSL/硬件以文档形式接入，不改 Agent 代码
- 类似当前 `croq-dsl-<dsl>` 模式但更结构化

**7. 编译器级补充优化 (CompileIQ)**
- 源码级优化到瓶颈后，编译器参数还能再提 5-15%
- ACF 文件可版本控制、可复现

**8. LLM + 搜索器混合 (Helion / Two-Stage)**
- LLM seeding 快速到达好的起点
- 传统搜索器做精细调优
- 减少 10x 编译/benchmark 次数

---

## 七、对 CroqTile-Tuner 转型的启示

### 当前设计的优势

1. **深度 Profiling**: ncu + SASS 逆向是行业领先水平
2. **多 DSL 支持**: 7 个 DSL 覆盖面广
3. **强制多样性规则**: 避免 macro-only sweep 的局部最优
4. **完整实验记录**: 每轮 STORE 保证可追溯
5. **纯技能驱动**: 简洁，无需复杂基础设施

### 当前设计的不足

1. **纯线性循环**: 无分支探索能力（vs KernelEvolve 的图搜索）
2. **无分层搜索**: 结构变换和参数调优混在一起（vs Two-Stage）
3. **知识硬编码**: 6 层规则固定在 SKILL.md 中（vs RAG 动态注入）
4. **单阶段验证**: 只有编译+运行（vs AutoKernel 5 阶段）
5. **无自动编排**: 需要人工指定 shape_key（vs Amdahl 自动排序）
6. **Context 压力**: 长 session transcript 作为 memory 效率低
7. **无编译器级优化**: 缺少 CompileIQ 类的"最后一英里"

### 潜在转型方向

| 方向 | 对标 | 改动量 | 预期收益 |
|------|------|--------|---------|
| 搜索分层 | Two-Stage / MTMC | 中 | 稳定性+可复现 |
| RAG 知识库 | KernelEvolve / AKG | 大 | 可扩展性+新硬件适配 |
| 图搜索 | KernelEvolve | 大 | 探索效率 |
| 多阶段验证 | AutoKernel | 小 | 正确性保证 |
| Amdahl 编排 | AutoKernel | 中 | 自动化+影响力优先 |
| CompileIQ 集成 | NVIDIA CompileIQ | 小 | 额外 5-15% |
| LLM-guided 搜索 | Helion hybrid | 中 | 减少迭代次数 |
| 结构化 Memory | AKG Trace / KernelEvolve KB | 中 | Context 效率 |

---

## 八、编排模式分析：Program-Chained vs Skill-Driven

这是你最关心的核心问题。行业系统的编排模式可分为三类：

### 8.1 分类

| 编排模式 | 定义 | 系统 | 占比 |
|---------|------|------|------|
| **纯代码编排 (无 LLM Agent)** | 整个流程由代码/编译器驱动，无 LLM 参与决策 | TVM/Ansor, XLA, Triton autotuner, Helion LFBO, TileLang/Carver, CompileIQ | 6/14 (43%) |
| **代码编排 + Agent 执行** | 程序代码定义工作流，LLM Agent 只负责步骤内的具体任务 | KernelEvolve, Astra, TritonForge, MTMC, Two-Stage Tuner, AKG Agent | 6/14 (43%) |
| **Skill/指令驱动 (Agent 自编排)** | Agent 读取长指令文档，自己决定何时执行什么步骤 | **AutoKernel**, **CroqTile-Tuner** | 2/14 (14%) |

### 8.2 详细对比

#### A. 代码编排 + Agent 执行 (主流 LLM 方案, 6/8 Agent 系统)

```
┌── 代码 Orchestrator (Python/Framework) ──┐
│                                           │
│  while not terminated:                    │
│    candidate = llm_synthesizer(context)   │  ← Agent 只做生成
│    result = evaluator.run(candidate)      │  ← 代码驱动评测
│    if result.correct and result.faster:   │  ← 代码做决策
│      tree.keep(candidate)                 │
│    else:                                  │
│      tree.revert(candidate)              │
│    context = update_context(result)       │  ← 代码管理状态
│                                           │
└───────────────────────────────────────────┘
```

**KernelEvolve (Meta):**
- Python job harness 驱动迭代
- Tree Search Engine 是代码实现的图搜索
- LLM 是被调用的 "Synthesizer" 组件
- 状态管理、termination、backtracking 全在代码中
- 引述: "A purpose-built long-running job harness drives each iteration"

**Astra (Stanford):**
- OpenAI Agents SDK 定义 4 个 Agent + 工具
- 代码控制 R=5 轮迭代
- Agent 只做单步任务 (profiling/planning/coding/testing)
- 引述: "We implement our multi-agent system with the OpenAI Agents SDK framework"

**AKG Agent (Huawei):**
- LangGraph 显式定义工作流 DAG
- Designer → Coder → Verifier 是代码编排的状态机
- Conductor 做错误路由（也是代码逻辑）
- 引述: "LangGraphTask replaces original Task Orchestration scheme"

**MTMC/QiMeng:**
- Python pipeline 交替调用 Macro (RL模型) 和 Micro (LLM)
- 代码决定何时停止、何时回退
- RL agent 本身的训练也是代码编排的

**Two-Stage Tuner:**
- Algorithm 1 (论文伪代码) 定义完整流程
- 4 个 Agent 被代码调度
- 引述: "Algorithm 1 presents a two-level GPU tuning pipeline driven by four collaborative agents"

**TritonForge:**
- Pipeline stages 在代码中显式串联
- Agent 只在每个 stage 内生成/修改代码

#### B. Skill/指令驱动 (Agent 自编排, 2/8 Agent 系统)

```
┌── Agent (LLM) ──────────────────────────────────┐
│                                                   │
│  读取 program.md / SKILL.md (800-900行)           │
│    ↓                                             │
│  Agent 自己:                                     │
│    - 决定做什么 (选择优化方向)                     │
│    - 决定何时做 (判断何时 profile/implement/store) │
│    - 管理状态 (读写 results.tsv, checkpoints)     │
│    - 判断停止条件                                 │
│    - 调用工具执行 (shell scripts)                 │
│                                                   │
└───────────────────────────────────────────────────┘
```

**AutoKernel (RightNow AI):**
- Agent 读取 `program.md` (909行)
- 包含完整的 6 层 playbook + 决策框架 + crash handling
- `orchestrate.py` 只做 Amdahl 排序，不编排内循环
- 内循环（编辑→bench→keep/revert）完全由 Agent 自主驱动
- Git 作为状态管理，Agent 自己执行 commit/reset
- 引述: "The agent reads program.md -- the 'research org code' -- which contains comprehensive instructions for autonomous operation"

**CroqTile-Tuner (本项目):**
- Agent 读取 `SKILL.md` (~750行)
- 包含完整的 round loop protocol + 规则 + 状态管理
- 所有 harness scripts 是工具，不是编排者
- Agent 自己: 决定优化方向、管理 active bases、判断 KEEP/DISCARD、处理 resume
- 引述: "The loop runs until interrupted. You are autonomous."

### 8.3 两种模式的优劣

| 维度 | 代码编排 + Agent 执行 | Skill/指令驱动 (Agent 自编排) |
|------|---------------------|------------------------------|
| **可靠性** | ✅ 高 — 流程不会偏离 | ⚠️ 中 — 依赖 Agent 遵守指令 |
| **可调试性** | ✅ 高 — 代码可断点/日志 | ⚠️ 低 — Agent 行为不透明 |
| **灵活性** | ⚠️ 中 — 受代码路径约束 | ✅ 高 — Agent 可即兴应变 |
| **基础设施成本** | ❌ 高 — 需要实现 orchestrator 代码 | ✅ 低 — 只需写 SKILL.md |
| **可扩展性** | ✅ 高 — 加组件不影响其他 | ⚠️ 中 — SKILL.md 越长越难维护 |
| **状态一致性** | ✅ 高 — 代码保证 | ⚠️ 依赖 Agent 正确读写 |
| **并行/分布式** | ✅ 天然支持 | ❌ 难 — Agent 是单线程思维 |
| **长时间运行** | ✅ 代码不疲劳 | ⚠️ Context 压力+指令遗忘 |
| **新人上手** | ⚠️ 需要理解代码 | ✅ 读 SKILL 即可理解 |
| **迭代速度** | ⚠️ 改流程需改代码 | ✅ 改 SKILL.md 即可 |

### 8.4 行业趋势

**明确趋势: 代码编排是主流方向 (75% 的 Agent 系统)。**

原因:
1. **规模**: Meta、Huawei 需要 7x24 无人值守运行，不能依赖 Agent 自律
2. **复现性**: 学术论文和生产系统需要确定性可复现的结果
3. **并行**: KernelEvolve "评估数千候选并行" — 只有代码编排能做到
4. **状态管理**: 图搜索、backtracking 等复杂状态逻辑不适合文本指令
5. **关注点分离**: LLM 擅长生成代码，但不擅长长期状态管理

**但 Skill-Driven 也有存在理由:**
1. **原型速度**: AutoKernel 从 0 到工作系统只需写 program.md
2. **表达力**: 自然语言比代码更容易表达复杂决策规则
3. **适应性**: Agent 可以处理 SKILL 没预见到的情况
4. **成本**: 不需要工程团队维护 orchestrator 代码

### 8.5 混合模式 (推荐方向)

最优设计可能是混合模式:

```
┌── 轻量代码编排 (确定性骨架) ──┐
│                                │
│  代码保证:                     │
│    - 状态持久化                │
│    - 正确性验证                │
│    - Termination 逻辑          │
│    - Profile 调度              │
│    - 结果记录                  │
│                                │
│  Agent 负责 (Skill 指导):      │
│    - 优化方向决策 (IDEA)       │
│    - 代码生成 (IMPLEMENT)      │
│    - 瓶颈解读 (PROFILE 分析)  │
│    - Web search + 知识综合     │
│                                │
└────────────────────────────────┘
```

**类比:**
- 代码编排 = "公司制度" — 不变的流程保障
- Skill 指导 = "岗位职责" — 灵活的决策空间
- 现在的 CroqTile-Tuner = "所有规则都写在岗位手册里，期望员工自觉遵守"
- 目标 = "公司制度确保流程不跑偏，岗位手册指导员工做好本职工作"

### 8.6 模型能力 vs 编排模式：哪种适合中等/廉价模型？

**结论: 代码编排模式对中等/廉价模型友好得多。Skill 驱动模式几乎只能用前沿模型。**

#### 各系统实际使用的模型

| 系统 | 编排模式 | 使用模型 | 成本等级 |
|------|---------|---------|---------|
| AutoKernel | Skill-driven | Claude Opus / GPT-4o (frontier) | 💰💰💰 高 |
| CroqTile-Tuner | Skill-driven | Claude Opus / GPT-5 (frontier) | 💰💰💰 高 |
| KernelEvolve | Code-chained | 自训练小模型 (post-trained) | 💰 低 |
| Astra | Code-chained | o4-mini | 💰 低 |
| MTMC/QiMeng | Code-chained | DeepSeek-Coder-**1.3B** (策略) + Gemini 2.5 Pro (实现) | 💰💰 中 |
| AKG Agent | Code-chained | 通用 LLM (可配置) | 💰💰 中 |
| Two-Stage Tuner | Code-chained | 通用 LLM | 💰💰 中 |
| TritonForge | Code-chained | 通用 LLM | 💰💰 中 |
| Helion LLM-guided | Code-chained | 通用 LLM (只做 seeding) | 💰 低 |

#### 为什么代码编排适合廉价模型

**Skill-driven 对模型的要求：**
```
需要能力:
1. 在 800+ 行指令中精确遵循每一条规则 (长 context 指令跟随)
2. 跨越 50+ 轮迭代保持一致行为 (长期状态保持)
3. 自主判断何时 profile, 何时 store, 何时停止 (自律)
4. 正确管理文件状态 (results.tsv, checkpoints, idea-log.jsonl)
5. 不遗忘规则 (如 "强制结构多样性", "5 次失败后放弃")
6. 处理意外情况时不崩溃 (crash recovery, GPU contention)

只有前沿模型 (Opus, GPT-5) 才能做到这些。
中等模型 (Sonnet, GPT-4o-mini) 会:
- 忘记规则 (第 30 轮开始忘记 "profile before every idea")
- 状态管理出错 (写错 TSV 格式, 跳过 STORE)
- 决策质量下降 (context window 被历史填满)
```

**Code-chained 对模型的要求：**
```
每次 LLM 调用只需要:
1. 读取当前内核 + profiling 数据 + 少量历史
2. 提出一个优化想法并实现代码
3. 就这么多。

代码负责:
- 调用 ncu (不需要 LLM 判断"该不该 profile")
- 管理 results.tsv (不需要 LLM 写文件)
- 判断 KEEP/DISCARD (比较数字，代码做)
- Termination (代码检查条件)
- 错误恢复 (代码捕获异常)
- 状态持久化 (代码保证原子性)

LLM 只做它最擅长的: 理解代码 + 生成代码。
```

#### 量化对比

| 维度 | Skill-driven (每轮) | Code-chained (每次调用) |
|------|--------------------|-----------------------|
| Prompt 长度 | ~800 行 SKILL + 累积历史 | ~100-200 行 (内核+profile+任务) |
| 需要记住的规则 | 所有 (~30条) | 当前步骤的 (~3-5条) |
| 决策复杂度 | 全局 (方向+实现+状态管理) | 局部 (只做一个步骤) |
| 容错空间 | 无 — 一步错全链崩 | 高 — 代码捕获+重试 |
| Context 使用效率 | 低 — 大量用于规则+历史 | 高 — 几乎全给当前任务 |

#### MTMC 的启示 — 极端廉价化

MTMC/QiMeng 最极端: 策略模型只用 **DeepSeek-Coder-1.3B** (极小模型)。原因：

1. 代码编排将"选优化动作"和"实现优化"分离
2. 策略模型只输出一个 token 序列: `<action_type> <target_region>`
3. 输出空间极小，不需要大模型
4. RL 训练让小模型学会正确的策略排序
5. 具体实现才用大模型 (Gemini 2.5 Pro)，但每次只做一个原子修改

**这意味着:** 如果用代码编排，甚至可以用 RL 训练一个 1B 参数模型做策略决策，只在代码实现步骤用 frontier 模型。

#### KernelEvolve 的启示 — 自训练飞轮

Meta 的路径更激进:
1. Frontier 模型运行 → 产生 agentic trajectories 训练数据
2. Post-train 小模型 (agentic RL, reward = 实测性能)
3. 小模型替代 frontier 模型运行 → 产生更多数据
4. 循环迭代 → 越来越便宜越来越好

引述: "This compounding flywheel enables us to self-host increasingly efficient models that are compact enough to run cost-effectively at scale while retaining the optimization capability of much larger frontier models."

**但前提是:** 必须是代码编排模式，因为:
- 需要结构化 trajectory 数据来训练（代码编排天然产生）
- 小模型无法遵循 800 行 SKILL.md
- 需要确定性 pipeline 保证训练数据质量

#### 对 CroqTile-Tuner 的直接影响

| 如果保持 Skill-driven | 如果转向代码编排 |
|---------------------|----------------|
| 必须用 Opus/GPT-5 级模型 | 可以用 Sonnet/GPT-4o-mini |
| ~$0.15-0.60/轮 (长 context) | ~$0.01-0.05/轮 (短 prompt) |
| 50 轮 ≈ $7.50-30.00 | 50 轮 ≈ $0.50-2.50 |
| 可靠性依赖模型能力 | 可靠性由代码保证 |
| 无法用自训练小模型 | 可以走 MTMC/KernelEvolve 路径 |
| Context 填满 → 需要 compaction hack | 每次调用独立 → 无 context 压力 |

**结论: 如果目标是降低成本或使用中等模型，转向代码编排是必要条件，不是可选项。**

### 8.7 自训练飞轮 (Self-Training Flywheel) 详解

KernelEvolve 的终极目标不是"用大模型调内核"，而是**用大模型产生数据来训练小模型，最终替代大模型**。

#### 运作机制

```
┌──── 飞轮循环 ────────────────────────────────────────────────┐
│                                                               │
│  Phase 1: Frontier 模型运行优化                               │
│    → 产出: 结构化 trajectory (state, action, reward)          │
│    → reward = 实测 TFLOPS 提升                                │
│                                                               │
│  Phase 2: RL Post-training                                    │
│    → 输入: trajectory 数据                                    │
│    → 方法: Agentic RL (PPO/DPO)                               │
│    → 输出: 专用小模型 (7B-14B)                                │
│                                                               │
│  Phase 3: 小模型替代运行                                      │
│    → 相同框架，更低成本                                       │
│    → 产出: 新的 trajectory 数据 (可能质量稍低但量大)          │
│                                                               │
│  Phase 4: 数据回流 → Phase 2                                  │
│    → 小模型的数据 + frontier 数据混合训练                     │
│    → 小模型迭代改进                                           │
│                                                               │
│  收敛: 小模型性能 ≈ frontier，成本低 100x                     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

#### 为什么代码编排是飞轮前提

| 要求 | 代码编排如何满足 | Skill-driven 为什么不行 |
|------|-----------------|----------------------|
| 结构化数据 | 每步产出格式化 (state, action, reward) | Chat transcript 非结构化 |
| 质量保证 | 代码确保每条 trajectory 完整有效 | Agent 可能跳步/错序 |
| 窄任务适配 | 小模型只需做"内核+profile→代码"窄任务 | 小模型无法遵循 800 行指令 |
| 确定性复现 | 相同输入→相同流程→可对比 | Agent 每次行为可能不同 |

#### 实际案例

**MTMC (静态飞轮)**:
- 收集 KernelBench 上的专家 trajectory
- PPO 训练 DeepSeek-Coder-1.3B (仅做策略选择)
- 结果: 1.3B 模型的策略选择能力 > 未经训练的 70B 模型

**KernelEvolve (动态飞轮)**:
- 持续运行 → 持续产数据 → 持续训练
- 最终目标: 完全自托管
- 引述: "This compounding flywheel enables us to self-host increasingly efficient models that are compact enough to run cost-effectively at scale"

#### 对 CroqTile-Tuner 的战略意义

```
路径 A (保持 Skill-driven):
  今天: 用 Opus $30/50轮
  1年后: 仍然用 Opus $30/50轮 (或更贵)
  数据: 非结构化 transcript (无法训练)

路径 B (转向代码编排):
  今天: 用 Opus $30/50轮 (积累 trajectory 数据)
  3月后: 用 Sonnet $2.50/50轮 (代码保证可靠性)
  6月后: 攒够数据 → RL 训练 7B 模型
  1年后: 自托管 $0.10/50轮 + 性能接近 frontier
```

---

## 九、CroqTile-Tuner v2 架构方案

### 9.1 当前架构 (v1) — Skill-Driven + 外挂 Monitor

```
┌─────────────────────────────────────────────────────────────────┐
│ Monitor (外挂观察层)                                            │
│  React UI ←──SSE──→ FastAPI Backend                            │
│                        │                                        │
│                   Scheduler                                     │
│                   (dispatch + scan artifacts)                   │
└────────────────────────┼────────────────────────────────────────┘
                         │ spawn subprocess
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Agent (自编排, 不可控)                                           │
│  读 SKILL.md (750行) → 自主决定: 何时 profile/implement/store   │
│  调用 harness scripts (bash) → 写 results.tsv, idea-log.jsonl   │
│  管理 checkpoints, iter naming, git commits                    │
└─────────────────────────────────────────────────────────────────┘
                         │ writes files
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Filesystem (耦合于 skill rules)                                 │
│  tuning/<gpu>/<dsl>/logs/<key>/<model>/results.tsv              │
│  tuning/<gpu>/<dsl>/logs/<key>/<model>/idea-log.jsonl           │
│  tuning/<gpu>/<dsl>/checkpoints/<key>/<model>/current_idea.json │
│  tuning/<gpu>/<dsl>/srcs/<key>/<model>/iter<NNN>_<tag>.<ext>    │
└─────────────────────────────────────────────────────────────────┘
```

**问题:**
1. Monitor 是 passive scanner — 只能事后观察，不能主动控制
2. Agent 是黑盒 — 可能遗忘规则、跳步、崩溃
3. 存储格式由 SKILL.md 规则定义 — UI 要 parse TSV/JSONL
4. 没有结构化 trajectory 数据输出
5. 单进程 Agent — 不支持并行 profiling、分支探索

### 9.2 目标架构 (v2) — 代码编排 + Agent 执行

```
┌─────────────────────────────────────────────────────────────────┐
│ Control Plane (Web UI + API)                                    │
│                                                                 │
│  React Dashboard                                                │
│  ├── Task 创建/管理                                             │
│  ├── 实时进度 (SSE)                                             │
│  ├── TFLOPS 图表 + 迭代历史                                     │
│  ├── 搜索树可视化 (optional)                                    │
│  ├── Agent 输出 Live Log                                        │
│  └── Terminate / Pause / Resume 控制                            │
│                                                                 │
│  FastAPI Backend                                                │
│  ├── REST API (CRUD tasks, view results)                        │
│  ├── SSE event stream                                           │
│  └── Orchestrator API (start/stop/status)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │ control signals
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Orchestrator (Python, 确定性代码)                                │
│                                                                 │
│  ┌─── Round Loop (代码控制) ───────────────────────────────┐    │
│  │                                                         │    │
│  │  1. PROFILE:  调用 ncu_profile.sh + profile_extract.sh  │    │
│  │  2. IDEA:     调用 LLM (→ 优化想法 + 代码方案)          │    │
│  │  3. IMPLEMENT: 调用 LLM (→ 内核源码)                    │    │
│  │  4. BUILD:    调用 build script (确定性)                 │    │
│  │  5. VERIFY:   运行正确性检查 (代码判断 pass/fail)       │    │
│  │  6. MEASURE:  运行 benchmark (代码提取 TFLOPS)          │    │
│  │  7. DECIDE:   代码比较 (new > best → KEEP, else DISCARD)│    │
│  │  8. STORE:    代码写 DB + 文件系统                       │    │
│  │  9. CONTINUE: 代码判断停止条件 → 循环                   │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  状态机:                                                        │
│  ├── Task 状态: pending → running → completed/failed            │
│  ├── Round 状态: profile → idea → implement → build → ...       │
│  └── Resume: 从 DB 恢复精确状态                                 │
│                                                                 │
│  LLM 调用接口:                                                  │
│  ├── idea_agent(profile_data, history, kernel_src) → Idea       │
│  ├── implement_agent(idea, base_kernel, dsl_spec) → KernelCode  │
│  └── debug_agent(compile_error, kernel_src) → Fix               │
└────────────────────────┬────────────────────────────────────────┘
                         │ reads/writes
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Data Layer (程序管理, UI 直接读取)                               │
│                                                                 │
│  SQLite/PostgreSQL:                                             │
│  ├── tasks (id, shape_key, status, best_tflops, ...)            │
│  ├── iterations (task_id, iter_num, tflops, decision, idea, ..) │
│  ├── trajectories (task_id, step, state_json, action, reward)   │
│  └── profiles (task_id, iter_num, bottleneck, metrics_json)     │
│                                                                 │
│  Filesystem (内核源码 + 二进制 + ncu 报告):                      │
│  ├── kernels/<task_uid>/iter<NNN>_<tag>.<ext>                   │
│  ├── builds/<task_uid>/iter<NNN>_<tag>                          │
│  ├── profiles/<task_uid>/ncu_<tag>.ncu-rep                      │
│  └── logs/<task_uid>/build_<iter>.txt                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 组件职责划分

| 组件 | 由什么控制 | 职责 |
|------|-----------|------|
| **Round Loop** | Python 代码 (asyncio) | 步骤顺序、状态转换、停止条件、错误恢复 |
| **PROFILE** | Shell scripts (不变) | ncu 采集 + 瓶颈分类 |
| **IDEA** | LLM 调用 (短 prompt) | 分析 profile 数据 → 提出一个优化想法 |
| **IMPLEMENT** | LLM 调用 (短 prompt) | 给定 idea + base kernel → 生成新内核代码 |
| **BUILD/VERIFY/MEASURE** | Shell scripts (不变) | 编译、验证、benchmark — 确定性代码 |
| **DECIDE** | Python 代码 (一行比较) | `new_tflops > best_tflops` |
| **STORE** | Python 代码 (DB + FS write) | 原子性写入 DB 和文件系统 |
| **状态管理** | DB (SQLAlchemy) | 精确恢复、UI 实时读取 |
| **Trajectory 记录** | Python 代码 | 每步自动记录 (state, action, reward) 用于未来训练 |

### 9.4 LLM 调用接口设计

核心理念: **每次 LLM 调用是窄任务、短 prompt、独立 context**

```python
# IDEA 步骤 — 输入精简, 输出结构化
class IdeaRequest:
    kernel_src: str           # 当前最优内核 (全文)
    profile_summary: dict     # bottleneck, key_metrics
    recent_history: list[dict] # 最近 5 轮 idea+result (不是全部)
    dsl_constraints: str      # DSL 允许/禁止列表 (从 dsl_spec 提取)

class IdeaResponse:
    idea_summary: str         # 人类可读一句话
    hypothesis: str           # 为什么这会有帮助
    category: str             # tiling|pipeline|memory|compute
    expected_gain: str        # +X TFLOPS 估计

# IMPLEMENT 步骤
class ImplementRequest:
    idea: IdeaResponse        # 从上一步
    base_kernel: str          # 完整源码
    dsl_spec: str             # DSL 语法参考 (压缩版)
    build_template: str       # build script 模板

class ImplementResponse:
    kernel_code: str          # 新内核源码
    build_script: str         # 可选: 如果需要修改 build

# DEBUG 步骤 (compile fail 时)
class DebugRequest:
    kernel_code: str          # 当前失败代码
    error_output: str         # 编译器/运行时错误
    attempt_number: int       # 第几次尝试 (1-5)
    dsl_spec: str             # DSL 约束

class DebugResponse:
    fixed_code: str           # 修复后的代码
    fix_description: str      # 改了什么
```

### 9.5 状态机设计

```python
class RoundState(Enum):
    PROFILE = "profile"
    IDEA = "idea"
    IMPLEMENT = "implement"
    BUILD = "build"
    VERIFY = "verify"
    MEASURE = "measure"
    DECIDE = "decide"
    STORE = "store"

class TaskState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"       # 用户暂停
    WAITING = "waiting"     # 等待重试
    COMPLETED = "completed"
    FAILED = "failed"

# 状态转换由代码控制，不依赖 Agent 自律
async def run_round(task: Task, round_num: int) -> RoundResult:
    # 每个步骤失败都有明确处理
    profile = await do_profile(task)          # 失败 → retry 1x → fail task
    idea = await call_llm_idea(profile, task) # LLM 调用
    code = await call_llm_implement(idea, task) # LLM 调用
    
    for attempt in range(5):                  # 代码控制重试
        build_ok = await do_build(code, task)
        if build_ok:
            break
        code = await call_llm_debug(code, error, task)  # LLM 修复
    else:
        return RoundResult(decision="COMPILE_FAIL")
    
    if not await do_verify(task):
        return RoundResult(decision="VERIFY_FAIL")
    
    tflops = await do_measure(task)
    decision = "KEEP" if tflops > task.best_tflops else "DISCARD"
    
    await store_round(task, round_num, tflops, decision, idea, code)
    return RoundResult(decision=decision, tflops=tflops)
```

### 9.6 Trajectory 记录格式 (为未来训练做准备)

```json
{
  "task_uid": "abc123",
  "round": 15,
  "timestamp": "2026-06-25T19:30:00Z",
  "state": {
    "kernel_src_hash": "sha256:...",
    "best_tflops": 142.5,
    "baseline_tflops": 180.0,
    "bottleneck": "memory_bound",
    "key_metrics": {"l2_hit_rate": 0.45, "dram_throughput_pct": 78.2},
    "recent_decisions": ["KEEP", "DISCARD", "DISCARD"]
  },
  "action": {
    "idea": "Use vectorized 128-bit loads instead of 32-bit for global memory",
    "category": "memory",
    "code_diff_hash": "sha256:..."
  },
  "reward": {
    "tflops": 148.2,
    "improvement": 0.04,
    "decision": "KEEP"
  },
  "metadata": {
    "model": "claude-4-sonnet",
    "llm_tokens_used": 2400,
    "wall_time_s": 85
  }
}
```

### 9.7 迁移路径 (渐进式)

| 阶段 | 做什么 | 保留什么 | 替换什么 |
|------|--------|---------|---------|
| **Phase 0** | 设计 + 原型 | 全部现有系统 | 无 |
| **Phase 1** | 实现 Orchestrator core | harness scripts (profile, build, measure) | SKILL.md 流程控制 → Python 状态机 |
| **Phase 2** | LLM 调用接口 | ncu profiling pipeline, DSL 知识 | Agent 自编排 → 结构化 LLM 调用 |
| **Phase 3** | 数据层迁移 | 内核源码文件 | results.tsv → DB, idea-log.jsonl → DB |
| **Phase 4** | UI 升级 | Monitor frontend 基础 | 被动观察 → 主动控制 |
| **Phase 5** | Trajectory 系统 | 一切 | 新增: 自动记录结构化数据 |
| **Phase 6** | 模型降级实验 | 一切 | Opus → Sonnet/小模型测试 |

### 9.8 设计决策总结 — 锁定 vs 开放

#### ✅ 已锁定 (General Plan)

| 决策 | 选择 | 理由 |
|------|------|------|
| 编排模式 | 代码编排 + Agent 执行 | 行业主流、支持廉价模型、支持自训练飞轮 |
| 迁移方式 | 渐进式 (Phase 0→4) | 保证迁移期间 tuning 不中断 |
| LLM 调用粒度 | 每步独立调用 (无累积 context) | 支持廉价模型、无 context 压力 |
| Harness 层存在 | 必须有，作为可替换抽象 | 安全迁移桥梁、支持 A/B 测试 |
| 存储解耦 | 不再由 Agent skill rules + harness 决定格式 | Web UI 直读、程序化管理 |
| 知识模块化 | 从 1 个 SKILL.md → 多模块按需注入 | 降低 context 压力、支持廉价模型 |
| 停止/决策/正确性 | 代码强制执行 | 不依赖 Agent 自律 |
| DSL knowledge 保留 | 保留并结构化 | 领域知识不变，载体变 |
| Harness scripts (bash) | 保留不变 | 已验证可靠 (profile, build, measure) |
| Trajectory 记录 | 自动产生结构化 (state, action, reward) | 为未来自训练/RL 做准备 |

#### 🔓 保持开放 (Concrete Infra/Technique)

| 决策维度 | 候选项 | 决策时机 |
|----------|--------|---------|
| **实现语言** | Python / TypeScript / Rust | Phase 1 启动前决定 |
| **LLM 接口** | litellm / 直接 SDK / 自建 | Phase 1 启动前决定 |
| **Harness 实现** | Cursor CLI wrapper / 自建 tool executor / OpenAI Agents SDK / 其他 | Phase 1 (先用现有工具) |
| **数据库** | SQLite / PostgreSQL / 其他 | Phase 1 (可从 SQLite 开始) |
| **前端** | 保留现有 React / 重写 | Phase 4 再决定 |
| **搜索策略** | 线性 hill-climb / tree search / 图搜索 | Phase 2+ (先做简单的) |
| **是否引入 framework** | 纯手写 / LangGraph / OpenAI Agents SDK / 其他 | 看 Phase 1 复杂度再定 |
| **Agent harness 是否自建** | 自建 / 用 Cursor SDK / 用其他 agent runtime | Phase 2 (Phase 1 先用 Cursor) |

### 9.9 为什么不用 Agentic Framework？

#### 行业实际选择

| 系统 | 选择 | 理由 |
|------|------|------|
| KernelEvolve (Meta) | 自建 Python | 生产级需要完全控制 |
| Astra (Stanford) | OpenAI Agents SDK | 研究原型求快，4 Agent 用 SDK 连接 |
| AKG Agent (Huawei) | 自建 → LangGraph | 迁移原因: 工作流可视化 + 状态管理 + 通用化需求 |
| Two-Stage / MTMC | 自建 Python | 学术代码，循环简单 |

#### 框架提供 vs 我们需要

| 框架能力 | 需要? | 替代方案 |
|---------|------|---------|
| Tool calling 路由 | ❌ | 我们直接 subprocess 调 shell scripts |
| Memory 管理 | ❌ | 每次 LLM 调用独立，DB 管跨轮状态 |
| 多模型切换 | ✅ | `litellm` (1个包，不是框架) |
| 重试/错误 | ⚠️ | `tenacity` (1个装饰器) |
| 可观测性 | ⚠️ | 已有 monitor backend |
| 状态持久化 | ✅ | SQLAlchemy + DB (比框架 checkpoint 更灵活) |
| 工作流图 | ⚠️ | 核心是 while 循环，不需要图 |
| 多 Agent 通信 | ❌ | 我们是顺序 pipeline，非并行对话 |

#### 核心论点: 我们的循环太简单

```python
# 整个核心不到 30 行，不需要框架
while not should_stop(task):
    profile = await run_ncu(task)
    idea = await call_llm("idea", profile, task)
    code = await call_llm("implement", idea, task)
    binary = await build(code, task)
    if not binary:
        for _ in range(5):
            code = await call_llm("debug", error, task)
            binary = await build(code, task)
            if binary: break
    tflops = await benchmark(binary)
    await store(task, tflops, "KEEP" if tflops > task.best else "DISCARD")
```

LangGraph 把这变成 ~200 行的图定义 + 节点函数。不值得。

#### 什么时候值得引入框架

| 条件 | 当前 | 未来可能 |
|------|------|---------|
| 复杂分支/图搜索 | ❌ 线性循环 | ✅ 如果加 tree search |
| 动态工具选择 | ❌ 固定 pipeline | 可能不会 |
| 多 Agent 并行 | ❌ 单 Agent | ✅ 如果加 parallel IDEA exploration |
| 通用化平台 | ❌ 只做内核调优 | ✅ 如果扩展到其他 AI infra 任务 |

**结论:** 现阶段纯 Python + 轻量依赖。如果将来加图搜索/并行探索，考虑 LangGraph。

#### 语言选择分析 (🔓 开放决策)

| 语言 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| **Python** | LLM SDK 生态最丰富 / GPU toolchain 直接可用 / 已有 monitor backend / 未来 RL 训练 | GIL 限制真并发 / 类型系统弱 | 快速原型 + 与 ML 生态集成 |
| **TypeScript** | 类型安全 / async 原生 / 前后端同栈 / Cursor SDK 是 TS | GPU 工具链需 subprocess / LLM 库较少 | 如果 orchestrator 与 UI 深度耦合 |
| **Rust** | 性能 / 类型安全 / 内存安全 / 长时间运行可靠 | 开发速度慢 / LLM 生态不成熟 / 学习曲线 | 如果系统需要长期高可靠运行 |
| **Go** | 并发模型优秀 / 编译快 / 部署简单 | LLM 生态差 / GPU 工具链需 FFI | 如果系统偏 infra |

**当前倾向 Python** (与已有代码最兼容，快速迭代)，但不锁定。

#### 可能的技术栈组合 (🔓 全部开放)

```
方案 A (Python 全栈):
  Orchestrator: Python asyncio (嵌入 FastAPI backend)
  LLM:          litellm 或直接 SDK
  重试:         tenacity 或自建
  数据:         pydantic / SQLAlchemy / SQLite
  GPU:          subprocess → harness scripts

方案 B (TypeScript 全栈):
  Orchestrator: Node.js / Bun + @cursor/sdk 或 openai SDK
  数据:         Drizzle/Prisma + SQLite
  GPU:          child_process → harness scripts
  
方案 C (Rust core + Python glue):
  Orchestrator: Rust (tokio) — 状态机 + scheduling
  LLM:          Python wrapper 通过 pyo3/FFI 或 HTTP microservice
  数据:         Rust sqlx + SQLite
  GPU:          Command → harness scripts

方案 D (混合):
  Orchestrator: 任意语言
  Agent Harness: 独立微服务 (任意语言)
  通信:         HTTP/gRPC 或 message queue
```

### 9.10 Skills 在新架构中的角色

#### 当前: Cursor 就是我们的 Agentic Framework

```
Cursor/OpenCode (框架层) + SKILL.md (领域知识 + 编排指令)
= 当前完整系统
```

Cursor 提供的框架能力:
- Tool calling (Shell/Read/Write/WebSearch)
- Model routing + API 管理
- Context window 管理
- Skills 加载机制

#### 转型后: Skills 形式变化，重要性不变

```
当前: 1 个 SKILL.md (750行) = 工作流 + 领域知识 + 规则 (一体化)
目标: 拆分为:
  - 工作流 → 代码 (orchestrator.py)
  - 领域知识 → 结构化知识模块 (按需注入 LLM)
  - 规则 → 代码逻辑 (if/else 强制执行)
```

#### 知识模块设计

```
knowledge/
├── dsl/
│   ├── croqtile.md      # Choreo 语法、primitive 参考 (~200行)
│   ├── cuda.md          # CUDA 内核模式、intrinsics (~150行)
│   ├── triton.md        # Triton tile API、autotuner 参数 (~150行)
│   ├── helion.md        # Helion kernel API (~100行)
│   └── tilelang.md      # TileLang T.gemm/T.copy API (~100行)
├── playbook/
│   ├── memory.md        # 内存优化: coalesce, bank conflict, prefetch
│   ├── compute.md       # 计算优化: tensorcore, warp specialization
│   ├── pipeline.md      # 流水线: async copy, multi-stage
│   └── structural.md    # 结构变换: persistent kernel, split-K, CTA swizzle
├── hardware/
│   ├── sm90.md          # H100: shared_mem, registers, SMs, bandwidth
│   └── sm86.md          # RTX 3070: 限制参数
└── templates/
    ├── idea_prompt.md    # IDEA 步骤 prompt 模板
    ├── implement_prompt.md # IMPLEMENT 步骤 prompt 模板
    └── debug_prompt.md   # DEBUG 步骤 prompt 模板
```

#### 代码如何使用知识模块

```python
async def call_llm_idea(profile: ProfileResult, task: Task) -> Idea:
    # 代码决定注入哪些知识
    dsl_spec = load_knowledge(f"dsl/{task.dsl}.md")
    
    # 根据瓶颈选择相关 playbook
    if profile.bottleneck == "memory_bound":
        playbook = load_knowledge("playbook/memory.md")
    elif profile.bottleneck == "compute_bound":
        playbook = load_knowledge("playbook/compute.md")
    else:
        playbook = load_knowledge("playbook/structural.md")
    
    hardware = load_knowledge(f"hardware/{task.gpu_arch}.md")
    
    # 组装 prompt — 比 750 行 SKILL 短得多
    prompt = render_template("templates/idea_prompt.md",
        kernel=task.best_kernel_src,
        profile=profile.summary,
        history=task.recent_history(5),  # 只最近 5 轮
        dsl_spec=dsl_spec,
        playbook=playbook,
        hardware=hardware,
    )
    
    # 调用 LLM — 短 prompt, 结构化输出
    return await llm.call(prompt, response_model=Idea)
```

#### 关键区别对比

| 维度 | 当前 (Agent 读 SKILL) | v2 (代码注入知识) |
|------|---------------------|------------------|
| 知识总量 | 相同 | 相同 |
| 每次 LLM 看到多少 | 全部 750 行 + 累积历史 | 只看相关部分 ~200-300 行 |
| 谁决定注入什么 | Agent 自己 (不可控) | 代码 (确定性) |
| 知识可维护性 | 1 大文件难改 | 小模块独立更新 |
| 新 DSL 接入 | 写新 SKILL 全文 | 加一个 dsl/new.md |
| 可测试性 | 无法单元测试 | 每个模块可独立验证 |

### 9.11 安全迁移: Agent Harness Layer 的必要性

#### 核心洞察

其他团队从零建起不需要迁移。我们有已运行系统，需要 harness layer 作为桥梁。

#### IMPLEMENT 步骤的真实复杂度

```
不是简单的 "prompt → code":
  1. 读取当前最优内核源码
  2. 读取 DSL 示例/参考
  3. 生成新内核代码
  4. 生成 build script
  5. 编译 → 失败 → 读错误 → 修改 → 重试 (loop)
  6. 可能需要 web search
```

这需要 Agent 有 tool 能力 (文件操作 + 命令执行)，不是纯文本生成。

#### Harness Layer 抽象

```python
class AgentHarness(ABC):
    """可替换的 Agent 执行层"""
    
    @abstractmethod
    async def call(self,
        task: str,              # "idea" | "implement" | "debug"
        system_prompt: str,     # 知识模块组装
        user_prompt: str,       # 当前任务
        tools: list[Tool],      # 允许的工具
        max_attempts: int,      # 步骤内重试
        timeout_s: int,         # 超时
    ) -> HarnessResult: ...

# Phase 1: 用 Cursor CLI 实现
class CursorHarness(AgentHarness):
    """启动 cursor-agent 执行单步窄任务"""
    ...

# Phase 2: 用 litellm + 简单 tool executor 实现
class LiteLLMHarness(AgentHarness):
    """直接 API 调用 + subprocess tool executor"""
    ...

# Phase 3+: 本地模型
class LocalModelHarness(AgentHarness):
    """本地推理 (vLLM/TGI) + tool executor"""
    ...
```

#### 安全迁移路径 (修订版)

```
Phase 0: 现状
  Cursor = harness + orchestrator + 知识载体
  SKILL.md = 编排指令 + 领域知识 (一体化)

Phase 1: 分离编排 (低风险)
  新增: Python Orchestrator (round loop + 状态机)
  保留: CursorHarness (执行 IDEA/IMPLEMENT/DEBUG)
  拆分: SKILL.md → 短 task prompts (~50行/步) + 知识模块
  验证: 对比 v1 vs v2 在相同 shape 上的 TFLOPS 收敛速度

Phase 2: 替换 Harness (中等风险)
  替换: CursorHarness → LiteLLMHarness
  原因: 去除 Cursor overhead (workspace indexing, rules loading)
  前提: Phase 1 验证 orchestrator 可靠

Phase 3: 模型降级 (低风险)
  在 LiteLLMHarness 上切换: Opus → Sonnet → o4-mini
  用积累的 trajectory 数据评估质量

Phase 4: 自训练 (高投入)
  积累足够 trajectory → RL 训练专用模型
  LocalModelHarness 部署
```

#### 为什么这比其他团队多了一层

| 我们 | 其他团队 |
|------|---------|
| 已有 Cursor 生态（rules, skills, hooks） | 从零开始 |
| 已有运行中的 tuning sessions | 无历史包袱 |
| 需要保证迁移期间 tuning 不中断 | 不存在此问题 |
| Harness 接口允许渐进替换 | 一步到位 |

这层抽象的价值: **迁移时可以 A/B 测试不同 harness 实现，在相同 orchestrator 下对比效果。**

---

## 十、Decision Journal — 完整顾虑与分析记录

> 记录日期: 2026-06-25
> 目的: 捕获架构讨论中的每一个顾虑、挑战、分析和决策理由，不遗漏任何细节。

### 10.1 触发点: 为什么要转型？

**用户原话:** "This project need a transition. current design is purely skills driven to run the tuning loop."

**核心问题:**
- 当前系统 100% 依赖 LLM Agent 自觉遵循 SKILL.md 指令来编排整个 tuning loop
- Agent 行为不可控 — 可能遗忘规则、跳步、context 填满后质量退化
- 无法降级模型 — 只有 frontier 模型 (Opus/GPT-5) 才能可靠遵循 750 行复杂指令
- 无法积累结构化训练数据 — chat transcript 不是结构化 trajectory
- 存储格式被 skill rules 耦合 — UI 要 parse agent 写的 TSV/JSONL
- Monitor 只能被动观察，不能主动控制

**决策:** 需要转向代码编排模式。这不是优化，是架构根本性变更。

---

### 10.2 行业分析: 谁在用哪种模式？

**用户原话:** "analyse how many framework is chaining the workflow with program, and only use agent to work for stepwise task, how many is skill system, that rely agent do the whole workflow following skills like us current design?"

**调研结果 (14 个系统):**
- 纯代码编排 (无 LLM): 6/14 (43%) — TVM, XLA, Triton, Helion, TileLang, CompileIQ
- 代码编排 + Agent 执行: 6/14 (43%) — KernelEvolve, Astra, TritonForge, MTMC, Two-Stage, AKG
- Skill/指令驱动 (Agent 自编排): **仅 2/14 (14%)** — AutoKernel, CroqTile-Tuner

**关键洞察:**
- 在 8 个使用 LLM 的系统中，6/8 (75%) 选择代码编排
- 只有 AutoKernel 和我们选择了 skill-driven
- AutoKernel 的 orchestrate.py 其实也做了部分编排 (Amdahl 排序)
- **我们可能是唯一一个完全依赖 Agent 自编排的生产系统**

**理由:** 这不是品味选择，是行业验证的工程实践。6/8 的 Agent 系统做出相同选择不是巧合。

---

### 10.3 成本分析: 模型降级可行性

**用户原话:** "which one is more fit for moderate LLMs? or cheap models?"

**分析:**
- Skill-driven 对模型要求极高:
  - 需要在 800+ 行指令中精确遵循每一条规则
  - 跨 50+ 轮保持一致行为
  - 自主管理文件状态
  - 只有 Opus/GPT-5 级才能胜任
  - 成本: ~$0.15-0.60/轮, 50 轮 ≈ $7.50-30.00

- 代码编排对模型要求极低:
  - 每次调用只需: 读内核 + profile → 提出/实现一个优化
  - 无需记住全局规则
  - 无需管理状态
  - Sonnet/GPT-4o-mini 甚至更小模型即可
  - 成本: ~$0.01-0.05/轮, 50 轮 ≈ $0.50-2.50

**极端案例 — MTMC:**
- 策略模型用 DeepSeek-Coder-**1.3B** (RL 训练后)
- 1.3B 的策略选择能力 > 未经训练的 70B 通用模型
- 前提: 代码编排将任务分解到极窄

**决策理由:** 如果目标是成本可控或模型可降级，代码编排是**必要条件**，不是可选项。

---

### 10.4 自训练飞轮: 长期战略

**用户原话:** "what is self-train fly-wheel"

**KernelEvolve (Meta) 的核心机制:**
1. Frontier 模型运行 → 产出结构化 trajectory (state, action, reward)
2. RL post-training 小模型 (PPO/DPO, reward = 实测 TFLOPS)
3. 小模型替代运行 → 产出更多 trajectory
4. 数据回流 → 持续迭代 → 成本递减

**为什么这与我们相关:**
- 如果保持 skill-driven: 产出非结构化 transcript → 无法训练 → 永远依赖 frontier 模型
- 如果转向代码编排: 每步自动产出结构化 trajectory → 可以训练 → 长期自托管

**前提条件:**
- 代码编排 (结构化数据的来源)
- 窄任务定义 (小模型能处理)
- 确定性 pipeline (保证数据质量)
- 足够的运行量 (数据积累)

**战略意义:**
```
今天: Opus $30/50轮 → 3月后: Sonnet $2.50/50轮 → 1年后: 自托管 $0.10/50轮
```

---

### 10.5 存储解耦 + Web UI 集成

**用户原话:** "our final system should also involve the monitoring/control web UI, and the storage filesystem will not coupled by agent skill rule + harness to fit web ui. they will all be programs"

**当前问题:**
- 存储格式由 SKILL.md 规则定义: "写入 results.tsv 格式为..."
- UI (monitor) 需要 parse TSV/JSONL — 耦合于 agent 写入格式
- Agent 如果写错格式，UI 就坏了
- Monitor 只能被动 scan artifacts，不能主动控制

**目标:**
- 存储由程序直接管理 (DB + programmatic file write)
- UI 从 DB 直读，不需要 parse agent artifacts
- Web UI 不是观察者，而是 control plane (start/stop/pause/resume)
- 所有数据流是 "程序 → DB → UI" 而非 "Agent → 文件 → UI parse"

**审查了现有 monitor 代码后确认:**
- `monitor/backend/app/scheduler.py`: 已有 Python 调度逻辑 (dispatch opencode)
- `monitor/backend/app/models.py`: 已有 SQLAlchemy ORM (Task, IterationLog, AgentLog)
- `monitor/backend/app/agent.py`: 目前靠 regex parse agent output (如 `STORE_PATTERN`)

**结论:** 已有 monitor 基础设施可以扩展为 orchestrator host，不需要从零建。

---

### 10.6 为什么不用 Agentic Framework？— 挑战与反思

**用户原话:** "nonono, do not hurry, let us challenge our self. one question, why not use agentic framework? what is the difference between we build by python or on framework or other lang?"

**挑战意图:** 用户在测试 "纯手写" 方案是否经过充分论证，还是仅仅是本能偏好。

**分析框架:**

| 维度 | 自建 | 用 Framework (LangGraph 等) |
|------|------|---------------------------|
| 控制力 | 完全 | 受限于框架抽象 |
| 学习成本 | 低 (熟悉 Python) | 中 (学框架 API) |
| 维护 | 自己维护一切 | 框架升级可能 breaking |
| 可调试 | 直接 debug | 框架内部不透明 |
| 社区支持 | 无 | 有 |
| 工作流可视化 | 需自建 | 框架可能自带 |
| 适合复杂度 | 简单到中等 | 中等到复杂 |

**行业实际选择:**
- KernelEvolve (Meta): 自建 — 生产级需要完全控制
- Astra (Stanford): OpenAI Agents SDK — 研究原型求快
- AKG Agent (Huawei): 自建 → LangGraph — 需求增长后引入

**核心论点:**
- 当前核心循环 ~30 行 Python 可表达
- 框架引入抽象层但不简化核心逻辑
- 如果未来加图搜索/并行，可以再引入

**但用户明确指出:** 决策保持开放。可能用框架，可能不用，取决于实际实现时的复杂度。

---

### 10.7 Agent Harness Layer — 关键洞察

**用户原话:** "so if we want to do safe migration, we still need agent harness layer..... that other most work did not do"

**这是整个讨论中最重要的洞察之一。**

**用户的逻辑:**
1. 其他团队 (KernelEvolve, Astra) 从零建起 — 不需要迁移
2. 我们有已运行系统 (Cursor + SKILL.md) — 需要安全过渡
3. Cursor 不仅仅是 LLM 调用 — 它是一个 **agent runtime** (tools, retry, streaming, context)
4. 如果直接替换为 "litellm + raw API"，我们要重建 Cursor 提供的一切
5. 所以需要一个 harness 抽象层，允许渐进替换底层实现

**Cursor 作为 harness 提供了什么:**
- Tool calling (Shell/Read/Write/WebSearch) → Agent 可以操作文件系统、执行命令
- 自动重试 (tool call 失败时)
- Context window 管理 (截断、summarize)
- 模型路由 + fallback
- 流式输出
- 错误恢复

**IMPLEMENT 步骤的真实复杂度:**
- 不是 "prompt → code" 的简单调用
- 是 "读文件 → 理解 → 写代码 → 编译 → 读错误 → 修改 → 重试" 的多步骤过程
- 需要 agent 有 tool 能力
- 这正是 harness 要提供的

**其他团队为什么不需要这层:**

| 团队 | 为什么不需要单独的 harness layer |
|------|-------------------------------|
| KernelEvolve | 从零自建 "LLM Synthesizer" — 就是个带 retry 的 API wrapper |
| Astra | OpenAI Agents SDK **就是** harness |
| AKG Agent | LangGraph + ReAct Agent **就是** harness |
| MTMC | LLM 只做纯文本生成，不需要 tool — 代码做文件操作 |

**他们都有 harness，只是不需要迁移。** 我们的独特挑战是: 从 Cursor 迁移到自建/其他 harness。

**解决方案: AgentHarness 抽象接口**

```
AgentHarness (abstract)
├── CursorHarness    → Phase 1: 用 Cursor CLI 执行单步 (已有能力)
├── LiteLLMHarness   → Phase 2: 直接 API + 简单 tool executor (自建)
├── SDKHarness       → 可选: 用 OpenAI Agents SDK 或 Cursor SDK
└── LocalHarness     → Phase 4: 本地模型推理 + tool executor
```

迁移时只换 harness 实现，orchestrator 不变。可以 A/B 测试对比效果。

---

### 10.8 决策保持开放的具体原因

**用户原话:** "keep the concrete infra, technique open, i may use rust, ts, python, i may use litellm or other shit, i may crafting the agent harness components my self"

**为什么语言保持开放:**
- Python: 与 ML 生态最兼容，快速迭代，已有 backend
- TypeScript: 如果 orchestrator 深度集成 Cursor SDK (TS)，可能更自然
- Rust: 如果系统需要长时间无人值守运行，Rust 的内存安全和可靠性有价值
- 混合: 核心循环可能用一种语言，harness 用另一种

**为什么 LLM 接口保持开放:**
- litellm: 快速统一多模型，但是第三方依赖
- 直接 SDK: 控制力强，少一层抽象
- 自建: 如果需要特殊逻辑 (如 structured output 解析、retry 策略)
- 可能根本不用现有库 — 如果 Agent Harness 已经封装了这些

**为什么 Harness 实现保持开放:**
- 可能保持用 Cursor (如果性能足够)
- 可能用 Cursor SDK (TypeScript) 构建轻量 agent runtime
- 可能完全自建 (Python subprocess + LLM API)
- 可能用 OpenAI Agents SDK (如果需要多 tool agent)
- 决策取决于 Phase 1 的实际体验

---

### 10.9 未解决的开放问题

以下问题需要在实际实现时解决:

| # | 问题 | 影响 | 决策时机 |
|---|------|------|---------|
| 1 | IMPLEMENT 步骤: agent 在步骤内部是否需要 tool calling? 还是拆分为更细的步骤由 orchestrator 控制? | 决定 harness 的复杂度 | Phase 1 原型期 |
| 2 | 如果 agent 在 IMPLEMENT 中需要编译重试，是 harness 内部处理还是 orchestrator 循环控制? | 控制粒度 | Phase 1 |
| 3 | 知识模块 (.md 文件) 如何版本管理? 是否需要动态更新机制 (RAG)? | 可扩展性 | Phase 2+ |
| 4 | 多 task 并行时的 GPU 资源管理: orchestrator 如何调度? | 吞吐量 | Phase 2+ |
| 5 | trajectory 数据的存储量级和清理策略? | 磁盘/DB 管理 | Phase 3+ |
| 6 | 如果引入 tree search / 图搜索，现有的线性循环设计如何扩展? | 架构韧性 | 远期 |
| 7 | 现有 SKILL.md 中的 "soft knowledge" (如 "结构多样性强制") 如何迁移为代码? 所有都能 codify 吗? | 迁移完整性 | Phase 1 |
| 8 | CursorHarness 的实际 overhead 是多少? workspace indexing / rules loading 花多少时间? | Phase 2 是否必要 | Phase 1 测量 |
| 9 | Web UI 是否需要实时展示 agent 的 thought process? 如果是，harness 需要 streaming 接口 | UX 设计 | Phase 4 |
| 10 | 如果用 Rust 做 orchestrator，如何与 Python ML 生态 (ncu parser, torch 等) 集成? | 语言选择 | 启动前 |

---

### 10.10 分析方法论记录

本次调研采用的分析方法:

1. **横向对比**: 14 个系统按多维度对比（编排模式、模型需求、状态管理、知识注入）
2. **定量分析**: 编排模式占比统计、成本估算、prompt 长度对比
3. **垂直深入**: 对 KernelEvolve/AutoKernel/MTMC 做深度架构剖析
4. **代码审查**: 读取现有 monitor 代码确认集成可行性
5. **挑战驱动**: 用户提出的每个挑战都作为分析起点，不预设结论
6. **分离 "What" vs "How"**: 锁定架构方向（what），保持技术选择开放（how）

---

### 10.11 讨论时间线

| 序号 | 用户问题 | 产出 | 关键决策 |
|------|---------|------|---------|
| 1 | "do a comprehensive investigation on ai kernel tuning" | §1-7: 行业全景 + 12 系统分析 | 确认行业趋势 |
| 2 | "analyse how many is program-chained, how many is skill system" | §8.1-8.5: 编排模式分类与对比 | 代码编排是主流 (75%) |
| 3 | "which one is more fit for moderate LLMs?" | §8.6: 成本分析 + MTMC/KernelEvolve 案例 | 代码编排是降本必要条件 |
| 4 | "what is self-train fly-wheel" | §8.7: 飞轮机制详解 | 代码编排是飞轮前提 |
| 5 | "read our current code, involve web UI, storage all programs" | §9.1-9.7: v2 架构设计 | 代码编排 + DB + Web control |
| 6 | "why not use agentic framework?" | §9.9: 框架分析 | 保持开放，倾向轻量 |
| 7 | "we still need agent harness layer" | §9.11: Harness 必要性分析 | Harness 抽象接口 + 渐进替换 |
| 8 | "fix general plan, keep infra open, note every detail" | §10: Decision Journal (本章) | 锁定方向，开放实现 |
| 9 | "usage scenario: config file + standalone binary" | §11.1: 使用场景锁定 | 独立 binary，不依赖 IDE |
| 10 | "research Pi, Omnigent, Pydantic AI Harness" | §11.2-11.5: 10 方案深入对比 | 全面评估 agent runtime |
| 11 | "LLM 需要熟悉环境 + 外部控制" | §11.3-11.4: 双层模型 + 熟悉环境原则 | Belt AND suspenders |
| 12 | "production-quality product, compare Pi vs Pydantic AI" | §11.5-11.6: 生产就绪度对比 | Pydantic AI 方向确认 |

---

## 十一、Agent Runtime 选型研究 (Standalone Binary 内置)

### 11.1 使用场景锁定

最终产品使用方式:
```bash
# 配置文件指定 model url + api key
$ cat config.yaml
model:
  provider: anthropic
  api_key: sk-...
  model: claude-sonnet-4-6
task:
  dsl: croqtile
  shape: 16384x16384x16384
  dtype: fp16

# 以独立 binary 运行，不依赖任何 IDE
$ ./croqtile-tuner --config config.yaml
```

**明确排除**: 不从 Cursor/Claude Code/OpenCode 等 IDE 中调用 skills 来 tune。

### 11.2 候选方案全景 (10 个)

#### 1. Anthropic Claude SDK (tool-use API)

Anthropic 的 Python/TS SDK 直接支持 tool_use — 在 messages API 中定义 tools schema，Claude 回复时返回 tool_use content block，执行后作为 tool_result 发回。架构极简：while loop 检查 response 里有没有 tool_use block。没有内置 agent loop runner — 完全自己控制。优势：零抽象、完全控制、直接访问 Claude 所有能力（extended thinking、cache、vision）。劣势：只支持 Claude、tool dispatch 自己写（~50 行）、无 retry/guardrail 内置。

#### 2. OpenAI Agents SDK

官方 Python agent 框架。核心 Agent + Runner，`Runner.run(agent, input)` 自动执行 tool-calling loop 直到产出 final output。内置: tool dispatch (函数注册自动 schema)、handoff (agent 切换)、guardrails (I/O 校验)、max_turns、hooks (on_tool_start/end, on_llm_start/end)、tracing (OpenTelemetry)。`tool_use_behavior` 控制 tool 结果处理。轻量 runtime 不是编排框架。劣势：默认只支持 OpenAI models (需 adapter)、Python only、不提供跨步骤持久化。

#### 3. Pydantic AI

Pydantic 团队的 agent 框架。核心理念: 类型安全 agent — Pydantic BaseModel 定义 output schema，自动验证+重试。模型无关 (OpenAI/Anthropic/Gemini/Bedrock/Groq/Ollama)。Tool 通过 @agent.tool 装饰器注册，自动 schema 生成。Capabilities 系统: 可组合 bundle (tools + hooks + instructions)。Pydantic AI Harness: 官方 capability library (CodeMode, Skills, Memory, Context compaction)。Durable Execution: 跨 failure 恢复。YAML Agent Specs: 无代码定义。内置 OpenTelemetry。

#### 4. Humanize (PolyArch)

Claude Code plugin。RLCR loop: Claude 做 implementation，Codex 独立做 code review，issue feed back 直到 acceptance criteria 满足。通过 hooks 管控 tool-use 边界和状态。有 Swarm Mode 做并行。**关键限制: 完全依赖 Claude Code 运行，不能嵌入独立 binary。**

#### 5. Claude Code CLI (subprocess 模式)

把 `claude` CLI 当 subprocess 调用。零 harness 开发（Claude Code 已是完整 runtime）。劣势：依赖安装 Claude Code CLI、进程级调用（启动 ~2-5s）、输出解析不够结构化、只支持 Claude、内部行为控制有限。

#### 6. Cursor SDK (@cursor/sdk)

TypeScript SDK，程序化创建/管理 agent sessions。劣势：TS only、依赖 Cursor 生态、不确定支持 headless。

#### 7. litellm + 自建 tool loop

litellm 提供统一 LLM API 接口 (100+ 模型)。不提供 agent loop — 自己写 while 循环 (~100-150 行)。完全控制 retry/timeout/tool 白名单/输出验证。最模型无关。劣势：所有 harness 逻辑自建、无 tracing/observability 内置。

#### 8. 完全自建 (raw HTTP)

不用任何 SDK，直接调 REST API。零依赖、任意语言。要处理: streaming SSE、tool_use 格式差异、auth、rate limiting、retry。Rust 实现性能最高但开发最慢。

#### 9. Omnigent (Databricks, Apache 2.0, 2026年6月)

Meta-harness — 坐在 Claude Code/Codex/Pi 之上的编排层。统一 API wrap 所有 terminal-based agents 和 SDK agents。内置 "Polly" 编排器: 把任务分配给 sub-agents 在并行 git worktrees 执行，交叉 review。Contextual policies (spend caps, risk escalation)。多设备协作 (terminal/web/mobile/REST)。**是运行时环境 (Node.js 22+)，不是可嵌入库。Alpha 阶段。**

#### 10. Pi (Mario Zechner, 开源, 2026年5月)

极简 coding agent harness — 4 个核心 tool (read, write, edit, bash)，TypeScript Extensions 扩展。Provider-agnostic (20+ 模型)。四种模式: Interactive, Print/JSON, RPC, **SDK** (嵌入应用)。SDK 通过 `createAgentSession()` 直接实例化。OpenClaw 是真实 SDK 集成案例。无内置 sub-agents/plan mode。Node.js 生态。

### 11.3 核心架构决策: 双层鲁棒性模型

```
┌─── Orchestrator (确定性锁) ────────────────────────────────┐
│                                                             │
│  IMPLEMENT 步骤:                                            │
│  ┌─── Agent (有 tools，可以自己修) ────────────────┐       │
│  │  write code → compile → 编译错误 → fix → 再试    │       │
│  │  (agent 自己搞定 = 低成本 low-hanging fruit)     │       │
│  └──────────────────────────────────────────────────┘       │
│  │                                                          │
│  │  Orchestrator 检查 gate:                                 │
│  │  - 编译成功？ max_turns 超了？ 代码质量达标？            │
│  │  If pass → MEASURE; if fail → retry/放弃                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

原则: **确定性锁 + Agent 能力 = 最大鲁棒性**。Agent 有 tool-calling 让它发挥训练能力（low-hanging fruit），Orchestrator 在外层做确定性检查防止失控。

### 11.4 "给 LLM 熟悉的环境" 原则

LLM (Claude/GPT) 被训练的 coding agent 模式:
- `read file.cu` → 查看代码
- `bash "nvcc file.cu"` → 编译
- 看到 error → `edit file.cu:42` → 修复
- 再 `bash` 编译

这是 Claude Code/Cursor 训练数据的核心模式。Agent 对这个 workflow 有强"直觉"。

**关键 insight**: LLM 和 runtime 的交互在 API 层面是统一的 tool_use 协议。无论 harness 是 Claude Code、Pi 还是 Pydantic AI，LLM 看到的都是 tool_use/tool_result 交互。只要 tool 命名 (bash, read, write, edit) 和返回格式 (raw stdout/stderr) 与训练一致，LLM 体验无差异。

→ **不需要用 Pi 软件来获得"Pi 式体验"。在 Pydantic AI 中注册同名同行为的 tools 即可。**

### 11.5 Pi vs Pydantic AI 深入对比

#### 运行模型差异

- **Pi (shell-based)**: Agent 像在终端操作。所有 I/O 是 text in/text out。Agent 看到 = 终端输出。可自由探索 (rg, cat, ls)。
- **Pydantic AI (Python function-based)**: Agent 调 typed Python 函数。可返回结构化数据。你控制 agent 看到什么。

#### 生产就绪度

| 维度 | Pi | Pydantic AI |
|------|---|------------|
| API 稳定性 | 0.x (pre-1.0) | post-1.0, backward-compat |
| Observability | 无内置 | 原生 OpenTelemetry |
| Structured output | 无 (text only) | Pydantic model 强制验证 + 重试 |
| Error handling | basic | ModelRetry + custom handlers |
| Testing/Eval | 无内置 | 内置 eval, conversation replay |
| 社区 | 小 (2 月龄) | 大 (Pydantic 生态) |
| 语言 | TypeScript/Node.js | Python |

#### Prompt Caching / Session / Memory / Skills

| 方面 | Pi | Pydantic AI |
|------|---|------------|
| Prompt caching | 透传 provider | model_settings + context compaction |
| Session 持久化 | 内置 SessionManager | Durable Execution |
| 跨 session memory | 无内置 | Harness Memory capability |
| Skills/知识注入 | Markdown skills + auto-discovery | SkillsCapability + deferred_loading |

**结论**: 对于我们的架构 (外部 orchestrator 控制一切)，这些方面都在 orchestrator 层解决更优。Agent runtime 越轻越好。

### 11.6 最终方向: 全栈 TypeScript — Pi SDK + Zod + 自建 Orchestrator

> **方向变更记录**: 初始倾向 Pydantic AI (Python)，经过深入讨论后反转。
> 核心原因: **tuning quality 优先** — LLM 在真实 coding 环境 (bash/read/write/edit) 中发挥最好，
> Pi 提供这个环境且是 TypeScript 原生。使用同语言栈消除跨语言桥接开销。

**全栈 TypeScript 架构:**

| 层 | 技术 | 语言 |
|----|------|------|
| Orchestrator | 自建 (macro loop + state machine) | TypeScript |
| Agent Runtime | Pi SDK (embedded via `createAgentSession`) | TypeScript |
| Agent Tools | Pi 内置 (bash/read/write/edit) + custom tools | TypeScript |
| LLM Provider | Pi 内置 (20+ provider, model-agnostic) | TypeScript |
| Verification | Zod (schema validation + type inference) | TypeScript |
| Monitor | Fastify/Express + SSE (取代现有 Python monitor) | TypeScript |
| Config | YAML → Zod schema validated | TypeScript |
| Binary | bun build --compile / pkg | TypeScript |

**嵌入 Pi SDK 的实际代码 (基于 OpenClaw 集成):**

```typescript
import { createAgentSession, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";

const { session } = await createAgentSession({
  cwd: taskWorkspaceDir,
  model: config.model,                    // 从 config.yaml 读取
  tools: builtInTools,                    // Pi 内置 read/write/edit/bash
  customTools: [compileKernelTool, benchmarkTool],  // 可注入自定义
  sessionManager: new SessionManager(...),
  settingsManager: new SettingsManager(...),
});

// 注入 tuning 上下文作为 system prompt
applySystemPromptOverrideToSession(session, buildTuningPrompt(task));

// 运行 agent (Pi 管理内部 tool-calling loop)
const result = await session.run(userPrompt, { maxTurns: 30 });

// Orchestrator 验证 (Zod)
const validated = KernelOutputSchema.safeParse(extractResult(result));
if (!validated.success) { /* retry or escalate */ }
```

**选择理由:**
1. **Tuning quality 优先** — LLM 在真实 coding env (Pi) 中产出更高质量代码
2. **同语言栈** — 无 Python↔TS 桥接，开发/调试统一
3. **Pi 原生能力** — session 管理、20+ model provider、extensions、RPC 全内置
4. **Zod = TS 的 Pydantic** — schema 验证 + 类型推导，用于 orchestrator 验证 gate
5. **可编译为 binary** — bun build --compile 产出独立可执行文件
6. **Pi 设计哲学** — 极简核心 + 用户扩展，与我们"orchestrator 控制一切"理念一致
7. **LLM 熟悉环境** — bash/read/write/edit 就是 Claude Code/Cursor 训练时的 tools

### 11.7 排除方案记录

| 方案 | 排除原因 |
|------|---------|
| Humanize | 完全依赖 Claude Code，不能嵌入独立 binary |
| Claude Code subprocess | 依赖安装、只支持 Claude、输出不够结构化 |
| Cursor SDK | 依赖 Cursor 生态，不够独立 |
| Omnigent | 运行时环境不是可嵌入库、Alpha 阶段 |
| Pydantic AI | Python (跨语言桥接) + 自定义 function tools 限制 LLM 创造力 |
| OpenAI Agents SDK | 主要只支持 OpenAI models + Python |
| litellm + 自建 (Python) | 跨语言桥接 + 需自建所有 agent infra |
| 完全自建 | 开发成本高，Pi SDK 已提供所需原语 |

### 11.8 Scope Boundary (范围边界)

**v1 (MVP):**
- 单任务 tuning loop (一个 shape + op + DSL)
- Pi SDK 嵌入 — LLM 在 coding env 中工作
- compile + benchmark + profile 自动化
- config.yaml 读取 (model endpoint + API key + task params)
- model 配置可切换 (config 里指定 step→model 映射)
- 结果 + trajectory 保存到 filesystem (JSONL)
- Zod 验证 gates

**v1 架构预留 (不实现但接口准备好):**
- Multi-model grey testing (同 task 多模型比较)
- 自定义 RL-tuned model 接入 (只需 litellm/Pi 兼容 API)
- Monitor web UI (接口预留, 事件格式定义好)

**明确 v1 之外:**
- RL 训练本身 (只做轨迹收集)
- Web UI / Monitor 实现
- 跨任务学习
- 分布式多机
- Self-training flywheel

### 11.9 关键设计原则总结

1. **给 LLM 真实 coding 环境** — Pi 的 bash/read/write/edit，不做 summarization
2. **双层鲁棒性** — agent 内部能力 (low-hanging fruit) + orchestrator 确定性锁
3. **Tuning quality > 架构整洁** — 优先让 LLM 发挥最大编码能力
4. **Skills 作为 markdown** — 知识以文本注入 system prompt (Pi native)
5. **全 trajectory 记录** — 为未来 grey test + RL 训练留数据
6. **Model-agnostic** — Pi 支持 20+ provider，config 切换即可

---

1. AutoKernel — https://github.com/RightNow-AI/autokernel (2026)
2. KernelEvolve — Meta ISCA 2026, https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve/
3. Astra — Stanford, NeurIPS DL4C 2025, https://github.com/Anjiang-Wei/Astra
4. TritonForge — arxiv 2512.09196 (2025)
5. MTMC/QiMeng-Kernel — AAAI 2026, arxiv 2511.20100
6. Two-Stage Tuner — arxiv 2601.12698 (2026)
7. AKG Agent — Huawei/MindSpore, arxiv 2512.23424, https://github.com/mindspore-ai/akg
8. NVIDIA CompileIQ — CUDA 13.3, https://github.com/nvidia/compileiq
9. TVM/Ansor — Apache TVM, https://tvm.apache.org/
10. XLA Autotuning — https://openxla.org/xla/persisted_autotuning
11. Triton Autotuner — https://triton-lang.org/
12. Helion — PyTorch, https://pytorch.org/projects/helion/
13. TileLang — Microsoft, arxiv 2504.17577
14. KernelBench — Stanford, https://github.com/ScalingIntelligence/KernelBench
15. Towards Automated Kernel Generation in the Era of LLMs — Survey, arxiv 2601.15727
16. tritonBLAS — arxiv 2512.04226
17. LLM-Guided Autotuning for Helion — PyTorch Blog 2026
18. Pi Coding Agent — https://pi.dev/, npm: @earendil-works/pi-coding-agent (2026)
19. Pydantic AI — https://pydantic.dev/pydantic-ai, https://github.com/pydantic/pydantic-ai
20. Pydantic AI Harness — https://github.com/pydantic/pydantic-harness
21. OpenAI Agents SDK — https://openai.github.io/openai-agents-python/
22. Omnigent — https://omnigent.ai/, https://github.com/omnigent-ai/omnigent (Databricks, 2026)
23. Humanize — https://github.com/PolyArch/humanize (Claude Code plugin)
24. OpenClaw Pi Integration — https://github.com/openclaw/openclaw (Pi SDK 实际集成案例)
25. litellm — https://github.com/BerriAI/litellm (统一 LLM API)
