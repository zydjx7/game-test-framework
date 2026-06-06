# 基于 LLM Agent 的 FPS 游戏自动化测试框架

[English](README.md) | 中文

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-ff69b4.svg)](https://langchain-ai.github.io/langgraph/)
[![BDD](https://img.shields.io/badge/Spec-Goal--level%20Gherkin-yellow.svg)](https://cucumber.io/docs/gherkin/)
[![VLM](https://img.shields.io/badge/Perception-VLM%20%2B%20GroundTruth-green.svg)](https://vizdoom.farama.org/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-purple.svg)](https://api.deepseek.com)
[![Tests](https://img.shields.io/badge/tests-117%20passing-brightgreen.svg)](#)

这是一个用于 **FPS 游戏自动化测试** 的 LLM Agent 框架。你用 Gherkin 写测试
目标，而不是写逐步脚本；Agent 会感知游戏状态（VLM 或 ground truth）、通过
function calling 自主选择动作、判断目标是否达成，并在动作结果异常时进行反思：
判断失败更像 perception / execution / logic，随后恢复或报告 bug。

> 硕士研究项目（M1，立命馆大学情报理工学研究科）。项目同时覆盖两个方向：
> **AI 测试**（BDD、VLM 游戏状态识别、Agent 执行测试、bug 检测）和
> **Agent 工程**（Planning、Tool Use / Function Calling、Reflection、
> LangGraph 编排、Agent 评估）。

---

## 为什么有意思

- **Goal-level BDD，而不是 step scripts。** 场景只描述测试目标，例如
  `Success: ammo_before - ammo_after >= 1`，Agent 自己决定如何达成目标。
- **Agent 通过 DeepSeek 原生 function calling 自主选动作。** 使用真实的
  `tools` / `tool_calls`，不是靠 prompt 解析伪装工具调用。
- **VLM 感知可以自动评估。** ViZDoom 暴露真实游戏状态，因此可以零人工标注
  评估 VLM 读 HUD 的准确率。Phase 1 spike 中 Qwen3-VL-Flash 的精确 ammo 读取
  达到 **100%**。
- **诊断式恢复阶梯（diagnostic recovery ladder）。** LangGraph 状态图在异常时
  先 **re_observe**（重读，不动作），再 **retry**（重做动作），两者都失败才
  **报告疑似 logic bug**。它**不依赖** LLM 的 perception/execution 判断来选恢复，
  而是用"需要哪一步才能恢复"来反推根因。持续性（logic-like）故障被报告 **5/5**，
  而没有反思的 baseline 静默超时 **0/5**。
- **已经展示可迁移性。** 同一个 Agent 层同时跑在 **ViZDoom** 和纯 Python
  **ToyFPS** 上，支持 ammo、health、score 多指标目标，且 Agent 代码不需要改。

## 架构

**Agent 层游戏无关、原样复用**；只有 **Adapter 层** 按每个游戏实现。`goals.feature`
被解析成 `Goal`，然后一个 LangGraph 状态机驱动 observe → decide → act → check 循环，
失败时沿"诊断式恢复阶梯"逐级升级。

```mermaid
flowchart TB
    GF["📝 goals.feature<br/>goal-level BDD"] -->|parse| GOAL["🎯 Goal<br/>success = 对 cumulative state 的表达式"]

    subgraph AGENT["🧠 Agent 层 · 游戏无关，原样复用 · LangGraph StateGraph"]
        direction TB
        OBS["👁 observe<br/>VLM / GroundTruth"] --> DEC["🧩 decide<br/>DeepSeek function calling"]
        DEC --> ACT["🎮 act<br/>运行 composite template"]
        ACT --> CHK{"✅ 期望达成?"}
        CHK -->|success| DONE(["done"])
        CHK -->|continue| DEC
        CHK -->|anomaly| REF["🔎 reflect<br/>记录三类诊断"]
        REF --> RO["① re_observe<br/>重读, 不动作"]
        RO -->|recovered| CHK
        RO -->|still failing| RT["② retry<br/>重做动作"]
        RT -->|recovered| CHK
        RT -->|阶梯耗尽| BUG["🐞 REPORT<br/>疑似 logic bug"]
    end

    GOAL --> OBS

    subgraph ADAPTER["🔌 Adapter 层 · 每个游戏实现"]
        direction LR
        ENV["🗺 Env / State<br/>game_variables · screen?"]
        PER["🔬 Perceptor<br/>VLM 或 GroundTruth"]
        AL["🛠 Action library<br/>composite test templates"]
    end

    OBS -. reads .-> PER
    ACT -. drives .-> AL
    AL -. wraps .-> ENV
    ADAPTER --> IMPL["实现示例:  env/ + actions/ → ViZDoom  ·  toy_fps/ → 纯 Python"]

    classDef spec fill:#f3e9ff,stroke:#8a5cc2,color:#2e1a47;
    classDef agent fill:#e7f0ff,stroke:#4a78c2,color:#10243e;
    classDef ladder fill:#fff1e0,stroke:#d98324,color:#5a3210;
    classDef report fill:#ffe1e1,stroke:#c23b3b,color:#4a1010;
    classDef adapter fill:#e6f7ec,stroke:#3aa364,color:#0f3a23;
    class GF,GOAL spec;
    class OBS,DEC,ACT,CHK,REF,DONE agent;
    class RO,RT ladder;
    class BUG report;
    class ENV,PER,AL,IMPL adapter;
```

**恢复阶梯是研究核心。** perception 和 execution 故障在异常发生时现象相同，所以
Agent **不**信任 LLM 的 `perception`/`execution` 标签来选恢复，而是按副作用从小到大
升级：① **re_observe**（重读，零副作用）恢复 perception 故障；② **retry**（重做动作）
恢复 execution 故障；③ 两者都失败，存活的故障才**报告为疑似 logic bug**。恢复的
*结果*就是诊断信号 —— 见 [`Doc/adr/0004`](Doc/adr/0004-diagnostic-recovery-ladder.md)。

测试能力对应 Agent 能力：BDD 规格、VLM/CV 游戏状态识别、bug 检测，对应
Planning、Tool Use、Reflection、LangGraph 和 Agent evaluation。

## 当前状态与关键结果

| Phase | 状态 | 结果 |
|---|---|---|
| 0：ViZDoom 环境 + perception 接口 | 完成 | wrapper + `GameStatePerceptor` ABC |
| 1：VLM perception vs ground truth | 完成 | Qwen3-VL-Flash 精确 ammo 准确率 **100%** |
| 2：Action library + goal-level Gherkin + agent loop | 完成 | 3/3 goals，Agent 通过 function calling 自主选动作 |
| 3 + step3：诊断式恢复阶梯（LangGraph） | 完成 | 持续性故障报告 **5/5** vs baseline 静默 **0/5**；perception/execution 由恢复阶梯区分 |
| 3.5：核心 schema 泛化 + ToyFPS 第二游戏 | 完成 | 同一 Agent 跑在两个游戏上，支持多指标 |
| 4：LLM oracle + mutation testing | 计划中 | 论文 track |

当前有 117 个 unit tests 通过。AssaultCube 本科 CV/BDD baseline 保留为后续对比。

## 快速开始

```bash
git clone https://github.com/zydjx7/game-test-framework.git
cd game-test-framework
pip install -r requirements.txt
```

项目根目录 `.env`：

```bash
DEEPSEEK_API_KEY=sk-...          # 或 OPENAI_API_KEY（OpenAI-compatible DeepSeek）
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DASHSCOPE_API_KEY=sk-...         # Qwen3-VL-Flash perception（阿里云百炼）
```

三个常用命令：

```bash
python -m pytest                         # 117 tests；不需要 API key 或 ViZDoom
python experiments/phase2_agent_demo.py  # ViZDoom 上跑 3 个 Agent goals（需要 ViZDoom + DeepSeek）
python experiments/toy_fps_demo.py       # 同一个 Agent 跑纯 Python ToyFPS（可迁移性 demo）
```

## 接入一个新游戏

Agent 层固定不动；新游戏只需要提供一个 **adapter**：

- 一个 state，包含 `game_variables` 字典；`screen` 可选。
- 一组 primitives，例如 `reset()`、`observe()`、`fire_once()`、`heal()`。
- 一个 action library，提供 Agent 可选择的 composite test templates。
- 一个 `goals.feature`，用 `<metric>_before` / `<metric>_after` / `steps`
  写成功条件。
- 如果游戏能提供结构化变量，可以复用 `GroundTruthPerceptor`；如果只能看画面，
  则写 VLM/CV perceptor。

完整契约和 checklist 见
[`Doc/adapter-contract.md`](Doc/adapter-contract.md)。参考实现见
[`toy_fps/`](toy_fps/) 和 [`tests/test_toy_fps.py`](tests/test_toy_fps.py)。

## 项目结构

```text
perception/   GameStatePerceptor ABC；GroundTruth / VLM / CV perceptors
actions/      primitives、composite test templates、result schema helpers
agent/        goal parser、function-calling loop、reflection、LangGraph graph
env/          ViZDoom wrapper + trajectory recorder
toy_fps/      纯 Python 第二游戏（可迁移性证明 + 快速测试 fixture）
experiments/  perception/reflection eval、agent demos、failure injection
tests/        117 pytest unit tests
Doc/          research plan、各 phase design notes、adapter contract、reports
Code/ src/    AssaultCube baseline（本科系统，保留用于对比）
```

## 技术栈

`LLM Agent` / `Function Calling / Tool Use` / `LangGraph` /
`Reflection / failure taxonomy` / `VLM(Qwen3-VL) perception` /
`Goal-level BDD / Gherkin` / `Agent evaluation` / `ViZDoom` /
`DeepSeek` / `Python`

## 设计文档

- [`Doc/research-plan.md`](Doc/research-plan.md)：5-phase 研究计划
- [`Doc/phase1-design.md`](Doc/phase1-design.md)：VLM perception spike
- [`Doc/phase2-design.md`](Doc/phase2-design.md)：Action library + Agent loop
- [`Doc/phase3-design.md`](Doc/phase3-design.md)：三类失败反思
- [`Doc/phase3-reflection-report.md`](Doc/phase3-reflection-report.md)：反思评估报告
- [`Doc/adapter-contract.md`](Doc/adapter-contract.md)：新游戏接入契约
- [`Doc/v2-roadmap.md`](Doc/v2-roadmap.md)：MCP / RAG / multi-agent 后续路线
