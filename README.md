# 🎮 Agent-Based FPS Game Testing Framework

English | [中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-ff69b4.svg)](https://langchain-ai.github.io/langgraph/)
[![BDD](https://img.shields.io/badge/Spec-Goal--level%20Gherkin-yellow.svg)](https://cucumber.io/docs/gherkin/)
[![VLM](https://img.shields.io/badge/Perception-VLM%20%2B%20GroundTruth-green.svg)](https://vizdoom.farama.org/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-purple.svg)](https://api.deepseek.com)
[![Tests](https://img.shields.io/badge/tests-117%20passing-brightgreen.svg)](#)

An **LLM-agent framework for automatically testing FPS games**. You write a test
*goal* in plain Gherkin; an agent perceives the game (VLM or ground truth),
**chooses its own actions** via function calling, judges success, and — when an
action doesn't do what it should — **reflects on why** (perception / execution /
logic) and either recovers or reports a bug.

> 🎓 修士研究 (M1, 立命館大学情報理工学研究科). The same system spans two angles:
> **AI-testing** (BDD, VLM game-state recognition, agent test execution, bug
> detection) and **agent engineering** (Planning, Tool Use / Function Calling,
> Reflection, LangGraph orchestration, evaluation).

---

## ✨ Why it's interesting

- **Goal-level BDD, not step scripts.** A scenario states the *goal*
  (`Success: ammo_before - ammo_after >= 1`); the agent decides *how*. No
  per-step Python functions to maintain.
- **The agent picks its own actions** with DeepSeek **native function calling**
  (real `tools`/`tool_calls`, not prompt-parsed).
- **VLM perception with automatic ground truth.** ViZDoom exposes the true state,
  so VLM accuracy is measured with zero human labels — concrete ammo reading hit
  **100%** in the Phase 1 spike.
- **Diagnostic recovery ladder** as a **LangGraph** state machine: on an anomaly
  the agent **re-observes** (re-read, no action), then **retries** (re-run), then
  **reports a suspected logic bug** once both fail. It does *not* trust the LLM's
  `perception`/`execution` label to pick a recovery — it disambiguates by *what
  it takes to recover*. A persistent (logic-like) fault is reported **5/5** vs a
  no-reflection baseline that silently times out **0/5**.
  *(The reliably-evaluated boundary is logic vs non-logic — whether to report a
  bug; perception vs execution share an observable and are not claimed as
  accurate classification. See [`Doc/phase3-reflection-report.md`](Doc/phase3-reflection-report.md)
  and [`Doc/adr/0004`](Doc/adr/0004-diagnostic-recovery-ladder.md).)*
- **Demonstrated portability.** The *same* agent layer runs on **two games** —
  ViZDoom and a pure-Python **ToyFPS** — with multi-metric goals (ammo, health,
  score) and **zero agent-code changes**. Plugging in a new game means writing an
  adapter, not touching the agent.

## 🧠 Architecture

The **agent layer is game-agnostic and reused as-is**; only the **adapter layer**
is written per game. A `goals.feature` spec is parsed into a `Goal`, then a
LangGraph state machine drives an observe → decide → act → check loop, escalating
through a diagnostic recovery ladder on failure.

```mermaid
flowchart TB
    GF["📝 goals.feature<br/>goal-level BDD"] -->|parse| GOAL["🎯 Goal<br/>success = expression over cumulative state"]

    subgraph AGENT["🧠 AGENT LAYER · game-agnostic, reused as-is · LangGraph StateGraph"]
        direction TB
        OBS["👁 observe<br/>VLM / GroundTruth"] --> DEC["🧩 decide<br/>DeepSeek function calling"]
        DEC --> ACT["🎮 act<br/>run composite template"]
        ACT --> CHK{"✅ expectation met?"}
        CHK -->|success| DONE(["done"])
        CHK -->|continue| DEC
        CHK -->|anomaly| REF["🔎 reflect<br/>record 3-type diagnosis"]
        REF --> RO["① re_observe<br/>re-read, no action"]
        RO -->|recovered| CHK
        RO -->|still failing| RT["② retry<br/>re-run the action"]
        RT -->|recovered| CHK
        RT -->|ladder spent| BUG["🐞 REPORT<br/>suspected logic bug"]
    end

    GOAL --> OBS

    subgraph ADAPTER["🔌 ADAPTER LAYER · per game"]
        direction LR
        ENV["🗺 Env / State<br/>game_variables · screen?"]
        PER["🔬 Perceptor<br/>VLM or GroundTruth"]
        AL["🛠 Action library<br/>composite test templates"]
    end

    OBS -. reads .-> PER
    ACT -. drives .-> AL
    AL -. wraps .-> ENV
    ADAPTER --> IMPL["implementations:  env/ + actions/ → ViZDoom  ·  toy_fps/ → pure-Python"]

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

**The recovery ladder is the research core.** Perception and execution failures
look identical at the moment of an anomaly, so the agent does *not* trust the
LLM's `perception`/`execution` label to choose a fix. It escalates by
side-effect: ① **re_observe** (re-read the state, zero side effects) recovers a
perception fault; ② **retry** (re-run the action) recovers an execution fault;
③ once both fail, the surviving fault is **reported as a suspected logic bug**.
The recovery *outcome* is the diagnosis — see
[`Doc/adr/0004`](Doc/adr/0004-diagnostic-recovery-ladder.md).

Testing capabilities ride on agent capabilities: BDD specs, VLM/CV game-state
recognition, bug detection ↔ Planning, Tool Use, Reflection, LangGraph, agent
evaluation.

## 📊 Status & key results

| Phase | Done | Result |
|---|---|---|
| 0 — ViZDoom env + perception interface | ✅ | wrapper + `GameStatePerceptor` ABC |
| 1 — VLM perception vs ground truth | ✅ | **concrete ammo accuracy 100%** (Qwen3-VL-Flash), ~¥0.01/run |
| 2 — Action library + goal-level Gherkin + agent loop | ✅ | 3/3 goals, agent self-selects actions via function calling |
| 3 + step 3 — diagnostic recovery ladder (LangGraph) | ✅ | persistent fault reported **5/5** vs baseline silent **0/5**; perception/execution disambiguated by the ladder |
| 3.5 — generalize core beyond ammo + **ToyFPS** 2nd game | ✅ | same agent runs on 2 games, multi-metric |
| 4 — LLM oracle + mutation testing | ⏳ | thesis track |

117 unit tests passing. AssaultCube baseline (undergrad CV/BDD system) is kept
for comparison.

## 🚀 Quickstart

```bash
git clone https://github.com/zydjx7/game-test-framework.git
cd game-test-framework
pip install -r requirements.txt
```

`.env` (project root) — text LLM (DeepSeek) + VLM (Qwen via DashScope):

```bash
DEEPSEEK_API_KEY=sk-...          # or OPENAI_API_KEY (OpenAI-compatible to DeepSeek)
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DASHSCOPE_API_KEY=sk-...         # Qwen3-VL-Flash perception (Alibaba Bailian)
```

Three commands:

```bash
python -m pytest                         # 117 tests (no API key, no ViZDoom needed)
python experiments/phase2_agent_demo.py  # agent self-tests 3 goals on ViZDoom (needs ViZDoom + DeepSeek)
python experiments/toy_fps_demo.py       # SAME agent on the pure-Python ToyFPS (portability; no ViZDoom)
```

## 🔌 Plug in a new game

The agent layer is fixed; a new game is an **adapter**. Provide a state with
`game_variables` (screen optional), primitives, an action library of composite
test templates, and a `goals.feature`. Reuse `GroundTruthPerceptor`, or write a
`VLMPerceptor` for pixel-based perception. Full contract + checklist:
**[`Doc/adapter-contract.md`](Doc/adapter-contract.md)**. Worked example:
[`toy_fps/`](toy_fps/) (~150 lines) + [`tests/test_toy_fps.py`](tests/test_toy_fps.py).

## 🗂️ Project structure

```
perception/   GameStatePerceptor ABC; GroundTruth / VLM / CV perceptors
actions/      primitives + composite test templates + result schema helpers
agent/        goal parser, function-calling loop, reflection, LangGraph graph
env/          ViZDoom wrapper + trajectory recorder
toy_fps/      pure-Python 2nd game (portability proof + fast test fixture)
experiments/  perception/reflection eval, agent demos, failure injection
tests/        117 pytest unit tests
Doc/          research plan, per-phase design notes, adapter contract, reports
Code/ src/    AssaultCube baseline (undergrad system, kept for comparison)
```

## 🧰 Tech stack

`LLM Agent` · `Function Calling / Tool Use` · `LangGraph` · `Reflection / failure
taxonomy` · `VLM (Qwen3-VL) perception` · `Goal-level BDD / Gherkin` · `agent
evaluation` · `ViZDoom` · `DeepSeek` · `Python`

## 📚 Design docs

`Doc/research-plan.md` (5-phase plan) · `Doc/phase{1,2,3}-design.md` ·
`Doc/phase3-reflection-report.md` (eval) · `Doc/adapter-contract.md` ·
`Doc/v2-roadmap.md` (MCP / RAG / multi-agent — future).
