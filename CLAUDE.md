# 项目上下文 — Game Testing Research

## 项目身份

研究方向：基于 LLM Agent 的游戏自动化测试框架。
起源：本科 (RiverGame on AssaultCube) → 硕士 ViZDoom agent loop → **当前：转向真引擎 (Unity) 的 gameplay regression 测试**。

**当前方向（2026-06 起 / forward 权威文档 → `Doc/project-direction.md`）**：
GameTest Agent System —— 复用已建的 Python agent core，把测试目标从 ViZDoom 的
toy 机制 (ammo/health) 升级到真实引擎里的**集成层 / 呈现层 / 进度 softlock** bug
（单元测试天然漏掉、需要玩家可见 oracle 的那类）。
ViZDoom 轨迹 (Phases 0–3.5) **已完成、117 tests green、保留为 Python core + 可移植性 baseline**。

> 决策前必读：`Doc/project-direction.md`（go-forward 权威）。`Doc/research-plan.md`
> 是 ViZDoom 轨迹的历史 master plan（已完成部分 + v1 不变量），**不再是 forward direction**。

## 一句话北极星

> 在真实引擎里，从需求生成分层测试，跑出真实 gameplay bug（尤其单元测试漏掉的
> 呈现层 / 进度 softlock bug），拿到 screenshot + trace + debug_state，生成开发者
> 可复现的报告。

## 铁律：live-smoke 优先（最重要）

ViZDoom 轨迹能成，是因为每步可验证：改 → `python -m pytest` → 跑 demo → 读
trace/state → **知道 AI 写的东西是否真的工作**。Unity 会削弱这个循环（故障藏在
editor/scene/prefab 状态里，AI 看不全、初学者难 debug）。所以：

> **任何 Unity 改动，不经过机器可判定的 PASS 不算完成。** Unity 版 `pytest` 是命令行
> PlayMode runner：
> `Unity -runTests -batchmode -projectPath <proj> -testPlatform PlayMode -testResults results.xml`
> 不能被 PlayMode test / smoke 脚本（写 PASS/FAIL）确认的改动，就是没做完。

## 当前 Gate（详见 `Doc/project-direction.md`，不要 start Gate N+1 前先过 Gate N）

- **Gate 0**（现在）：Unity **测试夹具骨架**（空/最小 3D 项目，不是游戏，无玩家/枪/相机/HUD/敌人）+
  一个**琐碎**机制（开关门）+ 命令行 PlayMode test + 导出 `debug_state.json`（screenshot 尽力而为）。
  判据：不开 Editor 也能从命令行知道 pass/fail。**从空项目起步，不导入 FPS 模板**；无 agent / 无 VLM / 无 bridge。
- Gate 1：checkpoint softlock fixture + 注入 door-not-persisted bug。
- Gate 2：Python runtime bridge（reset/action/observe/...），no-LLM 跑通。
- Gate 3：Gameplay Agent（复用 v1 core）接入，报 progression_softlock。
- Gate 4：Spec-to-Test Agent（先 Test Plan IR + 模板，不直接 LLM 生成 C#）。
- Gate 5：VLM 作视觉证据（不单独定罪）。

## AI 硬规则（Claude Code / Codex 都遵守，详见 `Doc/project-direction.md`）

1. Unity live-smoke 没过，不加新功能。
2. 没有 PlayMode test / smoke 写 PASS/FAIL，不声称 Unity 改动可用。
3. checkpoint softlock 端到端（Gate 3）跑通前，不做 multi-agent 编排。
4. VLM 不当唯一 oracle，只当 debug_state 旁边的视觉证据。
5. 第一条 vertical slice（Gate 3）稳定前，不碰 coverage / mutation。
6. **ViZDoom v1 项目保持 green、不动**——复用的 Python core + 可移植性证据。
7. runtime bridge 实现 `Doc/adapter-contract.md`；editor MCP 只管 authoring，绝不当 runtime oracle。
8. 不为后续 Gate 预写 spec/代码（如 bridge 协议在 Gate 2 才写）——沿用 v1 "不预建"纪律。

## 复用资产（v1 → 保持 green，不重写）

- `agent/` —— loop / goal (goal-level Gherkin) / reflection / graph (LangGraph) / recovery ladder
- `perception/` —— GroundTruthPerceptor + VLMPerceptor (Qwen3-VL-Flash)
- `actions/result.py` —— flat `<metric>_before/_after` schema
- `Doc/adapter-contract.md` —— runtime bridge 规格
- ViZDoom + ToyFPS adapters + tests —— 可移植性证据

## 三个新意点（防伪，任何"重设计"必须保留）

仍然是：(1) Goal-level Gherkin (2) failure 反思 + 诊断恢复 ladder (3) 注入 bug 评估 + 报告 oracle。
v2 落地：(1) Gate 3 的 goal；(2) 复用 reflection/recovery，新增 progression_softlock 类；
(3) Gate 1 注入 door-not-persisted bug 当 mutation，Gate 3/5 bug report + VLM 证据当 oracle。

## 关键设计决定（不要回退）

- **DeepSeek 是唯一文本/决策 LLM provider**（OpenAI-compatible SDK 接 DeepSeek，配置入口 `src/llm/client_helpers.py`）。**VLM 感知是例外**：用 Qwen3-VL-Flash / DashScope（`VLMPerceptor`），不要把这两者混为一谈。
- **Ground truth 来自引擎内部状态**（ViZDoom `game_variables` / Unity debug_state exporter），不用 OCR。
- **Gherkin 是"目标描述"不是"步骤描述"**。
- **perception / actions / agent / oracle 解耦，能独立测试。**
- **runtime bridge ≠ editor MCP**（见硬规则 7）。

## 关键规划文档（决策前先读）

| 文档 | 内容 |
|---|---|
| `Doc/project-direction.md` ⭐ **forward 权威** | GameTest Agent System 方向 + Gate 0–5 + 硬规则 |
| `Doc/adapter-contract.md` | runtime bridge 规格（新游戏要实现的接口） |
| `Doc/research-plan.md` | ViZDoom 轨迹 master plan（Phases 0–3.5 已完成；v1 不变量 + baseline） |
| `Doc/adr/README.md` | 长期决策（ADR-0001 result schema / 0003 failure 边界 等） |

## 项目路径约定

| 路径 | 用途 |
|---|---|
| `F:\game-testing-main\` | 主项目（Python agent core + 适配器 + 文档） |
| `Code/` / `src/rivergame/` | 本科 AssaultCube/RiverGame baseline，**永不动** |
| `env/` `agent/` `actions/` `perception/` `toy_fps/` | v1 已建、复用、**保持 green** |
| `unity/`（repo 内） | v2 新增 Unity 项目，**定为仓库内**；建项目前先在 `.gitignore` 加 Unity block（`Library/`/`Temp/`/`Obj/` 等生成物） |
| `F:\OBSIDIAN\Obsidian Vault\论文\` | 规划/笔记/汇报草稿，**不放代码**；AI 不改 Obsidian |

## ViZDoom 关键技术事实（v1 core 仍在用，避免 LLM 幻觉）

- `state.screen_buffer` 是 (C, H, W)，存图要 `np.transpose(buf, (1,2,0))`
- `state.game_variables` 是 numpy array，按 `.cfg` 位置顺序索引，没有 dict 形式
- 没有 `game.is_alive()`，从 `is_episode_finished()` + game_variables 推断
- Windows 装不上 80% 是缺 Microsoft Visual C++ 2015-2022 Redistributable
- 稳定组合：ViZDoom 1.2.x + Python 3.10-3.12

## 法律/职业红线（绝不写代码）

- **不为任何商业游戏写 Memory Hook / Input Hook / API Hook 代码**。反作弊语境在中国就业敏感，简历有这类代码会被嫌弃。
- bug 注入只在**自建** scenario / **自建** Unity 项目里做（改自己源码 + 重新编译的合法路径），不动商业游戏、不走 hook。
- Unity 测试目标是**自己搭的 MiniFPS / 自己改的开源模板**，不 reverse engineer 任何商业游戏。

## 验证命令

```powershell
# v1 Python core（保持 green）
python -m pytest
python experiments\vizdoom\hello_doom.py
python experiments\toy_fps_demo.py

# v2 Unity（Gate 0 落地后）
# Unity -runTests -batchmode -projectPath <proj> -testPlatform PlayMode -testResults results.xml
# python scripts\unity_smoke.py   # Gate 2 起
```

## 行为约定（给我自己看的）

0. **多 agent 协作**：开工前读 `AGENTS.md` + `WORKLOG.md`，按 fetch/rebase/push 协议执行。共享文件（README/.gitignore/requirements/pytest.ini/CLAUDE.md/AGENTS.md/WORKLOG.md）commit 用 `shared:` 前缀。
1. **学习标签**：每次产出实质代码/设计/决策后附 🎓 学习标签（见 ~/.claude/CLAUDE.md）。Plumbing 可省。
2. **决策前看 `Doc/project-direction.md`**，不要忽视它另起炉灶。
3. **新意点防伪**：见上"三个新意点"段，任何"重设计"必须保留那三点。
4. **不要扩 scope**：用户基础有限 + 无 DDL，最大风险是 scope creep + "设计跑在执行前面"。守 Gate 纪律 + live-smoke 优先。
5. **legacy 谨慎**：`Code/` / `src/rivergame/` 是本科产物，不"清理"，作 baseline 保留。
