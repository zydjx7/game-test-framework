# 项目上下文 — Game Testing Research

## 项目身份

研究方向：基于 LLM Agent 的游戏自动化测试框架。
起源：本科论文 (RiverGame extension on AssaultCube) → 硕士扩展。
当前状态：从 AssaultCube 静态截图测试迁移到 **ViZDoom 动态 agent loop**。

## 5 Phase 全貌

| Phase | 月份 | 核心目标 |
|---|---|---|
| **0**（当前）| M1 | 重构 + ViZDoom 环境就绪 + 老师 buy-in |
| 1 | M2-3 | Perception（VLM + Ground Truth 对比） |
| 2 | M4-5 | Action Executor + Goal-level Gherkin + 最小 agent loop |
| 3 | M6-7 | Reflection（三类 failure 分类与恢复） |
| 4 | M8-9 | LLM Oracle + Mutation Testing |
| 5 | M10+ | 论文 + 求职 |

跨 Phase 提前讨论某模块细节会让用户失焦。**提到未来 Phase 时给一句话点位即可，不要展开**。
完整 Phase 内容看 → `F:\OBSIDIAN\Obsidian Vault\论文\扩展构想-ViZDoom版.md`

## 当前阶段：Phase 0

**目标**：
1. 完成原 AssaultCube 项目的模块化重构（perception 接口抽出）
2. 在 sandbox（沙盒环境）中验证 ViZDoom 可以跑通
3. 准备好向丸山老师定期汇报的成果（demo + 数据 + 简短报告）

**Phase 0 完成定义**：
- ✅ AssaultCube baseline 测试仍能跑通（不被改坏）
- ✅ ViZDoom hello world 能产出 ammo trajectory（轨迹）+ 截图数据集
- ✅ 主项目"安全重构"完成：4 模块目录 + `GameStatePerceptor` 接口就位
- ✅ 一份给老师的进度汇报材料就绪（不需要他先批准才推进）

具体步骤看 → `F:\OBSIDIAN\Obsidian Vault\论文\ViZDoom-hello-world-两天路线.md`

## 当前阻塞 / 依赖

- **没有 hard blocker（硬阻塞）**。研究室文化是"看成果给建议"，不在每个决策上等老师。
- 进行中：ViZDoom sandbox 验证（已到 Step 2.1+）+ 主项目安全重构 **并行推进**
- 给老师的汇报应定期主动准备（每 Phase 末或关键决策后），但**不是推进的 gate（关卡）**

## 目标架构（基于既有重构状态）

**项目已经过一轮重构**（详见 `F:\OBSIDIAN\Obsidian Vault\论文\目前已进行的重构及项目状况.md`）：
- `Code/bdd/` = bdd_runner 角色（DeepSeek + behave 主流程，已跑通）
- `Code/bdd/test_generator/` = gherkin_generator 角色
- `Code/bdd/features/steps/` = assertion 角色
- `Code/GameStateChecker/` = CV 感知（待 Phase 0.1 包装）
- `src/llm/client_helpers.py` = DeepSeek 共享层（已统一）
- `src/gherkin/` = Gherkin parser
- `src/rivergame/` = legacy，不动
- `tests/` = pytest（legacy 标记已分离）

**因此不需要建 gherkin_generator/ bdd_runner/ assertion/ 新目录**。既有 Code/ 结构功能上就是它们。

```
project/
├── Code/                既有 AssaultCube baseline（不动）
├── src/                 既有共享层 + legacy（不动）
├── tests/               既有 pytest（持续维护）
│
├── perception/          ⭐ Phase 0.1 唯一新增
│   ├── base.py          GameStatePerceptor 抽象接口
│   └── cv_perceptor.py  包装 Code/GameStateChecker
│
├── env/                 Phase 1: ViZDoom 环境（从 sandbox 迁入）
├── perception/          Phase 1+ 扩充: ground_truth.py + vlm_perceptor.py
├── actions/             Phase 2
├── agent/               Phase 2-3
├── oracle/              Phase 4
└── experiments/         Phase 1+ 渐增
```

**关键原则**：既有 Code/ 和 src/ **不重命名、不迁移、不"清理"**。改名无功能收益，纯增风险。

## 关键设计决定（不要回退）

- **DeepSeek 是唯一 LLM provider**。代码用 OpenAI-compatible SDK 接 DeepSeek。不要重新引入 OpenAI 分支。配置入口：`src/llm/client_helpers.py`
- **ViZDoom 是主平台**，AssaultCube 降级为论文 baseline。不要把 AssaultCube 当主线开发。
- **Ground truth 用 ViZDoom 的 `state.game_variables`**，不要再用 OCR 做 ammo 识别。
- **Gherkin 从"步骤描述"升级为"目标描述"**。`weapon_steps.py` 那种逐步 step function 是 legacy。
- **保留模块解耦**：perception / actions / agent / oracle 必须能独立测试。

## 关键规划文档（决策前先读）

| 文档 | 内容 |
|---|---|
| `F:\OBSIDIAN\Obsidian Vault\论文\扩展构想-ViZDoom版.md` | 5 个 Phase 的完整 master plan |
| `F:\OBSIDIAN\Obsidian Vault\论文\ViZDoom-hello-world-两天路线.md` | 当前 sprint 的具体步骤 |
| `F:\OBSIDIAN\Obsidian Vault\论文\项目架构.md` | 旧（AssaultCube）状态，参考用 |
| `F:\OBSIDIAN\Obsidian Vault\论文\扩展构想.md` | 旧 Phase 规划（AssaultCube 版本），已被 ViZDoom 版替代 |

## 项目路径约定

| 路径 | 用途 |
|---|---|
| `F:\game-testing-main\` | 主项目代码（重构后的目标态）|
| `F:\vizdoom-sandbox\` | Phase 0.2 验证沙盒，**独立环境**，验证完前不要混入主项目 |
| `F:\OBSIDIAN\Obsidian Vault\论文\` | 规划文档、笔记、汇报草稿。**不放代码** |
| `Code/`（项目内）| 本科 AssaultCube 代码，作为论文 Section IV baseline 保留 |
| 实验产物（CSV / screenshots / demo 视频）| Phase 0 期间先放沙盒，Phase 0 结束后迁入主项目 `experiments/` |

## ViZDoom 关键技术事实（避免 LLM 幻觉）

- `state.screen_buffer` 是 **(C, H, W)** 通道在前，不是 OpenCV/PIL 的 (H, W, C)。存图要 `np.transpose(buf, (1, 2, 0))`
- `state.game_variables` 是 numpy array，按 `.cfg` 里 `available_game_variables` 的**位置顺序**索引，没有 dict 形式
- 没有 `game.is_alive()`，要从 `game.is_episode_finished()` + game_variables 自己推断
- Windows 装不上时 80% 是缺 Microsoft Visual C++ 2015-2022 Redistributable
- 稳定版本组合：ViZDoom 1.2.x + Python 3.10-3.12

## 法律/职业红线（绝不写代码）

- **不为任何商业游戏写 Memory Hook / Input Hook / API Hook 代码**。反外挂、反作弊法规在中国就业语境敏感，简历有这类代码会被嫌弃
- bug 注入只在 ViZDoom 的 ACS 脚本（Doom 的内置脚本语言）或自建 scenario（场景配置）里做，不动商业游戏
- 即使是 AssaultCube 这种开源游戏，也只走"改源码 + 重新编译"的合法路径，不走 hook

## 验证命令

```powershell
# 当前 AssaultCube baseline（保留）
python -m pytest
python Code\bdd\run_tests.py --mode predefined --feature generated_test.feature --target assaultcube

# 未来 ViZDoom 主线（Phase 0 之后）
python env\vizdoom_env.py    # self-test
python generate_trajectory.py
```

## 行为约定（给我自己看的）

1. **学习标签**：每次产出实质性代码 / 设计 / 决策后，附 🎓 学习标签（见 ~/.claude/CLAUDE.md）。Plumbing 类操作（移文件、pip install）可省略。
2. **决策前看规划文档**：用户对计划文档投入了大量思考，不要忽视它们另起炉灶。
3. **新意点防伪**：用户研究的差异化点是 (1) Goal-level Gherkin (2) 三类 failure 反思 (3) Mutation Testing + LLM Oracle。任何"重新设计架构"的建议必须保留这三点。
4. **不要扩 scope**：用户基础有限 + 无 DDL，最大风险是 scope creep。看到"顺便加个 X"的冲动先停一下。
5. **legacy 谨慎处理**：`Code/` 目录下 AssaultCube 代码是本科产物，不要"清理"它，作为 baseline 保留。
