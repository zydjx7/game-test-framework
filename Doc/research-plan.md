# 扩展构想 - ViZDoom 路线版（v2.0）

> **Authoritative location**: 本文件以 `Doc/research-plan.md`（git 仓库内）为准。
> 修改后用户会手动同步到 Obsidian 镜像（`F:\OBSIDIAN\Obsidian Vault\论文\扩展构想-ViZDoom版.md`，用户阅读用，方向是 repo → Obsidian）。
> AI agent 不要修改 Obsidian 那份；要改请改 `Doc/research-plan.md`，commit message 用 `shared:` 前缀。

> 本文档替代原 `扩展构想.md` 作为新的主线规划。
> 原 md 保留作为历史参考（AssaultCube + 静态截图思路的最终版本）。

---

## 0. 核心战略

### 本科 → 硕士的真正升级是什么

| 维度 | 本科（AssaultCube） | 硕士（ViZDoom） |
|---|---|---|
| 测试驱动 | Behave 线性 step | Agent loop + 反思 |
| Gherkin 角色 | 每步操作描述 | **高层测试目标描述** |
| 视觉 | 模板匹配 + OCR | VLM（多 backend） + 可与 ground truth 对比 |
| 动作执行 | 无（静态截图） | **Python API 直接调用** |
| 失败处理 | 立即停止 | 反思 → 重试 → 报告 |
| Oracle | 硬编码断言 | LLM-based oracle + Mutation Testing |
| 评估 | 手动准备截图 | 自动化跑批 + 注入 bug |

### 不变的东西（方法论继承）

- LLM 生成测试规约（保留）
- 模块解耦设计（保留并强化）
- BDD 风格（保留，但 Gherkin 降级为目标层）
- 静态视觉验证作为 baseline（在论文里对比用）

### 学术新意点（论文卖点）

```
LLM Agent + Mutation Testing + 多类型 Failure Recovery
       on Lightweight FPS Testing
```

跟 TITAN 的差异：TITAN 做 MMORPG 任务完成，你做 FPS 局部机制的 bug 检测。
跟 RiverGame 的差异：RiverGame 静态规约，你动态 agent loop。
跟 SIMA/Voyager 的差异：它们关心任务完成，你关心 bug detection。

---

## 1. 整体架构（目标态）

**心智模型**：项目分两层 —— **既有 AssaultCube baseline 层（已重构完成，不动）** 和 **ViZDoom 主线层（Phase 0.1 抽接口，Phase 1+ 渐进新增）**。两层通过 `perception/base.py` 的统一接口连接。

```
project/
│
├── 既有 baseline（已重构，本 Phase 不动）
│   ├── Code/
│   │   ├── bdd/                        # bdd_runner 角色（DeepSeek + behave）
│   │   │   ├── run_tests.py            # AssaultCube 主入口
│   │   │   ├── test_generator/         # gherkin_generator 角色
│   │   │   │   └── llm_generator.py
│   │   │   └── features/
│   │   │       └── steps/              # assertion 角色
│   │   │           └── weapon_steps.py
│   │   └── GameStateChecker/           # CV 感知（待包装）
│   │       ├── LogicLayer.py
│   │       ├── VisionUtils.py
│   │       └── main_flask_server.py
│   ├── src/
│   │   ├── llm/                        # DeepSeek 共享层
│   │   │   └── client_helpers.py
│   │   ├── gherkin/                    # Gherkin parser
│   │   └── rivergame/                  # legacy（标 legacy，不动）
│   └── tests/                          # pytest（非 legacy + legacy 标记分离）
│
├── ⭐ Phase 0.1（✅ 已完成）—— 唯一新增 `perception/` 目录
│   └── perception/
│       ├── base.py                     # GameStatePerceptor 抽象接口
│       └── cv_perceptor.py             # 包装 Code/GameStateChecker
│
├── ⭐ Phase 1+ 渐进新增（每个 Phase 只加它需要的，不预建空目录）
│   ├── env/                            # Phase 0.2: ViZDoom 环境封装（从 sandbox 迁入）
│   │   ├── vizdoom_env.py
│   │   └── scenarios/
│   ├── perception/                     # 既有目录扩充（不重复新建）
│   │   ├── ground_truth.py             # Phase 1: ViZDoom GameVariables
│   │   └── vlm_perceptor.py            # Phase 1: VLM 多 backend
│   ├── actions/                        # Phase 2: 动作层
│   │   ├── primitives.py
│   │   └── composites.py
│   ├── agent/                          # Phase 2-3: 主循环 + 反思
│   │   ├── loop.py                     # Phase 2
│   │   ├── reflection.py               # Phase 3
│   │   └── memory.py                   # Phase 3
│   ├── oracle/                         # Phase 4: 判定器 + 监控
│   │   ├── llm_oracle.py
│   │   └── monitors.py
│   └── experiments/                    # Phase 0.2 起持续积累实验脚本
│       ├── eval_perception.py
│       ├── eval_reflection.py
│       └── eval_oracle.py
```

**架构演进原则**（重要）：

- 概念上有 6 个顶层模块目录：`perception/` / `env/` / `actions/` / `agent/` / `oracle/` / `experiments/`
- **但不是 Phase 0 一次性建好**，而是**每个 Phase 只新建该 Phase 需要的目录**
- 已有的目录（perception/）由后续 Phase 在其中**加文件**，不重建
- `Code/` 和 `src/` 在所有 Phase 都**不动**

**关键原则**：
- **既有 Code/ 和 src/ 是 baseline，不重命名、不迁移、不"清理"**。Code/bdd/ 就是 bdd_runner，Code/bdd/test_generator/ 就是 gherkin_generator，无需另起目录
- **Phase 0.1 唯一新增的目录是 `perception/`**（✅ 已完成），定义了统一接口 + 包装既有 CV
- **Phase 0.2 新增 `env/` 和 `experiments/vizdoom/`**（✅ 已完成），把 ViZDoom sandbox 核心 wrapper 和 hello-world 脚本迁入主项目
- Phase 1+ 新模块按需新增在项目根目录，跟既有 Code/ src/ 并列，**不要预建空目录**

**为什么 Code/ 下既有模块"功能上就是 4 模块"也不重命名**：既有结构已经验证通过（pytest 全绿 + run_tests.py 跑通 DeepSeek + behave 全链路），改名/搬家无功能收益，纯增风险。论文里需要时直接讲"我们的 bdd_runner 模块（位于 Code/bdd/）"即可。

---

## 2. Phase 0：环境与重构（第 1 个月）

### 目标

把项目从"AssaultCube + 静态截图"切换到"ViZDoom + Python API"，并把原代码重构成符合长期架构的模块。

### 具体任务

#### 0.1 真实的 Phase 0.1（基于既有重构状态修正）

**重要前提**：项目已经过一轮重构（详见 `目前已进行的重构及项目状况.md` / `项目架构.md`）：
- DeepSeek 已统一到 `src/llm/client_helpers.py`
- `.env` 已集中到根目录
- pytest legacy/非 legacy 已分类
- `Code/bdd/` 主流程已跑通

因此 **Phase 0.1 不需要建新的 4 模块目录**（既有结构功能上就是那 4 个模块）。

**Phase 0.1 实际只做一件事**：在项目根目录新建 `perception/`，仅 2 个文件：

```python
# perception/base.py — 约 30 行
class GameStatePerceptor(ABC):
    @abstractmethod
    def perceive(self, screenshot, **kwargs) -> GameState: ...

@dataclass
class GameState:
    ammo: int | None
    health: int | None
    weapon: str | None
    # ... 其他字段
```

```python
# perception/cv_perceptor.py — 约 100 行
class CVPerceptor(GameStatePerceptor):
    """包装 Code/GameStateChecker，提供统一接口"""
    def __init__(self):
        from Code.GameStateChecker.LogicLayer import ...
        # 适配既有 CV
    def perceive(self, screenshot, **kwargs) -> GameState:
        # 调既有 CV，转成 GameState
        ...
```

**约束**：
- 不动 `Code/` 任何文件
- 不动 `src/` 任何文件
- 不动 `behave.ini` / `config.yaml` / `pytest.ini`
- 完成后 `python -m pytest` 与 `python Code\bdd\run_tests.py ...` 结果与之前完全一致

**完成定义**：
- ✅ `perception/base.py` 接口写完，有 docstring
- ✅ `perception/cv_perceptor.py` 包装好，能调通既有 `LogicLayer`
- ✅ 新增 `tests/test_cv_perceptor.py` 测包装层（不需要真跑 Flask，mock 即可）
- ✅ 既有所有测试和命令仍然全绿

#### 0.2 ViZDoom 环境搭建 ✅

```bash
pip install vizdoom
```

写一个 `env/vizdoom_env.py`，封装 ViZDoom 启动、状态读取、动作执行。

Phase 0.2 已将 sandbox 中的核心 wrapper 迁入 `env/`，并将 4 个 hello-world / trajectory 脚本迁入 `experiments/vizdoom/`。生成的截图、trajectory 和 `_vizdoom.ini` 仍是本地产物，不进入 Git。

最小可运行示例（你这一周就要跑通）：

```python
import vizdoom as vzd

game = vzd.DoomGame()
game.set_doom_scenario_path("basic.wad")
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
game.set_available_buttons([
    vzd.Button.ATTACK,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
])
game.set_available_game_variables([
    vzd.GameVariable.AMMO2,
    vzd.GameVariable.HEALTH,
])
game.init()

for _ in range(100):
    state = game.get_state()
    if state is None:
        break
    screen = state.screen_buffer        # numpy array
    ammo = state.game_variables[0]
    health = state.game_variables[1]
    print(f"ammo={ammo}, health={health}")
    game.make_action([1, 0, 0])         # fire

game.close()
```

#### 0.3 熟悉 ViZDoom 内置 scenarios

跑一遍这几个内置场景，理解 ACS 脚本和 .wad 结构：
- `basic.wad` —— 最简单，固定靶子
- `defend_the_center.wad` —— 多敌人
- `health_gathering.wad` —— 资源管理
- `deadly_corridor.wad` —— 多武器

理解的重点不是"怎么玩"，而是：
- 状态变量怎么暴露
- 动作空间长什么样
- scenario 是怎么定义的（为 Phase 4 mutation testing 铺路）

> 原 Phase 0.4「多 backend LLM 客户端」已合并到 Phase 1.3（VLMPerceptor 多 backend 实现）。
> 单一 text backend（DeepSeek，通过 `src/llm/client_helpers.py`）已经存在，Phase 0 不再单独处理 backend 抽象。

### 输出

- [x] `env/vizdoom_env.py` 能跑 basic.wad，能打印 ammo/health（核心 wrapper 已迁入；真实运行用 `experiments/vizdoom/hello_doom.py`）
- [x] 原 AssaultCube 测试仍然能跑（baseline 保留）
- [ ] （可选）一段简短 demo 录屏（ViZDoom 跑起来 + ammo 数字打印），作为 Phase 0 总结资料

### 成功标准

- 在 ViZDoom 里能从 Python 读到 ammo/health 数字（不需要 VLM，直接用 game_variables）
- 能调用 `make_action()` 让 agent 开火
- 原 AssaultCube + Behave + DeepSeek 跑通的状态没被搞坏

### 这一阶段不要做的事

- ❌ 不要碰 VLM （下阶段再做）
- ❌ 不要设计反思机制（更后面）
- ❌ 不要直接抛弃 AssaultCube 代码——它作为 baseline 有价值
- ❌ 不要追求 ViZDoom 的图形效果——研究用就行，性能优先

### 需要学的东西（≤ 1 周自学）

1. ViZDoom 官方文档 https://vizdoom.farama.org/
   - 重点读：Quickstart / Available scenarios / API
2. Gymnasium 接口（可选，先不用，但要知道）
3. `numpy` 处理 `screen_buffer`（你已经会 OpenCV，没难度）

---

## 3. Phase 1：感知层 - 多 backend VLM + Ground Truth 对比（第 2-3 个月）

### 目标

实现新的 `VLMPerceptor`，并**利用 ViZDoom 的 game_variables 作为 ground truth**，自动评估 VLM 的准确率。这是 AssaultCube 给不了你的关键能力——**有了 GT，准确率实验完全自动化**。

### 具体任务

#### 1.1 稳定 perception 接口 ✅（已在 Phase 0.1 完成）

接口已实现在 `perception/base.py`（`GameStatePerceptor` ABC + `GameState` dataclass）。
Phase 1 在此基础上**只新增** `ground_truth.py` / `vlm_perceptor.py`，不改 `base.py` 的契约。
若 Phase 1 实际使用中发现 `GameState` 缺字段（比如需要 `position` / `score`），
作为 backward-compatible 添加（新字段都设为 `Optional[...] = None`）。

#### 1.2 Ground Truth Perceptor

```python
# perception/ground_truth.py
class GroundTruthPerceptor(GameStatePerceptor):
    """直接从 ViZDoom 的 game_variables 读，作为评估基准"""
    def perceive(self, screenshot, vizdoom_state=None) -> GameState:
        gv = vizdoom_state.game_variables
        return GameState(
            ammo=int(gv[0]),
            health=int(gv[1]),
            ...
        )
```

**这是关键设计**：GT perceptor 让你能自动算出 VLM 的准确率，不用人工标注。

#### 1.3 VLM Perceptor（多 backend）

```python
# perception/vlm_perceptor.py
class VLMPerceptor(GameStatePerceptor):
    def __init__(self, backend="deepseek-vl2"):
        self.backend = backend
        self.prompt_template = load_prompt("vizdoom_state.txt")

    def perceive(self, screenshot, **kwargs) -> GameState:
        # 1. 截图编码成 base64
        # 2. 调用对应 backend
        # 3. 解析 JSON 输出
        # 4. 失败时返回 None（让上层 reflection 处理）
```

**Backend 选型**：

| Backend | 角色 | 备注 |
|---|---|---|
| DeepSeek-VL2 | 主力国产 | 跟 DeepSeek 是同一家，账号已有 |
| Qwen-VL-Max（阿里）| 备用国产 | 国内 API，求职加分项 |
| [Western baseline, TBD] | 论文 baseline 对比 | 见下方说明 |

**Western baseline 选型推迟到 Phase 1 启动时决定**（VLM 领域演进太快，提前定容易踩坑）。
候选池：Claude 4.x Sonnet / GPT-5 系列 / Gemini 2.5 Pro。
按当时 (1) 价格 (2) 海外信用卡可用性 (3) vision 能力 benchmark 选一个。
**仅用于论文 baseline 对比，不进 production loop**——不要在 Phase 1.4 之后还依赖它。

Prompt 设计（关键）：参考 TITAN §3.2 的"先抽象后具体"思路。
不要直接问 "ammo 是多少"，而是分两步：
- "ammo 状态属于 low / medium / high 哪一类？"
- "在 X 范围内的具体数字是多少？"

可以从 ViZDoom HUD 直接读，但 HUD 字体清晰，VLM 应该不难。

#### 1.4 对比实验脚本

```python
# experiments/eval_perception.py
"""
跑 N 局 basic.wad，每帧分别用：
- GroundTruthPerceptor (GT)
- CVPerceptor (旧 baseline)
- VLMPerceptor (多个 backend)
比较准确率、延迟、成本
输出 CSV
"""
```

输出表格大概长这样（论文里直接用）：

| Perceptor | Field | Accuracy | Avg Latency | Cost/1k frames |
|---|---|---|---|---|
| CV (template) | ammo | 95.6% | 12ms | 0 |
| VLM DeepSeek-VL2 | ammo | 92.1% | 1.8s | ¥0.5 |
| VLM Qwen-VL-Max | ammo | 94.3% | 2.1s | ¥0.8 |
| VLM [Western baseline, TBD] | ammo | TBD | TBD | TBD |

#### 1.5 把本科 CV 模块迁移作为 baseline ✅（已在 Phase 0.1 完成）

`CVPerceptor` 已实现在 `perception/cv_perceptor.py`，包装 `Code/GameStateChecker/LogicLayer.py`。
单元测试在 `tests/test_cv_perceptor.py`（15 tests，mock LogicLayer），
冒烟测试在 `scripts/smoke_test_cv_perceptor.py`（用真 LogicLayer + 真 PNG 验证）。
原 AssaultCube 截图库继续作为 baseline 评估对象 ——
**论文里这就是 Section "Comparison with prior art"**。

### 输出

- [ ] `perception/{base,ground_truth,cv_perceptor,vlm_perceptor}.py` 跑通
- [ ] 3 个 VLM backend 至少 2 个能跑通
- [ ] `eval_perception.py` 跑一次产出对比 CSV
- [ ] 一份对比表（论文 Section IV.A 草稿）

### 成功标准

- 在 ViZDoom basic.wad 上，VLM 对 ammo 字段的准确率 ≥ 90%
- 能产出 CV vs VLM vs GT 的三方对比数据
- 至少一个国产 backend 跑通（找工作加分）

### 这一阶段不要做的事

- ❌ 不要做 agent loop（下阶段）
- ❌ 不要做反思（更后面）
- ❌ 不要 fine-tune VLM（成本太高且对论文新意贡献小）
- ❌ 不要追求 100% 准确率——VLM 的"模糊但泛化"才是后面反思机制的素材

### 需要学的东西

1. Vision LLM 的 prompt 工程：
   - "structured output"（JSON mode / function calling）
   - 你本科已经做过 Gherkin 输出约束，迁移过来即可
2. 对比实验的设计：confusion matrix、precision/recall、bootstrap CI
3. 大致看一遍 TITAN §3.2 Perception Abstraction，理解 "先抽象后具体" 的思路

---

## 4. Phase 2：Action Executor + 最小 Agent Loop（第 4-5 个月）

### 目标

填补本科系统**最大的缺口** —— 让系统从 "Observer-only" 升级到 "Observer + Actor"。
并把 Gherkin 从"步骤描述"降级为"测试目标描述"，让 Agent 自己决定动作序列。

### 具体任务

#### 2.1 动作原语层

```python
# actions/primitives.py
class ActionPrimitives:
    def __init__(self, vizdoom_game):
        self.game = vizdoom_game

    def fire_once(self):
        self.game.make_action([1, 0, 0, ...], tics=4)

    def reload(self):
        self.game.make_action([0, 0, 1, ...], tics=20)

    def switch_weapon(self, slot):
        ...

    def observe(self):
        return self.game.get_state()

    def wait(self, tics):
        self.game.advance_action(tics)
```

#### 2.2 测试动作模板（高层）

```python
# actions/composites.py
class TestActions:
    """这是研究价值所在 —— 第三层 action template"""

    def fire_and_check_ammo(self, perceptor) -> dict:
        before = perceptor.perceive(self.observe().screen_buffer,
                                     vizdoom_state=self.observe())
        self.primitives.fire_once()
        after = perceptor.perceive(self.observe().screen_buffer,
                                    vizdoom_state=self.observe())
        return {
            "ammo_before": before.ammo,
            "ammo_after": after.ammo,
            "delta": (before.ammo or 0) - (after.ammo or 0),
        }

    def reload_and_check_ammo(self, perceptor) -> dict: ...
    def switch_and_check_crosshair(self, perceptor) -> dict: ...
```

这一层是 TITAN-style 的关键。LLM 操作的是这些模板，不是底层按键。

#### 2.3 Gherkin 降级为目标层

**Goal 数据模型**（agent loop 内部表示）：

```python
# agent/goal.py
from dataclasses import dataclass
from typing import Callable, List, Any

@dataclass
class Goal:
    """A test goal parsed from a Gherkin Scenario block.

    `success_criteria` 是从 Gherkin Success criteria 行编译出来的可调用对象，
    接受 agent loop 的累积 result dict，返回 bool。
    """
    description: str                   # 自然语言目标
    available_actions: List[str]       # 允许使用的 composite action 名字
    success_criteria: Callable[[dict], bool]
    metadata: dict                     # 原始 Gherkin / scenario 文件路径等

    def is_satisfied(self, result: dict) -> bool:
        return self.success_criteria(result)
```

**Gherkin 语法对比**：

旧（本科）：
```gherkin
When the player fires the weapon
Then the ammo should decrease by 1
```

新（硕士）：
```gherkin
Scenario: Verify ammo decreases after firing
  Goal: The agent should fire the weapon and confirm ammo decreases by exactly 1.
  Available actions: fire_and_check_ammo, reload_and_check_ammo, observe
  Success criteria: ammo_before - ammo_after == 1
```

LLM 解析 Gherkin → 构造 `Goal` 实例 → 自动选择 `fire_and_check_ammo`，
不需要预定义 step function。`Success criteria` 行编译为 lambda 注入到 `Goal.success_criteria`。

#### 2.4 最小 Agent Loop

```python
# agent/loop.py
def run_agent_loop(goal, env, perceptor, llm, max_steps=20):
    history = []
    for step in range(max_steps):
        # 1. observe
        state = perceptor.perceive(env.get_screen(), vizdoom_state=env.get_state())

        # 2. ask LLM: which action template?
        action_name, params = llm.decide(
            goal=goal,
            current_state=state,
            history=history,
            available_actions=TestActions.list(),
        )

        # 3. execute
        result = TestActions.run(action_name, params, env, perceptor)
        history.append((state, action_name, params, result))

        # 4. check success
        if goal.is_satisfied(result):
            return {"status": "success", "history": history}

    return {"status": "max_steps_exceeded", "history": history}
```

注意这里 **暂时没有反思**，那是 Phase 3 的事。
现在的失败 = "max_steps_exceeded" 或者断言 fail，先简单粗暴地报告。

#### 2.5 端到端最小 demo

跑通这个：
```
Goal: "Verify ammo decreases by exactly 1 after firing once"
↓
Agent observes → ammo=50
Agent calls fire_and_check_ammo
Agent observes → ammo=49
Agent: SUCCESS
```

并且**故意造一个失败 case**：把 perception 的 ammo 字段随机扰动 ±2，看 agent loop 怎么 fail。这就是 Phase 3 反思机制的素材。

### 输出

- [ ] `actions/{primitives,composites}.py`
- [ ] `agent/loop.py` 跑通最简单的 fire-and-check
- [ ] 第一段录屏 demo：完整 agent loop（口头讲 30 秒能说清楚）
- [ ] 一份 "前后对比" 文档：本科系统 vs 现在系统

### 成功标准

- 给一个 Goal 字符串，agent 能自己决定调哪些 action template
- 不需要预先写 Behave step function（这是研究价值！）
- 能跑通 3 个测试目标：fire/reload/switch

### 这一阶段不要做的事

- ❌ 不要急着做反思（看到失败就先记录，下个 Phase 处理）
- ❌ 不要扩动作集——5 个 composite 就够
- ❌ 不要追求 LLM 多么聪明——能正确选 action template 就行
- ❌ **不要扔掉 Behave**——把 Behave 保留作为"轻量级 runner"跑这些 goal-based scenario

### 需要学的东西

1. **LangGraph 或 AutoGen 之一**（找工作必备）
   - 推荐 LangGraph（更简单、状态图明确）
   - 你的 agent loop 可以用 LangGraph 重写，简历上加一条
2. ReAct paper（Yao et al. 2023）——TITAN 用的 baseline，10 分钟看完
3. Function calling / tool use（OpenAI / DeepSeek 都支持）

---

## 5. Phase 3：反思机制（第 6-7 个月）

### 目标

实现讨论 1.0.md 提到的**三类 failure** 的反思与恢复。
这是论文真正的核心贡献之一。

### 具体任务

#### 3.1 失败类型分类（关键设计）

```python
# agent/reflection.py
class FailureType(Enum):
    PERCEPTION = "perception"   # VLM 看错了
    EXECUTION = "execution"     # 动作没执行成功
    LOGIC = "logic"             # 游戏真的有 bug

@dataclass
class FailureContext:
    failure_type: FailureType
    expected: Any
    actual: Any
    history: list
    screenshot: np.ndarray
```

#### 3.2 反思 prompt

参考 TITAN §3.4 的反思 prompt 设计：

```
[Current state]: {abstract_state}
[Recent actions]: {history}
[Failure]: expected ammo to decrease by 1, but observed delta=0

Questions:
1. What are possible causes?
   (a) Perception error — VLM misread the digit
   (b) Execution error — fire didn't go through
   (c) Logic bug — game didn't decrement ammo
2. What concrete recovery action should I try?
3. If you believe this is a logic bug, what evidence supports it?

Respond ONLY with JSON: {"hypothesis": ..., "recovery_action": ..., "confidence": 0-1}
```

#### 3.3 恢复动作（每类 failure 一套策略）

| 失败类型 | 恢复策略 |
|---|---|
| Perception | 重新截图 → 换 backend → 多次投票 |
| Execution | 重试动作 → 等待动画 → 检查窗口焦点 |
| Logic | 标记为可疑 bug → 生成报告 → 继续测试其他场景 |

#### 3.4 注入失败的实验环境

故意制造每一类失败：
- Perception：在截图上加高斯噪声 / 降分辨率 / 裁剪偏移
- Execution：以 30% 概率"假装"没执行（不调用 `make_action`）
- Logic：（这一类留到 Phase 4 用 mutation testing 做真实 bug）

#### 3.5 评估实验

```python
# experiments/eval_reflection.py
"""
在 N=100 次注入失败的运行里，比较：
- baseline: 没反思，第一次失败就停
- proposed: 带反思，最多重试 3 次
输出：
- 恢复成功率
- 平均重试次数
- 误报率（把真 bug 当 perception error 反思掉了）
"""
```

### 输出

- [ ] `agent/reflection.py` 跑通
- [ ] 一份失败恢复率对比表（论文 Section IV.B 草稿）
- [ ] 三类失败的 case study（论文里展示 1-2 个具体反思流程）

### 成功标准

- Perception failure 恢复率 ≥ 70%
- Execution failure 恢复率 ≥ 80%
- 反思机制不会把真 bug 误判成 perception error（精确率 ≥ 80%）

### 这一阶段不要做的事

- ❌ 不要追求 TITAN 那种"跨 episode 记忆"——M1 没那么多时间
- ❌ 不要追求 LLM agent 自己学会工具——预定义 3 种恢复策略就够

### 需要学的东西

1. ReAct / Reflexion paper（Shinn et al. 2023）
2. 简单的 A/B 实验设计

---

## 6. Phase 4：Oracle + Mutation Testing（第 8-9 个月）

### 目标

实现 ViZDoom **独有的杀手锏**：通过修改 ACS 脚本注入 seeded bug，让 agent 自动检测出来。
这是论文最有说服力的实验。

### 具体任务

#### 4.1 Crash / Hang / Time 监控

参考 TITAN §3.5：

```python
# oracle/monitors.py
class CrashMonitor: ...     # 检测 ViZDoom 进程崩溃
class HangMonitor: ...      # 检测 agent N 步无进展
class TimeMonitor: ...      # 检测动作执行时间异常
```

#### 4.2 LLM Oracle

```python
# oracle/llm_oracle.py
def llm_judge(goal, history, final_state) -> dict:
    """
    给定测试目标、动作历史、最终状态，让 LLM 判断：
    - 测试是否真的通过了
    - 是否存在隐藏 bug
    - 给出诊断 markdown 报告
    """
```

这比硬编码断言强的地方是：可以发现"测试通过但实际不对"的场景。

#### 4.3 Mutation Testing（核心实验）

写至少 8 个、目标 10 个 seeded bug 的 ACS 脚本：

```
mutation_1_no_ammo_decrease.wad      # 开火不扣弹
mutation_2_reload_broken.wad         # reload 不补弹
mutation_3_weapon_switch_no_hud.wad  # 切枪 HUD 不更新
mutation_4_health_overflow.wad       # 拿包后血量 > 100
mutation_5_kill_count_wrong.wad      # 杀敌计数错位
mutation_6_enemy_hp_immortal.wad     # 敌人不死
mutation_7_ammo_negative.wad         # 弹药能减到负数
mutation_8_double_pickup.wad         # 物品能被捡两次
mutation_9_reload_during_fire.wad    # 开火中能 reload（卡死）
mutation_10_death_no_reset.wad       # 死亡后状态不重置
```

每一个 bug 都是 ACS 脚本里改一行就能造的。

#### 4.4 Bug Detection 实验

```python
# experiments/eval_oracle.py
"""
对每个 mutated scenario：
- TITAN-style agent (proposed) 跑 5 次
- Baseline (no reflection, no LLM oracle) 跑 5 次
- 看谁能检测到 bug
输出 confusion matrix
"""
```

预期结果（论文里写）：

| Method | Bugs Found | False Positives | Avg Time |
|---|---|---|---|
| Behave + Hardcoded assertion | 4 / 10 | 0 | 30s |
| + VLM Perception | 6 / 10 | 1 | 45s |
| + Reflection | 7 / 10 | 1 | 60s |
| **+ LLM Oracle (full)** | **9 / 10** | 2 | 90s |

#### 4.5 诊断报告生成

让 LLM 在发现 bug 时自动生成 markdown 报告（参考 TITAN 的 diagnosis report）：

```markdown
## Bug Report: Ammo Counter Not Decreasing After Firing

### Symptom
After executing `fire_once()`, the ammo counter did not decrement.

### Evidence
- Step 5: ammo=20, weapon=pistol
- Step 6: fire_once() executed, action confirmed
- Step 7: ammo=20 (expected 19)

### Hypothesis
Likely a logic bug in the ammo update routine triggered by firing.

### Reproducer
Scenario: basic_mutated_1.wad, weapon=pistol, initial_ammo=20
```

### 输出

- [ ] 至少 8 个，目标 10 个 mutated scenarios（.wad / .acs 文件）
- [ ] Bug detection 实验数据
- [ ] LLM Oracle 模块
- [ ] 自动生成的诊断报告样本 × 5

### 成功标准

- 在 10 个 seeded bugs 上，full system detection rate ≥ 80%
- 至少能展示 3 份高质量的自动 bug report
- False positive rate ≤ 30%（参考 TITAN 的 30%）

### 这一阶段不要做的事

- ❌ 不要追求"发现未知 bug"——seeded bug 已经够论文写了
- ❌ 不要做太复杂的 mutation（保持每个 bug 单点改动可解释）

### 需要学的东西

1. **ACS 脚本基础**（Doom 的脚本语言）—— 一天就够入门
   - 教程：https://zdoom.org/wiki/ACS
2. SLADE 工具（编辑 .wad / .acs）
3. Mutation testing 基础概念（mu_python / PIT 之类的 paper 看一两篇）

---

## 7. Phase 5：论文与求职（第 10 个月起）

### 论文写作时间表

- Month 10：初稿（先投 workshop / 小会议练手）
  - 目标：ICST Workshop / ASE Workshop / AsianTest
- Month 11-12：根据 review 改稿 + 投正会议
  - 候选：ICST 2027 main / ASE 2027 / ISSTA 2027
- 同步：把硕士论文（修士論文）大纲写出来

### 论文结构建议

```
1. Introduction
2. Background
   2.1 BDD for game testing (RiverGame)
   2.2 LLM agents (TITAN, Voyager, etc.)
   2.3 Mutation testing
3. Motivation: 本科系统的局限
4. Approach: 5 个模块
   4.1 Perception (VLM)
   4.2 Action Executor
   4.3 Goal-level Gherkin
   4.4 Reflection
   4.5 LLM Oracle + Mutation
5. Implementation: ViZDoom
6. Evaluation
   6.1 Perception accuracy (CV vs VLM vs GT)
   6.2 Reflection recovery rate
   6.3 Mutation bug detection
7. Discussion & Limitations
8. Conclusion
```

### 求职准备

> 暂留空。实习时机、目标公司、简历策略等内容待与 Claude 详细讨论后再补。
> 现在不要在这里做决定，避免 scope creep。

---

## 8. 关键陷阱 & 风险控制

### 陷阱 1：没 DDL → 永远在调研

**对策**：每个 Phase 月底写一份 ≤ 1 页的"做完了什么"自我复盘。
即使无人催，自我复盘本身就是 deadline。

### 陷阱 2：扩 scope

**对策**：每周自问一次"这件事是不是 Phase X 该做的"，不是就先记到 backlog.md，不立即做。

### 陷阱 3：被 ViZDoom 局限性卡住

**已知局限**：
- ViZDoom 是 Doom-like，不是现代 FPS。审稿人可能问"在现代游戏上能不能用？"
- **答辩话术**：本研究专注于 testing methodology 的设计与验证，ViZDoom 作为受控环境提供了：(1) 可重现的 ground truth，(2) 通过 ACS 注入 seeded bug 的能力，(3) 与 RL/Agent 文献对齐的标准 benchmark。方法本身可移植到任何提供状态接口的游戏。

### 陷阱 4：法律 / 商业禁忌

- ❌ 不要 reverse engineer 商业游戏（Apex / CSGO / Valorant 等）
- ❌ 不要写 Memory Hook / Input Hook 相关代码
- ❌ 不要把研究包装成"外挂检测对抗"——这在中国就业语境敏感

### 陷阱 5：审稿 / 答辩 / 面试可能被问的问题（防御性答案预演）

| 可能被问 | 你的答 |
|---|---|
| 为什么换平台？ | ViZDoom 提供 Python API 和 ground truth，让 testing methodology 研究专注于方法本身，AssaultCube 静态系统作为本科 baseline 在论文中保留对比 |
| ViZDoom 太老了吧？ | 它是 FPS AI 研究的学术标配（NeurIPS/ICML 等多篇 paper），且我们关心 testing methodology 的可移植性，不是图形真实感 |
| 跟 TITAN 区别？ | TITAN 关注 MMORPG 任务完成；本研究关注 FPS 局部机制的 bug 检测，引入 mutation testing 作为 evaluation 标准 |
| 创新点是什么？ | (1) Goal-level Gherkin（Gherkin 从步骤到目标的范式转移）; (2) 三类 failure 分类与反思恢复; (3) Mutation testing + LLM oracle 的结合 |

---

## 9. 速查表（贴墙上）

```
Phase 0 (M1):  ViZDoom 跑起来 + 旧代码重构
Phase 1 (M2-3): Perception (VLM + GT 对比)
Phase 2 (M4-5): Action Executor + Goal-level Gherkin + 最小 Agent Loop
Phase 3 (M6-7): Reflection (三类 failure)
Phase 4 (M8-9): Oracle + Mutation Testing
Phase 5 (M10+): 论文 + 求职

绝对要继承的：
  - LLM 生成测试规约的能力（本科已有）
  - 模块解耦设计
  - 静态 CV 作为 baseline

必须升级的：
  - Gherkin: 步骤层 → 目标层
  - 系统角色: Observer-only → Observer + Actor
  - 失败处理: 立即停 → 反思 + 重试
  - Oracle: 硬编码 → LLM + Mutation

技术栈关键词（简历用）：
  ViZDoom, LangGraph, ReAct, DeepSeek-VL2, Qwen-VL,
  Mutation Testing, BDD, LLM Agent, Reflection, Oracle
```

