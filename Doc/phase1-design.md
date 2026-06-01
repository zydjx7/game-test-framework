# Phase 1 Spike Design Doc

> **Status**: 2026-05-20 锁定。`research-plan.md` §3 的 Phase 1 具体化为此文档。
> **Scope**: 仅覆盖 Phase 1 spike（单 backend × basic.wad × ~50 次 VLM 调用）。
> Phase 4 的 4-backend 完整对比另开 design doc。

## 1. Goal（这次要回答的问题）

> **Qwen3-VL-Flash 在 ViZDoom basic.wad 的 ammo HUD 识别上，能不能跑通端到端链路，accuracy 大致在什么量级？**

不是"达到 90%"，是**看到第一个数字**。数字出来后再决定下一步（调 prompt / 换 backend / 进 Phase 2）。

## 2. Scope（明确不做什么）

| 在做 | 不做（留到后续 Phase） |
|---|---|
| 1 backend（Qwen3-VL-Flash） | 4 backend 对比 → Phase 4 |
| 1 scenario（basic.wad） | defend_the_center / health_gathering / deadly_corridor → Phase 2-3 |
| 1 字段（ammo） | health / weapon / enemy_visible → Phase 2-3 |
| 5 episodes × 10 keyframes ≈ 50 次调用 | 50 episodes × sampling sensitivity → Phase 4 |
| Exact match metric | F1 / tolerance metric → Phase 4 |
| 全程 ATTACK 录数据 | 随机 / agent-driven 采样 → Phase 2 |

## 3. GameState 字段定义（Phase 1 spike 范围）

`perception/base.py` 已有 `GameState` dataclass。Phase 1 spike **只填 `ammo` 字段**，其他字段保持 `None`。

```python
GameState(
    ammo=int,             # 主要填这个
    health=None,          # basic.wad 不暴露
    weapon=None,          # Phase 2 再加
    crosshair_red=None,
    enemies_visible=None,
)
```

**为什么只测 ammo**：`env/vizdoom_env.py` 的 `VARIABLE_NAMES["basic"] = ["ammo"]`，basic.cfg 只暴露 ammo 一个 game variable。这是 wrapper 既有约束，不改。

## 4. VLM Prompt v1（TITAN 风格："先抽象后具体"）

文件：`prompts/vizdoom_ammo_v1.txt`

```
You are analyzing a single frame from a Doom-like first-person
shooter game. Your task: identify the player's current ammo
count visible in the HUD (heads-up display).

The HUD is at the bottom of the screen. The ammo number is
shown in large yellow digits near the right side.

Step 1: Classify the ammo level into one of:
  - "high":   ammo >= 20
  - "medium": ammo between 10 and 19 (inclusive)
  - "low":    ammo < 10

Step 2: Identify the exact ammo number visible.

Respond ONLY with valid JSON in this exact format, no other
text, no markdown fences:

{"ammo_level": "<low|medium|high>", "ammo": <integer>}

If you cannot read the ammo number, return:
{"ammo_level": "unknown", "ammo": null}
```

**为什么 level 边界是 20 / 10**：basic.wad 起始 ammo = 26，自然分到 high（26 ~ 20）/ medium（19 ~ 10）/ low（9 ~ 0）三段，每段约 7-9 个值，分布大致均衡。

**TITAN 设计哲学**（论文卖点）：先抽象判断（level）训练 VLM 的"粗看"能力，再细看具体值。两个数字单独评估，论文里会出现"abstract accuracy vs concrete accuracy"对比。

## 5. Metric（怎么算"准不准"）

Phase 1 spike 只算 ammo 一个字段，**两个 accuracy 都要算**：

| Metric | 定义 | 用途 |
|---|---|---|
| `ammo_concrete_accuracy` | `pred.ammo == gt.ammo` 的比例 | 主指标，Section IV.A 用 |
| `ammo_abstract_accuracy` | `pred.ammo_level == gt.ammo_level` 的比例 | 验证 TITAN 抽象层假设 |

GT 端的 `ammo_level` 由相同 level 边界函数计算（保证算法一致）。

**Exact match，不容忍**（Q2 选 a）：`pred.ammo == gt.ammo` 必须严格相等。差 1 算错。

**失败处理**：
- VLM 返回 `ammo=null` → 算作错（不丢样本，记为 perception failure）
- JSON parse 失败 → 算作错（记为 parse failure）
- 这两类失败要单独计数，是 Phase 3 reflection 的素材

## 6. Trajectory 采集策略（Q4 选 a）

`scripts/record_basic_trajectories.py`：

```python
# 5 局 basic.wad，全程 ATTACK
for episode in range(5):
    state = env.reset()
    while not state.done:
        state = env.step([0, 0, 1])   # MOVE_LEFT=0, MOVE_RIGHT=0, ATTACK=1
        recorder.append(tick, state.screen, state.game_variables)
```

**为什么全程 ATTACK**：
1. spike 目的是测 VLM 识别 ammo 的能力，不是测 agent 决策
2. 全 ATTACK → ammo 单调从 26 降到 0 → 26 个 level / 数字都被覆盖
3. 数据可重现，不引入随机噪声

**Keyframe sampling**：每个 episode 等距取 10 帧（如 episode 长 26 tick，取 tick 0, 2, 5, 7, 10, 13, 15, 18, 20, 23）。5 × 10 = 50 次 VLM 调用。

## 7. Success Criteria（Q3 选 c：不设阈值）

Spike 完成 = 满足以下**全部**：
- [ ] 50 次 VLM 调用全部完成（含失败也算完成，不能崩溃）
- [ ] 输出 CSV 含 `tick, gt_ammo, vlm_ammo, vlm_level, concrete_correct, abstract_correct, latency_ms, cost_cny`
- [ ] 两个 accuracy 数字算出来（无论高低）
- [ ] 总成本 ≤ ¥5
- [ ] 至少 1 类 failure case 能口头解释原因（数字识别 / level 边界判断错 / JSON malformed）

**看到数字后再决定**：
- `concrete_accuracy ≥ 70%` → 路通，进 Phase 2
- `concrete_accuracy < 50%` → prompt v2 / 换 backend，再 spike 一次
- 中间区间 → case-by-case 讨论

## 8. Stage B 实现清单（Phase 1 spike 代码）

| 文件 | 行数估计 | 依赖 | 顺序 |
|---|---|---|---|
| `env/trajectory_recorder.py` | ~80 | VizDoomEnv | 1 |
| `scripts/record_basic_trajectories.py` | ~30 | recorder | 2 |
| `perception/ground_truth.py` | ~40 | GameState | 3 |
| `prompts/vizdoom_ammo_v1.txt` | ~15 | — | 4 |
| `perception/backends/qwen3_vl_flash.py` | ~60 | DashScope SDK + .env API key | 5 |
| `perception/vlm_perceptor.py` | ~80 | base.py + backend | 6 |
| `experiments/eval_perception_spike.py` | ~50 | 上面全部 | 7 |
| **总计** | **~355 行** | | |

**关键依赖**：DashScope API key（阿里云百炼 Qwen3-VL-Flash 的入口）需要先注册账号 + 拿 key + 加到 `.env`。这一步用户必须自己做（账号涉及实名）。

## 9. Out of Scope（这次不做，确认）

- ❌ Reflection / retry：失败就记，不重试。Phase 3 才做。
- ❌ 多 backend：只跑 Qwen3-VL-Flash。Phase 4 才扩。
- ❌ Sampling sensitivity：固定"等距 10 帧"。Phase 4 才做 3 种采样对比。
- ❌ Agent loop：spike 用离线 trajectory，不跑 agent。Phase 2 才做。
- ❌ LLM Oracle：accuracy 由代码算，不用 LLM judge。Phase 4 才做。
- ❌ Mutation testing：basic.wad 用原版，不注入 bug。Phase 4 才做。

## 10. 下一步（Stage A 完成后做什么）

Stage B 实现顺序按 §8 表格的"顺序"列：trajectory recorder → record script → ground truth → prompt → backend → VLM perceptor → eval。

**第一个 PR 单位**：`env/trajectory_recorder.py` + `scripts/record_basic_trajectories.py` + 单元测试。这两个文件不依赖 VLM API key，可以先写完跑通，再去申请 DashScope key。
