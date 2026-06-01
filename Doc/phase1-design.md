# Phase 1 Spike Design Doc

> **Status**: 2026-05-20 锁定；2026-06-01 经真环境冒烟修正 scenario。
> **Scope**: 仅覆盖 Phase 1 spike（单 backend × defend_the_center × ~50-100 次 VLM 调用）。
> Phase 4 的 4-backend 完整对比另开 design doc。

## 0. 真环境冒烟发现（2026-06-01，Stage B Step 1 后）

录制脚本在真 ViZDoom 上跑过后，推翻了两个 design 假设，scenario 因此从
basic 改为 defend_the_center：

1. **basic.wad 起始 ammo = 50，不是 26**（design 写错）。
2. **手枪稳态射击周期 = 14 tic/发**（Doom 源码状态机：PISTOL2→3→4→A_ReFire→1→2 = 6+4+4）。
   首发延迟约 4 tic（READY→PISTOL1→开火），稳态间隔 14 tic。
3. **因此 basic.wad 的 ammo 永远停在 [46,50]**：episode 要么打死靶子秒结束（~6 帧），
   要么超时 ~300 tic（只够 ~20 发，ammo 降到 ~30）。打不到 medium/low，TITAN 抽象层退化。
4. **改用 defend_the_center**：纯 ATTACK 跑 272 步，ammo ∈ [7,26]（high/med/low 全覆盖），
   health ∈ [24,100]，20 次 ammo 变化。ATTACK 在两 scenario 都是 button index 2，policy `[0,0,1]` 不变。

**对 keyframe 采样的影响**：ammo 每 ~14 tic 才变一次，所以采样必须 **event-driven
（ammo 值变化时取帧）**，不能等距乱采，否则会采到大量 ammo 相同的冗余帧。

## 1. Goal（这次要回答的问题）

> **Qwen3-VL-Flash 在 ViZDoom defend_the_center 的 ammo HUD 识别上，能不能跑通端到端链路，accuracy 大致在什么量级？**

不是"达到 90%"，是**看到第一个数字**。数字出来后再决定下一步（调 prompt / 换 backend / 进 Phase 2）。

## 2. Scope（明确不做什么）

| 在做 | 不做（留到后续 Phase） |
|---|---|
| 1 backend（Qwen3-VL-Flash） | 4 backend 对比 → Phase 4 |
| 1 scenario（defend_the_center） | health_gathering / deadly_corridor / 多场景 → Phase 2-3 |
| **评估 1 字段（ammo）** | health 录进 GT 但 spike 不评估；weapon / enemy_visible → Phase 2-3 |
| 5 episodes × ~10-20 keyframes（ammo-change 事件驱动） | 50 episodes × sampling sensitivity → Phase 4 |
| Exact match metric | F1 / tolerance metric → Phase 4 |
| 全程 ATTACK 录数据 | 随机 / agent-driven 采样 → Phase 2 |

## 3. GameState 字段定义（Phase 1 spike 范围）

`perception/base.py` 已有 `GameState` dataclass。defend_the_center 暴露 ammo + health
两个变量（`VARIABLE_NAMES["defend_the_center"] = ["ammo", "health"]`），但 **spike 只评估 ammo**。

```python
GameState(
    ammo=int,             # spike 评估这个（VLM vs GT 对比）
    health=int,           # GT 会填（defend_the_center 暴露），但 spike 不评估，留给 Phase 2-3
    weapon=None,          # Phase 2 再加
    crosshair_red=None,
    enemies_visible=None,
)
```

**为什么只评估 ammo**：spike 目的是打通链路 + 看第一个数字，单字段足够。health 顺带录进 GT，
Phase 2-3 扩多字段评估时直接能用，不用重录数据。

## 4. VLM Prompt v1（TITAN 风格："先抽象后具体"）

文件：`prompts/vizdoom_ammo_v1.txt`

```
You are analyzing a single frame from a Doom-like first-person
shooter game. Your task: identify the player's current ammo
count visible in the HUD (heads-up display).

The HUD is at the bottom of the screen. The ammo number is
shown in large yellow digits near the right side.

Step 1: Classify the ammo level into one of:
  - "high":   ammo >= 18
  - "medium": ammo between 9 and 17 (inclusive)
  - "low":    ammo < 9

Step 2: Identify the exact ammo number visible.

Respond ONLY with valid JSON in this exact format, no other
text, no markdown fences:

{"ammo_level": "<low|medium|high>", "ammo": <integer>}

If you cannot read the ammo number, return:
{"ammo_level": "unknown", "ammo": null}
```

**为什么 level 边界是 18 / 9**：defend_the_center 起始 ammo = 26，纯 ATTACK 实测降到 7。
按 [0,26] 三等分：high（26~18）/ medium（17~9）/ low（8~0），每段约 8-9 个值，分布均衡。
GT 端用相同边界函数算 `ammo_level`，保证算法一致。

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

## 6. Trajectory 采集策略（Q4 选 a，2026-06-01 改 scenario）

`scripts/record_spike_trajectories.py`（默认 scenario = defend_the_center）：

```python
# 5 局 defend_the_center，全程 ATTACK
# buttons: ['TURN_LEFT', 'TURN_RIGHT', 'ATTACK']，ATTACK = index 2
for episode in range(5):
    trajectory = record_episode(env, lambda s: [0, 0, 1], scenario="defend_the_center")
    save_trajectory(trajectory, ...)
```

**为什么全程 ATTACK**：
1. spike 目的是测 VLM 识别 ammo 的能力，不是测 agent 决策
2. 纯 ATTACK 实测：272 步，ammo 26→7（high/med/low 全覆盖），20 次 ammo 变化
3. 数据可重现，不引入随机噪声

**Keyframe sampling（event-driven，不是等距）**：因为手枪 14 tic 才打 1 发，ammo 每 ~14 tic
才变一次。采样策略 = **在每次 ammo 值发生变化后取 1 帧**（取变化后的稳定帧）。
一个 episode ~20 个 ammo-change → ~20 keyframes。5 episodes ≈ 100 次 VLM 调用（spike 上限）。
若想压到 ~50 次，可每隔一个 ammo-change 取一帧。

> ⚠️ 不要等距采样：等距会采到大量 ammo 相同的冗余帧（14 tic 内 ammo 不变），
> 既浪费 API 调用又让 accuracy 被重复值带偏。event-driven 保证每个 ammo 值大致只测一次。

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

| 文件 | 行数估计 | 依赖 | 顺序 | 状态 |
|---|---|---|---|---|
| `env/trajectory_recorder.py` | ~80 | VizDoomEnv | 1 | ✅ 已完成 + 测试 |
| `scripts/record_spike_trajectories.py` | ~50 | recorder | 2 | ✅ 已完成（defend_the_center） |
| `perception/ground_truth.py` | ~50 | GameState | 3 | ⬅ 下一个 |
| `prompts/vizdoom_ammo_v1.txt` | ~20 | — | 4 | |
| `perception/backends/qwen3_vl_flash.py` | ~70 | openai SDK（OpenAI 兼容）+ DASHSCOPE_API_KEY | 5 | |
| `perception/vlm_perceptor.py` | ~80 | base.py + backend | 6 | |
| `experiments/sampling/ammo_change_keyframes.py` | ~40 | trajectory | 7 | |
| `experiments/eval_perception_spike.py` | ~60 | 上面全部 | 8 | |
| **总计** | **~450 行** | | | |

**关键技术决定**：
- **不装 dashscope SDK**。阿里云百炼 Qwen3-VL 支持 OpenAI 兼容模式，复用项目已有的 `openai` SDK，
  跟 DeepSeek 走同一套（CLAUDE.md 设计哲学）。
- base_url（北京区）：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 模型名：`qwen3-vl-flash`
- 图片：base64 data URI（`data:image/png;base64,...`），OpenAI vision 标准格式
- API key 从 `.env` 的 `DASHSCOPE_API_KEY` 读，复用 `src/llm/client_helpers.load_project_dotenv()`

**关键依赖**：DASHSCOPE_API_KEY（阿里云百炼）需用户自己注册 + 拿 key + 加到 `.env`（实名认证，AI 做不了）。

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
