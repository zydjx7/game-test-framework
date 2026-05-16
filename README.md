# 🎮 Game Test Automation Framework

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org)
[![BDD](https://img.shields.io/badge/BDD-Gherkin-yellow.svg)](https://cucumber.io/docs/gherkin/)
[![ViZDoom](https://img.shields.io/badge/ViZDoom-1.2+-orange.svg)](https://vizdoom.farama.org/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-purple.svg)](https://api.deepseek.com)

> 🎓 **修士研究项目**（立命館大学情報理工学研究科 M1, 2026 年 4 月入学）
> 研究方向：基于 LLM Agent + VLM 的轻量级 FPS 游戏自动化测试框架
> 当前阶段：**Phase 0.1 完成** — 从 AssaultCube 静态截图测试迁移到 ViZDoom 动态 agent loop

## 📖 项目概述

本仓库实现一个**基于 LLM Agent 的 FPS 游戏自动化测试框架**，研究核心创新点：

1. **Goal-level Gherkin** — Gherkin 从"步骤描述"升级为"测试目标描述"
2. **三类 failure 反思恢复**（Perception / Execution / Logic）
3. **Mutation Testing + LLM Oracle** — 通过 ACS 脚本注入 seeded bug 自动评估

项目分两层：
- **AssaultCube baseline**（已完成）：本科论文系统，作为论文 Section IV 的对比基线保留
- **ViZDoom main line**（Phase 1+ 渐进新增）：Python API + ground truth + ACS 脚本可注入 bug

两层通过 [perception/base.py](perception/base.py) 的 `GameStatePerceptor` 统一接口连接。

## 🗺️ 研究路线图

| Phase | 月份 | 状态 | 核心目标 |
|---|---|---|---|
| **0** | M1 | 🟡 进行中 | ViZDoom 环境就绪 + perception 接口抽出 |
| 0.1 | M1 | ✅ 完成 | `GameStatePerceptor` 接口 + `CVPerceptor` 包装层 |
| 1 | M2-3 | ⚪ 未开始 | VLM Perception + Ground Truth 对比（多 backend） |
| 2 | M4-5 | ⚪ 未开始 | Action Executor + Goal-level Gherkin + 最小 agent loop |
| 3 | M6-7 | ⚪ 未开始 | Reflection（三类 failure 分类与恢复） |
| 4 | M8-9 | ⚪ 未开始 | LLM Oracle + Mutation Testing |
| 5 | M10+ | ⚪ 未开始 | 论文写作 + 求职 |

## 🏗️ 框架架构

```
game-testing-main/
│
├── 📁 Code/                            # AssaultCube baseline（论文 baseline，不动）
│   ├── 🎯 GameStateChecker/            # 经典 CV 感知：模板匹配 + OCR
│   │   ├── LogicLayer.py               # 弹药/准星检测主入口
│   │   ├── VisionUtils.py              # CV 工具（SIFT 特征匹配等）
│   │   └── AmmoTemplateRecognizer.py   # 数字模板识别器
│   ├── 🧪 bdd/                         # BDD 主流程（DeepSeek + behave）
│   │   ├── run_tests.py                # 入口
│   │   ├── test_generator/             # LLM 生成 Gherkin
│   │   └── features/steps/             # behave step functions
│   └── 🎵 SoundTestingSupport/         # 音频测试 PoC
│
├── 📁 src/                             # 共享层
│   ├── llm/client_helpers.py           # DeepSeek 统一入口（OpenAI-compatible SDK）
│   ├── gherkin/                        # Gherkin parser
│   └── rivergame/                      # legacy（标 legacy，不动）
│
├── 📁 perception/                      # ⭐ Phase 0.1 新增：统一感知接口
│   ├── base.py                         # GameStatePerceptor ABC + GameState dataclass
│   └── cv_perceptor.py                 # 包装 Code/GameStateChecker
│
├── 🧪 tests/                           # pytest（含新 test_cv_perceptor.py）
│
├── 📁 env/                             # ⏳ Phase 1 新增：ViZDoom 环境封装
├── 📁 actions/                         # ⏳ Phase 2
├── 📁 agent/                           # ⏳ Phase 2-3
├── 📁 oracle/                          # ⏳ Phase 4
└── 📁 experiments/                     # ⏳ Phase 1+ 渐增（对比实验脚本）
```

## 🚀 快速开始

### 环境要求

- Python 3.10+（ViZDoom 兼容性要求）
- OpenCV 4.5+
- Windows 用户额外需要：Microsoft Visual C++ 2015-2022 Redistributable（ViZDoom 依赖）

### 安装

```bash
git clone https://github.com/zydjx7/game-test-framework.git
cd game-test-framework
pip install -r requirements.txt
```

### LLM 配置

项目用 **OpenAI-compatible SDK** 接入 **DeepSeek**（不是 OpenAI）。在根目录 `.env`：

```bash
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
USE_LLM_ANALYSIS=true
```

### 运行

```bash
# 跑 AssaultCube baseline 测试（既有功能）
cd Code/bdd
python run_tests.py --mode predefined --feature generated_test.feature --target assaultcube

# 跑 pytest（含 Phase 0.1 新单元测试）
python -m pytest
```

## 🧩 Perception 接口（Phase 0.1）

新模块 [perception/](perception/) 把感知后端抽象为统一接口，让 Phase 1+ 的 VLM 和 ground truth 后端能无侵入接入：

```python
from perception import CVPerceptor

# 包装既有 AssaultCube CV pipeline
perceptor = CVPerceptor(target_name="assaultcube")
state = perceptor.perceive(screenshot, expected_ammo=20, check_crosshair=True)

print(state.ammo)           # 20（来自模板匹配 / OCR 兜底）
print(state.crosshair_red)  # True/False
```

Phase 1+ 将新增两个 backend，对外接口不变：

- `GroundTruthPerceptor` — 直接读 ViZDoom `state.game_variables`，作为评估基准
- `VLMPerceptor` — DeepSeek-VL2 / Qwen-VL / GPT-4o-mini 多 backend 视觉模型

## 🎯 核心功能（AssaultCube baseline）

### 1. 游戏状态检测

- ✅ 弹药数量识别（单/双位数模板匹配 + OCR 兜底）
- ✅ 准星状态分析（SIFT 特征匹配）
- ✅ 调试图像自动保存到 `debug/` 目录便于排查

### 2. BDD 自动化测试

```gherkin
Feature: 武器系统测试
  Scenario: 弹药消耗验证
    Given the game is started
    When player equips a primary weapon
    Then the ammo displayed should be 20
    When player fires the weapon
    Then the ammo count should decrease
```

### 3. LLM 智能测试生成

通过 DeepSeek 自动从自然语言描述生成 Gherkin 测试用例。

## 🔧 配置说明

`Code/GameStateChecker/config.yaml`（不在 git 跟踪，参考结构）：

```yaml
active_target: assaultcube
targets:
  assaultcube:
    cv_params:
      ammo_bbox_rel: [0.68, 0.92, 0.05, 0.064]  # 相对坐标
      crosshair_region: [0.45, 0.4, 0.1, 0.2]
```

## 🎥 演示视频

AssaultCube baseline 演示：https://youtu.be/qFfWvaLtOU0

ViZDoom main line demo 计划在 Phase 1 末发布。

## 📚 相关文档

- 完整研究计划（5-Phase）：内部文档 `扩展构想-ViZDoom版.md`
- 架构设计：`Doc/` 目录
- Phase 0.1 接口设计意图：见 [perception/base.py](perception/base.py) 模块 docstring

## 🤝 贡献指南

本仓库当前是个人研究项目，欢迎 issue 讨论。

## 📄 许可证

License 暂未确定，请勿用于商业用途。学术参考请通过 Issue 联系作者。

## 📞 联系

- 🐛 Issues：[GitHub Issues](https://github.com/zydjx7/game-test-framework/issues)

## 🙏 致谢

- 本科指导：大连理工大学
- 修士指导：丸山 勝久 先生（立命館大学）
- 启发来源：TITAN (NetEase), RiverGame, ReAct, Voyager
