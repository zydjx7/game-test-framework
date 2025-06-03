# 🎮 游戏测试自动化框架 (Game Test Automation Framework)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org)
[![Gherkin](https://img.shields.io/badge/BDD-Gherkin-yellow.svg)](https://cucumber.io/docs/gherkin/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

## 📖 项目概述

这是一个基于计算机视觉技术的游戏测试自动化框架，旨在通过机器替代人工测试员，降低测试成本并提高测试效率。该框架具有跨引擎、跨平台的特性，可以在不同设备（PC、游戏主机）和操作系统上工作。

### ✨ 主要特点

- 🔍 **计算机视觉驱动**：基于图像识别技术，无需游戏引擎集成
- 🚀 **跨平台兼容**：支持PC、游戏主机等多种平台
- 🎯 **引擎无关**：适用于Unity、Unreal Engine等各种游戏引擎
- 🤖 **LLM集成**：支持DeepSeek、OpenAI等大语言模型辅助测试
- 📊 **BDD测试**：使用Gherkin语法编写可读性强的测试用例
- 📈 **全面覆盖**：支持UI状态检测、弹药系统、准星识别、声音分析等

## 🏗️ 框架架构

```
game-testing-main/
├── 📁 Code/                           # 核心代码目录
│   ├── 🎯 GameStateChecker/           # 游戏状态检测引擎
│   │   ├── AmmoTemplateRecognizer.py  # 弹药数量识别
│   │   ├── VisionUtils.py             # 计算机视觉工具
│   │   └── LogicLayer.py              # 逻辑处理层
│   ├── 🧪 bdd/                        # BDD测试框架
│   │   ├── features/                  # 测试用例特性文件
│   │   ├── steps/                     # 测试步骤定义
│   │   └── test_generator/            # LLM测试生成器
│   ├── 🎵 SoundTestingSupport/        # 音频测试工具
│   └── 🎬 AnimationtestingSupport/    # 动画测试工具
├── 📚 Doc/                            # 技术文档
└── 📋 requirements.txt                # 项目依赖
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- OpenCV 4.5+
- 必要的Python包（见requirements.txt）

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/zydjx7/game-test-framework.git
cd game-test-framework
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境变量**
```bash
# 如需使用LLM功能，配置API密钥
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-chat"
```

### 运行示例

```bash
# 运行BDD测试
cd Code/bdd
python run_tests.py

# 生成LLM测试用例
python test_generator/llm_generator.py
```

## 🎯 核心功能

### 1. 🎮 游戏状态检测

**支持的检测类型：**
- ✅ 弹药数量识别（单位数/双位数）
- ✅ 准星状态分析（武器类型、瞄准目标）
- ✅ 玩家生命状态检测
- ✅ 武器切换状态识别

**示例代码：**
```python
from Code.GameStateChecker.AmmoTemplateRecognizer import AmmoTemplateRecognizer

recognizer = AmmoTemplateRecognizer()
ammo_count, confidence = recognizer.recognize_number(screenshot)
print(f"检测到弹药数量: {ammo_count}, 置信度: {confidence}")
```

### 2. 🧪 BDD自动化测试

**支持的测试步骤：**
```gherkin
Feature: 武器系统测试
  Scenario: 弹药消耗验证
    Given the game is started
    When player equips a primary weapon
    Then the ammo displayed should be 20
    When player fires the weapon
    Then the ammo count should decrease
```

### 3. 🤖 LLM智能测试生成

框架集成了大语言模型，可自动生成测试用例：

```python
from Code.bdd.test_generator.llm_generator import TestGenerator

generator = TestGenerator()
test_case = generator.generate_test_case("测试玩家切换武器时准星的变化")
```

### 4. 🎵 音频测试支持

- 音频信号分析
- 语音识别验证
- 游戏音效检测

## 📊 支持的游戏

目前框架主要针对以下游戏进行了优化：

- 🎯 **AssaultCube**：完整的FPS游戏测试支持
- 🎮 **Unity示例项目**：展示框架在Unity引擎中的应用

## 🔧 配置说明

### config.yaml 配置文件

```yaml
active_target: assaultcube
targets:
  assaultcube:
    cv_params:
      ammo_bbox_rel: [0.68, 0.92, 0.05, 0.064]  # 弹药区域相对坐标
      crosshair_region: [0.45, 0.4, 0.1, 0.2]   # 准星检测区域
```

### 环境变量配置

```bash
# LLM配置
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
USE_LLM_ANALYSIS=true

# 调试配置
DEBUG_MODE=true
LOG_LEVEL=DEBUG
```

## 📈 测试报告

框架自动生成详细的测试报告：

- 📊 JSON格式测试结果
- 🖼️ 调试截图和标注
- 📋 详细的错误日志
- 📈 测试覆盖率统计

## 🎥 演示视频

观看我们的工具演示视频：
[https://youtu.be/qFfWvaLtOU0](https://youtu.be/qFfWvaLtOU0)

## 📚 文档与教程

### 快速入门指南

1. **BDD测试入门**：查看 `Code/bdd/` 目录中的Unity和Python示例
2. **音频测试**：参考 `Code/SoundTestingSupport/` 目录
3. **动画测试**：查看 `Code/AnimationtestingSupport/` 目录  
4. **计算机视觉模型**：探索 `Code/TestApp/` 和 `Code/TestsPriorityApp/` 目录

### 详细文档

完整的技术文档位于 **Doc** 目录中，包括：
- 架构设计文档
- API参考手册
- 最佳实践指南
- 常见问题解答

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系我们

- 📧 邮箱：[项目邮箱]
- 🐛 问题反馈：[GitHub Issues](https://github.com/zydjx7/game-test-framework/issues)
- 💬 讨论：[GitHub Discussions](https://github.com/zydjx7/game-test-framework/discussions)

## 🙏 致谢

感谢所有为这个项目贡献代码、报告问题和提供建议的开发者们！

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！
