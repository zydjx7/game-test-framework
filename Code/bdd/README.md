# Gherkin游戏测试框架

这个框架使用Gherkin语法和LLM自动生成游戏测试用例，结合现有的图像识别引擎进行游戏界面测试。

## 系统要求

- Python 3.7+
- OpenAI API密钥
- 游戏截图样本（位于`../../GameStateChecker/unitTestResources/`目录）

## 安装依赖

```bash
pip install behave python-dotenv openai opencv-python
```

## 配置API密钥

在运行测试之前，您需要设置OpenAI API密钥。有两种方式：

1. 通过命令行工具设置：

```bash
python update_api_key.py --key YOUR_API_KEY --test
```

2. 手动创建`.env`文件，内容为：

```
OPENAI_API_KEY=YOUR_API_KEY
```

## 运行测试

### 运行预定义测试

```bash
python run_tests.py --mode predefined
```

### 使用LLM生成并运行单个测试

```bash
python run_tests.py --mode generated --req "测试玩家装备武器后准星是否显示"
```

### 使用LLM生成并运行多个测试

```bash
python run_tests.py --mode batch --req "测试武器系统，包括切换武器和弹药变化" --count 3
```

## 文件结构

- `features/`: Gherkin特性文件
- `steps/`: 步骤定义文件
- `test_generator/`: LLM测试生成器
- `run_tests.py`: 测试运行脚本
- `update_api_key.py`: API密钥更新工具

## 添加新的测试

1. 在`features/`目录中创建新的`.feature`文件
2. 在`steps/`目录中实现对应的步骤函数
3. 运行`python run_tests.py --mode predefined`执行测试

## 使用LLM自动生成测试

示例：

```bash
python run_tests.py --mode generated --req "测试当玩家切换武器时，准星是否变化"
```

这将生成一个Gherkin测试用例，并自动执行它。

## 批量生成测试用例

示例：

```bash
python run_tests.py --mode batch --req "测试玩家在不同场景下的武器交互" --count 5
```

这将生成5个不同的测试场景，并将它们合并到一个`.feature`文件中执行。

## 测试报告

测试结果报告保存在`reports/`目录中，格式为JSON。 