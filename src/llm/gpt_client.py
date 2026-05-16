import yaml
from openai import AsyncOpenAI
from typing import Dict, Any, Optional
from loguru import logger
from .client_helpers import create_async_openai_client, load_deepseek_config

class GPTClient:
    def __init__(self, config_path: str = "config.yaml"):
        """初始化DeepSeek-compatible LLM客户端"""
        self.config = self._load_config(config_path)
        deepseek_config = self.config["api"]["deepseek"]
        self.client = create_async_openai_client(
            AsyncOpenAI,
            api_key=deepseek_config["api_key"],
            base_url=deepseek_config.get("base_url"),
        )
        self.model = deepseek_config["model"]
        self.temperature = deepseek_config["temperature"]
        self.max_tokens = deepseek_config["max_tokens"]
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            loaded = load_deepseek_config(config)
            config.setdefault("api", {})["deepseek"] = {
                "api_key": loaded.api_key,
                "base_url": loaded.base_url,
                "model": loaded.model,
                "temperature": loaded.temperature,
                "max_tokens": loaded.max_tokens,
            }
            logger.info(
                "Loaded DeepSeek config for model={} base_url={}",
                loaded.model,
                loaded.base_url,
            )
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    async def generate_test_scenario(self, prompt: str) -> Optional[str]:
        """生成测试场景"""
        try:
            # 构建系统提示，引导生成标准Gherkin格式
            system_prompt = """你是一个游戏测试专家，专注于生成Gherkin格式的测试场景。
请按照以下格式生成测试场景：

Feature: [特性名称]
  [特性描述]

  Scenario: [场景名称]
    Given [初始条件]
    When [执行动作]
    Then [期望结果]

确保生成的场景：
1. 包含完整的Feature和至少一个Scenario
2. 每个Scenario都有Given, When, Then步骤
3. 使用清晰和具体的描述
4. 保持步骤简单明了
"""

            # 添加具体的游戏测试要求
            test_requirements = """
针对游戏测试，请特别关注：
- 游戏状态验证
- 玩家动作响应
- 游戏规则遵守
- 边界条件测试
"""

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": test_requirements},
                    {"role": "user", "content": f"请为以下游戏场景生成测试用例：\n{prompt}"}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            generated_content = response.choices[0].message.content
            logger.info(f"成功生成测试场景")
            logger.debug(f"生成的内容：\n{generated_content}")
            
            return generated_content
            
        except Exception as e:
            logger.error(f"调用DeepSeek API失败，将使用 deterministic fallback 场景: {e}")
            return self._fallback_test_scenario(prompt)

    def _fallback_test_scenario(self, prompt: str) -> str:
        """Return a deterministic scenario when the LLM endpoint is unavailable."""
        return """Feature: RiverGame basic flow
  Scenario: Player crosses the first platform
    Given the game is started and the player is at the start
    When the player jumps to the first platform
    Then the player should be on the first platform"""

    async def analyze_game_state(self, state_description: str) -> Optional[Dict[str, Any]]:
        """分析游戏状态"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个游戏状态分析专家，专注于分析游戏状态并提供关键信息。"},
                    {"role": "user", "content": f"分析以下游戏状态并提供关键信息：\n{state_description}"}
                ],
                temperature=0.3,
                max_tokens=self.max_tokens
            )
            
            analysis = response.choices[0].message.content
            logger.info("游戏状态分析完成")
            
            return {
                "analysis": analysis,
                "success": True
            }
        except Exception as e:
            logger.error(f"游戏状态分析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
