from typing import Optional
from .gpt_client import GPTClient
from loguru import logger

class AmmoScenarioGenerator:
    def __init__(self, gpt_client: GPTClient):
        self.gpt_client = gpt_client
        self.template = """你是一个游戏测试专家，专注于生成Gherkin格式的测试场景。
请严格按照以下格式生成弹药同步测试场景：

Feature: [特性名称]
  Scenario: [场景名称]
    Given the player has [数字] bullets
    When the player shoots [once|数字 times]
    Then the ammo displayed should [be 数字|match the internal ammo]

生成的场景必须：
1. 完全匹配上述Given/When/Then模式
2. 确保弹药数量计算正确
3. 不要添加任何额外的步骤或注释"""

    async def generate_ammo_scenario(self, initial_ammo: Optional[int] = None, shots: Optional[int] = None) -> str:
        """生成弹药测试场景"""
        try:
            # 构建提示
            if initial_ammo is not None and shots is not None:
                prompt = f"{self.template}\n\n请生成一个场景，初始弹药为{initial_ammo}，射击{shots}次。"
            else:
                prompt = f"{self.template}\n\n请生成一个合理的弹药测试场景。"
            
            # 调用GPT生成场景
            gherkin_content = await self.gpt_client.generate_test_scenario(prompt)
            if not gherkin_content:
                raise ValueError("生成测试场景失败")
            
            logger.info("成功生成弹药测试场景")
            logger.debug(f"生成的场景内容：\n{gherkin_content}")
            
            return gherkin_content
            
        except Exception as e:
            logger.error(f"生成弹药测试场景失败: {e}")
            return None