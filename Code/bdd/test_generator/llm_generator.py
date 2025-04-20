import os
from dotenv import load_dotenv
from openai import OpenAI
import sys
from loguru import logger

# 设置日志
logger.add("logs/test_generator.log", level="DEBUG", rotation="1 MB")

class TestGenerator:
    def __init__(self, api_key=None):
        # 加载环境变量
        load_dotenv()
        
        # 优先使用参数传入的API密钥，如果没有则使用环境变量
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("未找到API密钥，请设置OPENAI_API_KEY环境变量")
            raise ValueError("No API key found in parameters or environment variables")
        
        # 从环境变量获取模型名称，默认为deepseek-chat
        self.model = os.getenv("OPENAI_MODEL", "deepseek-chat")
        logger.info(f"使用模型: {self.model}")
        
        # 获取API类型和基础URL
        self.api_type = os.getenv("API_TYPE", "deepseek")
        self.base_url = os.getenv("OPENAI_BASE_URL", "")
        
        # 创建OpenAI客户端
        client_args = {"api_key": self.api_key}
        if self.base_url:
            client_args["base_url"] = self.base_url
            logger.info(f"使用自定义API基础URL: {self.base_url}")
            
        self.client = OpenAI(**client_args)
        
    def create_test_prompt(self, test_requirement):
        return f"""
        生成一个针对以下游戏测试需求的Gherkin测试场景:
        {test_requirement}
        
        只使用以下预定义步骤:
        - Given the game is started (游戏已启动)
        - When player equips a weapon (玩家装备武器)
        - When player fires the weapon (玩家开火)
        - When player switches to a different weapon (玩家切换到不同武器)
        - When player reloads the weapon (玩家重新装填武器)
        - Then the crosshair should be visible (准星应该可见)
        - Then the ammo count should decrease (弹药数量应该减少)
        - Then the ammo count should match the expected value (弹药数量应该与预期值匹配)
        
        注意：本框架不驱动真实游戏，只验证静态截图，场景里不要引入需要实时交互的步骤。
        当前测试针对的是AssaultCube游戏，请确保测试场景符合这款FPS游戏的特性。
        
        仅输出Gherkin场景，不要包含任何解释。使用Feature:和Scenario:关键字，并确保格式正确。
        """
    
    def generate_test_case(self, requirement):
        try:
            logger.info(f"开始生成测试用例，需求: {requirement[:50]}...，使用{self.api_type} API")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个游戏测试用例生成器，专门生成Gherkin格式的测试用例。"},
                    {"role": "user", "content": self.create_test_prompt(requirement)}
                ]
            )
            
            test_case = response.choices[0].message.content
            logger.info(f"成功生成测试用例，长度: {len(test_case)}")
            return test_case
        except Exception as e:
            logger.error(f"生成测试用例时出错: {e}")
            return None
            
    def generate_multiple_test_cases(self, requirement, count=5):
        """生成多个测试用例"""
        try:
            logger.info(f"开始生成{count}个测试用例，需求: {requirement[:50]}...")
            
            prompt = f"""
            为以下游戏测试需求生成{count}个不同的Gherkin测试场景:
            {requirement}
            
            每个场景应该测试不同的方面或情况。只使用以下预定义步骤:
            - Given the game is started (游戏已启动)
            - When player equips a weapon (玩家装备武器)
            - When player fires the weapon (玩家开火)
            - When player switches to a different weapon (玩家切换到不同武器)
            - When player reloads the weapon (玩家重新装填武器)
            - Then the crosshair should be visible (准星应该可见)
            - Then the ammo count should decrease (弹药数量应该减少)
            - Then the ammo count should match the expected value (弹药数量应该与预期值匹配)
            
            注意：本框架不驱动真实游戏，只验证静态截图，场景里不要引入需要实时交互的步骤。
            当前测试针对的是AssaultCube游戏，请确保测试场景符合这款FPS游戏的特性。
            
            使用Feature:和多个Scenario:关键字，并确保格式正确。
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个游戏测试用例生成器，专门生成Gherkin格式的测试用例。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            test_cases = response.choices[0].message.content
            logger.info(f"成功生成多个测试用例，长度: {len(test_cases)}")
            return test_cases
        except Exception as e:
            logger.error(f"生成多个测试用例时出错: {e}")
            return None