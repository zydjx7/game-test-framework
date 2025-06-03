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
        # 检查是否使用LLM分析准星
        use_llm_analysis = os.getenv("USE_LLM_ANALYSIS", "false").lower() == "true"
        
        # 检查测试需求是否涉及弹药
        ammo_related = any(word in test_requirement.lower() for word in ['弹药', '子弹', 'ammo', '射击', '开火', 'fire'])
        
        # 根据USE_LLM_ANALYSIS环境变量调整可用步骤
        if use_llm_analysis:
            crosshair_steps = """
        - Then the crosshair should be visible (准星应该可见)
        - Then the crosshair should indicate aiming at an enemy (准星应该表明瞄准敌人)
        - Then the crosshair should indicate aiming at a teammate (准星应该表明瞄准队友)
        - Then the crosshair should be for a grenade (准星应该是手雷的准星)
        - Then the crosshair should be for a knife (准星应该是刀的准星)
        - Then the crosshair should be for a secondary weapon (准星应该是副武器的准星)
        - Then the crosshair should be for a primary weapon (准星应该是主武器的准星)
        - Then the crosshair should indicate reloading (准星应该表明正在换弹)
        - Then the player view should indicate death (玩家视图应该表明死亡状态)
        - Then the crosshair should match the current weapon state (准星应该匹配当前武器状态)
        - Then the crosshair should be analyzed by LLM (准星应该由LLM分析)"""
        else:
            crosshair_steps = """
        - Then the crosshair should be visible (准星应该可见)
        - Then the crosshair should indicate aiming at an enemy (准星应该表明瞄准敌人)
        - Then the crosshair should indicate aiming at a teammate (准星应该表明瞄准队友)
        - Then the crosshair should be for a grenade (准星应该是手雷的准星)
        - Then the crosshair should be for a knife (准星应该是刀的准星)
        - Then the crosshair should be for a secondary weapon (准星应该是副武器的准星)
        - Then the crosshair should be for a primary weapon (准星应该是主武器的准星)
        - Then the crosshair should indicate reloading (准星应该表明正在换弹)
        - Then the player view should indicate death (玩家视图应该表明死亡状态)"""
        
        # 根据是否涉及弹药决定是否包含弹药相关步骤
        ammo_steps = ""
        if ammo_related:
            ammo_steps = """
        - Given the player has {number} ammo (玩家有特定数量的弹药)
        - Then the ammo count should decrease (弹药数量应该减少)
        - Then the ammo count should match the expected value (弹药数量应该匹配预期值)
        - Then the ammo displayed should be {number} (弹药显示应为特定数值)
        - Then the ammo count should be verified by LLM (弹药数量应该由LLM验证)"""
        
        # 组合生成最终的提示模板
        return f"""
        生成一个针对以下游戏测试需求的Gherkin测试场景:
        {test_requirement}
        
        只使用以下预定义步骤:
        - Given the game is started (游戏已启动)
        - When player equips a weapon (玩家装备武器)
        - When player equips a primary weapon (玩家装备主武器)
        - When player fires the weapon (玩家开火)
        - When player aims at an enemy (玩家瞄准敌人)
        - When player aims at a teammate (玩家瞄准队友)
        - When player switches to grenade (玩家切换到手雷)
        - When player switches to knife (玩家切换到刀)
        - When player switches to secondary weapon (玩家切换到副武器)
        - When player reloads the weapon (玩家重新装填武器)
        - When player dies (玩家死亡)
        {crosshair_steps}{ammo_steps}
        
        注意：本框架不驱动真实游戏，只验证静态截图，场景里不要引入需要实时交互的步骤。
        当前测试针对的是AssaultCube游戏，请确保测试场景符合这款FPS游戏的特性:
        - AssaultCube中默认突击步枪的满弹量是20发
        - 可用的弹药数值通常为0-20之间
        - 请不要使用超过20的弹药数，因为测试框架中没有这样的截图
        
        测试步骤明确规则：
        - 使用"When player equips a primary weapon"来表示玩家装备主武器
        - 使用"Then the crosshair should be for a primary weapon"来验证主武器准星
        - 仅在测试需求明确涉及弹药时才使用弹药相关步骤
        - 当测试武器切换时，使用专用的步骤验证准星（如"Then the crosshair should be for a knife"），不要使用通用步骤
        - 仅在测试需求明确要求使用LLM分析时才使用"Then the crosshair should be analyzed by LLM"步骤
        
        测试场景参考示例（当测试要求验证不同武器的准星时）：
        ```
        Feature: Weapon Switching and Crosshair Verification
        Scenario: Verify crosshair changes when switching between weapons
            Given the game is started
            When player equips a primary weapon
            Then the crosshair should be for a primary weapon
            When player switches to secondary weapon
            Then the crosshair should be for a secondary weapon
            When player switches to knife
            Then the crosshair should be for a knife
        ```
        
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
            
            # 检查是否使用LLM分析准星
            use_llm_analysis = os.getenv("USE_LLM_ANALYSIS", "false").lower() == "true"
            
            # 检查测试需求是否涉及弹药
            ammo_related = any(word in requirement.lower() for word in ['弹药', '子弹', 'ammo', '射击', '开火', 'fire'])
            
            # 根据USE_LLM_ANALYSIS环境变量调整可用步骤
            if use_llm_analysis:
                crosshair_steps = """
            - Then the crosshair should be visible (准星应该可见)
            - Then the crosshair should indicate aiming at an enemy (准星应该表明瞄准敌人)
            - Then the crosshair should indicate aiming at a teammate (准星应该表明瞄准队友)
            - Then the crosshair should be for a grenade (准星应该是手雷的准星)
            - Then the crosshair should be for a knife (准星应该是刀的准星)
            - Then the crosshair should be for a secondary weapon (准星应该是副武器的准星)
            - Then the crosshair should be for a primary weapon (准星应该是主武器的准星)
            - Then the crosshair should indicate reloading (准星应该表明正在换弹)
            - Then the player view should indicate death (玩家视图应该表明死亡状态)
            - Then the crosshair should match the current weapon state (准星应该匹配当前武器状态)
            - Then the crosshair should be analyzed by LLM (准星应该由LLM分析)"""
            else:
                crosshair_steps = """
            - Then the crosshair should be visible (准星应该可见)
            - Then the crosshair should indicate aiming at an enemy (准星应该表明瞄准敌人)
            - Then the crosshair should indicate aiming at a teammate (准星应该表明瞄准队友)
            - Then the crosshair should be for a grenade (准星应该是手雷的准星)
            - Then the crosshair should be for a knife (准星应该是刀的准星)
            - Then the crosshair should be for a secondary weapon (准星应该是副武器的准星)
            - Then the crosshair should be for a primary weapon (准星应该是主武器的准星)
            - Then the crosshair should indicate reloading (准星应该表明正在换弹)
            - Then the player view should indicate death (玩家视图应该表明死亡状态)"""
            
            # 根据是否涉及弹药决定是否包含弹药相关步骤
            ammo_steps = ""
            if ammo_related:
                ammo_steps = """
            - Given the player has {number} ammo (玩家有特定数量的弹药)
            - Then the ammo count should decrease (弹药数量应该减少)
            - Then the ammo count should match the expected value (弹药数量应该匹配预期值)
            - Then the ammo displayed should be {number} (弹药显示应为特定数值)
            - Then the ammo count should be verified by LLM (弹药数量应该由LLM验证)"""
            
            prompt = f"""
            为以下游戏测试需求生成{count}个不同的Gherkin测试场景:
            {requirement}
            
            每个场景应该测试不同的方面或情况。只使用以下预定义步骤:
            - Given the game is started (游戏已启动)
            - When player equips a weapon (玩家装备武器)
            - When player equips a primary weapon (玩家装备主武器)
            - When player fires the weapon (玩家开火)
            - When player aims at an enemy (玩家瞄准敌人)
            - When player aims at a teammate (玩家瞄准队友)
            - When player switches to grenade (玩家切换到手雷)
            - When player switches to knife (玩家切换到刀)
            - When player switches to secondary weapon (玩家切换到副武器)
            - When player reloads the weapon (玩家重新装填武器)
            - When player dies (玩家死亡)
            {crosshair_steps}{ammo_steps}
            
            注意：本框架不驱动真实游戏，只验证静态截图，场景里不要引入需要实时交互的步骤。
            当前测试针对的是AssaultCube游戏，请确保测试场景符合这款FPS游戏的特性:
            - AssaultCube中默认突击步枪的满弹量是20发
            - 可用的弹药数值通常为0-20之间
            - 请不要使用超过20的弹药数，因为测试框架中没有这样的截图
            
            测试步骤明确规则：
            - 使用"When player equips a primary weapon"来表示玩家装备主武器
            - 使用"Then the crosshair should be for a primary weapon"来验证主武器准星
            - 仅在测试需求明确涉及弹药时才使用弹药相关步骤
            - 当测试武器切换时，使用专用的步骤验证准星（如"Then the crosshair should be for a knife"），不要使用通用步骤
            - 仅在测试需求明确要求使用LLM分析时才使用"Then the crosshair should be analyzed by LLM"步骤
            
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