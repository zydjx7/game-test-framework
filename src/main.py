import asyncio
from typing import Dict, Any
from loguru import logger
from llm.gpt_client import GPTClient
from gherkin.parser import GherkinParser
from rivergame.test_runner import RiverGameTestRunner
from utils.config import ConfigManager

class GameTestingFramework:
    def __init__(self):
        """初始化测试框架"""
        self.config_manager = ConfigManager()
        if not self.config_manager.validate_config():
            raise ValueError("配置验证失败")
            
        self.gpt_client = GPTClient()
        self.gherkin_parser = GherkinParser()
        self.test_runner = RiverGameTestRunner()
        
    async def generate_and_run_tests(self, game_description: str) -> Dict[str, Any]:
        """生成并运行测试场景"""
        try:
            # 使用GPT生成测试场景
            gherkin_content = await self.gpt_client.generate_test_scenario(
                f"为以下游戏场景生成Gherkin测试用例：\n{game_description}"
            )
            
            if not gherkin_content:
                raise ValueError("生成测试场景失败")
                
            # 验证生成的Gherkin内容
            if not self.gherkin_parser.validate_gherkin(gherkin_content):
                raise ValueError("生成的Gherkin内容无效")
                
            # 解析测试场景
            feature = self.gherkin_parser.parse_feature(gherkin_content)
            if not feature:
                raise ValueError("解析Gherkin内容失败")
                
            # 执行测试场景
            results = []
            for scenario in feature['scenarios']:
                scenario_result = await self.test_runner.execute_test_scenario(scenario)
                results.append(scenario_result)
                
            return {
                'feature': feature['name'],
                'scenarios_results': results,
                'success': all(r['success'] for r in results)
            }
            
        except Exception as e:
            logger.error(f"测试执行失败: {e}")
            return {
                'error': str(e),
                'success': False
            }
            
    async def cleanup(self):
        """清理资源"""
        await self.test_runner.cleanup()

async def main():
    """主函数"""
    framework = GameTestingFramework()
    try:
        # 示例游戏场景描述
        game_description = """
        RiverGame是一个简单的过河游戏：
        - 玩家需要控制角色过河
        - 河面上有多个浮动的平台
        - 玩家可以在平台间跳跃
        - 如果玩家掉入水中则游戏失败
        - 到达对岸则游戏胜利
        """
        
        results = await framework.generate_and_run_tests(game_description)
        logger.info(f"测试结果: {results}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
    finally:
        await framework.cleanup()

if __name__ == "__main__":
    asyncio.run(main())