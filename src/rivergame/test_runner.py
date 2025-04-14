import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger
import aiohttp
import yaml

class RiverGameTestRunner:
    def __init__(self, config_path: str = "config.yaml"):
        """初始化测试运行器"""
        self.config = self._load_config(config_path)
        self.base_url = f"http://{self.config['rivergame']['host']}:{self.config['rivergame']['port']}"
        self.session = None
        self._initialized = False
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    async def initialize(self):
        """初始化HTTP会话"""
        if not self._initialized:
            self.session = aiohttp.ClientSession()
            self._initialized = True
            logger.info("RiverGameTestRunner初始化完成")

    async def cleanup(self):
        """清理资源"""
        if self._initialized and self.session:
            await self.session.close()
            self._initialized = False
            logger.info("RiverGameTestRunner清理完成")
            
    async def execute_test_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """执行测试场景"""
        if not self._initialized:
            await self.initialize()
            
        results = {
            'scenario_name': scenario['name'],
            'steps_results': [],
            'success': True
        }
        
        try:
            for step in scenario['steps']:
                step_result = await self._execute_step(step)
                results['steps_results'].append(step_result)
                if not step_result['success']:
                    results['success'] = False
                    break
                    
            return results
            
        except Exception as e:
            logger.error(f"执行测试场景失败: {e}")
            results['success'] = False
            results['error'] = str(e)
            return results
            
    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个测试步骤"""
        result = {
            'step_type': step['type'],
            'step_content': step['content'],
            'success': False
        }
        
        try:
            if not self.session:
                await self.initialize()
                
            # 根据步骤类型执行相应的操作
            if step['type'] == 'Given':
                result.update(await self._setup_game_state(step['content']))
            elif step['type'] == 'When':
                result.update(await self._perform_action(step['content']))
            elif step['type'] == 'Then':
                result.update(await self._verify_game_state(step['content']))
            
            return result
            
        except Exception as e:
            logger.error(f"执行测试步骤失败: {e}")
            result['error'] = str(e)
            return result
            
    async def _setup_game_state(self, state_description: str) -> Dict[str, Any]:
        """设置游戏初始状态"""
        try:
            async with self.session.post(
                f"{self.base_url}/setup",
                json={"state": state_description}
            ) as response:
                response_data = await response.json()
                return {
                    'success': response.status == 200,
                    'response': response_data
                }
        except Exception as e:
            logger.error(f"设置游戏状态失败: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _perform_action(self, action_description: str) -> Dict[str, Any]:
        """执行游戏动作"""
        try:
            async with self.session.post(
                f"{self.base_url}/action",
                json={"action": action_description}
            ) as response:
                response_data = await response.json()
                return {
                    'success': response.status == 200,
                    'response': response_data
                }
        except Exception as e:
            logger.error(f"执行游戏动作失败: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _verify_game_state(self, expected_state: str) -> Dict[str, Any]:
        """验证游戏状态"""
        try:
            async with self.session.get(
                f"{self.base_url}/state"
            ) as response:
                current_state = await response.json()
                # 这里需要实现具体的状态验证逻辑
                verification_result = self._compare_states(current_state, expected_state)
                return {
                    'success': verification_result['success'],
                    'details': verification_result
                }
        except Exception as e:
            logger.error(f"验证游戏状态失败: {e}")
            return {'success': False, 'error': str(e)}
            
    def _compare_states(self, current_state: Dict[str, Any], expected_state: str) -> Dict[str, Any]:
        """比较当前状态和期望状态"""
        # 这里需要实现具体的状态比较逻辑
        # 可以使用LLM来协助理解和比较状态
        return {
            'success': True,  # 临时返回值，需要实现实际的比较逻辑
            'details': {
                'current_state': current_state,
                'expected_state': expected_state
            }
        }