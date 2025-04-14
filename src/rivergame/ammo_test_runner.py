from typing import Dict, Any, Optional
from loguru import logger
import sys
import os
import cv2
import yaml

# 添加GameStateChecker目录到Python路径
GAME_CHECKER_PATH = os.path.join(os.path.dirname(__file__), '../../Code/GameStateChecker')
sys.path.append(GAME_CHECKER_PATH)
from LogicLayer import LogicLayer

class AmmoTestRunner:
    def __init__(self):
        """初始化弹药测试执行器"""
        self._current_ammo = 0
        self._logic_layer = LogicLayer()
        self._screenshot_base_path = GAME_CHECKER_PATH
        
        # 加载配置
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
                logger.info("成功加载配置文件")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            self._config = {}
            
        logger.info(f"初始化AmmoTestRunner, 截图路径: {self._screenshot_base_path}")
        
        self._vision_params = {
            "bbox": [912, 1015, 79, 49],
            "hsv_min": 129,
            "hsv_max": 130
        }
        
    def setup_vision_params(self, bbox=None, hsv_min=None, hsv_max=None):
        """设置视觉检测参数"""
        if bbox:
            self._vision_params["bbox"] = bbox
        if hsv_min is not None:
            self._vision_params["hsv_min"] = hsv_min
        if hsv_max is not None:
            self._vision_params["hsv_max"] = hsv_max
            
        logger.info(f"更新视觉检测参数: {self._vision_params}")

    async def execute_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """执行弹药测试场景"""
        try:
            logger.info(f"开始执行测试场景: {scenario['name']}")
            results = {
                'scenario_name': scenario['name'],
                'steps_results': [],
                'success': True
            }
            
            # 列出可用的截图
            screenshots = [f for f in os.listdir(self._screenshot_base_path) 
                         if f.startswith(('screenShot_', 'aug_screenShot_'))]
            logger.debug(f"可用的截图文件: {screenshots[:5]}...")
            
            for step in scenario['steps']:
                logger.debug(f"执行步骤: {step['type']} - {step.get('content', '')}")
                step_result = await self._execute_step(step)
                results['steps_results'].append(step_result)
                
                if not step_result['success']:
                    logger.error(f"步骤执行失败: {step_result.get('error', '未知错误')}")
                    results['success'] = False
                    break
                else:
                    logger.info(f"步骤执行成功: {step_result['message']}")
            
            return results
            
        except Exception as e:
            error_msg = f"执行测试场景失败: {str(e)}"
            logger.error(error_msg)
            return {
                'scenario_name': scenario.get('name', 'Unknown'),
                'error': error_msg,
                'success': False,
                'steps_results': []
            }
    
    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个测试步骤"""
        try:
            step_type = step['type']
            params = step.get('params', {})
            logger.debug(f"执行步骤类型: {step_type}, 参数: {params}")
            
            if step_type == 'given':
                if 'initial_ammo' in params:
                    self._current_ammo = params['initial_ammo']
                    logger.debug(f"设置初始弹药数为: {self._current_ammo}")
                    screenshot = self._load_screenshot(self._current_ammo)
                    
                    if screenshot is None:
                        error_msg = f"无法加载初始弹药数为 {self._current_ammo} 的截图"
                        logger.error(error_msg)
                        return {
                            'type': step_type,
                            'message': error_msg,
                            'success': False,
                            'error': error_msg
                        }
                    
                    # 使用更新后的视觉参数进行验证
                    is_sync = self._verify_ammo_sync(
                        screenshot, 
                        self._current_ammo,
                        self._vision_params
                    )
                    
                    return {
                        'type': step_type,
                        'message': f"设置初始弹药数为: {self._current_ammo}",
                        'success': is_sync
                    }
                    
            elif step_type == 'when':
                if 'shots' in params:
                    shots = params['shots']
                    old_ammo = self._current_ammo
                    self._current_ammo = max(0, self._current_ammo - shots)
                    logger.debug(f"发射子弹: {shots} 发，弹药数从 {old_ammo} 变为 {self._current_ammo}")
                    
                    screenshot = self._load_screenshot(self._current_ammo)
                    if screenshot is None:
                        error_msg = f"无法加载射击后弹药数为 {self._current_ammo} 的截图"
                        logger.error(error_msg)
                        return {
                            'type': step_type,
                            'message': error_msg,
                            'success': False,
                            'error': error_msg
                        }
                        
                    return {
                        'type': step_type,
                        'message': f"发射了 {shots} 发子弹，剩余 {self._current_ammo}",
                        'success': True
                    }
                    
            elif step_type == 'then':
                screenshot = self._load_screenshot(self._current_ammo)
                if screenshot is None:
                    error_msg = "无法加载验证阶段的游戏截图"
                    logger.error(error_msg)
                    return {
                        'type': step_type,
                        'message': error_msg,
                        'success': False,
                        'error': error_msg
                    }
                
                if 'check_sync' in params and params['check_sync']:
                    is_sync = self._verify_ammo_sync(screenshot, self._current_ammo)
                    result_msg = f"验证弹药显示同步状态: {'成功' if is_sync else '失败'}"
                    logger.info(result_msg)
                    return {
                        'type': step_type,
                        'message': result_msg,
                        'success': is_sync
                    }
                elif 'expected_ammo' in params:
                    expected = params['expected_ammo']
                    is_sync = self._verify_ammo_sync(screenshot, expected)
                    result_msg = f"验证弹药数量: 期望={expected}, 实际={self._current_ammo}"
                    logger.info(f"{result_msg}, 同步状态: {'成功' if is_sync else '失败'}")
                    return {
                        'type': step_type,
                        'message': result_msg,
                        'success': is_sync and self._current_ammo == expected
                    }
            
            error_msg = f"未知的测试步骤类型: {step_type}"
            logger.warning(error_msg)
            return {
                'type': step_type,
                'message': error_msg,
                'success': False,
                'error': error_msg
            }
            
        except Exception as e:
            error_msg = f"执行测试步骤失败: {e}"
            logger.error(error_msg)
            return {
                'type': step_type,
                'error': error_msg,
                'success': False,
                'message': error_msg
            }
            
    def _load_screenshot(self, ammo_count: int) -> Optional[Any]:
        """加载对应弹药数量的游戏截图"""
        try:
            # 构建截图文件名模式以支持不同的命名格式
            screenshot_patterns = [
                f"screenShot_{ammo_count*100}.png",
                f"aug_screenShot_{ammo_count*100}.png",
                f"screenShot_{ammo_count}.png",
                f"aug_screenShot_{ammo_count}.png"
            ]
            
            for pattern in screenshot_patterns:
                screenshot_path = os.path.join(self._screenshot_base_path, pattern)
                logger.debug(f"尝试加载截图: {screenshot_path}")
                
                if os.path.exists(screenshot_path):
                    image = cv2.imread(screenshot_path)
                    if image is not None:
                        logger.info(f"成功加载截图: {screenshot_path}")
                        return image
                    else:
                        logger.warning(f"无法读取截图内容: {screenshot_path}")
            
            logger.error(f"找不到对应弹药数{ammo_count}的截图，尝试过的模式: {screenshot_patterns}")
            return None
            
        except Exception as e:
            logger.error(f"加载截图失败: {e}")
            return None
            
    def _verify_ammo_sync(self, screenshot: Any, expected_ammo: int, vision_params: Dict[str, Any] = None) -> bool:
        """验证弹药同步状态"""
        try:
            if vision_params is None:
                vision_params = self._vision_params
                
            logger.debug(f"开始验证弹药同步，期望值: {expected_ammo}，参数: {vision_params}")
            
            result = self._logic_layer.check_ammo_state(
                screenshot,
                expected_ammo,
                region=vision_params["bbox"]
            )
            
            logger.info(f"弹药验证结果: {result}")
            return result[0]  # 返回验证是否成功
            
        except Exception as e:
            logger.error(f"验证弹药同步失败: {e}")
            return False