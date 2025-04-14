from typing import List, Dict, Any, Optional
from loguru import logger
import re

class GherkinParser:
    def __init__(self):
        self.step_patterns = {
            'given_ammo': r'the player has (\d+) bullets?',
            'when_shoot': r'the player shoots? (?:once|(\d+) times?)',
            'then_ammo': r'the ammo displayed should be (\d+)',
            'then_sync': r'the ammo displayed should match the internal ammo'
        }
    
    def parse_feature(self, content: str) -> Optional[Dict[str, Any]]:
        """解析Gherkin特性文件内容"""
        if not content or not isinstance(content, str):
            logger.error("无效的Gherkin内容")
            return None
            
        try:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            feature = {
                'name': '',
                'scenarios': []
            }
            
            current_scenario = None
            
            for line in lines:
                if line.startswith('Feature:'):
                    feature['name'] = line[8:].strip()
                elif line.startswith('Scenario:'):
                    if current_scenario:
                        feature['scenarios'].append(current_scenario)
                    current_scenario = {
                        'name': line[9:].strip(),
                        'steps': []
                    }
                else:
                    # 解析步骤
                    step = self._parse_step(line)
                    if step and current_scenario:
                        current_scenario['steps'].append(step)
            
            if current_scenario:
                feature['scenarios'].append(current_scenario)
            
            if not feature['name'] or not feature['scenarios']:
                logger.error("无效的特性文件：缺少特性名称或场景")
                return None
                
            return feature
            
        except Exception as e:
            logger.error(f"解析Gherkin内容失败: {e}")
            return None
    
    def _parse_step(self, line: str) -> Optional[Dict[str, Any]]:
        """解析单个步骤"""
        try:
            for keyword in ['Given', 'When', 'Then']:
                if line.startswith(keyword):
                    content = line[len(keyword):].strip()
                    step_type = keyword.lower()
                    params = self._extract_params(step_type, content)
                    
                    if params is not None:
                        return {
                            'type': step_type,
                            'content': content,
                            'params': params
                        }
            return None
            
        except Exception as e:
            logger.error(f"解析步骤失败: {e}")
            return None
    
    def _extract_params(self, step_type: str, content: str) -> Optional[Dict[str, Any]]:
        """提取步骤参数"""
        try:
            if step_type == 'given':
                match = re.match(self.step_patterns['given_ammo'], content)
                if match:
                    return {'initial_ammo': int(match.group(1))}
                    
            elif step_type == 'when':
                match = re.match(self.step_patterns['when_shoot'], content)
                if match:
                    shots = match.group(1)
                    return {'shots': 1 if not shots else int(shots)}
                    
            elif step_type == 'then':
                match = re.match(self.step_patterns['then_ammo'], content)
                if match:
                    return {'expected_ammo': int(match.group(1))}
                    
                match = re.match(self.step_patterns['then_sync'], content)
                if match:
                    return {'check_sync': True}
            
            logger.error(f"无法提取参数，步骤类型: {step_type}, 内容: {content}")
            return None
            
        except Exception as e:
            logger.error(f"提取参数失败: {e}")
            return None
    
    def validate_gherkin(self, content: str) -> bool:
        """验证Gherkin语法是否正确"""
        if not content or not isinstance(content, str):
            return False
            
        try:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            has_feature = False
            has_scenario = False
            current_scenario_steps = []
            
            for line in lines:
                if line.startswith('Feature:'):
                    has_feature = True
                elif line.startswith('Scenario:'):
                    if current_scenario_steps:
                        if not self._validate_scenario_steps(current_scenario_steps):
                            return False
                    has_scenario = True
                    current_scenario_steps = []
                else:
                    step = self._parse_step(line)
                    if step:
                        current_scenario_steps.append(step)
            
            # 验证最后一个场景
            if current_scenario_steps:
                if not self._validate_scenario_steps(current_scenario_steps):
                    return False
            
            return has_feature and has_scenario
            
        except Exception as e:
            logger.error(f"验证Gherkin语法失败: {e}")
            return False
    
    def _validate_scenario_steps(self, steps: list) -> bool:
        """验证场景步骤的有效性"""
        try:
            if not steps:
                return False
                
            # 验证步骤顺序
            step_types = [step['type'] for step in steps]
            required_types = ['given', 'when', 'then']
            
            for required_type, actual_type in zip(required_types, step_types):
                if required_type != actual_type:
                    logger.error(f"步骤顺序错误: 期望 {required_type}, 实际 {actual_type}")
                    return False
            
            # 验证参数
            for step in steps:
                if not step.get('params'):
                    logger.error(f"步骤缺少参数: {step}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"验证场景步骤失败: {e}")
            return False
    
    def format_gherkin(self, feature: Dict[str, Any]) -> str:
        """将解析后的特性字典格式化为Gherkin文本"""
        try:
            lines = [f"Feature: {feature['name']}\n"]
            
            # 添加描述
            if feature['description']:
                lines.extend([f"  {line}" for line in feature['description']])
                lines.append("")
            
            # 添加场景
            for scenario in feature['scenarios']:
                lines.append(f"Scenario: {scenario['name']}")
                for step in scenario['steps']:
                    lines.append(f"  {step['type']} {step['content']}")
                lines.append("")
            
            return '\n'.join(lines)
            
        except Exception as e:
            logger.error(f"格式化Gherkin内容失败: {e}")
            return ""

    def add_step_definition(self, step_type: str, pattern: str, implementation: callable):
        """添加步骤定义"""
        pass  # 待实现
        
    def run_step(self, step: Dict[str, str]):
        """执行步骤"""
        pass  # 待实现

    def validate_ammo_scenario(self, scenario: Dict[str, Any]) -> bool:
        """验证弹药场景的正确性"""
        try:
            has_given = False
            has_when = False
            has_then = False
            
            for step in scenario['steps']:
                if step['type'] == 'given' and 'initial_ammo' in step['params']:
                    has_given = True
                elif step['type'] == 'when' and 'shots' in step['params']:
                    has_when = True
                elif step['type'] == 'then' and ('expected_ammo' in step['params'] or 'check_sync' in step['params']):
                    has_then = True
            
            return has_given and has_when and has_then
            
        except Exception as e:
            logger.error(f"验证弹药场景失败: {e}")
            return False