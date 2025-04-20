from behave.runner import Runner
from behave.configuration import Configuration
from .llm_generator import TestGenerator
import os
import time
import json
import re
from loguru import logger

class TestExecutor:
    def __init__(self, api_key=None, target=None):
        self.generator = TestGenerator(api_key)
        # 使用空列表作为command_args，避免Behave解析sys.argv
        self.config = Configuration(command_args=[])
        # 设置格式为pretty，避免NoneType错误
        self.config.format = ["pretty"]
        # 设置失败时不停止，继续执行后续步骤
        self.config.stop_on_failure = False
        self.feature_dir = os.path.join(os.path.dirname(__file__), '../features')
        # 确保feature_dir目录存在
        os.makedirs(self.feature_dir, exist_ok=True)
        # 保存测试目标
        self.target = target
        # 如果提供了target，设置环境变量
        if self.target:
            os.environ["GAMECHECK_TARGET"] = self.target
        
    def execute_from_requirement(self, requirement):
        """根据需求生成并执行单个测试用例"""
        # 生成测试用例
        gherkin_test = self.generator.generate_test_case(requirement)
        if not gherkin_test:
            return {"status": "error", "message": "无法生成测试用例"}
        
        # 记录原始输出并清理LLM返回的Gherkin文本
        logger.debug(f"LLM 原始输出:\n----\n{gherkin_test}\n----") 
        cleaned_gherkin = gherkin_test.strip()  # 去除首尾空白

        # 尝试去除常见的Markdown代码块标记
        lines = cleaned_gherkin.splitlines()
        if lines:
            # 移除开头的```gherkin或```
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            # 移除结尾的```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

        cleaned_gherkin = "\n".join(lines).strip()

        # 检查是否包含Feature:关键字
        if not re.search(r"Feature:", cleaned_gherkin):
            logger.error("无法在LLM输出中找到有效的'Feature:'关键字")
            return {"status": "error", "message": "LLM输出格式错误，未找到Feature关键字"}

        logger.debug(f"清理后的Gherkin:\n----\n{cleaned_gherkin}\n----")
        
        # 保存为.feature文件
        feature_path = os.path.join(self.feature_dir, 'generated_test.feature')
        
        with open(feature_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_gherkin)
        
        # 执行测试
        self.config.paths = [feature_path]
        runner = Runner(self.config)
        exit_code = runner.run()
        success = (exit_code == 0)
        
        # 返回成功/失败状态
        return {"status": "success" if success else "failure", "exit_code": exit_code}
    
    def execute_batch(self, requirement, count=5):
        """生成并执行多个测试用例"""
        # 生成多个测试用例
        gherkin_tests = self.generator.generate_multiple_test_cases(requirement, count)
        if not gherkin_tests:
            return {"status": "error", "message": "无法生成测试用例"}
        
        # 记录原始输出并清理LLM返回的Gherkin文本
        logger.debug(f"LLM 原始批量输出:\n----\n{gherkin_tests}\n----") 
        cleaned_gherkin = gherkin_tests.strip()  # 去除首尾空白

        # 尝试去除常见的Markdown代码块标记
        lines = cleaned_gherkin.splitlines()
        if lines:
            # 移除开头的```gherkin或```
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            # 移除结尾的```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

        cleaned_gherkin = "\n".join(lines).strip()

        # 检查是否包含Feature:关键字
        if not re.search(r"Feature:", cleaned_gherkin):
            logger.error("无法在LLM批量输出中找到有效的'Feature:'关键字")
            return {"status": "error", "message": "LLM批量输出格式错误，未找到Feature关键字"}

        logger.debug(f"清理后的批量Gherkin:\n----\n{cleaned_gherkin}\n----")
        
        # 保存为.feature文件
        feature_path = os.path.join(self.feature_dir, 'batch_test.feature')
        
        with open(feature_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_gherkin)
        
        # 执行测试
        self.config.paths = [feature_path]
        runner = Runner(self.config)
        exit_code = runner.run()
        success = (exit_code == 0)
        
        # 返回成功/失败状态
        return {"status": "success" if success else "failure", "exit_code": exit_code}
    
    def generate_report(self, result, generation_time=None):
        """生成测试报告"""
        # 检查result是字典还是对象
        if isinstance(result, dict):
            report = {
                "status": result.get("status", "unknown"),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            # 添加其他可能的字段
            for key in ["exit_code", "message"]:
                if key in result:
                    report[key] = result[key]
        else:
            # 旧版兼容，但实际已不会执行到这里
            report = {
                "status": "success" if result else "failure",
                "scenarios_passed": getattr(result, 'scenarios_passed', 0),
                "scenarios_failed": getattr(result, 'scenarios_failed', 0),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        
        if generation_time:
            report["generation_time"] = generation_time
        
        # 添加测试目标信息
        if self.target:
            report["target"] = self.target
            
        report_path = os.path.join(os.path.dirname(__file__), '../reports')
        if not os.path.exists(report_path):
            os.makedirs(report_path)
            
        report_file = os.path.join(report_path, f'report_{int(time.time())}.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
            
        return report