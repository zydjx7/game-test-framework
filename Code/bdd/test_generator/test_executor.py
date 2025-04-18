from behave.runner import Runner
from behave.configuration import Configuration
from .llm_generator import TestGenerator
import os
import time
import json

class TestExecutor:
    def __init__(self, api_key=None, target=None):
        self.generator = TestGenerator(api_key)
        # 使用空列表作为command_args，避免Behave解析sys.argv
        self.config = Configuration(command_args=[])
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
        
        # 保存为.feature文件
        feature_path = os.path.join(self.feature_dir, 'generated_test.feature')
        
        with open(feature_path, 'w', encoding='utf-8') as f:
            f.write(gherkin_test)
        
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
        
        # 保存为.feature文件
        feature_path = os.path.join(self.feature_dir, 'batch_test.feature')
        
        with open(feature_path, 'w', encoding='utf-8') as f:
            f.write(gherkin_tests)
        
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