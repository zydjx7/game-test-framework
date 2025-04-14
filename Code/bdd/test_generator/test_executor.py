from behave.runner import Runner
from behave.configuration import Configuration
from .llm_generator import TestGenerator
import os
import time
import json

class TestExecutor:
    def __init__(self, api_key=None):
        self.generator = TestGenerator(api_key)
        self.config = Configuration()
        self.feature_dir = os.path.join(os.path.dirname(__file__), '../features')
        
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
        return runner.run()
    
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
        return runner.run()
    
    def generate_report(self, result, generation_time=None):
        """生成测试报告"""
        report = {
            "status": "success" if result else "failure",
            "scenarios_passed": getattr(result, 'scenarios_passed', 0),
            "scenarios_failed": getattr(result, 'scenarios_failed', 0),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        if generation_time:
            report["generation_time"] = generation_time
            
        report_path = os.path.join(os.path.dirname(__file__), '../reports')
        if not os.path.exists(report_path):
            os.makedirs(report_path)
            
        report_file = os.path.join(report_path, f'report_{int(time.time())}.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
            
        return report 