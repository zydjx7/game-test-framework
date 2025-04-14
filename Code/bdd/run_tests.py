#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gherkin测试自动化脚本
使用LLM自动生成并执行Gherkin格式的游戏测试用例
"""

import os
import sys
import time
from test_generator.test_executor import TestExecutor
from dotenv import load_dotenv

# 加载环境变量中的API密钥
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

def run_predefined_tests():
    """运行预定义的测试用例"""
    from behave.runner import Runner
    from behave.configuration import Configuration
    
    config = Configuration()
    config.paths = ["features/"]
    runner = Runner(config)
    return runner.run()

def run_generated_tests(api_key=None, requirement=None):
    """运行自动生成的测试用例"""
    if not api_key:
        print("未提供API密钥，将尝试从环境变量加载")
    
    # 如果未提供需求，使用默认需求
    if not requirement:
        requirement = "测试游戏中的武器系统，包括准星显示和弹药计数功能"
    
    # 创建测试执行器
    executor = TestExecutor(api_key)
    
    # 开始计时
    start_time = time.time()
    
    # 生成并执行测试用例
    result = executor.execute_from_requirement(requirement)
    
    # 计算生成时间
    generation_time = time.time() - start_time
    
    # 生成报告
    report = executor.generate_report(result, generation_time)
    
    print(f"测试完成！结果：{report}")
    return report

def run_batch_tests(api_key=None, requirement=None, count=5):
    """运行批量生成的测试用例"""
    if not api_key:
        print("未提供API密钥，将尝试从环境变量加载")
    
    # 如果未提供需求，使用默认需求
    if not requirement:
        requirement = "测试游戏中的武器系统，包括准星显示、弹药计数和武器切换功能"
    
    # 创建测试执行器
    executor = TestExecutor(api_key)
    
    # 开始计时
    start_time = time.time()
    
    # 生成并执行测试用例
    result = executor.execute_batch(requirement, count)
    
    # 计算生成时间
    generation_time = time.time() - start_time
    
    # 生成报告
    report = executor.generate_report(result, generation_time)
    
    print(f"批量测试完成！结果：{report}")
    return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gherkin测试自动化脚本")
    parser.add_argument("--api-key", dest="api_key", help="OpenAI API密钥")
    parser.add_argument("--req", dest="requirement", help="测试需求描述")
    parser.add_argument("--count", type=int, default=5, help="批量生成的测试用例数量")
    parser.add_argument("--mode", choices=["predefined", "generated", "batch"], 
                        default="generated", help="测试模式：预定义/生成单个/批量生成")
    
    args = parser.parse_args()
    
    # 使用命令行参数或环境变量中的API密钥
    api_key = args.api_key or API_KEY
    
    if args.mode == "predefined":
        print("运行预定义测试用例...")
        result = run_predefined_tests()
    elif args.mode == "generated":
        print("生成并运行单个测试用例...")
        result = run_generated_tests(api_key, args.requirement)
    elif args.mode == "batch":
        print(f"生成并运行{args.count}个测试用例...")
        result = run_batch_tests(api_key, args.requirement, args.count)
    
    sys.exit(0 if result else 1) 