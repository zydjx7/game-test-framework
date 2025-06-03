#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gherkin测试自动化脚本
使用LLM自动生成并执行Gherkin格式的游戏测试用例
"""

import os
import sys
import time
import subprocess
import signal
import atexit
from test_generator.test_executor import TestExecutor
from dotenv import load_dotenv
from loguru import logger

# 确保logs目录存在
logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# 设置日志
logger.add(os.path.join(logs_dir, "run_tests.log"), level="DEBUG", rotation="1 MB")

# 加载环境变量中的API密钥
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# 启动Flask服务器
def start_flask_server(target=None):
    """启动视觉检测服务器"""
    logger.info("正在启动Flask视觉检测服务器...")
    server_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'GameStateChecker/main_flask_server.py')
    
    # 确保日志目录存在
    flask_logs_dir = os.path.join(logs_dir, "flask")
    os.makedirs(flask_logs_dir, exist_ok=True)
    
    try:
        # 检查服务器是否已经在运行
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 5000))
        sock.close()
        
        if result == 0:
            logger.info("Flask服务器似乎已经在运行，端口5000被占用")
            return None
            
        # 启动服务器进程，重定向日志到文件，如果指定了target则传递环境变量
        env = os.environ.copy()
        if target:
            env["GAMECHECK_TARGET"] = target
            logger.info(f"设置测试目标为: {target}")
            
        server = subprocess.Popen(
            [sys.executable, server_path],
            stdout=open(os.path.join(flask_logs_dir, "flask_stdout.log"), "w"),
            stderr=open(os.path.join(flask_logs_dir, "flask_stderr.log"), "w"),
            env=env
        )
        
        logger.info(f"Flask服务器启动 (PID: {server.pid})")
        
        # 注册关闭函数
        def cleanup():
            logger.info("正在关闭Flask服务器...")
            if server.poll() is None:  # 如果进程还在运行
                server.send_signal(signal.SIGTERM)
                server.wait(timeout=5)  # 等待最多5秒
                if server.poll() is None:  # 如果还没有退出
                    server.kill()  # 强制终止
            logger.info("Flask服务器已关闭")
        
        atexit.register(cleanup)
        
        # 给服务器更多的启动时间
        wait_time = 5  # 增加到5秒
        logger.info(f"等待Flask服务器启动 ({wait_time}秒)...")
        time.sleep(wait_time)
        
        # 确认服务器是否正常启动
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 5000))
        sock.close()
        
        if result != 0:
            logger.error("Flask服务器似乎没有正常启动，端口5000未被占用")
            # 检查服务器是否已经退出
            if server.poll() is not None:
                logger.error(f"Flask服务器进程已退出，退出码: {server.returncode}")
                logger.error("请查看日志文件获取更多信息: " + os.path.join(flask_logs_dir, "flask_stderr.log"))
                return None
        else:
            logger.info("Flask服务器已成功监听端口5000")
        
        return server
    
    except Exception as e:
        logger.error(f"启动Flask服务器失败: {e}")
        return None

# 1. 预定义测试模式
def run_predefined_tests(target=None, feature_file=None):
    """运行预定义的测试用例"""
    # 启动Flask服务器
    server = start_flask_server(target)
    
    from behave.runner import Runner
    from behave.configuration import Configuration
    
    # 获取当前目录作为根目录
    root = os.path.dirname(os.path.abspath(__file__))
    
    # 使用空列表作为command_args，避免Behave解析sys.argv
    config = Configuration(command_args=[])
    
    # 设置正确的特性文件和格式
    if feature_file:
        # 如果指定了特定的feature文件
        feature_path = os.path.join(root, "features", feature_file)
        if os.path.exists(feature_path):
            config.paths = [feature_path]
            logger.info(f"使用特定的feature文件: {feature_path}")
        else:
            logger.error(f"指定的feature文件不存在: {feature_path}")
            return False
    else:
        # 使用整个features目录
        config.paths = [os.path.join(root, "features")]
        logger.info(f"使用特性目录: {config.paths[0]}")
    
    config.format = ["pretty"]
    
    # 设置环境变量以便传递target给steps文件
    if target:
        os.environ["GAMECHECK_TARGET"] = target
    
    runner = Runner(config)
    result = runner.run()
    
    return result == 0  # 返回布尔值表示成功/失败

# 2. 生成单个测试模式
def run_generated_tests(api_key=None, requirement=None, target=None):
    """运行自动生成的测试用例"""
    if not api_key:
        logger.info("未提供API密钥，将尝试从环境变量加载")
    
    # 如果未提供需求，使用默认需求
    if not requirement:
        requirement = "测试游戏中的武器系统，包括准星显示和弹药计数功能"
    
    # 启动Flask服务器
    server = start_flask_server(target)
    
    # 创建测试执行器
    executor = TestExecutor(api_key, target=target)
    
    # 开始计时
    start_time = time.time()
    
    # 生成并执行测试用例
    result = executor.execute_from_requirement(requirement)
    
    # 计算生成时间
    generation_time = time.time() - start_time
    
    # 生成报告
    report = executor.generate_report(result, generation_time)
    
    logger.info(f"测试完成！结果：{report}")
    print(f"测试完成！结果：{report}")
    
    # 返回布尔值以便于sys.exit()使用
    return report["status"] == "success"

# 3. 批量生成测试模式
def run_batch_tests(api_key=None, requirement=None, count=5, target=None):
    """运行批量生成的测试用例"""
    if not api_key:
        logger.info("未提供API密钥，将尝试从环境变量加载")
    
    # 如果未提供需求，使用默认需求
    if not requirement:
        requirement = "测试游戏中的武器系统，包括准星显示、弹药计数和武器切换功能"
    
    # 启动Flask服务器
    server = start_flask_server(target)
    
    # 创建测试执行器
    executor = TestExecutor(api_key, target=target)
    
    # 开始计时
    start_time = time.time()
    
    # 生成并执行测试用例
    result = executor.execute_batch(requirement, count)
    
    # 计算生成时间
    generation_time = time.time() - start_time
    
    # 生成报告
    report = executor.generate_report(result, generation_time)
    
    logger.info(f"批量测试完成！结果：{report}")
    print(f"批量测试完成！结果：{report}")
    
    # 返回布尔值以便于sys.exit()使用
    return report["status"] == "success"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gherkin测试自动化脚本")
    parser.add_argument("--api-key", dest="api_key", help="OpenAI API密钥")
    parser.add_argument("--req", dest="requirement", help="测试需求描述")
    parser.add_argument("--count", type=int, default=5, help="批量生成的测试用例数量")
    parser.add_argument("--target", dest="target", help="设置测试目标，如'p1_legacy'或'assaultcube'")
    parser.add_argument("--mode", choices=["predefined", "generated", "batch"], 
                        default="predefined", help="测试模式：预定义/生成单个/批量生成")
    parser.add_argument("--feature", dest="feature_file", help="指定要运行的feature文件(仅在predefined模式下有效)")
    parser.add_argument("--use-llm-analysis", action="store_true", help="是否在测试中使用LLM进行准星分析")
    
    args = parser.parse_args()
    
    # 使用命令行参数或环境变量中的API密钥
    api_key = args.api_key or API_KEY
    
    # 设置是否使用LLM进行准星分析的环境变量
    if args.use_llm_analysis:
        os.environ["USE_LLM_ANALYSIS"] = "true"
    else:
        os.environ["USE_LLM_ANALYSIS"] = "false"
    
    if args.mode == "predefined":
        print("运行预定义测试用例...")
        result = run_predefined_tests(args.target, args.feature_file)
    elif args.mode == "generated":
        print("生成并运行单个测试用例...")
        result = run_generated_tests(api_key, args.requirement, args.target)
    elif args.mode == "batch":
        print(f"生成并运行{args.count}个测试用例...")
        result = run_batch_tests(api_key, args.requirement, args.count, args.target)
    
    sys.exit(0 if result else 1)