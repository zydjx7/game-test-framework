import pytest
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# 导入测试所需的组件
from src.llm.gpt_client import GPTClient
from src.gherkin.parser import GherkinParser
from src.rivergame.test_runner import RiverGameTestRunner
from src.rivergame.mock_server import MockGameServer
from src.utils.config import ConfigManager

@pytest.fixture(scope="session")
def event_loop():
    """创建一个事件循环，供所有测试使用"""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def mock_game_server(event_loop):
    """启动模拟游戏服务器"""
    server, runner = await MockGameServer.create_and_start()
    yield server
    await runner.cleanup()

@pytest.fixture(scope="function")
async def config_manager():
    """配置管理器实例"""
    return ConfigManager()

@pytest.fixture(scope="function")
async def gpt_client(config_manager):
    """GPT客户端实例"""
    return GPTClient()

@pytest.fixture(scope="function")
async def gherkin_parser():
    """Gherkin解析器实例"""
    return GherkinParser()

@pytest.fixture(scope="function")
async def test_runner(mock_game_server):
    """测试运行器实例"""
    runner = RiverGameTestRunner()
    await runner.initialize()
    yield runner
    await runner.cleanup()