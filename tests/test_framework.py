import pytest
import asyncio
from src.llm.gpt_client import GPTClient
from src.gherkin.parser import GherkinParser
from src.rivergame.test_runner import RiverGameTestRunner
from src.rivergame.mock_server import MockGameServer
from src.utils.config import ConfigManager

pytestmark = pytest.mark.asyncio

@pytest.fixture(scope="session")
async def mock_game_server():
    """启动模拟游戏服务器"""
    server, runner = await MockGameServer.create_and_start()
    yield server
    await runner.cleanup()

@pytest.fixture(scope="function")
async def config_manager():
    return ConfigManager()

@pytest.fixture(scope="function")
async def gpt_client(config_manager):
    return GPTClient()

@pytest.fixture(scope="function")
async def gherkin_parser():
    return GherkinParser()

@pytest.fixture(scope="function")
async def test_runner(mock_game_server):
    runner = RiverGameTestRunner()
    await runner.initialize()
    yield runner
    await runner.cleanup()

async def test_gherkin_generation(gpt_client: GPTClient, gherkin_parser: GherkinParser):
    """测试Gherkin场景生成"""
    # 测试场景描述
    game_description = """
    RiverGame是一个简单的过河游戏：
    - 玩家需要控制角色过河
    - 河面上有多个浮动的平台
    - 玩家可以在平台间跳跃
    - 如果玩家掉入水中则游戏失败
    - 到达对岸则游戏胜利
    """
    
    # 生成测试场景
    gherkin_content = await gpt_client.generate_test_scenario(game_description)
    assert gherkin_content is not None, "生成的Gherkin内容不能为空"
    print(f"\n生成的Gherkin内容:\n{gherkin_content}")  # 添加调试输出
    
    # 验证生成的内容
    assert gherkin_parser.validate_gherkin(gherkin_content), "生成的Gherkin内容无效"
    
    # 解析测试场景
    feature = gherkin_parser.parse_feature(gherkin_content)
    assert feature is not None, "无法解析Gherkin内容"
    assert len(feature['scenarios']) > 0, "没有生成测试场景"

async def test_game_state_verification(test_runner: RiverGameTestRunner, mock_game_server):
    """测试游戏状态验证"""
    # 测试场景
    test_scenario = {
        'name': '基本游戏流程测试',
        'steps': [
            {
                'type': 'Given',
                'content': '游戏已启动且玩家位于起始位置'
            },
            {
                'type': 'When',
                'content': '玩家跳跃到第一个平台'
            },
            {
                'type': 'Then',
                'content': '玩家应该成功站在第一个平台上'
            }
        ]
    }
    
    # 执行测试场景
    result = await test_runner.execute_test_scenario(test_scenario)
    assert result['success'], f"测试执行失败: {result.get('error', '未知错误')}"
    
    # 验证游戏状态
    game_state = mock_game_server.game_state
    assert game_state['player_position']['y'] > 0, "玩家没有成功跳跃"

async def test_config_validation(config_manager: ConfigManager):
    """测试配置验证"""
    # 验证配置完整性
    assert config_manager.validate_config(), "配置验证失败"
    
    # 验证必要配置项
    assert config_manager.get('api.openai.api_key') is not None, "缺少API密钥配置"
    assert config_manager.get('rivergame.host') is not None, "缺少游戏服务器主机配置"
    assert config_manager.get('rivergame.port') is not None, "缺少游戏服务器端口配置"

@pytest.mark.asyncio
async def test_mock_server_functionality(mock_game_server):
    """测试模拟服务器功能"""
    # 验证初始状态
    initial_state = mock_game_server.game_state
    assert initial_state['game_status'] == 'ready', "游戏初始状态不正确"
    assert initial_state['player_position']['x'] == 0, "玩家初始X坐标不正确"
    assert initial_state['player_position']['y'] == 0, "玩家初始Y坐标不正确"
    
    # 模拟状态变化
    mock_game_server.game_state['player_position']['x'] = 100
    mock_game_server.game_state['player_position']['y'] = 50
    
    # 验证状态更新
    updated_state = mock_game_server.game_state
    assert updated_state['player_position']['x'] == 100, "玩家X坐标更新失败"
    assert updated_state['player_position']['y'] == 50, "玩家Y坐标更新失败"