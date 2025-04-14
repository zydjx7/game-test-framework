import pytest
from src.gherkin.parser import GherkinParser
from src.rivergame.ammo_test_runner import AmmoTestRunner
import os
from loguru import logger
import cv2
import numpy as np

@pytest.fixture(scope="module")
def setup_test_environment():
    """设置测试环境"""
    logger.add("logs/test_debug.log", level="DEBUG", rotation="1 MB")
    logger.info("初始化测试环境")
    
    # 检查测试资源
    resources_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Code/GameStateChecker')
    assert os.path.exists(resources_path), f"测试资源目录不存在: {resources_path}"
    
    # 返回测试配置
    return {
        "resources_path": resources_path,
        "default_bbox": [912, 1015, 79, 49],
        "hsv_range": (129, 130)
    }

@pytest.mark.asyncio
async def test_basic_ammo_sync(setup_test_environment):
    """测试基本的弹药同步功能"""
    logger.info("开始执行基本弹药同步测试")
    
    # 准备测试场景
    gherkin_content = """Feature: Test ammo sync
  Scenario: Check ammo display
    Given the player has 30 bullets
    When the player shoots once
    Then the ammo displayed should be 29"""
    
    # 解析并执行测试
    try:
        feature = GherkinParser().parse_feature(gherkin_content)
        assert feature is not None, "Gherkin解析失败"
        
        runner = AmmoTestRunner()
        runner.setup_vision_params(
            bbox=setup_test_environment["default_bbox"],
            hsv_min=setup_test_environment["hsv_range"][0],
            hsv_max=setup_test_environment["hsv_range"][1]
        )
        
        results = await runner.execute_scenario(feature['scenarios'][0])
        
        # 详细记录执行结果
        for step_result in results['steps_results']:
            logger.info(f"{step_result['type']}: {step_result['message']}")
            if not step_result['success']:
                logger.error(f"步骤失败: {step_result.get('error', '未知错误')}")
        
        assert results['success'], f"测试执行失败: {results.get('error', '未知错误')}"
        
    except Exception as e:
        logger.exception("测试执行过程中发生错误")
        raise

@pytest.mark.asyncio
async def test_ammo_sync_with_multiple_shots(setup_test_environment):
    """测试多次射击场景"""
    logger.info("开始测试多次射击场景")
    
    gherkin_content = """Feature: Test ammo sync
  Scenario: Check ammo display with multiple shots
    Given the player has 15 bullets
    When the player shoots 5 times
    Then the ammo displayed should be 10"""
    
    try:
        feature = GherkinParser().parse_feature(gherkin_content)
        assert feature is not None, "Gherkin解析失败"
        
        runner = AmmoTestRunner()
        results = await runner.execute_scenario(feature['scenarios'][0])
        
        # 验证结果
        for step_result in results['steps_results']:
            logger.info(f"{step_result['type']}: {step_result['message']}")
            if not step_result['success']:
                logger.error(f"步骤失败详情: {step_result}")
                
        assert results['success'], f"多次射击测试失败: {results.get('error', '未知错误')}"
        
    except Exception as e:
        logger.exception("多次射击测试过程中发生错误")
        raise

@pytest.mark.asyncio
async def test_ammo_sync_with_edge_cases(setup_test_environment):
    """测试边界情况"""
    test_cases = [
        (0, "空弹夹测试"),
        (1, "单发子弹测试"),
        (999, "超大弹药数测试")
    ]
    
    for ammo_count, case_name in test_cases:
        logger.info(f"开始执行边界测试: {case_name}")
        
        gherkin_content = f"""Feature: Test ammo sync edge cases
  Scenario: {case_name}
    Given the player has {ammo_count} bullets
    Then the ammo displayed should be {ammo_count}"""
            
        try:
            feature = GherkinParser().parse_feature(gherkin_content)
            runner = AmmoTestRunner()
            results = await runner.execute_scenario(feature['scenarios'][0])
            
            assert results['success'], f"{case_name}失败: {results.get('error', '未知错误')}"
            logger.info(f"边界测试通过: {case_name}")
            
        except Exception as e:
            logger.error(f"边界测试失败 {case_name}: {str(e)}")
            raise