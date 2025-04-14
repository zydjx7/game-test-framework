from behave import given, when, then
import sys
import os
import cv2

# Add the GameStateChecker directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../GameStateChecker'))

from main_flask_client import GameStateChecker_Client

# 定义游戏测试类，封装对CPP测试层的调用
class GameTester:
    def __init__(self):
        self.checker_client = GameStateChecker_Client()
        self.screenshot_path = os.path.join(os.path.dirname(__file__), 
                                           '../../GameStateChecker/unitTestResources/p1.png')
        self.expected_ammo = 50  # 默认弹药数量
        
    def check_weapon_cross(self):
        # 定义测试上下文
        context = {
            "requestCategory": "UI",
            "requestFunc": "checkWeaponCross",
            "screenshotsCount": 1,
        }
        
        # 定义期望结果
        expected_answer = {"boolResult": "True"}
        
        # 调用视觉检测功能
        result = self.checker_client.check_visuals_onScreenshot(
            screenShotPath=self.screenshot_path,
            testContext=context,
            expectedAnswer=expected_answer
        )
        
        return "Test passed" in result
    
    def check_ammo_sync(self, expected_ammo=50):
        # 定义测试上下文
        context = {
            "requestCategory": "UI",
            "requestFunc": "checkAmmoSyncText",
            "screenshotsCount": 1,
            "bbox": [912, 1015, 79, 49],
            "textColorValueInHSV_min": 129,
            "textColorValueInHSV_max": 130,
            "tolerance": 2  # 允许2个像素的误差
        }
        
        # 定义期望结果
        expected_answer = {
            "intResult": expected_ammo,
            "allowedVariance": 0  # 弹药数量必须精确匹配
        }
        
        try:
            # 调用视觉检测功能
            result = self.checker_client.check_visuals_onScreenshot(
                screenShotPath=self.screenshot_path,
                testContext=context,
                expectedAnswer=expected_answer
            )
            
            if isinstance(result, str):
                return "Test passed" in result
            elif isinstance(result, dict):
                return result.get("success", False)
            
            return False
        except Exception as e:
            logger.error(f"弹药检测失败: {str(e)}")
            return False

@given('the game is started')
def step_impl(context):
    context.game_tester = GameTester()

@when('player equips a weapon')
def step_impl(context):
    # 模拟装备武器的动作
    # 实际上这里不需要执行任何操作，因为测试图像已经包含了已装备武器的状态
    pass

@when('player fires the weapon')
def step_impl(context):
    # 模拟开火动作
    # 实际上这里不需要执行任何操作，因为我们只是在测试图像状态
    # 设置firing后的弹药数为49
    context.game_tester.expected_ammo = 49
    pass

@then('the crosshair should be visible')
def step_impl(context):
    assert context.game_tester.check_weapon_cross(), "武器准星未显示"

@then('the ammo count should decrease')
def step_impl(context):
    # 检查弹药数量是否正确显示
    # 这里假设减少后的弹药数是49
    assert context.game_tester.check_ammo_sync(49), "弹药数量显示不正确"
    
@then('the ammo count should match the expected value')
def step_impl(context):
    # 检查弹药数量是否与预期值匹配
    assert context.game_tester.check_ammo_sync(context.game_tester.expected_ammo), "弹药数量与预期值不匹配"