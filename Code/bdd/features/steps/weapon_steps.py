from behave import given, when, then
import sys
import os
import cv2
import json
from loguru import logger

# 确保logs目录存在
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# 设置日志
logger.add(os.path.join(logs_dir, "weapon_steps.log"), level="DEBUG", rotation="1 MB")

# 使用绝对路径添加GameStateChecker目录
game_checker_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', 'GameStateChecker'))
sys.path.append(game_checker_path)
logger.info(f"已添加GameStateChecker路径: {game_checker_path}")

# 在导入前显示Python路径和目录内容，帮助调试
logger.debug(f"Python路径: {sys.path}")
logger.debug(f"GameStateChecker目录内容: {os.listdir(game_checker_path) if os.path.exists(game_checker_path) else '目录不存在'}")

# 尝试导入LogicLayer以获取配置信息
try:
    from LogicLayer import LogicLayer
    logger.info("成功导入LogicLayer")
except Exception as e:
    logger.exception(f"导入LogicLayer失败: {str(e)}")

# 尝试直接从绝对路径导入
import importlib.util
try:
    client_path = os.path.join(game_checker_path, 'main_flask_client.py')
    if os.path.exists(client_path):
        logger.info(f"主动加载模块: {client_path}")
        spec = importlib.util.spec_from_file_location("main_flask_client", client_path)
        main_flask_client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_flask_client)
        GameStateChecker_Client = main_flask_client.GameStateChecker_Client
        logger.info("成功导入GameStateChecker_Client")
    else:
        logger.error(f"主动加载失败，文件不存在: {client_path}")
        raise ImportError(f"文件不存在: {client_path}")
except Exception as e:
    logger.exception(f"导入GameStateChecker_Client失败: {str(e)}")
    raise

# 定义游戏测试类，封装对CPP测试层的调用
class GameTester:
    def __init__(self):
        logger.info("初始化GameTester")
        self.checker_client = GameStateChecker_Client()
        
        # 从环境变量获取测试目标，默认为'assaultcube'
        self.target_name = os.getenv("GAMECHECK_TARGET", "assaultcube")
        logger.info(f"当前测试目标: {self.target_name}")
        
        # 初始化LogicLayer来获取目标特定的配置
        self.logic_layer = LogicLayer(target_name=self.target_name)
        
        # 使用LogicLayer的get_test_image_path方法获取适当的截图路径
        if self.target_name == 'p1_legacy':
            self.screenshot_path = self.logic_layer.get_test_image_path()
        else:
            # 对于AssaultCube等其他目标，尝试获取有效的测试图像
            self.screenshot_path = self.logic_layer.get_test_image_path(category='Crosshair', file_name='cross_normal.png')
            if not self.screenshot_path:
                self.screenshot_path = self.logic_layer.get_test_image_path(category='Ammo', file_name='ammo_clip30_total90.png')
        
        logger.debug(f"使用测试目标: {self.target_name}, 路径: {self.screenshot_path}")
        
        # 验证截图文件是否存在
        if not self.screenshot_path or not os.path.exists(self.screenshot_path):
            logger.error(f"截图文件不存在: {self.screenshot_path}")
            # 尝试查找替代截图
            test_resources_dir = os.path.join(game_checker_path, 'test_images')
            if not os.path.exists(test_resources_dir):
                logger.warning(f"测试资源目录不存在: {test_resources_dir}")
                test_resources_dir = os.path.join(game_checker_path, 'unitTestResources')
                if not os.path.exists(test_resources_dir):
                    os.makedirs(test_resources_dir, exist_ok=True)
                    logger.warning(f"创建了测试资源目录: {test_resources_dir}")
            
            # 尝试查找其他可用的截图
            alt_screenshots = []
            for root, dirs, files in os.walk(test_resources_dir):
                for file in files:
                    if file.endswith('.png'):
                        alt_screenshots.append(os.path.join(root, file))
            
            if alt_screenshots:
                self.screenshot_path = alt_screenshots[0]
                logger.info(f"使用替代截图: {self.screenshot_path}")
            else:
                logger.error("没有找到可用的截图文件")
                # 最后尝试使用默认P1路径
                self.screenshot_path = os.path.join(game_checker_path, 'unitTestResources', 'p1.png')
        
        # 获取目标特定的参数
        self.expected_ammo = 50  # 默认弹药数量
        if self.target_name == 'assaultcube':
            self.expected_ammo = 30  # AssaultCube默认弹药数
        
        logger.debug(f"GameTester初始化完成，使用截图: {self.screenshot_path}")
        
    def check_weapon_cross(self):
        logger.info("检查武器准星")
        
        # 定义测试上下文
        context = {
            "requestCategory": "UI",
            "requestFunc": "checkWeaponCross",
            "screenshotsCount": 1,
        }
        
        # 定义期望结果
        expected_answer = {"boolResult": "True"}
        
        try:
            # 调用视觉检测功能
            result = self.checker_client.check_visuals_onScreenshot(
                screenShotPath=self.screenshot_path,
                testContext=context,
                expectedAnswer=expected_answer
            )
            
            # 处理JSON响应
            logger.debug(f"准星检测原始响应: {result}")
            try:
                if isinstance(result, str):
                    result_dict = json.loads(result)
                    success = result_dict.get("result", False)
                    logger.info(f"准星检测结果: {success}")
                    return success
                elif isinstance(result, dict):
                    success = result.get("result", False)
                    logger.info(f"准星检测结果: {success}")
                    return success
                logger.warning("未知的响应格式")
                return False
            except json.JSONDecodeError:
                logger.error(f"无法解析JSON响应: {result}")
                # 兼容旧格式
                success = "Test passed" in result
                logger.info(f"准星检测结果(旧格式): {success}")
                return success
        except Exception as e:
            logger.error(f"准星检测过程中出错: {e}")
            return False
    
    def check_ammo_sync(self, expected_ammo=None):
        if expected_ammo is None:
            expected_ammo = self.expected_ammo
            
        logger.info(f"检查弹药同步，期望值: {expected_ammo}")
        
        # 获取目标特定的bbox和HSV参数
        cv_params = self.logic_layer.target_config.get('cv_params', {})
        ammo_bbox = cv_params.get('ammo_bbox', [912, 1015, 79, 49])  # 默认值为p1_legacy的bbox
        hsv_min = cv_params.get('ammo_hsv_hue_min', 129)
        hsv_max = cv_params.get('ammo_hsv_hue_max', 130)
        
        logger.debug(f"使用bbox: {ammo_bbox}, HSV范围: {hsv_min}-{hsv_max}")
        
        # 定义测试上下文
        context = {
            "requestCategory": "UI",
            "requestFunc": "checkAmmoSyncText",
            "screenshotsCount": 1,
            "bbox": ammo_bbox,
            "textColorValueInHSV_min": hsv_min,
            "textColorValueInHSV_max": hsv_max,
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
            
            # 处理JSON响应
            logger.debug(f"弹药同步检测原始响应: {result}")
            try:
                if isinstance(result, str):
                    result_dict = json.loads(result)
                    success = result_dict.get("result", False)
                    logger.info(f"弹药同步检测结果: {success}")
                    return success
                elif isinstance(result, dict):
                    success = result.get("result", False)
                    logger.info(f"弹药同步检测结果: {success}")
                    return success
                logger.warning("未知的响应格式")
                return False
            except json.JSONDecodeError:
                logger.error(f"无法解析JSON响应: {result}")
                # 兼容旧格式
                success = "Test passed" in result
                logger.info(f"弹药同步检测结果(旧格式): {success}")
                return success
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
    # 设置firing后的弹药数为当前预期值减1
    context.game_tester.expected_ammo -= 1
    pass

@then('the crosshair should be visible')
def step_impl(context):
    assert context.game_tester.check_weapon_cross(), "武器准星未显示"

@then('the ammo count should decrease')
def step_impl(context):
    # 检查弹药数量是否正确显示
    assert context.game_tester.check_ammo_sync(context.game_tester.expected_ammo), "弹药数量显示不正确"
    
@then('the ammo count should match the expected value')
def step_impl(context):
    # 检查弹药数量是否与预期值匹配
    assert context.game_tester.check_ammo_sync(context.game_tester.expected_ammo), "弹药数量与预期值不匹配"

@then('the ammo count should be 50')
def step_impl(context):
    assert context.game_tester.check_ammo_sync(50), "弹药数量不是50"