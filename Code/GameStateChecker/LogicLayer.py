import os
import sys  # 添加 sys 模块导入
import time
import glob
from typing import List, Tuple, Dict
from VisionUtils import VisionUtils
import cv2
import re
import numpy as np
from loguru import logger
import yaml

class LogicLayer:
    def __init__(self, target_name=None, config=None):
        """
        初始化逻辑层
        @param target_name: 测试目标名称(如"p1_legacy"或"assaultcube")
        @param config: 配置字典
        """
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"初始化LogicLayer，基础路径: {self.base_path}")
        
        # 加载配置信息
        self.config = config
        if self.config is None:
            self.load_config()
            
        # 设置目标
        self.target_name = target_name or self.config.get('active_target', 'p1_legacy')
        logger.info(f"当前测试目标: {self.target_name}")
        
        # 获取目标特定的配置
        self.target_config = self.config.get('targets', {}).get(self.target_name, {})
        if not self.target_config:
            logger.warning(f"未找到目标 '{self.target_name}' 的配置，使用默认设置")
            self.target_config = {
                "screenshot_type": "static",
                "screenshot_path": "unitTestResources/p1.png",
                "cv_params": {
                    "crosshair_template": "unitTestResources/Cross_p.png",
                    "min_keypoints": 3,
                    "template_threshold": 0.6,
                    "ammo_bbox": [912, 1015, 79, 49],
                    "ammo_hsv_min": 129,
                    "ammo_hsv_max": 130
                }
            }
            
        self.preloadData()

    def load_config(self):
        """加载配置文件"""
        config_path = os.path.join(self.base_path, "config.yaml")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                logger.info(f"成功加载配置文件: {config_path}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            self.config = {
                "active_target": "p1_legacy",
                "targets": {
                    "p1_legacy": {
                        "screenshot_type": "static",
                        "screenshot_path": "unitTestResources/p1.png",
                        "cv_params": {
                            "crosshair_template": "unitTestResources/Cross_p.png",
                            "min_keypoints": 3,
                            "template_threshold": 0.6,
                            "ammo_bbox": [912, 1015, 79, 49],
                            "ammo_hsv_min": 129,
                            "ammo_hsv_max": 130
                        }
                    }
                }
            }

    def __convertScreenshotToGray(self, screenshot):
        """转换截图为灰度图像"""
        try:
            if len(screenshot.shape) > 2:  # BGR ?
                screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            else:
                screenshot_gray = screenshot
            return screenshot_gray
        except Exception as e:
            logger.error(f"转换图像失败: {e}")
            return None

    def preloadData(self, cv_params=None):
        """
        预加载所需数据
        @param cv_params: 视觉参数字典，如果提供则覆盖自动加载的参数
        """
        try:
            # 使用传入的参数或从配置中获取
            if cv_params is None:
                cv_params = self.target_config.get('cv_params', {})
                
            # 从配置中获取模板路径
            template_rel_path = cv_params.get('crosshair_template', "unitTestResources/Cross_p.png")
            template_path = os.path.join(self.base_path, template_rel_path)
            logger.debug(f"加载模板图像: {template_path}")
            
            if os.path.exists(template_path):
                self.templateImg_weaponCross = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if self.templateImg_weaponCross is None:
                    logger.error("无法读取武器准星模板图像")
            else:
                logger.error(f"模板图像不存在: {template_path}")
                self.templateImg_weaponCross = None
        except Exception as e:
            logger.error(f"预加载数据失败: {e}")
            self.templateImg_weaponCross = None

    def get_target_specific_param(self, param_name, default=None):
        """获取特定目标的参数"""
        return self.target_config.get('cv_params', {}).get(param_name, default)

    def testWeaponCrossPresence(self, screenshots: List[any],
                               context: Dict[any, any],
                               expectedAnswer: Dict[any, any],
                               cv_params=None,
                               debugEnabled=False,
                               debug_dir=None) -> bool:
        """
        测试武器准星是否存在
        @param screenshots: 截图列表
        @param context: 上下文信息
        @param expectedAnswer: 期望的答案
        @param cv_params: 视觉参数字典，优先级高于默认值
        @param debugEnabled: 是否启用调试
        @param debug_dir: 调试图像保存目录
        @return: bool - 是否检测到准星
        """
        try:
            if not screenshots or len(screenshots) == 0:
                logger.error("没有提供截图")
                return False
                
            screenshot_gray = self.__convertScreenshotToGray(screenshots[0])
            if screenshot_gray is None:
                logger.error("转换截图失败")
                return False
                
            # 修复：如果未加载模板，则根据target_config加载 - 不受cv_params参数的影响
            if self.templateImg_weaponCross is None:
                # 获取默认配置，如果未提供cv_params
                if cv_params is None:
                    cv_params = self.target_config.get('cv_params', {})
                    
                template_rel_path = cv_params.get('crosshair_template')
                if template_rel_path:
                    template_path = os.path.join(self.base_path, template_rel_path)
                    if os.path.exists(template_path):
                        self.templateImg_weaponCross = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                        logger.debug(f"懒加载准星模板: {template_path}")
            
            # 如果传入了cv_params中包含新的模板路径，也加载新模板替换原有的
            if cv_params and cv_params.get('crosshair_template'):
                template_rel_path = cv_params.get('crosshair_template')
                template_path = os.path.join(self.base_path, template_rel_path)
                if os.path.exists(template_path):
                    self.templateImg_weaponCross = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                    logger.debug(f"使用指定的准星模板: {template_path}")
            
            if self.templateImg_weaponCross is None:
                logger.error("武器准星模板未加载")
                return False
            
            # 提取ROI区域(屏幕中心区域)
            h, w = screenshot_gray.shape[:2]
            cx, cy = w // 2, h // 2
            
            # 从配置参数获取ROI大小
            if cv_params is None:
                cv_params = self.target_config.get('cv_params', {})
                
            roi_size = cv_params.get('roi_size')
            if roi_size is None:
                roi_factor = cv_params.get('roi_factor', 0.25)  # 默认为短边的25%
                roi_size = int(min(w, h) * roi_factor)
                roi_size = max(150, roi_size)  # 设置最小值
            
            y1, y2 = max(0, cy - roi_size // 2), min(h, cy + roi_size // 2)
            x1, x2 = max(0, cx - roi_size // 2), min(w, cx + roi_size // 2)
            roi_gray = screenshot_gray[y1:y2, x1:x2]
            
            if debugEnabled:
                logger.debug(f"提取ROI区域: 中心({cx},{cy}), 大小({x2-x1}x{y2-y1}), " +
                             f"使用{'固定ROI大小' if cv_params.get('roi_size') else '动态ROI系数'}")
                
                # 使用传入的调试目录保存ROI图像
                debug_save_dir = debug_dir if debug_dir else os.path.join(self.base_path, "debug")
                os.makedirs(debug_save_dir, exist_ok=True)
                
                # 构造文件名后缀
                filename_suffix = ""
                screenshot_path = context.get("screenshotFile", "")
                if screenshot_path:
                    filename_suffix = os.path.splitext(os.path.basename(screenshot_path))[0]
                else:
                    filename_suffix = f"img_{int(time.time())}"
                
                cv2.imwrite(os.path.join(debug_save_dir, f'cross_roi_{filename_suffix}.png'), roi_gray)
                logger.debug(f"已保存准星ROI区域图像: {os.path.join(debug_save_dir, f'cross_roi_{filename_suffix}.png')}")
            
            min_keypoints = cv_params.get('min_keypoints', 3)
            lowe_ratio = cv_params.get('lowe_ratio', 0.75)
            template_threshold = cv_params.get('template_threshold', 0.7)  # 降低阈值为0.7
            
            # 使用正确的参数名（minKeypoints 而不是 min_keypoints）并传递调试目录
            good_matches, min_keypoints_required = VisionUtils.matchTemplateImg(
                img_src=self.templateImg_weaponCross,
                img_target=roi_gray,
                minKeypoints=min_keypoints,
                useORB=False,
                useSIFT=True,
                debugEnabled=debugEnabled,
                debug_dir=debug_dir if debug_dir else os.path.join(self.base_path, "debug")
            )
            
            # 检查匹配结果
            isMatching = len(good_matches) >= min_keypoints_required
            
            if debugEnabled:
                logger.debug(f"准星匹配结果: {isMatching}, 找到特征点: {len(good_matches)}/{min_keypoints_required}")
                
            # 直接返回匹配结果，不再与期望值比较
            return isMatching
            
        except Exception as e:
            logger.error(f"测试武器准星失败: {e}")
            return False

    def testAmmoTextInSync(self, screenshots: List[any],
                          context: Dict[any, any],
                          expectedAnswer: Dict[any, any],
                          cv_params=None,
                          debugEnabled=False,
                          debug_dir=None) -> bool:
        """
        测试弹药文本同步
        @param screenshots: 截图列表
        @param context: 上下文信息
        @param expectedAnswer: 期望的答案
        @param cv_params: 视觉参数字典
        @param debugEnabled: 是否启用调试
        @param debug_dir: 调试图像保存目录
        @return: bool - 弹药文本是否同步
        """
        try:
            # 提取预期弹药数
            expected_ammo = int(expectedAnswer.get("intResult", -1))
            logger.info(f"预期弹药数: {expected_ammo}")
            
            if expected_ammo < 0:
                logger.error("无效的预期弹药数")
                return False
                
            # 截图判空
            if not screenshots or len(screenshots) == 0:
                logger.error("未提供截图")
                return False
                
            screenshot = screenshots[0]
            
            # 获取YAML文件中的视觉参数
            yaml_cv_params = self.get_target_specific_param("ammo_ocr_params", {})
            
            # 如果指定了cv_params，则覆盖默认值
            if cv_params:
                yaml_cv_params.update(cv_params)
                
            # 从配置获取弹药区域(如果有)
            roi_rect = yaml_cv_params.get("roi_rect")
            
            # 获取过滤器类型
            filter_type = yaml_cv_params.get("filter_type", "hsv")
            
            # 构建文件名后缀用于调试
            filename_suffix = ""
            screenshot_path = context.get("screenshotFile", "")
            if screenshot_path:
                filename_suffix = os.path.splitext(os.path.basename(screenshot_path))[0]
            else:
                filename_suffix = f"img_{int(time.time())}"
            
            # 使用VisionUtils读取文本
            # 将调试目录传递给OCR函数
            ocr_text = VisionUtils.readTextFromPicture(
                screenshot, 
                roi_rect, 
                filter_type, 
                yaml_cv_params, 
                debugEnabled, 
                filename_suffix, 
                debug_dir
            )
            
            # 将OCR结果保存到context中以便在测试循环中使用
            context["ocr_result"] = ocr_text
            
            logger.info(f"OCR识别文本: '{ocr_text}'")
            
            # 如果OCR结果为空，则提前返回失败
            if not ocr_text:
                logger.warning("未识别到有效文本")
                return False
                
            # 尝试将OCR结果转换为数字
            try:
                actual_ammo = int(ocr_text)
                logger.info(f"OCR识别弹药数: {actual_ammo}")
                
                # 比较实际弹药数与预期弹药数
                is_matching = actual_ammo == expected_ammo
                logger.info(f"弹药数对比 - 预期: {expected_ammo}, 实际: {actual_ammo}, 匹配: {is_matching}")
                
                return is_matching
            except ValueError as e:
                logger.error(f"OCR结果无法转换为数字: {e}")
                return False
                
        except Exception as e:
            logger.exception(f"弹药文本同步测试出错: {e}")
            return False

    def check_ammo_state(self, screenshot_path, expected_value, region=None, cv_params=None):
        """
        检查弹药状态
        @param screenshot_path: 截图路径
        @param expected_value: 期望的弹药数值
        @param region: 检测区域 [x, y, width, height]
        @param cv_params: 视觉参数字典
        @return: (bool, str) - (是否匹配, 详细信息)
        """
        try:
            if not isinstance(screenshot_path, (str, np.ndarray)):
                return False, "无效的截图输入"
                
            # 处理图像数组输入
            if isinstance(screenshot_path, np.ndarray):
                image = screenshot_path
            else:
                if not os.path.exists(screenshot_path):
                    return False, f"截图文件不存在: {screenshot_path}"
                image = cv2.imread(screenshot_path)
                
            if image is None:
                return False, "无法读取截图"
                
            # 使用默认的cv_params如果没有提供
            if cv_params is None:
                cv_params = self.target_config.get('cv_params', {})
                
            # 处理检测区域，使用参数提供的区域或配置默认区域
            if region is None:
                region = cv_params.get('ammo_bbox')
                
            if region:
                x, y, w, h = region
                if x >= 0 and y >= 0 and x + w <= image.shape[1] and y + h <= image.shape[0]:
                    roi = image[y:y+h, x:x+w]
                else:
                    return False, "检测区域超出图像范围"
            else:
                roi = image
                
            # 图像预处理
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 特殊情况处理：弹药为0
            if expected_value == 0:
                # 使用模板匹配或特定的零值检测逻辑
                zero_template = self._get_zero_template()
                if zero_template is not None:
                    match_result = cv2.matchTemplate(binary, zero_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(match_result)
                    if max_val > 0.8:  # 匹配阈值
                        return True, "检测到弹药数为0"
            
            # 常规数字识别
            try:
                import pytesseract
                config = '--psm 7 -c tessedit_char_whitelist=0123456789'
                detected_text = pytesseract.image_to_string(binary, config=config)
                detected_text = ''.join(filter(str.isdigit, detected_text))
                
                if detected_text:
                    detected_value = int(detected_text)
                    if detected_value == expected_value:
                        return True, f"弹药数量匹配: {detected_value}"
                    else:
                        return False, f"弹药数量不匹配: 期望 {expected_value}, 实际 {detected_value}"
                else:
                    return False, "未能识别到数字"
                    
            except Exception as e:
                return False, f"数字识别失败: {str(e)}"
                
        except Exception as e:
            logger.error(f"检测过程发生错误: {str(e)}")
            return False, f"检测过程错误: {str(e)}"
            
    def _get_zero_template(self):
        """获取数字0的模板图像"""
        try:
            template_path = os.path.join(os.path.dirname(__file__), 'templates', 'zero.png')
            if os.path.exists(template_path):
                return cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            return None
        except Exception:
            return None
            
    def get_test_image_path(self, category=None, file_name=None):
        """
        获取测试图像路径
        @param category: 图像类别(如'Ammo', 'Crosshair', 'NoHUD')
        @param file_name: 文件名
        @return: 图像文件的完整路径
        """
        if self.target_name == 'p1_legacy':
            # 针对旧逻辑，返回固定路径
            return os.path.join(self.base_path, self.target_config.get('screenshot_path', 'unitTestResources/p1.png'))
        else:
            # 针对AssaultCube，根据类别和文件名构建路径
            base_path = self.target_config.get('screenshot_base_path', 'test_images')
            if category and file_name:
                return os.path.join(self.base_path, base_path, category, file_name)
            return None

if __name__ == "__main__":
    # 配置日志
    logger.add("test_debug.log", level="DEBUG", rotation="1 MB")
    logger.info("开始AssaultCube视觉检测调试")
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        active_target = config.get('active_target', 'assaultcube')
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        active_target = 'assaultcube'
        config = {}
    
    logger.info(f"当前测试目标: {active_target}")
    
    # 创建目标特定的调试目录
    debug_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
    target_debug_dir = os.path.join(debug_base_dir, active_target)  # 创建目标子目录，例如 debug/assaultcube/
    os.makedirs(target_debug_dir, exist_ok=True)
    logger.info(f"创建目标特定调试目录: {target_debug_dir}")
    
    # 初始化LogicLayer
    logicLayer = LogicLayer(active_target, config)
    
    # 测试准星检测
    logger.info("===== 准星检测测试 =====")
    # 拼接完整路径
    full_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  config.get('targets', {}).get(active_target, {}).get('screenshot_base_path', 'unitTestResources'))
    crosshair_images = glob.glob(os.path.join(full_base_path, 'Crosshair', '*.png'))
    
    crosshair_pass = 0
    crosshair_total = 0
    
    for image_path in crosshair_images:
        try:
            # 从文件名判断期望结果
            filename = os.path.basename(image_path)
            expected_result = "dead" not in filename and "reload" not in filename and "teammate" not in filename
            
            # 加载图像
            screenshot = cv2.imread(image_path)
            
            if screenshot is None:
                logger.warning(f"无法加载图像: {image_path}")
                continue
                
            # 构造测试上下文
            context = {'screenshotFile': image_path}
            expected_answer = {"boolResult": "True" if expected_result else "False"}
            
            # 进行测试
            isMatching = logicLayer.testWeaponCrossPresence(
                [screenshot], 
                context, 
                expected_answer, 
                debugEnabled=True,
                debug_dir=target_debug_dir
            )
            
            # 比较结果
            result_str = "通过" if isMatching == expected_result else "失败"
            logger.info(f"准星测试: {filename} - 期望: {expected_result}, 结果: {isMatching}, 判定: {result_str}")
            
            crosshair_total += 1
            if isMatching == expected_result:
                crosshair_pass += 1
                
        except Exception as e:
            logger.error(f"准星测试出错: {e}")
    
    logger.info(f"准星测试结果: {crosshair_pass}/{crosshair_total} 通过, 通过率: {crosshair_pass/crosshair_total*100 if crosshair_total > 0 else 0:.2f}%")
    
    # 测试弹药识别
    logger.info("===== 弹药识别测试 =====")
    ammo_images = glob.glob(os.path.join(full_base_path, 'Ammo', '*.png'))
    
    ammo_pass = 0
    ammo_total = 0
    
    for image_path in ammo_images:
        try:
            # 从文件名提取期望的弹药数
            filename = os.path.basename(image_path)
            match = re.search(r'ammo_clip(\d+)_total(\d+)', filename)
            
            if not match:
                logger.warning(f"无法从文件名解析弹药数量: {filename}")
                continue
                
            expected_ammo = int(match.group(1))  # 当前弹匣数量
            
            # 加载图像
            screenshot = cv2.imread(image_path)
            
            if screenshot is None:
                logger.warning(f"无法加载图像: {image_path}")
                continue
                
            # 构造测试上下文
            context = {'screenshotFile': image_path}
            expected_answer = {"intResult": expected_ammo}
            
            # 进行测试，记录OCR的原始结果
            actual_result = logicLayer.testAmmoTextInSync(
                [screenshot], 
                context, 
                expected_answer, 
                debugEnabled=True,
                debug_dir=target_debug_dir
            )
            
            # 从context获取OCR结果
            ocr_text = context.get("ocr_result", "")
            
            # 比较结果
            result_str = "通过" if actual_result else "失败"
            logger.info(f"弹药测试: {filename} - 期望: {expected_ammo}, OCR结果: '{ocr_text}', 判定: {result_str}")
            
            ammo_total += 1
            if actual_result:
                ammo_pass += 1
                
        except Exception as e:
            logger.error(f"弹药测试出错: {e}")
    
    logger.info(f"弹药测试结果: {ammo_pass}/{ammo_total} 通过, 通过率: {ammo_pass/ammo_total*100 if ammo_total > 0 else 0:.2f}%")
    
    logger.info("视觉检测调试完成")

