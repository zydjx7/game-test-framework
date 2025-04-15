import os
import time
from typing import List, Tuple, Dict
from VisionUtils import VisionUtils
import cv2
import re
import numpy as np  # 添加numpy导入
from loguru import logger

class LogicLayer:
    def __init__(self):
        """初始化逻辑层"""
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"初始化LogicLayer，基础路径: {self.base_path}")
        self.preloadData()

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

    def preloadData(self):
        """预加载所需数据"""
        try:
            template_path = os.path.join(self.base_path, "unitTestResources", "Cross_p.png")
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

    def testWeaponCrossPresence(self, screenshots: List[any],
                               context: Dict[any, any],
                               expectedAnswer: Dict[any, any],
                               debugEnabled=False) -> bool:
        """测试武器准星是否存在"""
        try:
            if not screenshots or len(screenshots) == 0:
                logger.error("没有提供截图")
                return False
                
            screenshot_gray = self.__convertScreenshotToGray(screenshots[0])
            if screenshot_gray is None:
                logger.error("转换截图失败")
                return False
                
            if self.templateImg_weaponCross is None:
                logger.error("武器准星模板未加载")
                return False
                
            isMatching = VisionUtils.matchTemplateImg(
                img_target=screenshot_gray,
                img_src=self.templateImg_weaponCross,
                minKeypoints=6,
                debugEnabled=debugEnabled
            )
            
            if debugEnabled:
                logger.debug(f"准星匹配结果: {isMatching}")
                
            return expectedAnswer["boolResult"] == str(isMatching)
            
        except Exception as e:
            logger.error(f"测试武器准星失败: {e}")
            return False

    def testAmmoTextInSync(self, screenshots: List[any],
                          context: Dict[any, any],
                          expectedAnswer: Dict[any, any],
                          debugEnabled=False) -> bool:
        """测试弹药文本同步"""
        try:
            if not screenshots or len(screenshots) == 0:
                logger.error("没有提供截图")
                return False
                
            if not context.get('bbox'):
                logger.error("未提供弹药显示区域")
                return False
                
            logger.debug(f"开始识别弹药文本，期望值: {expectedAnswer.get('intResult')}")
            
            # 增加截图预处理
            img = screenshots[0]
            if debugEnabled:
                cv2.imwrite("debug_roi.png", img[context["bbox"][1]:context["bbox"][1]+context["bbox"][3], 
                                               context["bbox"][0]:context["bbox"][0]+context["bbox"][2]])
            
            # 读取文本
            res = VisionUtils.readTextFromPicture(
                srcImg=img,
                boundingBox=context["bbox"],
                textValueColor_inHSV_min=context.get("textColorValueInHSV_min", 129),
                textValueColor_inHSV_max=context.get("textColorValueInHSV_max", 130),
                do_imgProcessing=True,
                debugEnabled=debugEnabled
            )
            
            # 增强数字提取逻辑
            res_clean = re.findall(r'\d+', res)
            if res_clean:
                try:
                    res_int = int(res_clean[0])
                    logger.debug(f"识别到的弹药数: {res_int}")
                except ValueError:
                    logger.error(f"无法将识别结果转换为数字: {res_clean[0]}")
                    return False
            else:
                logger.error("未识别到有效数字")
                if debugEnabled:
                    logger.debug(f"OCR原始结果: {res}")
                return False
                
            is_match = expectedAnswer.get("intResult") == res_int
            logger.info(f"弹药数匹配结果: 期望={expectedAnswer.get('intResult')}, 实际={res_int}, 匹配={is_match}")
            
            # 保存调试信息
            if debugEnabled and not is_match:
                cv2.imwrite("debug_binary.png", VisionUtils.getBinaryImage(img, 
                    context.get("textColorValueInHSV_min", 129),
                    context.get("textColorValueInHSV_max", 130)))
            
            return is_match
            
        except Exception as e:
            logger.error(f"测试弹药同步失败: {e}")
            if debugEnabled:
                logger.exception("详细错误信息")
            return False

    def check_ammo_state(self, screenshot_path, expected_value, region=None):
        """
        检查弹药状态
        @param screenshot_path: 截图路径
        @param expected_value: 期望的弹药数值
        @param region: 检测区域 [x, y, width, height]
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
                
            # 处理检测区域
            if region:
                x, y, w, h = region
                if x >= 0 and y >= 0 and x + w <= image.shape[1] and y + h <= image.shape[0]:
                    image = image[y:y+h, x:x+w]
                else:
                    return False, "检测区域超出图像范围"
            
            # 图像预处理
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

if __name__ == "__main__":
    # 用于单元测试的代码
    logicLayer = LogicLayer()

    # 测试1：准星检测
    unitTestScreenShot = cv2.imread(os.path.join(logicLayer.base_path, "unitTestResources/p1.png"), cv2.IMREAD_GRAYSCALE)
    res = logicLayer.testWeaponCrossPresence(
        screenshots=[unitTestScreenShot],
        context={},
        expectedAnswer={"boolResult": "True"},
        debugEnabled=True
    )
    logger.info(f"准星检测结果: {res}")

    # 测试2：弹药同步
    unitTestScreenShot = cv2.imread(os.path.join(logicLayer.base_path, "unitTestResources/p1.png"))
    res = logicLayer.testAmmoTextInSync(
        screenshots=[unitTestScreenShot],
        context={
            "bbox": [912, 1015, 79, 49],
            "textColorValueInHSV_min": 129,
            "textColorValueInHSV_max": 130
        },
        expectedAnswer={"intResult": 50},
        debugEnabled=True
    )
    logger.info(f"弹药同步测试结果: {res}")

