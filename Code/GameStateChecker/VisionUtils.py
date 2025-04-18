import cv2
import os
import pytesseract
import yaml
from PIL import Image
import time
import numpy as np
from typing import List, Tuple, Dict, Union
from matplotlib import pyplot as plt
from loguru import logger

class VisionUtils:
    """
    视觉工具类 - 提供视觉检测相关的工具方法
    负责图像处理、特征匹配、文字识别等底层操作
    """
    # 初始化Tesseract路径
    # 修改：使用与LogicLayer相同的配置路径，避免在不同位置加载
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:  # 明确指定UTF-8编码
            config = yaml.safe_load(f)
            tesseract_path = config.get('vision', {}).get('tesseract', {}).get('path')
            # 加载其他视觉配置
            cross_threshold = config.get('vision', {}).get('cross', {}).get('threshold', 0.6)
            min_keypoints = config.get('vision', {}).get('cross', {}).get('min_keypoints', 3)
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
    except Exception as e:
        print(f"警告：无法加载Tesseract配置: {e}")
        cross_threshold = 0.6  # 默认模板匹配阈值
        min_keypoints = 3     # 默认最小特征点数

    @staticmethod
    def readTextFromPicture(srcImg, boundingBox, filter_type='hsv', context=None, 
                           do_imgProcessing=True, debugEnabled=False, filename_suffix=""):
        """
        从图像中读取文本
        @param srcImg: 源图像
        @param boundingBox: 边界框 [x, y, w, h]
        @param filter_type: 过滤类型，'hsv'或'value'
        @param context: 上下文字典，包含过滤参数
        @param do_imgProcessing: 是否进行图像处理
        @param debugEnabled: 是否启用调试
        @param filename_suffix: 调试文件名后缀，用于区分不同的ROI
        @return: 识别的文本
        """
        try:
            if context is None:
                context = {}
                
            # 提取ROI
            x, y, w, h = boundingBox
            roi = srcImg[y:y+h, x:x+w]
            
            if do_imgProcessing:
                # 根据过滤类型进行不同处理
                if filter_type == 'hsv':
                    # 使用HSV颜色空间过滤 - AI-B建议的参数
                    hsv_min = context.get("textColorValueInHSV_min", 0)
                    hsv_max = context.get("textColorValueInHSV_max", 179)
                    hsv_sat_max = context.get("hsv_sat_max", 60)  # 白色/灰白 → 饱和度低
                    hsv_val_min = context.get("hsv_val_min", 200)  # 高亮度
                    hsv_val_max = context.get("hsv_val_max", 255)
                    
                    # 转换到HSV空间
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    
                    # 创建掩码 - 使用饱和度和亮度值过滤白色数字
                    lower = np.array([hsv_min, 0, hsv_val_min])  # 饱和度最小为0
                    upper = np.array([hsv_max, hsv_sat_max, hsv_val_max])
                    mask = cv2.inRange(hsv, lower, upper)
                    
                    # 应用掩码
                    result = cv2.bitwise_and(roi, roi, mask=mask)
                    
                    # 转换为灰度图
                    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    
                elif filter_type == 'value':
                    # 使用亮度值过滤 - 针对AssaultCube的白色数字
                    val_min = context.get("valueThresholdMin", 200)  # 默认值提高到200，更适合白色数字
                    val_max = context.get("valueThresholdMax", 255)
                    
                    # 转换为灰度图
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    # 根据亮度值创建掩码
                    _, mask = cv2.threshold(gray, val_min, val_max, cv2.THRESH_BINARY)
                    gray = mask  # 直接使用二值图像
                else:
                    # 默认转灰度
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # 二值化
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                # 仅转为灰度，不做额外处理
                binary = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 图像预处理增强
            # 先进行膨胀操作，帮助连接数字部分，填充可能的空洞（AI-B建议）
            kernel = np.ones((3,3), np.uint8)  # 使用3x3内核
            binary = cv2.dilate(binary, kernel, iterations=1)
            
            # 然后进行闭运算清理噪点并连接断开的线段
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # OCR识别 - 使用PSM=8模式（AI-B建议）
            # PSM=8: 将图像视为单个字符块，更适合识别单个数字
            custom_config = r'--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(binary, config=custom_config).strip()
            
            if debugEnabled:
                logger.debug(f"OCR识别结果: '{text}', 过滤类型: {filter_type}")
                # 保存调试图像
                debug_dir = "debug"
                os.makedirs(debug_dir, exist_ok=True)
                
                # 使用后缀区分不同的ROI调试图
                suffix = filename_suffix if filename_suffix else ""
                cv2.imwrite(os.path.join(debug_dir, f'debug_roi{suffix}.png'), roi)
                cv2.imwrite(os.path.join(debug_dir, f'debug_binary{suffix}.png'), binary)
                
                # 如果有掩码，保存掩码
                if 'mask' in locals():
                    cv2.imwrite(os.path.join(debug_dir, f'debug_mask{suffix}.png'), mask)
            
            # 使用正则表达式提取数字，而不是简单过滤
            import re
            clean_text = ''.join(re.findall(r'\d+', text))
            return clean_text
            
        except Exception as e:
            logger.error(f"读取文本失败: {e}")
            return ""

    @staticmethod
    def matchTemplateImg(img_target, img_src, minKeypoints=3, lowe_ratio=0.75, 
                        template_threshold=None, debugEnabled=False, return_details=False):
        """
        匹配模板图像 - 使用SIFT特征点匹配
        @param img_target: 目标图像
        @param img_src: 源图像(模板)
        @param minKeypoints: 最小匹配点数
        @param lowe_ratio: Lowe比率测试阈值
        @param template_threshold: 模板匹配阈值
        @param debugEnabled: 是否启用调试
        @param return_details: 是否返回详细匹配信息
        @return: 布尔值(是否匹配成功)或包含详细信息的字典
        """
        try:
            # 使用默认值如果参数未提供
            if template_threshold is None:
                template_threshold = VisionUtils.cross_threshold
                
            # 结果字典
            result = {"matched": False, "method": "none", "score": 0.0}
            
            # 使用SIFT特征匹配
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img_src, None)
            kp2, des2 = sift.detectAndCompute(img_target, None)
            
            if des1 is None or des2 is None:
                logger.warning("未能提取特征点描述子，回退到模板匹配")
                # 如果SIFT失败，尝试使用模板匹配作为备选
                template_result = VisionUtils.matchTemplateSimple(
                    img_target, img_src, threshold=template_threshold, 
                    debugEnabled=debugEnabled, return_details=return_details
                )
                
                if return_details:
                    result.update(template_result)
                    return result
                else:
                    return template_result
                
            # FLANN参数
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            try:
                matches = flann.knnMatch(des1, des2, k=2)
            except Exception as e:
                logger.error(f"特征匹配失败: {e}")
                # 如果knnMatch失败，尝试使用模板匹配作为备选
                template_result = VisionUtils.matchTemplateSimple(
                    img_target, img_src, threshold=template_threshold, 
                    debugEnabled=debugEnabled, return_details=return_details
                )
                
                if return_details:
                    result.update(template_result)
                    return result
                else:
                    return template_result
            
            # 应用Lowe's ratio测试
            good_matches = []
            for m, n in matches:
                if m.distance < lowe_ratio * n.distance:  # 使用传入的lowe_ratio
                    good_matches.append(m)
            
            if debugEnabled:
                logger.debug(f"找到 {len(good_matches)} 个好的匹配点，需要 {minKeypoints} 个")
                # 创建匹配结果可视化
                debug_dir = "debug"
                os.makedirs(debug_dir, exist_ok=True)
                
                # 绘制匹配结果
                img_matches = cv2.drawMatches(
                    img_src, kp1, 
                    img_target, kp2, 
                    good_matches, None, 
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.imwrite(os.path.join(debug_dir, 'sift_matches.png'), img_matches)
                
            # 检查是否有足够的匹配点
            matched = len(good_matches) >= minKeypoints
            
            # 更新结果信息
            result["matched"] = matched
            result["method"] = "sift"
            result["score"] = len(good_matches)  # 使用匹配点数量作为分数
            
            if not matched and debugEnabled:
                logger.debug(f"SIFT匹配失败(找到{len(good_matches)}点<{minKeypoints}点)，尝试使用模板匹配")
                # SIFT匹配失败，尝试使用模板匹配
                template_result = VisionUtils.matchTemplateSimple(
                    img_target, img_src, threshold=template_threshold, 
                    debugEnabled=debugEnabled, return_details=True
                )
                
                # 只有当模板匹配成功时才更新结果
                if template_result.get("matched", False):
                    result.update(template_result)
            
            if return_details:
                return result
            else:
                return result["matched"]
                
        except Exception as e:
            logger.error(f"模板匹配失败: {e}")
            # 出错时尝试备选方法
            template_result = VisionUtils.matchTemplateSimple(
                img_target, img_src, threshold=template_threshold, 
                debugEnabled=debugEnabled, return_details=return_details
            )
            
            if return_details:
                result.update(template_result)
                return result
            else:
                return template_result
    
    @staticmethod
    def matchTemplateSimple(img_target, img_src, threshold=None, debugEnabled=False, 
                           return_details=False) -> Union[bool, Dict]:
        """
        简单的模板匹配方法，作为SIFT的备选
        @param img_target: 目标图像
        @param img_src: 源图像(模板)
        @param threshold: 匹配阈值
        @param debugEnabled: 是否启用调试
        @param return_details: 是否返回详细匹配信息
        @return: 布尔值(是否匹配成功)或包含详细信息的字典
        """
        try:
            if threshold is None:
                threshold = VisionUtils.cross_threshold
                
            # 结果字典
            result = {"matched": False, "method": "template", "score": 0.0}
            
            # 确保图像是灰度的
            if len(img_src.shape) > 2:
                img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
            else:
                img_src_gray = img_src
                
            if len(img_target.shape) > 2:
                img_target_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
            else:
                img_target_gray = img_target
            
            # 确保模板尺寸小于目标图像
            if img_src_gray.shape[0] > img_target_gray.shape[0] and img_src_gray.shape[1] > img_target_gray.shape[1]:
                # 模板比目标图像小，进行模板匹配
                result_mat = cv2.matchTemplate(img_target_gray, img_src_gray, cv2.TM_CCOEFF_NORMED)
            else:
                # 模板比目标图像大，需要调整大小
                scale = 0.5  # 缩放比例
                resized_template = cv2.resize(img_src_gray, None, fx=scale, fy=scale)
                result_mat = cv2.matchTemplate(img_target_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            
            # 获取最大匹配值及位置
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_mat)
            
            if debugEnabled:
                logger.debug(f"模板匹配最大值: {max_val:.4f}, 阈值: {threshold:.4f}")
                
                # 保存匹配结果可视化
                debug_dir = "debug"
                os.makedirs(debug_dir, exist_ok=True)
                
                # 绘制匹配结果
                h, w = img_src_gray.shape
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                
                # 在目标图像上画矩形标记匹配位置
                result_img = cv2.cvtColor(img_target_gray.copy(), cv2.COLOR_GRAY2BGR)
                cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)
                
                cv2.imwrite(os.path.join(debug_dir, 'template_match.png'), result_img)
                cv2.imwrite(os.path.join(debug_dir, 'template_correlation.png'), (result_mat * 255).astype(np.uint8))
            
            # 更新结果信息
            matched = max_val >= threshold
            result["matched"] = matched
            result["score"] = max_val
            
            if return_details:
                return result
            else:
                return matched
            
        except Exception as e:
            logger.error(f"简单模板匹配失败: {e}")
            if return_details:
                return {"matched": False, "method": "error", "score": 0.0, "error": str(e)}
            else:
                return False
    
    @staticmethod
    def getBinaryImage(srcImg, filter_type='hsv', context=None):
        """
        获取二值化图像用于调试
        @param srcImg: 源图像
        @param filter_type: 过滤类型，'hsv'或'value'
        @param context: 上下文字典，包含过滤参数
        @return: 二值化图像
        """
        try:
            if context is None:
                context = {}
                
            if filter_type == 'hsv':
                hsv_min = context.get("textColorValueInHSV_min", 0)
                hsv_max = context.get("textColorValueInHSV_max", 180)
                
                hsv = cv2.cvtColor(srcImg, cv2.COLOR_BGR2HSV)
                # 创建掩码
                lower = np.array([hsv_min, 50, 50])
                upper = np.array([hsv_max, 255, 255])
                mask = cv2.inRange(hsv, lower, upper)
                return mask
            elif filter_type == 'value':
                val_min = context.get("valueThresholdMin", 200)  # 默认阈值提高
                val_max = context.get("valueThresholdMax", 255)
                
                gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, val_min, val_max, cv2.THRESH_BINARY)
                return mask
            else:
                # 默认返回灰度图
                return cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
                
        except Exception as e:
            logger.error(f"获取二值化图像失败: {e}")
            return None

    @staticmethod
    def preprocessimg_demo():
        """图像预处理演示"""
        img = cv2.imread("Cross.png", cv2.COLOR_BGR2GRAY)
        if img is None:
            print("无法加载图像")
            return
            
        img[img < 150] = 0
        cv2.imshow("processed image", img)
        cv2.waitKey()
        cv2.imwrite("Cross_p.png", img)
