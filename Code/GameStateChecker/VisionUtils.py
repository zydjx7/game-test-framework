import cv2
import os
import pytesseract
import yaml
from PIL import Image
import time
import numpy as np
from typing import List, Tuple, Dict
from matplotlib import pyplot as plt
from loguru import logger

class VisionUtils:
    # 初始化Tesseract路径
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
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
    def readTextFromPicture(srcImg, boundingBox, textValueColor_inHSV_min, textValueColor_inHSV_max,
                           do_imgProcessing=True, debugEnabled=False):
        """从图像中读取文本"""
        try:
            # 提取ROI
            x, y, w, h = boundingBox
            roi = srcImg[y:y+h, x:x+w]
            
            if do_imgProcessing:
                # 转换到HSV空间
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # 创建掩码
                lower = np.array([textValueColor_inHSV_min, 50, 50])
                upper = np.array([textValueColor_inHSV_max, 255, 255])
                mask = cv2.inRange(hsv, lower, upper)
                
                # 应用掩码
                result = cv2.bitwise_and(roi, roi, mask=mask)
                
                # 转换为灰度图
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                
                # 二值化
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                binary = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 图像预处理增强
            # 使用形态学操作清理噪点
            kernel = np.ones((2,2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # OCR识别
            custom_config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(binary, config=custom_config)
            
            if debugEnabled:
                logger.debug(f"OCR识别结果: {text}")
                # 保存调试图像
                debug_dir = "debug"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, 'debug_roi.png'), roi)
                cv2.imwrite(os.path.join(debug_dir, 'debug_binary.png'), binary)
            
            # 清理结果，只保留数字
            clean_text = ''.join(filter(str.isdigit, text))
            return clean_text
            
        except Exception as e:
            logger.error(f"读取文本失败: {e}")
            return ""

    @staticmethod
    def matchTemplateImg(img_target, img_src, minKeypoints=3, debugEnabled=False):
        """匹配模板图像 - 使用SIFT特征点匹配"""
        try:
            # 使用SIFT特征匹配
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img_src, None)
            kp2, des2 = sift.detectAndCompute(img_target, None)
            
            if des1 is None or des2 is None:
                logger.warning("未能提取特征点描述子")
                # 如果SIFT失败，尝试使用模板匹配作为备选
                return VisionUtils.matchTemplateSimple(img_target, img_src, debugEnabled=debugEnabled)
                
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
                return VisionUtils.matchTemplateSimple(img_target, img_src, debugEnabled=debugEnabled)
            
            # 应用Lowe's ratio测试，阈值稍微提高
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:  # 从0.7调高到0.75
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
            if not matched and debugEnabled:
                logger.debug("SIFT匹配失败，尝试使用模板匹配")
                # SIFT匹配失败，尝试使用模板匹配
                return VisionUtils.matchTemplateSimple(img_target, img_src, debugEnabled=debugEnabled)
                
            return matched
            
        except Exception as e:
            logger.error(f"模板匹配失败: {e}")
            # 出错时尝试备选方法
            return VisionUtils.matchTemplateSimple(img_target, img_src, debugEnabled=debugEnabled)
    
    @staticmethod
    def matchTemplateSimple(img_target, img_src, threshold=None, debugEnabled=False):
        """简单的模板匹配方法，作为SIFT的备选"""
        try:
            if threshold is None:
                threshold = VisionUtils.cross_threshold
                
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
                result = cv2.matchTemplate(img_target_gray, img_src_gray, cv2.TM_CCOEFF_NORMED)
            else:
                # 模板比目标图像大，需要调整大小
                scale = 0.5  # 缩放比例
                resized_template = cv2.resize(img_src_gray, None, fx=scale, fy=scale)
                result = cv2.matchTemplate(img_target_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            
            # 获取最大匹配值及位置
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if debugEnabled:
                logger.debug(f"模板匹配最大值: {max_val}, 阈值: {threshold}")
                
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
                cv2.imwrite(os.path.join(debug_dir, 'template_correlation.png'), (result * 255).astype(np.uint8))
                
            # 返回匹配结果，使用较低的阈值0.6
            return max_val >= threshold
            
        except Exception as e:
            logger.error(f"简单模板匹配失败: {e}")
            return False
    
    @staticmethod
    def getBinaryImage(srcImg, textValueColor_inHSV_min, textValueColor_inHSV_max):
        """获取二值化图像用于调试"""
        try:
            hsv = cv2.cvtColor(srcImg, cv2.COLOR_BGR2HSV)
            
            # 创建掩码
            lower = np.array([textValueColor_inHSV_min, 50, 50])
            upper = np.array([textValueColor_inHSV_max, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            
            return mask
        except Exception as e:
            print(f"获取二值化图像失败: {e}")
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
