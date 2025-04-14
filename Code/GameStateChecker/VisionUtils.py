import cv2
import os
import pytesseract
import yaml
from PIL import Image
import time
import numpy as np
from typing import List, Tuple, Dict
from matplotlib import pyplot as plt

class VisionUtils:
    # 初始化Tesseract路径
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            tesseract_path = config.get('vision', {}).get('tesseract', {}).get('path')
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
    except Exception as e:
        print(f"警告：无法加载Tesseract配置: {e}")

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
                print(f"OCR识别结果: {text}")
                # 保存调试图像
                cv2.imwrite('debug_roi.png', roi)
                cv2.imwrite('debug_binary.png', binary)
            
            # 清理结果，只保留数字
            clean_text = ''.join(filter(str.isdigit, text))
            return clean_text
            
        except Exception as e:
            print(f"读取文本失败: {e}")
            return ""

    @staticmethod
    def matchTemplateImg(img_target, img_src, minKeypoints=6, debugEnabled=False):
        """匹配模板图像"""
        try:
            # 使用SIFT特征匹配
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img_src, None)
            kp2, des2 = sift.detectAndCompute(img_target, None)
            
            if des1 is None or des2 is None:
                return False
                
            # FLANN参数
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            try:
                matches = flann.knnMatch(des1, des2, k=2)
            except Exception as e:
                print(f"特征匹配失败: {e}")
                return False
            
            # 应用Lowe's ratio测试
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            if debugEnabled:
                print(f"找到 {len(good_matches)} 个好的匹配点")
                
            return len(good_matches) >= minKeypoints
            
        except Exception as e:
            print(f"模板匹配失败: {e}")
            return False

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
