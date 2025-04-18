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
    def readTextFromPicture(srcImg, boundingBox, filter_type='value', context=None, cv_params=None, do_imgProcessing=True, debugEnabled=False, filename_suffix="", debug_dir="debug"):
        """
        从图片中读取文本(使用OCR)
        支持HSV和Value两种过滤模式
        @param srcImg: 原始图像
        @param boundingBox: 边界框 [x, y, width, height]
        @param filter_type: 过滤类型，'hsv'或'value'
        @param context: 上下文参数
        @param cv_params: 视觉参数，优先级高于context
        @param do_imgProcessing: 是否进行图像处理
        @param debugEnabled: 是否启用调试
        @param filename_suffix: 调试文件名后缀，用于区分不同图像的调试输出
        @param debug_dir: 调试图像保存目录
        @return: 识别的文本
        """
        try:
            # 提取ROI区域
            x, y, w, h = boundingBox
            roi = srcImg[y:y+h, x:x+w] if y >= 0 and x >= 0 and y+h <= srcImg.shape[0] and x+w <= srcImg.shape[1] else srcImg
            
            if debugEnabled:
                # 保存ROI区域进行调试
                os.makedirs(debug_dir, exist_ok=True)
                roi_filename = os.path.join(debug_dir, f"ammo_roi_{filename_suffix}.png")
                cv2.imwrite(roi_filename, roi)
                logger.debug(f"已保存ROI区域图像到 {roi_filename}")
                
            # 检查ROI是否为空
            if roi is None or roi.size == 0:
                logger.warning("ROI区域为空或无效，无法进行OCR识别")
                return ""
            
            # 初始化参数
            if context is None:
                context = {}
                
            # 获取过滤类型 - 优先使用 cv_params
            if cv_params and 'ammo_filter_type' in cv_params:
                filter_type = cv_params.get('ammo_filter_type')
                
            # 根据过滤类型选择处理方法
            binary = None
            
            if filter_type == 'hsv':
                # 从 cv_params 或 context 获取 HSV 参数
                hue_min = cv_params.get('ammo_hsv_hue_min', context.get('textColorValueInHSV_min', 0))
                hue_max = cv_params.get('ammo_hsv_hue_max', context.get('textColorValueInHSV_max', 179))
                sat_min = cv_params.get('ammo_hsv_sat_min', context.get('hsv_sat_min', 0))
                sat_max = cv_params.get('ammo_hsv_sat_max', context.get('hsv_sat_max', 50))
                val_min = cv_params.get('ammo_hsv_val_min', context.get('hsv_val_min', 180))
                val_max = cv_params.get('ammo_hsv_val_max', context.get('hsv_val_max', 255))
                
                # HSV颜色空间过滤
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # 创建HSV范围
                lower = np.array([hue_min, sat_min, val_min])
                upper = np.array([hue_max, sat_max, val_max])
                
                # 应用HSV过滤
                binary = cv2.inRange(hsv, lower, upper)
                
                if debugEnabled:
                    logger.debug(f"使用HSV过滤: H({hue_min}-{hue_max}), S({sat_min}-{sat_max}), V({val_min}-{val_max})")
                    
                    # 保存HSV通道图像用于调试
                    h, s, v = cv2.split(hsv)
                    cv2.imwrite(os.path.join(debug_dir, f"hsv_h_channel_{filename_suffix}.png"), h)
                    cv2.imwrite(os.path.join(debug_dir, f"hsv_s_channel_{filename_suffix}.png"), s)
                    cv2.imwrite(os.path.join(debug_dir, f"hsv_v_channel_{filename_suffix}.png"), v)
                
            else:  # 'value'模式(灰度处理)
                # 从 cv_params 或 context 获取阈值参数
                value_min = cv_params.get('ammo_value_threshold_min', context.get('valueThresholdMin', 150))
                value_max = cv_params.get('ammo_value_threshold_max', context.get('valueThresholdMax', 255))
                
                # 转换为灰度图
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # 应用灰度阈值
                _, binary = cv2.threshold(gray, value_min, value_max, cv2.THRESH_BINARY)
                
                if debugEnabled:
                    logger.debug(f"使用Value过滤: ({value_min}-{value_max})")
                    # 保存灰度图像用于调试
                    cv2.imwrite(os.path.join(debug_dir, f"gray_{filename_suffix}.png"), gray)
            
            # 应用图像处理
            if do_imgProcessing and binary is not None:
                # 保存原始二值化图像用于比较
                if debugEnabled:
                    cv2.imwrite(os.path.join(debug_dir, f"binary_pre_processing_{filename_suffix}.png"), binary)
                
                # 应用膨胀和闭操作使文字更连贯
                kernel = np.ones((2, 2), np.uint8)
                binary = cv2.dilate(binary, kernel, iterations=1)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
                
                if debugEnabled:
                    logger.debug("应用了形态学处理: 膨胀(2x2内核,1次迭代) + 闭操作(2x2内核,1次迭代)")
            
            if debugEnabled and binary is not None:
                # 保存二值化图像(用于调试)
                binary_filename = os.path.join(debug_dir, f"debug_binary_{filename_suffix}.png")
                cv2.imwrite(binary_filename, binary)
                logger.debug(f"已保存二值化图像到 {binary_filename}")
            
            # 检查二值图像是否为有效
            if binary is None:
                logger.error("二值化图像生成失败")
                return ""
            
            # 设置Tesseract OCR选项 - PSM 8表示单字块模式
            custom_config = r'--psm 8 -c tessedit_char_whitelist=0123456789'
            
            # 使用Tesseract进行OCR识别
            text = pytesseract.image_to_string(binary, config=custom_config).strip()
            
            # 删除所有非数字字符(包括句点、空格等)
            import re
            text = re.sub(r'[^0-9]', '', text)
            
            if debugEnabled:
                logger.info(f"OCR识别结果: '{text}' (过滤类型: {filter_type})")
                
            # 记录OCR结果到context（用于调试和结果传递）
            if context is not None:
                context["ocr_result"] = text
                
            return text
            
        except Exception as e:
            logger.error(f"OCR文本识别失败: {e}")
            # 记录错误到context
            if context is not None:
                context["ocr_error"] = str(e)
            return ""

    @staticmethod
    def matchTemplateImg(img_src, img_target, minKeypoints=6, useORB=False, useSIFT=True, debugEnabled=False, debug_dir="debug", filename_suffix=""):
        """
        使用特征点匹配(SIFT/ORB)和备选的模板匹配来比对图像
        @param img_src: 源图像/模板
        @param img_target: 目标图像
        @param minKeypoints: 最小匹配特征点数量
        @param useORB: 是否使用ORB特征
        @param useSIFT: 是否使用SIFT特征
        @param debugEnabled: 是否启用调试
        @param debug_dir: 调试图像保存目录
        @param filename_suffix: 调试文件名后缀，用于区分不同图像的调试输出
        @return: 匹配的特征点列表和最小特征点要求
        """
        try:
            # 检查图片类型
            if type(img_target) is not np.ndarray:
                img_target = np.array(img_target)
            if type(img_src) is not np.ndarray:
                img_src = np.array(img_src)
                
            # 检查图像通道，转为灰度图
            if len(img_src.shape) > 2:
                img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
            if len(img_target.shape) > 2:
                img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
                
            # 创建SIFT检测器
            sift = cv2.SIFT_create()
            
            # 在图像中检测关键点和描述符
            kp1, des1 = sift.detectAndCompute(img_src, None)
            kp2, des2 = sift.detectAndCompute(img_target, None)
            
            # 记录关键点信息
            if debugEnabled:
                logger.debug(f"SIFT特征检测: 模板图像有 {len(kp1) if kp1 else 0} 个关键点, 目标图像有 {len(kp2) if kp2 else 0} 个关键点")
            
            # 使用FLANN匹配器进行特征匹配
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            good_matches = []
            
            # 确保检测到足够的特征点
            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                try:
                    # 进行KNN匹配
                    matches = flann.knnMatch(des1, des2, k=2)
                    
                    # 应用Lowe过滤条件
                    lowe_ratio = 0.7  # Lowe推荐的比例
                    for m, n in matches:
                        if m.distance < lowe_ratio * n.distance:
                            good_matches.append(m)
                    
                    if debugEnabled:
                        logger.debug(f"SIFT找到 {len(matches)} 对匹配, 应用Lowe比率({lowe_ratio})过滤后剩余 {len(good_matches)} 个好的匹配")
                        logger.debug(f"与目标值比较: {len(good_matches)}/{minKeypoints} 个匹配点, 要求比例: {lowe_ratio}")
                    
                except Exception as e:
                    logger.warning(f"特征匹配过程中出错: {e}")
            
            # 保存匹配结果图像（如果启用调试）
            if debugEnabled:
                # 创建调试目录
                os.makedirs(debug_dir, exist_ok=True)
                
                # 保存匹配结果图像
                if len(good_matches) > 0:
                    img_matches = cv2.drawMatches(img_src, kp1, img_target, kp2, good_matches[:min(10, len(good_matches))], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    
                    # 使用文件名后缀区分不同的匹配图像
                    suffix = f"_{filename_suffix}" if filename_suffix else ""
                    match_path = os.path.join(debug_dir, f"sift_matches{suffix}.png")
                    cv2.imwrite(match_path, img_matches)
                    logger.debug(f"已保存SIFT匹配可视化图像到 {match_path}")
                else:
                    logger.debug("未找到SIFT特征匹配，无法生成匹配图像")
                
            # 如果匹配点不足，尝试模板匹配
            if len(good_matches) < minKeypoints:
                # 进行模板匹配
                template_result = cv2.matchTemplate(img_target, img_src, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(template_result)
                
                logger.debug(f"特征点匹配不足({len(good_matches)}/{minKeypoints})，使用模板匹配备选，相关性得分:{max_val:.3f}")
                
                # 保存模板匹配结果
                if debugEnabled:
                    # 保存相关性得分图
                    suffix = f"_{filename_suffix}" if filename_suffix else ""
                    corr_path = os.path.join(debug_dir, f"template_correlation{suffix}.png")
                    cv2.imwrite(corr_path, template_result * 255)
                    
                    # 在目标图像上绘制最佳匹配位置
                    h, w = img_src.shape
                    top_left = max_loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    result_img = cv2.cvtColor(img_target.copy(), cv2.COLOR_GRAY2BGR) if len(img_target.shape) < 3 else img_target.copy()
                    cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)
                    pos_path = os.path.join(debug_dir, f"template_match_position{suffix}.png")
                    cv2.imwrite(pos_path, result_img)
                    
                    logger.debug(f"已保存模板匹配结果图像到 {debug_dir} 目录")
            
            matched = len(good_matches) >= minKeypoints
            logger.info(f"特征点匹配结果: {len(good_matches)}/{minKeypoints}, 匹配状态: {matched}")
            
            return good_matches, minKeypoints
            
        except Exception as e:
            logger.error(f"模板匹配过程中出错: {e}")
            return [], minKeypoints

    @staticmethod
    def matchTemplateSimple(img_target, img_src, threshold=None, debugEnabled=False, 
                           debug_dir="debug", return_details=False, filename_suffix="") -> Union[bool, Dict]:
        """
        简单的模板匹配方法，作为SIFT的备选
        @param img_target: 目标图像
        @param img_src: 源图像(模板)
        @param threshold: 匹配阈值
        @param debugEnabled: 是否启用调试
        @param debug_dir: 调试图像保存目录
        @param return_details: 是否返回详细匹配信息
        @param filename_suffix: 调试文件名后缀，用于区分不同图像的调试输出
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
            resized_template = None
            if img_src_gray.shape[0] > img_target_gray.shape[0] or img_src_gray.shape[1] > img_target_gray.shape[1]:
                # 模板比目标图像大，需要调整大小
                scale = 0.5  # 缩放比例
                logger.debug(f"模板({img_src_gray.shape})大于目标({img_target_gray.shape})，应用缩放: {scale}")
                resized_template = cv2.resize(img_src_gray, None, fx=scale, fy=scale)
                result_mat = cv2.matchTemplate(img_target_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            else:
                # 模板比目标图像小，进行模板匹配
                logger.debug(f"模板({img_src_gray.shape})小于或等于目标({img_target_gray.shape})，直接匹配")
                result_mat = cv2.matchTemplate(img_target_gray, img_src_gray, cv2.TM_CCOEFF_NORMED)
            
            # 获取最大匹配值及位置
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_mat)
            
            # 进行结果判断
            matched = max_val >= threshold
            
            # 添加详细调试日志
            if debugEnabled:
                logger.debug(f"模板匹配最大值: {max_val:.4f}, 阈值: {threshold:.4f}, 匹配状态: {matched}")
                
                # 保存匹配结果可视化
                os.makedirs(debug_dir, exist_ok=True)
                
                # 绘制匹配结果
                template_to_use = resized_template if resized_template is not None else img_src_gray
                h, w = template_to_use.shape[:2]
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                
                # 在目标图像上画矩形标记匹配位置
                result_img = cv2.cvtColor(img_target_gray.copy(), cv2.COLOR_GRAY2BGR) if len(img_target_gray.shape) < 3 else img_target_gray.copy()
                cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0) if matched else (0, 0, 255), 2)
                
                # 添加匹配分数文字
                text = f"Score: {max_val:.2f}, Threshold: {threshold:.2f}"
                cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if matched else (0, 0, 255), 2)
                
                # 保存结果图像，添加文件名后缀以区分不同测试
                suffix = f"_{filename_suffix}" if filename_suffix else ""
                match_result_path = os.path.join(debug_dir, f'template_match{suffix}.png')
                corr_result_path = os.path.join(debug_dir, f'template_correlation{suffix}.png')
                
                cv2.imwrite(match_result_path, result_img)
                cv2.imwrite(corr_result_path, (result_mat * 255).astype(np.uint8))
                
                logger.debug(f"已保存模板匹配结果图像到 {match_result_path}")
                logger.debug(f"已保存相关性分布图像到 {corr_result_path}")
            
            # 更新结果信息
            result["matched"] = matched
            result["score"] = max_val
            result["threshold"] = threshold
            
            # 日志记录匹配结果
            logger.info(f"模板匹配结果: 得分={max_val:.4f}, 阈值={threshold:.4f}, 匹配状态={matched}")
            
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

