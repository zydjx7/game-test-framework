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
from AmmoTemplateRecognizer import AmmoTemplateRecognizer

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
    
    # 根路径和默认调试目录
    base_path = os.path.dirname(os.path.abspath(__file__))
    default_debug_dir = os.path.join(base_path, "debug")
    
    @staticmethod
    def normalize_debug_dir(debug_dir):
        """
        标准化调试目录，确保使用绝对路径并创建目录
        @param debug_dir: 调试目录路径
        @return: 绝对路径
        """
        # 如果传入空路径或None，使用默认目录
        if not debug_dir:
            debug_dir = VisionUtils.default_debug_dir
            
        # 如果传入相对路径，转换为绝对路径
        if not os.path.isabs(debug_dir):
            debug_dir = os.path.join(VisionUtils.base_path, debug_dir)
            
        # 确保目录存在
        os.makedirs(debug_dir, exist_ok=True)
        
        # 使用DEBUG级别记录目录信息
        logger.debug(f"使用调试目录: {debug_dir}")
        return debug_dir

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
            # 标准化调试目录路径
            if debugEnabled:
                debug_dir = VisionUtils.normalize_debug_dir(debug_dir)
                # 保留处理图像文件的debug信息
                logger.debug(f"处理图像文件: {filename_suffix}")
            
            # 提取ROI区域
            x, y, w, h = boundingBox
            roi = srcImg[y:y+h, x:x+w] if y >= 0 and x >= 0 and y+h <= srcImg.shape[0] and x+w <= srcImg.shape[1] else srcImg
            
            if debugEnabled:
                # 确保文件名后缀不为空，如果为空则使用时间戳
                if not filename_suffix:
                    import datetime
                    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename_suffix = f"auto_{current_time}"
                
                # 保存ROI区域进行调试，但不记录路径
                os.makedirs(debug_dir, exist_ok=True)
                roi_filename = os.path.join(debug_dir, f"ammo_roi_{filename_suffix}.png")
                cv2.imwrite(roi_filename, roi)
                
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
                    # 只在特定条件下输出HSV参数信息，使用DEBUG级别
                    if "hsv_debug_output" not in context:
                        logger.debug(f"使用HSV过滤: H({hue_min}-{hue_max}), S({sat_min}-{sat_max}), V({val_min}-{val_max})")
                        context["hsv_debug_output"] = True
                    
                    # 保存HSV通道图像用于调试
                    h, s, v = cv2.split(hsv)
                    cv2.imwrite(os.path.join(debug_dir, f"hsv_h_channel_{filename_suffix}.png"), h)
                    cv2.imwrite(os.path.join(debug_dir, f"hsv_s_channel_{filename_suffix}.png"), s)
                    cv2.imwrite(os.path.join(debug_dir, f"hsv_v_channel_{filename_suffix}.png"), v)
                
            else:  # 'value'模式(灰度处理)
                # 从 cv_params 或 context 获取阈值参数
                value_min = cv_params.get('ammo_value_threshold_min', context.get('valueThresholdMin', 120))  # 降低默认值为120，提高弱对比度识别能力
                value_max = cv_params.get('ammo_value_threshold_max', context.get('valueThresholdMax', 255))
                
                # 转换为灰度图
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # 应用灰度阈值
                _, binary = cv2.threshold(gray, value_min, value_max, cv2.THRESH_BINARY)
                
                if debugEnabled:
                    # 简化，只在特定条件下输出参数信息，使用DEBUG级别
                    if "value_debug_output" not in context:
                        logger.debug(f"使用Value过滤: ({value_min}-{value_max})")
                        context["value_debug_output"] = True
                    
                    # 保存灰度图像用于调试
                    cv2.imwrite(os.path.join(debug_dir, f"gray_{filename_suffix}.png"), gray)
            
            # 应用图像处理
            if do_imgProcessing and binary is not None:
                # 保存原始二值化图像用于比较
                if debugEnabled:
                    cv2.imwrite(os.path.join(debug_dir, f"binary_pre_processing_{filename_suffix}.png"), binary)
                
                # 从ocr_params获取形态学处理参数
                ammo_ocr_params = cv_params.get('ammo_ocr_params', {}) if cv_params else {}
                
                # 从cv_params或context获取形态学处理参数
                ksize = ammo_ocr_params.get('morph_kernel_size', 
                       cv_params.get('morph_kernel_size', 
                       context.get('morph_kernel_size', 2)))
                
                iters = ammo_ocr_params.get('morph_iterations', 
                       cv_params.get('morph_iterations', 
                       context.get('morph_iterations', 1)))
                
                # 应用膨胀和闭操作使文字更连贯
                kernel = np.ones((ksize, ksize), np.uint8)
                binary = cv2.dilate(binary, kernel, iterations=iters)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iters)
                
                # 可选的图像缩放处理
                scale_factor = ammo_ocr_params.get('scale_factor', 
                              cv_params.get('scale_factor', 
                              context.get('scale_factor', 1.0)))
                
                if scale_factor > 1.0:
                    binary = cv2.resize(binary, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
                    if debugEnabled:
                        logger.debug(f"应用图像放大: 缩放因子 {scale_factor}")
                
                if debugEnabled:
                    # 只在第一次输出形态学处理信息，使用DEBUG级别
                    if "morph_info_output" not in context:
                        logger.debug(f"应用了形态学处理: 膨胀({ksize}x{ksize}内核,{iters}次迭代) + 闭操作({ksize}x{ksize}内核,{iters}次迭代)")
                        context["morph_info_output"] = True
            
            if debugEnabled and binary is not None:
                # 保存二值化图像(用于调试)
                binary_filename = os.path.join(debug_dir, f"debug_binary_{filename_suffix}.png")
                cv2.imwrite(binary_filename, binary)
            
            # 检查二值图像是否为有效
            if binary is None:
                logger.error("二值化图像生成失败")
                return ""
            
            # 从ammo_ocr_params获取OCR参数
            ammo_ocr_params = cv_params.get('ammo_ocr_params', {}) if cv_params else {}
            
            # 从context获取OCR参数，允许外部控制PSM模式
            # 优先级: ammo_ocr_params > cv_params > context > 默认值
            psm = ammo_ocr_params.get('psm', 
                 cv_params.get('psm', 
                 context.get('psm', 7)))  # 默认使用单行文本模式7，适合多位数字识别
                 
            whitelist = ammo_ocr_params.get('whitelist', 
                       cv_params.get('whitelist', 
                       context.get('whitelist', '0123456789')))
            
            # 设置Tesseract OCR选项
            custom_config = f'--psm {psm} -c tessedit_char_whitelist={whitelist}'
            
            try:
                # 使用Tesseract进行OCR识别
                text = pytesseract.image_to_string(binary, config=custom_config).strip()
                
                # 类型检查，确保返回值是字符串而不是布尔值
                if isinstance(text, bool):
                    logger.warning(f"OCR返回了一个布尔值 ({text}) 而不是文本，将其转换为空字符串")
                    text = ""
                    
                # 删除所有非数字字符(包括句点、空格等)
                import re
                text = re.sub(r'[^0-9]', '', text) if text else ""
                
                if debugEnabled:
                    logger.info(f"OCR识别结果: '{text}' (过滤类型: {filter_type}, PSM模式: {psm})")
                    
                # 记录OCR结果到context（用于调试和结果传递）
                if context is not None:
                    context["ocr_result"] = text
                    
                # 添加兜底处理，避免返回None导致后续处理出错
                return "" if text is None else text
                
            except Exception as inner_e:
                logger.warning(f"Tesseract OCR处理时发生内部错误: {inner_e}")
                if context is not None:
                    context["ocr_error"] = f"内部OCR错误: {inner_e}"
                return ""
            
        except Exception as e:
            logger.error(f"OCR文本识别失败: {e}")
            # 记录错误到context
            if context is not None:
                context["ocr_error"] = str(e)
            return ""

    @staticmethod
    def matchTemplateImg(img_src, img_target, minKeypoints=6, useORB=False, useSIFT=True, debugEnabled=False, debug_dir="debug", filename_suffix=""):
        """
        使用特征点匹配算法匹配两张图片，如果特征点不足则使用模板匹配作为备选
        @param img_src: 源图片(模板)
        @param img_target: 目标图片(场景)
        @param minKeypoints: 最少匹配特征点数
        @param useORB: 是否使用ORB算法
        @param useSIFT: 是否使用SIFT算法
        @param debugEnabled: 是否启用调试
        @param debug_dir: 调试图像保存目录
        @param filename_suffix: 调试文件名后缀
        @return: (good_matches, min_keypoints_required) - 匹配的特征点以及要求的最小特征点数
        """
        good_matches = []
        min_keypoints_required = minKeypoints  # 默认为传入的值
        
        try:
            # 标准化调试目录路径 
            if debugEnabled:
                debug_dir = VisionUtils.normalize_debug_dir(debug_dir)
                
                # 确保文件名后缀不为空
                if not filename_suffix:
                    # 生成一个基于时间戳的默认后缀
                    import datetime
                    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename_suffix = f"auto_{current_time}"
            
            # 检查图像尺寸和通道
            if img_src is None or img_target is None:
                logger.error("源图像或目标图像为空")
                return good_matches, min_keypoints_required
                
            # 确保输入图像是灰度图
            if len(img_src.shape) > 2:
                img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
            else:
                img_src_gray = img_src
                
            if len(img_target.shape) > 2:
                img_target_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
            else:
                img_target_gray = img_target
                
            # 选择使用的特征检测算法
            if useORB:
                # 使用ORB检测算法
                detector = cv2.ORB_create()
                # ORB默认使用汉明距离
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                
                if debugEnabled:
                    # 只在开启调试且指定ORB算法时输出一次，避免重复日志
                    logger.debug("使用ORB特征检测算法")
                
            elif useSIFT:
                # 使用SIFT检测算法
                detector = cv2.SIFT_create()
                # SIFT默认使用L2范数
                bf = cv2.BFMatcher()
                lowe_ratio = 0.75  # David Lowe的论文中推荐的比率阈值
                
                if debugEnabled:
                    # 只在特定条件下输出算法选择日志，避免重复
                    if "sift_info_output" not in globals():
                        globals()["sift_info_output"] = True
                        logger.debug("使用SIFT特征检测算法")
            else:
                logger.error("未指定特征检测算法，默认使用SIFT")
                detector = cv2.SIFT_create()
                bf = cv2.BFMatcher()
                lowe_ratio = 0.75
                
            # 检测特征点和计算描述符
            keypoints_src, descriptors_src = detector.detectAndCompute(img_src_gray, None)
            keypoints_target, descriptors_target = detector.detectAndCompute(img_target_gray, None)
            
            if debugEnabled:
                # 保存带有特征点的图片以进行调试
                # 但不每次都输出日志，减少日志冗余
                img_src_kp = cv2.drawKeypoints(img_src_gray, keypoints_src, None)
                img_target_kp = cv2.drawKeypoints(img_target_gray, keypoints_target, None)
                
                cv2.imwrite(os.path.join(debug_dir, f'src_keypoints_{filename_suffix}.png'), img_src_kp)
                cv2.imwrite(os.path.join(debug_dir, f'target_keypoints_{filename_suffix}.png'), img_target_kp)
            
            if descriptors_src is None or descriptors_target is None or len(descriptors_src) == 0 or len(descriptors_target) == 0:
                logger.warning("无法检测到特征点或计算描述符")
                return good_matches, min_keypoints_required
                
            # 使用不同的匹配方法
            if useSIFT:
                # 对SIFT使用knnMatch并应用Lowe的比率测试
                matches = bf.knnMatch(descriptors_src, descriptors_target, k=2)
                
                # 应用Lowe比率测试
                for m, n in matches:
                    if m.distance < lowe_ratio * n.distance:
                        good_matches.append(m)
            else:
                # 对ORB使用普通匹配
                matches = bf.match(descriptors_src, descriptors_target)
                good_matches = sorted(matches, key=lambda x: x.distance)
                
            if debugEnabled:
                # 根据图像大小动态调整图像尺寸，避免过大
                match_img_height = max(img_src.shape[0], img_target.shape[0])
                scale_factor = min(1.0, 800.0 / match_img_height) if match_img_height > 800 else 1.0
                
                # 只显示前N个最佳匹配，避免图像过于复杂
                max_matches_to_show = 20
                matches_to_draw = good_matches[:max_matches_to_show] if len(good_matches) > max_matches_to_show else good_matches
                
                # 绘制匹配结果
                img_matches = cv2.drawMatches(img_src_gray, keypoints_src, img_target_gray, keypoints_target, matches_to_draw, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                # 应用缩放
                if scale_factor < 1.0:
                    img_matches = cv2.resize(img_matches, None, fx=scale_factor, fy=scale_factor)
                
                # 保存匹配图像
                cv2.imwrite(os.path.join(debug_dir, f'matches_{filename_suffix}.png'), img_matches)
                
                # 只输出一次关键的匹配信息，避免日志冗余
                logger.debug(f"特征点数量 - 源图像: {len(keypoints_src)}, 目标图像: {len(keypoints_target)}, 匹配数: {len(good_matches)}/{min_keypoints_required}")
            
            # 如果特征点匹配不足，尝试模板匹配作为备选方案
            if len(good_matches) < minKeypoints:
                logger.debug(f"特征点匹配不足({len(good_matches)}/{minKeypoints})，使用模板匹配备选")
                
                # 进行模板匹配
                template_result = cv2.matchTemplate(img_target_gray, img_src_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(template_result)
                
                if debugEnabled:
                    # 保存相关性得分图
                    suffix = f"_{filename_suffix}" if filename_suffix else ""
                    corr_path = os.path.join(debug_dir, f"template_correlation{suffix}.png")
                    cv2.imwrite(corr_path, template_result * 255)
                    
                    # 在目标图像上绘制最佳匹配位置
                    h, w = img_src_gray.shape
                    top_left = max_loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    result_img = cv2.cvtColor(img_target_gray.copy(), cv2.COLOR_GRAY2BGR) if len(img_target_gray.shape) < 3 else img_target_gray.copy()
                    cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)
                    pos_path = os.path.join(debug_dir, f"template_match_position{suffix}.png")
                    cv2.imwrite(pos_path, result_img)
                
                # 如果模板匹配分数高于阈值，则认为匹配成功
                if max_val >= 0.8:  # 0.8是模板匹配的阈值
                    logger.info(f"模板匹配分数达到{max_val:.3f} >= 0.8，判定为匹配成功")
                    # 返回足够数量的匹配点以通过检查 
                    return [1] * minKeypoints, min_keypoints_required
            
            return good_matches, min_keypoints_required
            
        except Exception as e:
            logger.error(f"特征匹配失败: {e}")
            return good_matches, min_keypoints_required

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
            # 标准化调试目录路径
            if debugEnabled:
                debug_dir = VisionUtils.normalize_debug_dir(debug_dir)
                
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
            
            # 修改判断逻辑：
            # 1. 如果相关系数大于等于0.9，直接判定为匹配成功
            # 2. 否则使用配置的阈值判断
            if max_val >= 0.9:
                matched = True
                logger.info(f"模板匹配分数达到{max_val:.3f} >= 0.9，直接判定为匹配成功")
            else:
                matched = max_val >= threshold
            
            # 添加详细调试日志
            if debugEnabled:
                # 确保文件名后缀不为空
                if not filename_suffix:
                    import datetime
                    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename_suffix = f"auto_{current_time}"
                    
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

    def recognize_ammo_with_template(self, image, expected_value=None, debug_enabled=False, debug_dir=None):
        """使用模板识别弹药数量"""
        # 初始化带配置路径的识别器
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "digits")
        
        recognizer = AmmoTemplateRecognizer(templates_dir=template_dir, config_path=config_path)
        return recognizer.recognize_number(image, expected_value, 
                                          debugEnabled=debug_enabled, 
                                          debug_dir=self.normalize_debug_dir(debug_dir))

