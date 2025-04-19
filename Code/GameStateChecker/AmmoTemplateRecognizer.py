import os
import cv2
import numpy as np
from loguru import logger
import yaml

class AmmoTemplateRecognizer:
    def __init__(self, templates_dir=None, config_path=None):
        """初始化弹药数字模板识别器"""
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        
        # 设置模板目录
        if templates_dir is None:
            self.templates_dir = os.path.join(self.base_path, "templates", "digits")
        else:
            self.templates_dir = templates_dir
            
        # 加载配置文件
        if config_path is None:
            config_path = os.path.join(self.base_path, "config.yaml")
        
        logger.info(f"初始化AmmoTemplateRecognizer，配置文件路径: {config_path}")
        
        # 尝试从配置文件中加载相对坐标
        self.use_relative_coords = False
        self.single_digit_bbox_rel = None
        self.double_digit_bbox_rel = None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.debug(f"成功加载配置文件: {config_path}")
                
            # 从配置文件中获取相对坐标
            active_target = config.get('active_target', 'assaultcube')
            logger.debug(f"当前激活的目标: {active_target}")
            
            target_config = config.get('targets', {}).get(active_target, {})
            logger.debug(f"目标配置: {active_target} 是否存在: {active_target in config.get('targets', {})}")
            
            cv_params = target_config.get('cv_params', {})
            logger.debug(f"CV参数键: {list(cv_params.keys())}")
            
            # 检查是否有相对坐标配置
            if 'ammo_bbox_rel' in cv_params:
                rel_coords = cv_params['ammo_bbox_rel']
                logger.info(f"从配置文件加载相对坐标: {rel_coords}")
                
                # 计算单位数和双位数的边界框（相对坐标）
                # 相对坐标的格式为[x/width, y/height, w/width, h/height]
                self.single_digit_bbox_rel = rel_coords.copy()
                
                # 双位数相对坐标稍微调整，与单位数相同的y坐标，但x坐标靠右一点，宽度更大
                self.double_digit_bbox_rel = rel_coords.copy()
                # 将双位数的宽度设为单位数的2.4倍，确保完全包含较宽的数字(如2,8,9)
                self.double_digit_bbox_rel[2] *= 2.4
                
                self.use_relative_coords = True
                logger.info(f"使用相对坐标: {self.use_relative_coords}")
                logger.info(f"单位数相对坐标: {self.single_digit_bbox_rel}")
                logger.info(f"双位数相对坐标: {self.double_digit_bbox_rel}")
            else:
                logger.warning(f"配置文件中未找到相对坐标ammo_bbox_rel，将使用绝对坐标。可用参数: {list(cv_params.keys())}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            logger.warning("将使用默认的绝对坐标")
        
        # 单位数和双位数的默认绝对边界框（当无法使用相对坐标时）
        self.single_digit_bbox = [877, 1323, 66, 92]   # 单位数边界框
        self.double_digit_bbox = [887, 1323, 123, 92]  # 双位数边界框
        
        # 默认匹配阈值
        self.default_match_threshold = 0.45  # 提高阈值以增强鲁棒性
        
        # 加载数字模板
        self.digit_templates = self._load_digit_templates()
        
        logger.info(f"初始化弹药模板识别器，模板目录: {self.templates_dir}")
        logger.info(f"已加载 {len(self.digit_templates)} 个数字模板，匹配阈值: {self.default_match_threshold}")
        
        if not self.use_relative_coords:
            logger.info(f"单位数边界框(绝对): {self.single_digit_bbox}, 双位数边界框(绝对): {self.double_digit_bbox}")
        
        # 检查是否有足够的模板用于识别
        if len(self.digit_templates) < 5:  # 至少需要0-9中的5个数字才能进行基本识别
            logger.warning("加载的模板数量过少，可能会影响识别准确性")
        
        # 输出模板尺寸信息
        max_template_size = 0
        min_template_size = float('inf')
        for digit, template in self.digit_templates.items():
            h, w = template.shape[:2]
            max_template_size = max(max_template_size, h, w)
            min_template_size = min(min_template_size, h, w)
            logger.debug(f"模板 {digit} 尺寸: {h}x{w}像素")
        
        logger.debug(f"模板尺寸范围: 最小 {min_template_size} - 最大 {max_template_size} 像素")
    
    def _load_digit_templates(self):
        """加载所有数字模板"""
        templates = {}
        
        # 检查目录是否存在
        if not os.path.exists(self.templates_dir):
            logger.error(f"模板目录不存在: {self.templates_dir}")
            return templates
            
        # 加载0-9的数字模板
        for digit in range(10):
            template_path = os.path.join(self.templates_dir, f"{digit}.png")
            if os.path.exists(template_path):
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    # 检查模板尺寸
                    template_height, template_width = template.shape[:2]
                    logger.debug(f"加载数字模板: {digit}.png, 尺寸: {template_height}x{template_width}像素")
                    
                    # 直接存储原始模板，不进行大小调整
                    templates[str(digit)] = template
                    logger.debug(f"成功加载数字模板: {digit}.png")
                else:
                    logger.warning(f"无法读取数字模板: {template_path}")
            else:
                logger.warning(f"数字模板不存在: {template_path}")
        
        # 检查是否所有模板都已加载
        if len(templates) < 10:
            logger.warning(f"未能加载所有10个数字模板，只找到 {len(templates)} 个")
            missing_digits = [str(i) for i in range(10) if str(i) not in templates]
            if missing_digits:
                logger.warning(f"缺少以下数字模板: {', '.join(missing_digits)}")
                
        return templates
    
    def recognize_number(self, image, expected_value=None, 
                         match_threshold=None, debugEnabled=False, debug_dir="debug"):
        """
        识别图像中的弹药数字
        @param image: 输入图像
        @param expected_value: 预期值（用于选择bbox）
        @param match_threshold: 匹配阈值（可选，默认使用self.default_match_threshold）
        @param debugEnabled: 是否启用调试
        @param debug_dir: 调试目录
        @return: (识别的数字, 置信度)
        """
        try:
            # 检查输入图像是否有效
            if image is None or image.size == 0:
                logger.error("输入图像无效")
                return None, False
                
            # 设置匹配阈值
            if match_threshold is None:
                match_threshold = self.default_match_threshold
                
            # 检查模板是否加载
            if not self.digit_templates:
                logger.error("未加载数字模板，无法进行识别")
                return None, False
            
            # 根据预期值选择bbox或bbox_rel
            digit_count = 2 if expected_value is not None and expected_value >= 10 else 1
            
            # 获取真实的边界框坐标
            if self.use_relative_coords:
                # 使用相对坐标计算实际边界框
                img_height, img_width = image.shape[:2]
                
                # 选择相对坐标
                rel_bbox = self.double_digit_bbox_rel if digit_count == 2 else self.single_digit_bbox_rel
                
                # 转换为绝对坐标 [x, y, w, h]
                x = int(rel_bbox[0] * img_width)
                y = int(rel_bbox[1] * img_height)
                w = int(rel_bbox[2] * img_width)
                h = int(rel_bbox[3] * img_height)
                
                bbox = [x, y, w, h]
                
                logger.info(f"使用相对坐标: {rel_bbox} → 实际边界框: {bbox} (图像尺寸: {img_width}x{img_height})")
            else:
                # 使用绝对坐标
                bbox = self.double_digit_bbox if digit_count == 2 else self.single_digit_bbox
                logger.info(f"使用绝对坐标边界框: {bbox}")
            
            # 在原始图像上绘制边界框用于调试
            if debugEnabled:
                debug_img = image.copy()
                x, y, w, h = bbox
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 添加文本说明
                text = f"Digit{digit_count} Box"
                cv2.putText(debug_img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 确保目录存在
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir, exist_ok=True)
                
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 保存带边界框的原始图像
                bbox_debug_path = os.path.join(debug_dir, f"debug_bbox_{timestamp}.jpg")
                cv2.imwrite(bbox_debug_path, debug_img)
                logger.info(f"已保存带边界框的调试图像: {bbox_debug_path}")
            
            # 提取ROI
            x, y, w, h = bbox
            # 确保坐标有效
            if y < 0 or x < 0 or y+h > image.shape[0] or x+w > image.shape[1]:
                logger.warning(f"边界框坐标无效: {bbox}, 图像尺寸: {image.shape}")
                # 尝试调整边界框使其适合图像
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    logger.error("调整后的边界框无效，无法提取ROI")
                    return None, False
                
                logger.info(f"已调整边界框至: [{x}, {y}, {w}, {h}]")
            
            roi = image[y:y+h, x:x+w].copy()
            
            # 检查ROI是否有效
            if roi is None or roi.size == 0:
                logger.error("提取的ROI区域无效")
                return None, False
                
            # 保存调试图像
            if debugEnabled:
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir, exist_ok=True)
                
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                roi_path = os.path.join(debug_dir, f"template_roi_{timestamp}.png")
                cv2.imwrite(roi_path, roi)
                logger.debug(f"已保存模板ROI图像: {roi_path}, 使用bbox: [{x}, {y}, {w}, {h}]")
                logger.debug(f"ROI尺寸: {roi.shape}")
            
            # 预处理图像
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) > 2 else roi.copy()
            logger.debug(f"灰度图像尺寸: {gray.shape}")
            
            # 使用灰度图直接进行模板匹配，不做额外处理
            # 对单位数和两位数分别处理
            if digit_count == 1:
                return self._match_single_digit(gray, match_threshold, debugEnabled, debug_dir)
            else:
                return self._match_double_digits(gray, match_threshold, debugEnabled, debug_dir)
                
        except Exception as e:
            logger.error(f"模板识别数字失败: {e}")
            import traceback
            logger.debug(f"错误详情: {traceback.format_exc()}")
            return None, False
    
    def _match_single_digit(self, image, match_threshold, debugEnabled=False, debug_dir="debug"):
        """匹配单个数字模板"""
        # 创建形态学操作的kernel
        kernel = np.ones((2, 2), np.uint8)
        
        # 使用原始图像，不进行额外的自适应阈值处理
        roi_processed = image
        
        # 记录匹配过程
        logger.debug(f"直接使用灰度图像进行匹配，ROI尺寸: {roi_processed.shape}")
        
        if debugEnabled:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(debug_dir, f"roi_processed_{timestamp}.png"), roi_processed)
            logger.debug(f"ROI处理后尺寸: {roi_processed.shape}")

        # 匹配结果记录
        results = []
        processed_templates = {}  # 存储处理后的模板

        # 获取ROI的高度和宽度
        roi_height, roi_width = roi_processed.shape[:2]

        # 对每个数字模板进行相同的预处理，然后进行匹配
        for digit, template in self.digit_templates.items():
            # 使用原始模板，不进行额外处理
            template_processed = template.copy()
            
            # 检查模板尺寸是否大于ROI尺寸，如果是则调整模板尺寸
            template_height, template_width = template_processed.shape[:2]
            if template_height > roi_height or template_width > roi_width:
                # 如果模板尺寸大于ROI，则调整模板大小
                scale_h = roi_height / template_height * 0.9  # 留出10%的余量
                scale_w = roi_width / template_width * 0.9
                scale = min(scale_h, scale_w)  # 使用较小的缩放因子保持比例
                
                # 计算新尺寸
                new_width = int(template_width * scale)
                new_height = int(template_height * scale)
                
                if new_width > 0 and new_height > 0:
                    template_processed = cv2.resize(template_processed, (new_width, new_height), 
                                                  interpolation=cv2.INTER_CUBIC)  # 使用立方插值提高质量
                    
                    if debugEnabled:
                        logger.debug(f"调整模板 {digit} 尺寸: {template_height}x{template_width} -> {new_height}x{new_width}")
                else:
                    logger.warning(f"模板 {digit} 无法调整大小，跳过此模板")
                    continue
            
            # 保存处理后的模板供调试
            processed_templates[digit] = template_processed
            
            if debugEnabled:
                logger.debug(f"模板 {digit} 尺寸: {template_processed.shape}, ROI尺寸: {roi_processed.shape}")
            
            try:
                # 使用模板匹配 - 尝试多种匹配方法，找到最佳结果
                methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
                max_scores = []
                
                for method in methods:
                    try:
                        result = cv2.matchTemplate(roi_processed, template_processed, method)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        max_scores.append(max_val)
                    except Exception as method_err:
                        logger.warning(f"匹配方法 {method} 失败: {method_err}")
                        max_scores.append(0)
                
                # 使用所有方法中的最高分数
                max_val = max(max_scores) if max_scores else 0
                
                logger.debug(f"数字 {digit} 匹配分数: {max_val:.4f}")
                results.append((digit, max_val, None))  # 简化，不保存位置信息
            except Exception as e:
                logger.error(f"模板 {digit} 匹配失败: {e}")
                continue

        if not results:
            logger.warning("所有模板匹配都失败，无法识别数字")
            return None, False

        if debugEnabled:
            # 保存处理后的所有模板图像
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            for digit, proc_tmpl in processed_templates.items():
                cv2.imwrite(os.path.join(debug_dir, f"template_{digit}_processed_{timestamp}.png"), proc_tmpl)

        # 按匹配分数降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        best_match = results[0]
        
        # 如果最佳匹配的分数大于阈值，则认为匹配成功
        if best_match[1] >= match_threshold:
            logger.info(f"找到匹配的数字: {best_match[0]}, 分数: {best_match[1]:.4f}")
            return int(best_match[0]), True
        else:
            logger.warning(f"无匹配数字，最佳尝试: {best_match[0]} 分数: {best_match[1]:.4f}, 阈值: {match_threshold}")
            return int(best_match[0]), False
    
    def _match_double_digits(self, image, match_threshold, debugEnabled=False, debug_dir="debug"):
        """匹配两位数字，分别识别十位和个位"""
        try:
            # 获取图像宽度
            img_width = image.shape[1]
            
            # 检查图像宽度，确保可以分割
            if img_width < 20:  # 最小宽度，根据实际情况调整
                logger.warning(f"图像宽度过小无法分割: {img_width}像素")
                return None, False
            
            # 改进：使用智能分割点而非简单对半分
            # 估算合理的分割点：对于宽度为150-160px的双位数，第一个数字约占40%宽度
            # 特别是数字1很窄时，需要给第二个数字更多空间
            split_ratio = 0.4  # 分割比例，40%给第一个数字
            mid_point = int(img_width * split_ratio)
            
            # 分割图像为左半部分(十位数)和右半部分(个位数)
            left_half = image[:, :mid_point]
            right_half = image[:, mid_point:]
            
            # 检查分割后的图像是否有效
            if left_half.size == 0 or right_half.size == 0:
                logger.warning("分割后的图像无效")
                return None, False
            
            if debugEnabled:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                left_path = os.path.join(debug_dir, f"tens_digit_{timestamp}.png")
                right_path = os.path.join(debug_dir, f"ones_digit_{timestamp}.png")
                cv2.imwrite(left_path, left_half)
                cv2.imwrite(right_path, right_half)
                logger.debug(f"十位数图像尺寸: {left_half.shape}, 个位数图像尺寸: {right_half.shape}")
            
            # 匹配左半部分(十位数)
            tens_digit, tens_success = self._match_single_digit(
                left_half, match_threshold, debugEnabled, debug_dir)
            
            # 匹配右半部分(个位数)
            ones_digit, ones_success = self._match_single_digit(
                right_half, match_threshold, debugEnabled, debug_dir)
            
            # 检查结果是否有效
            if tens_digit is None and ones_digit is None:
                logger.warning("十位数和个位数均未识别")
                return None, False
            
            # 增加重复数字检查：如果左右识别出相同的数字，可能是误判
            if tens_digit == ones_digit:
                logger.warning(f"检测到重复数字: 十位={tens_digit}, 个位={ones_digit}，这可能是误判")
                # 降低置信度，但允许结果通过（1+1=11是合法的）
                confidence_penalty = 0.2
                if match_threshold > confidence_penalty:
                    match_threshold += confidence_penalty
            
            # 如果两位数字都匹配成功，返回组合结果
            if tens_success and ones_success:
                # 尝试转换为整数并组合
                try:
                    full_number = int(tens_digit) * 10 + int(ones_digit)
                    logger.info(f"双位数匹配成功: {full_number} (十位={tens_digit}, 个位={ones_digit})")
                    return full_number, True
                except (ValueError, TypeError) as e:
                    logger.error(f"数字转换失败: {e}, 十位={tens_digit}, 个位={ones_digit}")
                    return None, False
            # 如果只有十位数匹配成功，可能是个位处理有问题或真的只有一位数
            elif tens_success and not ones_success:
                logger.warning(f"只匹配到十位数字: {tens_digit}，个位匹配失败")
                # 尝试将其作为单个数字返回
                try:
                    return int(tens_digit), True
                except (ValueError, TypeError):
                    return None, False
            # 如果只有个位数匹配成功
            elif not tens_success and ones_success:
                logger.warning(f"只匹配到个位数字: {ones_digit}，十位匹配失败")
                try:
                    return int(ones_digit), True
                except (ValueError, TypeError):
                    return None, False
            else:
                logger.warning(f"双位数匹配失败: 十位={tens_digit if tens_digit is not None else 'None'}, 个位={ones_digit if ones_digit is not None else 'None'}")
                return None, False
        except Exception as e:
            logger.error(f"双位数匹配过程出错: {e}")
            return None, False