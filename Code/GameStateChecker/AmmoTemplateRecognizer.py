import os
import cv2
import numpy as np
from loguru import logger
import yaml

class AmmoTemplateRecognizer:
    def __init__(self, templates_dir=None, config_path=None):
        """Initializing ammo digital template recognizer"""
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Set template directory
        if templates_dir is None:
            self.templates_dir = os.path.join(self.base_path, "templates", "digits")
        else:
            self.templates_dir = templates_dir
            
        # Load config file
        if config_path is None:
            config_path = os.path.join(self.base_path, "config.yaml")
        
        logger.info(f"Initializing AmmoTemplateRecognizer, config file path: {config_path}")
        
        # Try to load relative coordinates from config file
        self.use_relative_coords = False
        self.single_digit_bbox_rel = None
        self.double_digit_bbox_rel = None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.debug(f"Successfully loaded config file: {config_path}")
                
            # Get relative coordinates from config file
            active_target = config.get('active_target', 'assaultcube')
            logger.debug(f"Current active target: {active_target}")
            
            target_config = config.get('targets', {}).get(active_target, {})
            logger.debug(f"Target config: {active_target} exists: {active_target in config.get('targets', {})}")
            
            cv_params = target_config.get('cv_params', {})
            logger.debug(f"CV parameter keys: {list(cv_params.keys())}")
            
            # Check if there is relative coordinate configuration
            if 'ammo_bbox_rel' in cv_params:
                rel_coords = cv_params['ammo_bbox_rel']
                logger.info(f"Loaded relative coordinates from config: {rel_coords}")
                
                # Calculate boundary boxes (relative coordinates) for single and double digits
                # Relative coordinates format: [x/width, y/height, w/width, h/height]
                self.single_digit_bbox_rel = rel_coords.copy()
                
                # Adjust double digit coordinates slightly, same y coordinate as single digit, but x coordinate to the right, wider width
                self.double_digit_bbox_rel = rel_coords.copy()
                # Set double digit width to 2.4 times single digit width to ensure full coverage of wider digits (like 2,8,9)
                self.double_digit_bbox_rel[2] *= 2.4
                
                self.use_relative_coords = True
                logger.info(f"Using relative coordinates: {self.use_relative_coords}")
                logger.info(f"Single digit relative coordinates: {self.single_digit_bbox_rel}")
                logger.info(f"Double digit relative coordinates: {self.double_digit_bbox_rel}")
            else:
                logger.warning(f"Relative coordinates ammo_bbox_rel not found in config file. Available parameters: {list(cv_params.keys())}")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            logger.warning("Will use default absolute coordinates")
        
        # Default absolute boundary boxes (when relative coordinates cannot be used)
        self.single_digit_bbox = [877, 1323, 66, 92]   # Single digit boundary box
        self.double_digit_bbox = [887, 1323, 123, 92]  # Double digit boundary box
        
        # Default match threshold
        self.default_match_threshold = 0.45  # Increase threshold for robustness
        
        # Load digit templates
        self.digit_templates = self._load_digit_templates()
        
        logger.info(f"Initialized ammo template recognizer, template directory: {self.templates_dir}")
        logger.info(f"Loaded {len(self.digit_templates)} digit templates, match threshold: {self.default_match_threshold}")
        
        if not self.use_relative_coords:
            logger.info(f"Single digit bbox (absolute): {self.single_digit_bbox}, Double digit bbox (absolute): {self.double_digit_bbox}")
        
        # Check if there are enough templates for recognition
        if len(self.digit_templates) < 5:  # At least 5 digits from 0-9 are needed for basic recognition
            logger.warning("Too few templates loaded, may affect recognition accuracy")
        
        # Output template size information
        max_template_size = 0
        min_template_size = float('inf')
        for digit, template in self.digit_templates.items():
            h, w = template.shape[:2]
            max_template_size = max(max_template_size, h, w)
            min_template_size = min(min_template_size, h, w)
            logger.debug(f"Template {digit} size: {h}x{w} pixels")
        
        logger.debug(f"Template size range: Min {min_template_size} - Max {max_template_size} pixels")
    
    def _load_digit_templates(self):
        """Load all digit templates"""
        templates = {}
        
        # Check if directory exists
        if not os.path.exists(self.templates_dir):
            logger.error(f"Template directory does not exist: {self.templates_dir}")
            return templates
            
        # Load digit templates 0-9
        for digit in range(10):
            template_path = os.path.join(self.templates_dir, f"{digit}.png")
            if os.path.exists(template_path):
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    # Check template size
                    template_height, template_width = template.shape[:2]
                    logger.debug(f"Loading digit template: {digit}.png, size: {template_height}x{template_width} pixels")
                    
                    # Store original template directly, no size adjustment
                    templates[str(digit)] = template
                    logger.debug(f"Successfully loaded digit template: {digit}.png")
                else:
                    logger.warning(f"Cannot read digit template: {template_path}")
            else:
                logger.warning(f"Digit template does not exist: {template_path}")
        
        # Check if all templates are loaded
        if len(templates) < 10:
            logger.warning(f"Failed to load all 10 digit templates, only found {len(templates)}")
            missing_digits = [str(i) for i in range(10) if str(i) not in templates]
            if missing_digits:
                logger.warning(f"Missing digit templates: {', '.join(missing_digits)}")
                
        return templates
    
    def recognize_number(self, image, expected_value=None, 
                         match_threshold=None, debugEnabled=False, debug_dir="debug"):
        """
        Recognize ammo digit from image
        @param image: Input image
        @param expected_value: Expected value (for selecting bbox)
        @param match_threshold: Match threshold (optional, default uses self.default_match_threshold)
        @param debugEnabled: Whether to enable debugging
        @param debug_dir: Debug directory
        @return: (Recognized digit, Confidence)
        """
        try:
            # Check if input image is valid
            if image is None or image.size == 0:
                logger.error("Invalid input image")
                return None, False
                
            # Set match threshold
            if match_threshold is None:
                match_threshold = self.default_match_threshold
                
            # Check if templates are loaded
            if not self.digit_templates:
                logger.error("No digit templates loaded, cannot perform recognition")
                return None, False
            
            # Select bbox or bbox_rel based on expected value
            digit_count = 2 if expected_value is not None and expected_value >= 10 else 1
            
            # Get actual boundary box coordinates
            if self.use_relative_coords:
                # Use relative coordinates to calculate actual boundary box
                img_height, img_width = image.shape[:2]
                
                # Select relative coordinates
                rel_bbox = self.double_digit_bbox_rel if digit_count == 2 else self.single_digit_bbox_rel
                
                # Convert to absolute coordinates [x, y, w, h]
                x = int(rel_bbox[0] * img_width)
                y = int(rel_bbox[1] * img_height)
                w = int(rel_bbox[2] * img_width)
                h = int(rel_bbox[3] * img_height)
                
                bbox = [x, y, w, h]
                
                logger.info(f"Using relative coordinates: {rel_bbox} → Actual bbox: {bbox} (Image size: {img_width}x{img_height})")
            else:
                # Use absolute coordinates
                bbox = self.double_digit_bbox if digit_count == 2 else self.single_digit_bbox
                logger.info(f"Using absolute coordinate bbox: {bbox}")
            
            # Draw boundary box on original image for debugging
            if debugEnabled:
                debug_img = image.copy()
                x, y, w, h = bbox
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add text description
                text = f"Digit{digit_count} Box"
                cv2.putText(debug_img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Ensure directory exists
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir, exist_ok=True)
                
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save original image with boundary box
                bbox_debug_path = os.path.join(debug_dir, f"debug_bbox_{timestamp}.jpg")
                cv2.imwrite(bbox_debug_path, debug_img)
                logger.info(f"Saved debug image with bbox: {bbox_debug_path}")
            
            # Extract ROI
            x, y, w, h = bbox
            # Ensure coordinates are valid
            if y < 0 or x < 0 or y+h > image.shape[0] or x+w > image.shape[1]:
                logger.warning(f"Invalid bbox coordinates: {bbox}, image size: {image.shape}")
                # Try to adjust boundary box to fit image
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    logger.error("Adjusted bbox is invalid, cannot extract ROI")
                    return None, False
                
                logger.info(f"Adjusted bbox to: [{x}, {y}, {w}, {h}]")
            
            roi = image[y:y+h, x:x+w].copy()
            
            # Check if ROI is valid
            if roi is None or roi.size == 0:
                logger.error("Extracted ROI is invalid")
                return None, False
                
            # Save debug image
            if debugEnabled:
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir, exist_ok=True)
                
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                roi_path = os.path.join(debug_dir, f"template_roi_{timestamp}.png")
                cv2.imwrite(roi_path, roi)
                logger.debug(f"Saved template ROI image: {roi_path}, using bbox: [{x}, {y}, {w}, {h}]")
                logger.debug(f"ROI size: {roi.shape}")
            
            # Preprocess image
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) > 2 else roi.copy()
            logger.debug(f"Grayscale image size: {gray.shape}")
            
            # Use grayscale image directly for template matching, no extra processing
            # Process single and double digits separately
            if digit_count == 1:
                return self._match_single_digit(gray, match_threshold, debugEnabled, debug_dir)
            else:
                return self._match_double_digits(gray, match_threshold, debugEnabled, debug_dir)
                
        except Exception as e:
            logger.error(f"Template number recognition failed: {e}")
            import traceback
            logger.debug(f"Error details: {traceback.format_exc()}")
            return None, False
    
    def _match_single_digit(self, image, match_threshold, debugEnabled=False, debug_dir="debug"):
        """Match single digit template"""
        # Create morphological operation kernel
        kernel = np.ones((2, 2), np.uint8)
        
        # Use original image, no extra adaptive threshold processing
        roi_processed = image
        
        # Record matching process
        logger.debug(f"Directly use grayscale image for matching, ROI size: {roi_processed.shape}")
        
        if debugEnabled:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(debug_dir, f"roi_processed_{timestamp}.png"), roi_processed)
            logger.debug(f"ROI processed size: {roi_processed.shape}")

        # Matching result record
        results = []
        processed_templates = {}  # Store processed templates

        # Get ROI height and width
        roi_height, roi_width = roi_processed.shape[:2]

        # Process each digit template in the same way, then perform matching
        for digit, template in self.digit_templates.items():
            # Use original template, no extra processing
            template_processed = template.copy()
            
            # Check if template size is larger than ROI size, if so adjust template size
            template_height, template_width = template_processed.shape[:2]
            if template_height > roi_height or template_width > roi_width:
                # If template size is larger than ROI, adjust template size
                scale_h = roi_height / template_height * 0.9  # Leave 10% margin
                scale_w = roi_width / template_width * 0.9
                scale = min(scale_h, scale_w)  # Use smaller scaling factor to maintain ratio
                
                # Calculate new size
                new_width = int(template_width * scale)
                new_height = int(template_height * scale)
                
                if new_width > 0 and new_height > 0:
                    template_processed = cv2.resize(template_processed, (new_width, new_height), 
                                                  interpolation=cv2.INTER_CUBIC)  # Use cubic interpolation for quality improvement
                    
                    if debugEnabled:
                        logger.debug(f"Adjusted template {digit} size: {template_height}x{template_width} -> {new_height}x{new_width}")
                else:
                    logger.warning(f"Template {digit} cannot be adjusted, skip this template")
                    continue
            
            # Save processed template for debugging
            processed_templates[digit] = template_processed
            
            if debugEnabled:
                logger.debug(f"Template {digit} size: {template_processed.shape}, ROI size: {roi_processed.shape}")
            
            try:
                # Use template matching - try multiple matching methods, find best result
                methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
                max_scores = []
                
                for method in methods:
                    try:
                        result = cv2.matchTemplate(roi_processed, template_processed, method)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        max_scores.append(max_val)
                    except Exception as method_err:
                        logger.warning(f"Matching method {method} failed: {method_err}")
                        max_scores.append(0)
                
                # Use highest score from all methods
                max_val = max(max_scores) if max_scores else 0
                
                logger.debug(f"Digit {digit} matching score: {max_val:.4f}")
                results.append((digit, max_val, None))  # Simplified, no position information saved
            except Exception as e:
                logger.error(f"Template {digit} matching failed: {e}")
                continue

        if not results:
            logger.warning("All template matches failed, cannot recognize digit")
            return None, False

        if debugEnabled:
            # Save processed all template images
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            for digit, proc_tmpl in processed_templates.items():
                cv2.imwrite(os.path.join(debug_dir, f"template_{digit}_processed_{timestamp}.png"), proc_tmpl)

        # Sort results by matching score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        best_match = results[0]
        
        # If best match score is greater than threshold, consider matching successful
        if best_match[1] >= match_threshold:
            logger.info(f"Found matching digit: {best_match[0]}, score: {best_match[1]:.4f}")
            return int(best_match[0]), True
        else:
            logger.warning(f"No matching digit, best attempt: {best_match[0]} score: {best_match[1]:.4f}, threshold: {match_threshold}")
            return int(best_match[0]), False
    
    def _match_double_digits(self, image, match_threshold, debugEnabled=False, debug_dir="debug"):
        """Match two digits, recognize tens and ones separately"""
        try:
            # Get image width
            img_width = image.shape[1]
            
            # Check image width, ensure it can be split
            if img_width < 20:  # Minimum width, adjust based on actual situation
                logger.warning(f"Image width too small to split: {img_width} pixels")
                return None, False
            
            # Improved: Use intelligent split point rather than simple half split
            # Estimate reasonable split point: For 150-160px wide double digits, first digit occupies about 40% width
            # Especially when digit 1 is very narrow, need to give second digit more space
            split_ratio = 0.4  # Split ratio, 40% for first digit
            mid_point = int(img_width * split_ratio)
            
            # Split image into left half (tens digit) and right half (ones digit)
            left_half = image[:, :mid_point]
            right_half = image[:, mid_point:]
            
            # Check if split images are valid
            if left_half.size == 0 or right_half.size == 0:
                logger.warning("Split images are invalid")
                return None, False
            
            if debugEnabled:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                left_path = os.path.join(debug_dir, f"tens_digit_{timestamp}.png")
                right_path = os.path.join(debug_dir, f"ones_digit_{timestamp}.png")
                cv2.imwrite(left_path, left_half)
                cv2.imwrite(right_path, right_half)
                logger.debug(f"Tens digit image size: {left_half.shape}, Ones digit image size: {right_half.shape}")
            
            # Match left half (tens digit)
            tens_digit, tens_success = self._match_single_digit(
                left_half, match_threshold, debugEnabled, debug_dir)
            
            # Match right half (ones digit)
            ones_digit, ones_success = self._match_single_digit(
                right_half, match_threshold, debugEnabled, debug_dir)
            
            # Check if results are valid
            if tens_digit is None and ones_digit is None:
                logger.warning("Neither tens nor ones digit recognized")
                return None, False
            
            # Add repeated digit check: If both recognize the same digit, it may be a false judgment
            if tens_digit == ones_digit:
                logger.warning(f"Detected repeated digit: Tens={tens_digit}, Ones={ones_digit}, This may be a false judgment")
                # Lower confidence, but allow result through (1+1=11 is legal)
                confidence_penalty = 0.2
                if match_threshold > confidence_penalty:
                    match_threshold += confidence_penalty
            
            # If both digits match successfully, return combined result
            if tens_success and ones_success:
                # Try to convert to integer and combine
                try:
                    full_number = int(tens_digit) * 10 + int(ones_digit)
                    logger.info(f"Double digit match successful: {full_number} (Tens={tens_digit}, Ones={ones_digit})")
                    return full_number, True
                except (ValueError, TypeError) as e:
                    logger.error(f"Digit conversion failed: {e}, Tens={tens_digit}, Ones={ones_digit}")
                    return None, False
            # If only tens digit matches successfully, possibly ones processing is有问题或真的只有一位数
            elif tens_success and not ones_success:
                logger.warning(f"Only matched tens digit: {tens_digit}, ones match failed")
                # Try to return it as a single digit
                try:
                    return int(tens_digit), True
                except (ValueError, TypeError):
                    return None, False
            # If only ones digit matches successfully
            elif not tens_success and ones_success:
                logger.warning(f"Only matched ones digit: {ones_digit}, tens match failed")
                try:
                    return int(ones_digit), True
                except (ValueError, TypeError):
                    return None, False
            else:
                logger.warning(f"Double digit match failed: Tens={tens_digit if tens_digit is not None else 'None'}, Ones={ones_digit if ones_digit is not None else 'None'}")
                return None, False
        except Exception as e:
            logger.error(f"Double digit matching process failed: {e}")
            return None, False