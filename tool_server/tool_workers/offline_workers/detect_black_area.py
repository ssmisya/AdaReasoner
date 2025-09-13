# detect_black_area.py
import cv2
import numpy as np
from PIL import Image
import io
import base64

from tool_server.utils.server_utils import build_logger
from tool_server.utils.utils import load_image
from tool_server.tool_workers.offline_workers.base_offline_worker import BaseOfflineWorker
from tool_server.utils.error_codes import *

logger = build_logger("detect_black_area_worker")

class DetectBlackArea(BaseOfflineWorker):
    """
    检测图像中的纯黑区域，并返回包围这些区域的方形边界框
    """
    
    def __init__(self):
        super().__init__(model_name="DetectBlackArea")
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Detect pure black areas in an image and return their bounding boxes",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The image to analyze (image identifier)"
                        },
                        "threshold": {
                            "type": "integer",
                            "description": "Brightness threshold (0-255) below which pixels are considered black (default: 1)"
                        },
                        "min_area": {
                            "type": "integer",
                            "description": "Minimum area (in pixels) for a region to be considered (default: 100)"
                        }
                    },
                    "required": ["image"]
                },
            }
        }

    def _execute(self, params):
        """执行黑色区域检测"""
        try:
            # 提取参数
            image = params["image"]
            threshold = params.get("threshold", 1)
            min_area = params.get("min_area", 200)
            
            # 将PIL图像转换为OpenCV格式
            pil_image = image
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # 检测黑色区域
            bounding_boxes = self.detect_black_regions(cv_image, threshold, min_area)
            
            # 将边界框格式从 [x, y, width, height] 转换为 '[x_min, y_min, x_max, y_max]'
            formatted_boxes = []
            for bbox in bounding_boxes:
                x, y, w, h = bbox
                formatted_box = f"[{x}, {y}, {x+w}, {y+h}]"
                formatted_boxes.append(formatted_box)
            
            return {
                "status": "success",
                "bounding_boxes": formatted_boxes,
                "message": f"Found {len(bounding_boxes)} black areas",
                "error_code": SUCCESS
            }
            
        except Exception as e:
            logger.error(f"Error detecting black areas: {str(e)}")
            return {
                "status": "failed",
                "message": f"Error detecting black areas: {str(e)}",
                "error_code": TOOL_RUN_FAILED
            }
    
    def detect_black_regions(self, image, threshold, min_area):
        """
        检测图像中的黑色区域并返回它们的边界框
        
        Args:
            image: OpenCV格式的图像
            threshold: 像素亮度阈值，低于此值被视为黑色
            min_area: 最小区域大小（像素数）
            
        Returns:
            list: 边界框列表，每个边界框为 [x, y, width, height]
        """
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 二值化图像以识别黑色区域
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选并获取边界框
        bounding_boxes_dict_list = []
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 仅考虑超过最小面积的轮廓
            if area >= min_area:
                # 获取边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes_dict_list.append({
                    "grounding_box": [x, y, w, h],
                    "area": area
                })
        bounding_boxes_dict_list = sorted(bounding_boxes_dict_list, key=lambda x: x["area"], reverse=True)
        bounding_boxes = [item["grounding_box"] for item in bounding_boxes_dict_list]
        
        return bounding_boxes
    
    def verify_tool_parameter(self, params):
        """验证工具的输入参数"""
        try:
            # 检查image是否存在
            if "image" not in params:
                raise ValueError("Missing required parameter: image")
            
            # 加载图像
            image = params["image"]
            loaded_image = load_image(image)
            
            # 验证可选参数
            threshold = params.get("threshold", 1)
            if isinstance(threshold, str):
                threshold = int(threshold)
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 255:
                raise ValueError("threshold must be an integer between 0 and 255")
            
            min_area = params.get("min_area", 200)
            if isinstance(min_area, str):
                min_area = int(min_area)
            if not isinstance(min_area, (int, float)) or min_area < 0:
                raise ValueError("min_area must be a positive integer")
            
            # 创建新的参数字典
            new_params = {
                "image": loaded_image,
                "threshold": threshold,
                "min_area": min_area
            }
            
            return {
                "params_qualified_reward": 1,
                "params_qualified": True,
                "new_params": new_params
            }
            
        except Exception as e:
            error_info = str(e)
            return {
                "params_qualified_reward": 0,
                "params_qualified": False,
                "error_info": error_info,
                "new_params": None
            }