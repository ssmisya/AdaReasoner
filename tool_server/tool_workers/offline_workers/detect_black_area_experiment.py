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
                "description": "Detect pure black areas in an image and return their bounding boxes, prioritizing rectangular regions",
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
                            "description": "Minimum area (in pixels) for a region to be considered (default: 200)"
                        },
                        "rect_similarity": {
                            "type": "number",
                            "description": "Threshold for rectangle similarity (0.0-1.0). Higher value means stricter rectangle detection (default: 0.85)"
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
            rect_similarity = params.get("rect_similarity", 0.85)
            
            # 将PIL图像转换为OpenCV格式
            pil_image = image
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # 检测黑色区域
            bounding_boxes = self.detect_black_regions(cv_image, threshold, min_area, rect_similarity)
            
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
    
    def detect_black_regions(self, image, threshold, min_area, rect_similarity=0.85):
        """
        检测图像中的黑色区域并返回它们的边界框，优先检测矩形区块
        
        Args:
            image: OpenCV格式的图像
            threshold: 像素亮度阈值，低于此值被视为黑色
            min_area: 最小区域大小（像素数）
            rect_similarity: 矩形相似度阈值 (0.0-1.0)，值越高要求越严格
            
        Returns:
            list: 边界框列表，每个边界框为 [x, y, width, height]
        """
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 二值化图像以识别黑色区域
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 存储所有符合条件的区域信息
        regions = []
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 仅考虑超过最小面积的轮廓
            if area >= min_area:
                # 获取边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                rect_area = w * h
                
                # 计算轮廓与其边界矩形的面积比
                # 如果比值接近1，则该轮廓接近矩形
                area_ratio = area / rect_area if rect_area > 0 else 0
                
                # 类型1：如果它本身就是矩形（面积比很高）
                if area_ratio >= rect_similarity:
                    regions.append({
                        "bbox": [x, y, w, h],
                        "area": area,
                        "type": "rectangle",
                        "score": area_ratio  # 使用面积比作为评分
                    })
                    continue
                
                # 类型2：近似矩形，但有一些不规则突出部分
                # 尝试使用多边形逼近，看是否可以得到一个矩形
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 如果逼近结果是一个四边形，那么可能是一个近似矩形
                if len(approx) == 4:
                    # 计算逼近多边形的面积
                    approx_area = cv2.contourArea(approx)
                    # 获取逼近多边形的边界矩形
                    approx_x, approx_y, approx_w, approx_h = cv2.boundingRect(approx)
                    approx_rect_area = approx_w * approx_h
                    # 计算面积比
                    approx_ratio = approx_area / approx_rect_area if approx_rect_area > 0 else 0
                    
                    if approx_ratio >= rect_similarity:
                        regions.append({
                            "bbox": [approx_x, approx_y, approx_w, approx_h],
                            "area": approx_area,
                            "type": "approx_rectangle",
                            "score": approx_ratio * 0.9  # 稍微降低评分，优先级低于完美矩形
                        })
                        continue
                
                # 类型3：不规则形状，查找其中的最大内接矩形
                # 创建轮廓的掩码
                mask = np.zeros(binary.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                
                # 使用距离变换查找最大内接矩形
                dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
                _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
                
                # 最大值给出了最大内接圆的半径，(max_loc.x, max_loc.y) 是内接圆的中心
                max_rect_size = int(max_val * 2)  # 矩形边长约为内接圆直径
                
                # 确保矩形不会超出轮廓范围
                max_rect_x = max(0, max_loc[0] - max_val)
                max_rect_y = max(0, max_loc[1] - max_val)
                max_rect_w = min(binary.shape[1] - max_rect_x, max_rect_size)
                max_rect_h = min(binary.shape[0] - max_rect_y, max_rect_size)
                
                # 确保找到的矩形确实在黑色区域内
                max_rect_area = max_rect_w * max_rect_h
                
                if max_rect_area >= min_area:
                    regions.append({
                        "bbox": [int(max_rect_x), int(max_rect_y), int(max_rect_w), int(max_rect_h)],
                        "area": max_rect_area,
                        "type": "max_inscribed_rectangle",
                        "score": 0.8 * (max_rect_area / area)  # 使用内接矩形与轮廓面积比作为评分
                    })
        
        # 按评分对区域进行排序
        regions.sort(key=lambda x: (x["score"], x["area"]), reverse=True)
        
        # 提取排序后的边界框
        bounding_boxes = [region["bbox"] for region in regions]
        
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
            
            rect_similarity = params.get("rect_similarity", 0.85)
            if isinstance(rect_similarity, str):
                rect_similarity = float(rect_similarity)
            if not isinstance(rect_similarity, (int, float)) or rect_similarity < 0 or rect_similarity > 1:
                raise ValueError("rect_similarity must be a number between 0.0 and 1.0")
            
            # 创建新的参数字典
            new_params = {
                "image": loaded_image,
                "threshold": threshold,
                "min_area": min_area,
                "rect_similarity": rect_similarity
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