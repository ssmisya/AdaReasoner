# insert_image.py
import cv2
import numpy as np
from PIL import Image
import io
import base64
import ast

from tool_server.utils.server_utils import build_logger
from tool_server.utils.utils import load_image, pil_to_base64
from tool_server.tool_workers.offline_workers.base_offline_worker import BaseOfflineWorker
from tool_server.utils.error_codes import *

logger = build_logger("insert_image_worker")

class InsertImage(BaseOfflineWorker):
    """
    将一张图片插入到另一张图片的指定位置
    """
    
    def __init__(self):
        super().__init__(model_name="InsertImage")
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Insert an image into a base image at a specified position defined by a bounding box",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "base_image": {
                            "type": "string",
                            "description": "The base image to insert into (image identifier)"
                        },
                        "image_to_insert": {
                            "type": "string",
                            "description": "The image to be inserted (image identifier)"
                        },
                        "coordinates": {
                            "type": "string",
                            "description": "Coordinates in format '[x_min, y_min, x_max, y_max]', eg., '[100, 100, 200, 200]'. Only absolute pixel values (integers) are supported."
                        },
                        "resize": {
                            "type": "boolean",
                            "description": "Whether to resize the image_to_insert to fit the bounding box (default: true)"
                        }
                    },
                    "required": ["base_image", "image_to_insert", "coordinates"]
                },
            }
        }

    def _execute(self, params):
        """执行图片插入操作"""
        try:
            # 提取参数
            base_image = params["base_image"]
            image_to_insert = params["image_to_insert"]
            coordinates = params["coordinates"]
            resize = params.get("resize", True)
            
            # 将坐标从字符串转换为列表
            if isinstance(coordinates, str):
                try:
                    coordinates = ast.literal_eval(coordinates)
                except:
                    raise ValueError("Invalid coordinates format. Must be '[x_min, y_min, x_max, y_max]'")
            
            # 检查坐标是否有效
            if not isinstance(coordinates, list) or len(coordinates) != 4:
                raise ValueError("Coordinates must be a list with 4 values [x_min, y_min, x_max, y_max]")
                
            x_min, y_min, x_max, y_max = map(int, coordinates)
            
            # 计算目标区域的宽度和高度
            target_width = x_max - x_min
            target_height = y_max - y_min
            
            if target_width <= 0 or target_height <= 0:
                raise ValueError("Invalid bounding box dimensions. Width and height must be positive")
            
            # 将PIL图像转换为OpenCV格式
            base_img = np.array(base_image)
            base_img = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)
            
            insert_img = np.array(image_to_insert)
            insert_img = cv2.cvtColor(insert_img, cv2.COLOR_RGB2BGR)
            
            # 如果需要，调整要插入的图片大小
            if resize:
                insert_img = cv2.resize(insert_img, (target_width, target_height), interpolation=cv2.INTER_AREA)
            else:
                # 如果不调整大小，截取图片或使用较小的尺寸
                h, w = insert_img.shape[:2]
                target_width = min(target_width, w)
                target_height = min(target_height, h)
                insert_img = insert_img[:target_height, :target_width]
            
            # 将图片插入到目标位置
            h, w = insert_img.shape[:2]
            base_img[y_min:y_min+h, x_min:x_min+w] = insert_img
            
            # 将结果转换回PIL格式
            result_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_img)
            
            # 转换为base64并返回结果
            image_base64 = pil_to_base64(result_pil)
            
            return {
                "status": "success",
                "edited_image": image_base64,
                "message": "Image inserted successfully",
                "error_code": SUCCESS
            }
            
        except Exception as e:
            logger.error(f"Error inserting image: {str(e)}")
            return {
                "status": "failed",
                "message": f"Error inserting image: {str(e)}",
                "error_code": TOOL_RUN_FAILED
            }
    
    def verify_tool_parameter(self, params):
        """验证工具的输入参数"""
        try:
            # 检查必要参数是否存在
            required_params = ["base_image", "image_to_insert", "coordinates"]
            for param in required_params:
                if param not in params:
                    raise ValueError(f"Missing required parameter: {param}")
            
            # 加载图像
            base_image = params["base_image"]
            base_image_loaded = load_image(base_image)
            
            image_to_insert = params["image_to_insert"]
            image_to_insert_loaded = load_image(image_to_insert)
            
            # 验证坐标
            coordinates = params["coordinates"]
            if isinstance(coordinates, str):
                try:
                    coords_list = ast.literal_eval(coordinates)
                    if not isinstance(coords_list, list) or len(coords_list) != 4:
                        raise ValueError("coordinates must be in format '[x_min, y_min, x_max, y_max]'")
                    x_min, y_min, x_max, y_max = coords_list
                    # 检查坐标是否为有效值
                    if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min:
                        raise ValueError("Invalid coordinate values")
                    # 检查坐标是否在基础图片范围内
                    width, height = base_image_loaded.size
                    if x_max > width or y_max > height:
                        raise ValueError(f"Coordinates exceed base image dimensions ({width}x{height})")
                except Exception as e:
                    raise ValueError(f"Invalid coordinates format: {str(e)}")
            else:
                raise ValueError("coordinates must be a string in format '[x_min, y_min, x_max, y_max]'")
            
            # 验证resize参数（如果存在）
            resize = params.get("resize", True)
            if isinstance(resize, str):
                if resize.lower() == "true":
                    resize = True
                elif resize.lower() == "false":
                    resize = False
                else:
                    raise ValueError("resize must be a boolean value ('true' or 'false')")
            
            # 创建新的参数字典
            new_params = {
                "base_image": base_image_loaded,
                "image_to_insert": image_to_insert_loaded,
                "coordinates": coordinates,
                "resize": resize
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