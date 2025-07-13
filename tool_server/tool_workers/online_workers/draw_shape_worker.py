"""
A model worker executes the model.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import base64
import uuid
import os
import traceback
import re
from io import BytesIO
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional
from fastapi import Request

from tool_server.utils.server_utils import build_logger
from tool_server.utils.utils import load_image, pil_to_base64
from tool_server.utils.worker_arguments import WorkerArguments
from tool_server.utils.error_codes import *
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"draw_shape_worker_{worker_id}.log")

@dataclass
class DrawShapeArguments(WorkerArguments):
    max_concurrency: int = field(
        default=10,
        metadata={"help": "Maximum number of concurrent requests the model can handle"}
    )

class DrawShapeWorker(BaseToolWorker):
    def __init__(self, worker_arguments: DrawShapeArguments = None):
        # 在调用父类初始化前先设置模型名称
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "DrawShape"
            
        # 设置最大并发数
        if worker_arguments:
            self.max_concurrency = worker_arguments.max_concurrency
            # 更新基类中的限制值
            worker_arguments.limit_model_concurrency = self.max_concurrency
            
        super().__init__(worker_arguments)
        
        # 初始化线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_concurrency)
        logger.info(f"Initialized thread pool with {self.max_concurrency} workers")
            
        self.instruction = {
            "type": "function",
            "function": {
                "name": "DrawShape",
                "description": 
                    "Draw geometric shapes (rectangle, ellipse, or circle) with red borders at specified bounding box locations on the image. Returns the edited image in base64 format.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier of the image to edit, e.g., 'img_1'"
                        },
                        "bboxes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "shape": {
                                        "type": "string",
                                        "enum": ["rectangle", "ellipse", "circle"],
                                        "description": "Type of shape to draw."
                                    },
                                    "coords": {
                                        "type": "array",
                                        "items": { "type": "integer" },
                                        "description": "Bounding box coordinates in [x_min, y_min, x_max, y_max] format."
                                    }
                                },
                                "required": ["shape", "coords"]
                            },
                            "description": "List of shapes to draw and their coordinates."
                        }
                    },
                    "required": ["image", "bboxes"]
                }
            }
        }
        
    def init_model(self):
        logger.info(f"No need to initialize model {self.model_name}.")
        self.model = None
        
    def generate(self, params):
        """执行绘制形状操作并返回结果"""
        return self._process_draw_shape_request(params)
        
    def _process_draw_shape_request(self, params):
        """分离出实际处理逻辑，方便并发调用"""
        tool_reward = 2.0
        # 计算Parameter Name Matching
        param_keys = set(params.keys())
        required_keys = set(self.instruction["function"]["parameters"]["required"])
        parameter_name_match_reward = len(param_keys & required_keys) / len(required_keys | param_keys)
        tool_reward = tool_reward + parameter_name_match_reward
        # 参数名称没有完全匹配，直接返回
        if parameter_name_match_reward < 1:
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": "Invalid parameters: expected keys: image, bboxes.",
                "error_code": INVALID_PARAMETERS,
                "tool_reward": tool_reward
            }
        
        required_keys_num = len(required_keys)
        # 初始化参数合规计数器
        correct_param_content_num = 0
        
        try:
            # 提取输入参数
            try:
                image_data = params["image"]
                bboxes = params.get("bboxes", [])
                
                if not bboxes:
                    raise KeyError("'bboxes' not found in params or empty")
            except Exception as e:
                message = f"Invalid parameters: expected keys: image, bboxes. Error: {str(e)}"
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": message,
                    "error_code": INVALID_PARAMETERS,
                    "tool_reward": tool_reward
                }
                return pred_dict
            
            # 加载图像
            try:
                image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")  # 图像加载成功，参数合规+1
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Failed to load image: {str(e)}",
                    "error_code": CANNOT_LOAD_IMAGE,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                }
                return pred_dict
            correct_param_content_num += 1
            
            # 创建绘图对象
            draw = ImageDraw.Draw(image)
            
            # 绘制所有形状
            for bbox in bboxes:
                try:
                    shape_type = bbox.get("shape")
                    coords = bbox.get("coords")
                    
                    if not shape_type or not coords:
                        logger.warning(f"Skipping invalid bbox: {bbox}")
                        continue
                    
                    if len(coords) != 4:
                        logger.warning(f"Skipping bbox with invalid coordinates: {coords}")
                        continue
                    
                    # 验证形状类型是否为支持的类型
                    if shape_type not in ["rectangle", "ellipse", "circle"]:
                        message = f"Unsupported shape type: {shape_type}. Supported types are: rectangle, ellipse, circle."
                        pred_dict = {
                            "tool_response_from": self.model_name,
                            "status": "failed",
                            "message": message,
                            "error_code": INVALID_PARAMETERS,
                            "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                        }
                        return pred_dict
                    
                    x_min, y_min, x_max, y_max = coords
                    
                    # 验证坐标是否在图像范围内
                    if x_min < 0 or y_min < 0 or x_max > image.width or y_max > image.height:
                        message = f"Coordinates {coords} are outside the image dimensions ({image.width}x{image.height})."
                        pred_dict = {
                            "tool_response_from": self.model_name,
                            "status": "failed",
                            "message": message,
                            "error_code": INVALID_PARAMETERS,
                            "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                        }
                        return pred_dict
                    
                    # 绘制不同类型的形状
                    if shape_type == "rectangle":
                        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
                    elif shape_type == "ellipse":
                        draw.ellipse([x_min, y_min, x_max, y_max], outline="red", width=2)
                    elif shape_type == "circle":
                        # 对于圆形，我们使用椭圆，但确保宽度和高度相等
                        center_x = (x_min + x_max) / 2
                        center_y = (y_min + y_max) / 2
                        radius = max((x_max - x_min) / 2, (y_max - y_min) / 2)
                        draw.ellipse([center_x - radius, center_y - radius, 
                                     center_x + radius, center_y + radius], 
                                     outline="red", width=2)
                except Exception as e:
                    logger.error(f"Error drawing shape {bbox}: {e}")
                    pred_dict = {
                        "tool_response_from": self.model_name,
                        "status": "failed",
                        "message": f"Error drawing shape: {str(e)}",
                        "error_code": INVALID_PARAMETERS,
                        "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                    }
                    return pred_dict
            
            correct_param_content_num += 1  # bboxes参数验证和处理成功，参数合规+1
            
            # 将修改后的图像转换为base64
            buffered = BytesIO()
            image_format = image.format if image.format else 'PNG'
            image.save(buffered, format=image_format)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "success",
                "edited_image": img_str,
                "message": f"Successfully drew {len(bboxes)} shapes on the image.",
                "image_dimensions_pixels": {
                    "width": image.width,
                    "height": image.height
                },
                "error_code": SUCCESS,
                "tool_reward": tool_reward+correct_param_content_num/required_keys_num
            }
            
            return pred_dict
                
        except Exception as e:
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error: {str(e)}\n Traceback:{traceback.format_exc()}\n",
                "error_code": TOOL_RUN_FAILED,
                "tool_reward": tool_reward+correct_param_content_num/required_keys_num
            }
            logger.error(f"Error during shape drawing operation: {e}")
            logger.error(traceback.format_exc())
            return pred_dict
    
    def get_tool_instruction(self):
        return self.instruction
        
    def __del__(self):
        """析构函数，确保线程池正确关闭"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)



if __name__ == "__main__":
    parser = HfArgumentParser((DrawShapeArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")

    worker = DrawShapeWorker(
        worker_arguments=args
    )
    worker.run()