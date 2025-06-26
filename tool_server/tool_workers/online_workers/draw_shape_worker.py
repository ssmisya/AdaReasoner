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

from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

from tool_server.utils.server_utils import build_logger
from tool_server.utils.utils import load_image, pil_to_base64
from tool_server.utils.worker_arguments import WorkerArguments
from tool_server.utils.error_codes import *
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"crop_worker_{worker_id}.log")


class DrawShapeWorker(BaseToolWorker):
    def __init__(self, worker_arguments: WorkerArguments = None):
        # 在调用父类初始化前先设置模型名称
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "DrawShape"
        super().__init__(worker_arguments)
            
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
                    "error_code": INVALID_PARAMETERS
                }
                return pred_dict
            
            # 加载图像
            try:
                if os.path.exists(image_data):
                    image = Image.open(image_data).convert("RGB")
                else:
                    image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Failed to load image: {str(e)}",
                    "error_code": CANNOT_LOAD_IMAGE
                }
                return pred_dict
            
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
                    
                    x_min, y_min, x_max, y_max = coords
                    
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
                    else:
                        logger.warning(f"Unsupported shape type: {shape_type}")
                        
                except Exception as e:
                    logger.error(f"Error drawing shape {bbox}: {e}")
            
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
                "error_code": SUCCESS
            }
            
            return pred_dict
                
        except Exception as e:
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error: {str(e)}\n Traceback:{traceback.format_exc()}\n",
                "error_code": TOOL_RUN_FAILED
            }
            logger.error(f"Error during shape drawing operation: {e}")
            logger.error(traceback.format_exc())
            return pred_dict
    
    def get_tool_instruction(self):
        return self.instruction



if __name__ == "__main__":
    parser = HfArgumentParser((WorkerArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")

    worker = DrawShapeWorker(
        worker_arguments=args
    )
    worker.run()