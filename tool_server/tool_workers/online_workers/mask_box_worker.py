"""
A model worker executes the model.
"""

import torch
import numpy as np
from PIL import Image
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



class MaskBoxWorker(BaseToolWorker):
    def __init__(self, worker_arguments: WorkerArguments = None):
        # 在调用父类初始化前先设置模型名称
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "MaskBox"
        super().__init__(worker_arguments)
            
        self.instruction = {
            "type": "function",
            "function": {
                "name": "MaskBox",
                "description": 
                    "Mask out all specified bounding box regions in the input image by overlaying white rectangles. Returns the edited image in base64 format.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier of the image to edit, e.g., 'img_1'."
                        },
                        "bboxes": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": { "type": "integer" },
                                "description": "Bounding box in the format [x_min, y_min, x_max, y_max] using absolute pixel coordinates."
                            },
                            "description": "List of bounding boxes to be masked."
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
                    raise KeyError("'bboxes' not found or empty in params")
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
            
            # 处理边界框参数
            try:
                # 创建可编辑的图像副本
                draw_image = image.copy()
                
                # 在图像上绘制白色矩形
                from PIL import ImageDraw
                draw = ImageDraw.Draw(draw_image)
                
                for bbox in bboxes:
                    if len(bbox) != 4:
                        raise ValueError(f"Invalid bounding box: {bbox}. Expected 4 values [x_min, y_min, x_max, y_max].")
                    
                    # 确保所有坐标都是整数
                    x_min, y_min, x_max, y_max = [int(c) for c in bbox]
                    
                    # 绘制白色矩形
                    draw.rectangle([x_min, y_min, x_max, y_max], fill="white")
                
                # 将处理后的图像转换为base64
                buffered = BytesIO()
                image_format = image.format if image.format else 'PNG'
                draw_image.save(buffered, format=image_format)
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "success",
                    "edited_image": img_str,
                    "message": f"Successfully masked {len(bboxes)} regions in the image.",
                    "image_dimensions_pixels": {
                        "width": draw_image.width,
                        "height": draw_image.height
                    },
                    "error_code": SUCCESS
                }
                
                return pred_dict
            
            except ValueError as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Error processing bounding boxes: {str(e)}",
                    "error_code": TOOL_RUN_FAILED
                }
                return pred_dict
                
        except Exception as e:
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error: {str(e)}\n Traceback:{traceback.format_exc()}\n",
                "error_code": TOOL_RUN_FAILED
            }
            logger.error(f"Error during mask box operation: {e}")
            logger.error(traceback.format_exc())
            return pred_dict
    
    def get_tool_instruction(self):
        return self.instruction



if __name__ == "__main__":
    parser = HfArgumentParser((WorkerArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")

    worker = MaskBoxWorker(
        worker_arguments=args
    )
    worker.run()