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
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"crop_worker_{worker_id}.log")

@dataclass
class CropArguments(WorkerArguments):
    pass

class CropToolWorker(BaseToolWorker):
    def __init__(self, worker_arguments: CropArguments = None):
        # 在调用父类初始化前先设置模型名称
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "Crop"
        super().__init__(worker_arguments)
            
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Crop an image using specified bounding box coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier or path of the image to crop, or base64 encoded image data."
                        },
                        "param": {
                            "type": "string",
                            "description": "Coordinates in format '[x_min, y_min, x_max, y_max]'. Can be absolute pixel values (integers) or normalized values between 0 and 1 (floats)."
                        }
                    },
                    "required": ["image", "param"]
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
                crop_param = params.get("param", "")
                
                if not crop_param:
                    raise KeyError("'param' not found in params")
            except Exception as e:
                message = f"Invalid parameters: expected keys: image, param. Error: {str(e)}"
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": message,
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
                }
                return pred_dict
            
            # 解析裁剪参数
            match = re.match(r'\[\s*([\d\.,\s]+)\s*\]', crop_param)
            if match:
                try:
                    # 提取坐标 - 允许浮点数
                    coords_str = match.group(1).split(',')
                    # 尝试转换为浮点数
                    crop_coords = [float(c) for c in coords_str]
                    
                    if len(crop_coords) != 4:
                        raise ValueError(f"Invalid number of coordinates: {crop_coords}. Expected 4 values [x_min, y_min, x_max, y_max].")
                    
                    # 解析坐标，格式为 [x_min, y_min, x_max, y_max]
                    x_min, y_min, x_max, y_max = crop_coords
                    
                    # 自动检测是否为归一化坐标
                    normalized = all(0 <= c <= 1 for c in crop_coords)
                    logger.info(f"自动检测坐标类型: {'归一化' if normalized else '绝对'} 坐标")
                    
                    # 如果是归一化坐标，将其转换为绝对坐标
                    if normalized:
                        width, height = image.size
                        absolute_coords = [
                            int(x_min * width),    # x_min
                            int(y_min * height),   # y_min
                            int(x_max * width),    # x_max
                            int(y_max * height)    # y_max
                        ]
                        logger.info(f"归一化坐标转换为绝对坐标: {absolute_coords}")
                    else:
                        # 确保绝对坐标为整数
                        absolute_coords = [int(c) for c in crop_coords]
                    
                    # 转换为PIL的裁剪格式 [left, upper, right, lower]
                    x_min, y_min, x_max, y_max = absolute_coords
                    pil_crop_coords = [x_min, y_min, x_max, y_max]
                    
                    # 裁剪图像
                    cropped_image = image.crop(pil_crop_coords)
                    
                    # 将裁剪后的图像转换为base64
                    buffered = BytesIO()
                    image_format = image.format if image.format else 'PNG'
                    cropped_image.save(buffered, format=image_format)
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    pred_dict = {
                        "tool_response_from": self.model_name,
                        "status": "success",
                        "edited_image": img_str,
                        "message": f"Image cropped successfully using {'normalized' if normalized else 'absolute'} coordinates.",
                        "image_dimensions_pixels": {
                            "width": cropped_image.width,
                            "height": cropped_image.height
                        },
                        # "crop_coordinates": {
                        #     "x_min": x_min,
                        #     "y_min": y_min,
                        #     "x_max": x_max,
                        #     "y_max": y_max,
                        #     "normalized": normalized,
                        #     "normalized_coords": {
                        #         "x_min": x_min / image.width,
                        #         "y_min": y_min / image.height,
                        #         "x_max": x_max / image.width,
                        #         "y_max": y_max / image.height
                        #     } if not normalized else {
                        #         "x_min": crop_coords[0],
                        #         "y_min": crop_coords[1],
                        #         "x_max": crop_coords[2],
                        #         "y_max": crop_coords[3]
                        #     }
                        # }
                    }
                    
                    return pred_dict
                
                except ValueError as e:
                    pred_dict = {
                        "tool_response_from": self.model_name,
                        "status": "failed",
                        "message": f"Error processing crop parameters '{crop_param}': {str(e)}",
                    }
                    return pred_dict
            else:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Parameter format mismatch: {crop_param}. Expected format '[x_min, y_min, x_max, y_max]'.",
                }
                return pred_dict
                
        except Exception as e:
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            logger.error(f"Error during crop operation: {e}")
            logger.error(traceback.format_exc())
            return pred_dict
    
    def get_tool_instruction(self):
        return self.instruction

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = HfArgumentParser((CropArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")

    worker = CropToolWorker(
        worker_arguments=args
    )
    worker.run()