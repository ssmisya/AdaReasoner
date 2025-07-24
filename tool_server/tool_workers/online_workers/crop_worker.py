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
logger = build_logger(__file__, f"crop_worker_{worker_id}.log")

@dataclass
class CropArguments(WorkerArguments):
    max_concurrency: int = field(
        default=120000,
        metadata={"help": "Maximum number of concurrent requests the model can handle"}
    )
    min_image_height_or_length: int = field(
        default=32,
        metadata={"help": "Minimum image size for cropping. Images smaller than this will be resized."}
    )
    min_image_height_or_length: int = field(
        default=32,
        metadata={"help": "Minimum image size for cropping. Images smaller than this will be resized."}
    )

class CropToolWorker(BaseToolWorker):
    def __init__(self, worker_arguments: CropArguments = None):
        # 在调用父类初始化前先设置模型名称
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "Crop"
        
        # 设置最大并发数
        if worker_arguments:
            self.max_concurrency = worker_arguments.max_concurrency
            # 更新基类中的限制值
            worker_arguments.limit_model_concurrency = self.max_concurrency
        
        super().__init__(worker_arguments)
        self.min_image_height_or_length = worker_arguments.min_image_height_or_length if worker_arguments else 32
        self.min_image_height_or_length = worker_arguments.min_image_height_or_length if worker_arguments else 32
        
        # 初始化线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_concurrency)
        logger.info(f"Initialized thread pool with {self.max_concurrency} workers")
            
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Crop an image using specified bounding box coordinates. This tool returns the cropped image in base64 format along with its dimensions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier of the image to crop, e.g., 'img_1'."
                        },
                        "coordinates": {
                            "type": "string",
                            "description": "Coordinates in format '[x_min, y_min, x_max, y_max]', eg., '[100, 100, 200, 200]'. Only absolute pixel values (integers) are supported."
                        }
                    },
                    "required": ["image", "coordinates"]
                }
            }
        }

    def init_model(self):
        logger.info(f"No need to initialize model {self.model_name}.")
        self.model = None
        
    def generate(self, params):
        """执行裁剪操作并返回结果"""
        return self._process_crop_request(params)
        
    def _process_crop_request(self, params):
        """分离出实际处理逻辑，方便并发调用"""
        tool_reward = 2.0
        # 计算Parameter Name Matching
        param_keys = set(params.keys())
        is_coordinates = False
        is_bbox = False
        required_keys = set(self.instruction["function"]["parameters"]["required"])
        if "coordinates" in param_keys:
            is_coordinates = True
        if "bbox" in param_keys:
            is_bbox = True
            required_keys = set(["image", "bbox"])
        parameter_name_match_reward = len(param_keys & required_keys) / len(required_keys | param_keys)
        tool_reward = tool_reward + parameter_name_match_reward
        # 参数名称没有完全匹配，直接返回
        if parameter_name_match_reward < 1:
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": "Invalid parameters: expected keys: image, coordinates.",
                "error_code": INVALID_PARAMETERS,
                "tool_reward": tool_reward
            }
        
        required_keys_num = len(required_keys)
        # 初始化参数合规计数器
        correct_param_content_num = 0
        try:

            image_data = params["image"]
            if is_coordinates:
                crop_param = str(params["coordinates"])
            elif is_bbox:
                crop_param = str(params["bbox"])
            print(f"DEBUG 2: crop_param = {crop_param}")

            # Load image
            try:
                image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
                original_width, original_height = image.size
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Failed to load image: {str(e)}",
                    "error_code": CANNOT_LOAD_IMAGE,
                    "tool_reward": tool_reward,
                }
                return pred_dict
            correct_param_content_num += 1
            
            # Parse crop parameters
            match = re.match(r'\[\s*([\d\.,\s]+)\s*\]', crop_param)
            if match:
                try:
                    # Extract coordinates
                    coords_str = match.group(1).split(',')
                    # Try to convert to float
                    crop_coords = [float(c) for c in coords_str]
                    
                    if len(crop_coords) != 4:
                        pred_dict = {
                            "tool_response_from": self.model_name,
                            "status": "failed",
                            "message": "Invalid number of coordinates: {crop_coords}. Expected 4 values [x_min, y_min, x_max, y_max].",
                            "error_code": INVALID_PARAMETERS,
                            "tool_reward": tool_reward+correct_param_content_num/required_keys_num,
                            "image_dimensions_pixels": {
                                "width": original_width,
                                "height": original_height
                            }
                        }
                        return pred_dict
                    
                    # Ensure all coordinates are absolute values (no normalized coordinates)
                    width, height = image.size
                    if any(0 <= c <= 1 for c in crop_coords) and all(c <= 1 for c in crop_coords):
                        pred_dict = {
                            "tool_response_from": self.model_name,
                            "status": "failed",
                            "message": "Normalized coordinates (0-1) are not supported. Please use absolute pixel values.",
                            "error_code": INVALID_PARAMETERS,
                            "tool_reward": tool_reward+correct_param_content_num/required_keys_num,
                            "image_dimensions_pixels": {
                                "width": width,
                                "height": height
                            }
                        }
                        return pred_dict
                    
                    # Convert to integers for PIL crop
                    absolute_coords = [int(c) for c in crop_coords]
                    
                    # Convert to PIL crop format [left, upper, right, lower]
                    x_min, y_min, x_max, y_max = absolute_coords
                    # 检查坐标是否超出图像范围
                    if x_min > width or x_max > width or y_min > height or y_max > height or x_min < 0 or x_max < 0 or y_min < 0 or y_max < 0 or x_min > x_max or y_min > y_max:
                        pred_dict = {
                            "tool_response_from": self.model_name,
                            "status": "failed",
                            "message": "Coordinates are out of image bounds.",
                            "error_code": INVALID_PARAMETERS,
                            "tool_reward": tool_reward+correct_param_content_num/required_keys_num,
                            "image_dimensions_pixels": {
                                "width": width,
                                "height": height
                            }
                        }
                        return pred_dict
                    pil_crop_coords = [x_min, y_min, x_max, y_max]
                    
                    # Crop image
                    cropped_image = image.crop(pil_crop_coords)
                    cropped_width, cropped_height = cropped_image.size
                    # Check if cropped image meets minimum size requirement
                    if cropped_height < self.min_image_height_or_length or cropped_width < self.min_image_height_or_length:
                        # 对较小的边进行填充
                        logger.info(f"Cropped image is too small ({cropped_width}x{cropped_height}), padding to meet minimum dimension requirement of {self.min_image_height_or_length}")
                        
                        # 创建一个新的白色背景图像
                        new_width = max(cropped_width, self.min_image_height_or_length)
                        new_height = max(cropped_height, self.min_image_height_or_length)
                        padded_image = Image.new("RGB", (new_width, new_height), (255, 255, 255))
                        
                        # 计算粘贴位置（居中）
                        paste_x = (new_width - cropped_width) // 2
                        paste_y = (new_height - cropped_height) // 2
                        
                        # 将裁剪的图像粘贴到新图像上
                        padded_image.paste(cropped_image, (paste_x, paste_y))
                        cropped_image = padded_image
                        logger.info(f"Padded image to {new_width}x{new_height}")
                        cropped_width, cropped_height = new_width, new_height
                    
                    # 填充后检查是否有极端的宽高比
                    aspect_ratio = max(cropped_width, cropped_height) / min(cropped_width, cropped_height)
                    if aspect_ratio > 200:
                        logger.info(f"Extreme aspect ratio detected ({aspect_ratio}), cropping from the middle of the longer dimension")
                        if cropped_width > cropped_height:
                            # 宽度过长，从中间裁剪
                            center = cropped_width // 2
                            crop_width = cropped_height * 200
                            left = max(0, center - crop_width // 2)
                            right = min(cropped_width, center + crop_width // 2)
                            cropped_image = cropped_image.crop((left, 0, right, cropped_height))
                        else:
                            # 高度过长，从中间裁剪
                            center = cropped_height // 2
                            crop_height = cropped_width * 200
                            top = max(0, center - crop_height // 2)
                            bottom = min(cropped_height, center + crop_height // 2)
                            cropped_image = cropped_image.crop((0, top, cropped_width, bottom))
                        cropped_width, cropped_height = cropped_image.size
                        logger.info(f"Cropped image to {cropped_width}x{cropped_height} after aspect ratio adjustment")
                    # Convert cropped image to base64
                    # buffered = BytesIO()
                    # image_format = image.format if image.format else 'PNG'
                    # cropped_image.save(buffered, format=image_format)
                    # img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    img_str = pil_to_base64(cropped_image)
                    # buffered = BytesIO()
                    # image_format = image.format if image.format else 'PNG'
                    # cropped_image.save(buffered, format=image_format)
                    # img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    img_str = pil_to_base64(cropped_image)
                    correct_param_content_num += 1
                    
                    pred_dict = {
                        "tool_response_from": self.model_name,
                        "status": "success",
                        "edited_image": img_str,
                        "message": f"Image cropped successfully using absolute coordinates.",
                        "image_dimensions_pixels": {
                            "width": cropped_width,
                            "height": cropped_height
                        },
                        "error_code":SUCCESS,
                        "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                    }
                    
                    return pred_dict
                
                except ValueError as e:
                    pred_dict = {
                        "tool_response_from": self.model_name,
                        "status": "failed",
                        "message": f"Error processing crop parameters '{crop_param}': {str(e)}",
                        "error_code": TOOL_RUN_FAILED,
                        "tool_reward": tool_reward+correct_param_content_num/required_keys_num,
                        "image_dimensions_pixels": {
                            "width": original_width,
                            "height": original_height
                        }
                    }
                    return pred_dict
            else:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Parameter format mismatch: {crop_param}. Expected format '[x_min, y_min, x_max, y_max]' with absolute pixel values.",
                    "error_code": INVALID_PARAMETERS,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num,
                    "image_dimensions_pixels": {
                        "width": original_width,
                        "height": original_height
                    }
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
            logger.error(f"Error during crop operation: {e}")
            logger.error(traceback.format_exc())
            return pred_dict
    
    def get_tool_instruction(self):
        return self.instruction
        
    def __del__(self):
        """析构函数，确保线程池正确关闭"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)

if __name__ == "__main__":
    parser = HfArgumentParser((CropArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")

    worker = CropToolWorker(
        worker_arguments=args
    )
    worker.run()