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
logger = build_logger(__file__, f"highlight_box_worker_{worker_id}.log")

@dataclass
class HighlightBoxArguments(WorkerArguments):
    max_concurrency: int = field(
        default=10,
        metadata={"help": "Maximum number of concurrent requests the model can handle"}
    )


class HighlightBoxWorker(BaseToolWorker):
    def __init__(self, worker_arguments: HighlightBoxArguments = None):
        # 在调用父类初始化前先设置模型名称
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "HighlightBox"
            
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
                "name": "HighlightBox",
                "description": 
                    "Highlight specified bounding box regions in the image using semi-transparent red overlays. Returns the edited image in base64 format.",
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
                                "type": "array",
                                "items": { "type": "integer" },
                                "description": "Bounding box in the format [x_min, y_min, x_max, y_max] using absolute pixel coordinates."
                            },
                            "description": "List of bounding boxes to be highlighted."
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
        """执行高亮框操作并返回结果"""
        return self._process_highlight_box_request(params)
        
    def _process_highlight_box_request(self, params):
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

            image_data = params["image"]
            bboxes = params.get("bboxes", [])
            # 加载图像
            try:
                image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGBA")
                correct_param_content_num += 1
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Failed to load image: {str(e)}",
                    "error_code": CANNOT_LOAD_IMAGE,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                }
                return pred_dict
            
            # 验证所有边界框坐标是否在图像范围内
            image_width, image_height = image.size
            for bbox in bboxes:
                if len(bbox) != 4:
                    pred_dict = {
                        "tool_response_from": self.model_name,
                        "status": "failed",
                        "message": f"Invalid bbox format: {bbox}. Expected 4 values [x_min, y_min, x_max, y_max].",
                        "error_code": INVALID_PARAMETERS,
                        "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                    }
                    return pred_dict
                
                x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
                
                # 检查坐标是否在图像范围内
                if x_min < 0 or y_min < 0 or x_max > image_width or y_max > image_height:
                    pred_dict = {
                        "tool_response_from": self.model_name,
                        "status": "failed",
                        "message": f"Bounding box coordinates {bbox} are outside of image dimensions ({image_width}x{image_height}).",
                        "error_code": INVALID_PARAMETERS,
                        "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                    }
                    return pred_dict
            
            # bboxes参数验证通过
            correct_param_content_num += 1
            
            # 创建一个透明层用于绘制高亮框
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            
            # 处理每个边界框
            for bbox in bboxes:
                try:
                    # 确保所有坐标都是整数
                    x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
                    
                    # 在透明层上绘制半透明红色矩形
                    bbox_overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                    red_color = (255, 0, 0, 128)  # 半透明红色
                    
                    # 填充矩形区域
                    for y in range(y_min, y_max):
                        for x in range(x_min, x_max):
                            if 0 <= x < image.width and 0 <= y < image.height:
                                bbox_overlay.putpixel((x, y), red_color)
                    
                    # 将当前边界框叠加到主透明层上
                    overlay = Image.alpha_composite(overlay, bbox_overlay)
                    
                except Exception as e:
                    logger.warning(f"Error processing bbox {bbox}: {str(e)}")
                    continue
            
            # 将透明层叠加到原始图像上
            result_image = Image.alpha_composite(image, overlay)
            
            # 转换为RGB以确保兼容性
            result_image = result_image.convert("RGB")
            
            # 将结果图像转换为base64
            buffered = BytesIO()
            image_format = image.format if image.format else 'PNG'
            result_image.save(buffered, format=image_format)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "success",
                "edited_image": img_str,
                "message": f"Successfully highlighted {len(bboxes)} bounding boxes on the image.",
                "image_dimensions_pixels": {
                    "width": result_image.width,
                    "height": result_image.height
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
                "tool_reward": tool_reward+(correct_param_content_num/required_keys_num if required_keys_num > 0 else 0)
            }
            logger.error(f"Error during highlight box operation: {e}")
            logger.error(traceback.format_exc())
            return pred_dict
    
    def get_tool_instruction(self):
        return self.instruction
        
    def __del__(self):
        """析构函数，确保线程池正确关闭"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)



if __name__ == "__main__":
    parser = HfArgumentParser((HighlightBoxArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")

    worker = HighlightBoxWorker(
        worker_arguments=args
    )
    worker.run()