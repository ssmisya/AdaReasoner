"""
A model worker executes the model.
"""

import uuid
import os
import re
import io
import numpy as np
from PIL import Image
import torch
import traceback
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from tool_server.utils.worker_arguments import WorkerArguments
from tool_server.utils.error_codes import *
import matplotlib.pyplot as plt

import easyocr

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None

np.random.seed(3)

class OCRToolWorker(BaseToolWorker):
    def __init__(self, worker_arguments: WorkerArguments = None):
        # 在调用父类初始化前先设置模型名称
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "ocr"
        super().__init__(worker_arguments)
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Extracts and localizes text from the given image using OCR. Returns bounding boxes, recognized text, and confidence scores.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier of the image to analyze, e.g., 'img_1'"
                        }
                    },
                    "required": ["image"]
                }
            }
        }
        
    def init_model(self):
        logger.info(f"Initializing model {self.model_name}...")
        self.ocr_model = easyocr.Reader(['ch_sim','en'])
        
    def get_tool_instruction(self):
        return self.instruction
        
    def generate(self, params):
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
                "message": "Invalid parameters: expected keys: image.",
                "error_code": INVALID_PARAMETERS,
                "tool_reward": tool_reward
            }
        
        required_keys_num = len(required_keys)
        # 初始化参数合规计数器
        correct_param_content_num = 0
        
        image = params["image"]
        text_threshold = params.get("text_threshold", 0.25)
        
        # If params are ok, continue
        try:
            try:
                img = base64_to_pil(image).convert("RGB")
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Cannot load image: {str(e)}",
                    "error_code": CANNOT_LOAD_IMAGE,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                }
                return pred_dict
            
            correct_param_content_num += 1

            result = self.ocr_model.readtext(np.array(img))
            detections = []

            for polygon, label, confidence in result:
                # Skip results with confidence below threshold
                if confidence < text_threshold:
                    continue
                    
                # Round confidence to 2 decimal places
                confidence = round(float(confidence), 2)
                
                # Extract polygon coordinates min/max values
                x_coords = [float(pt[0]) for pt in polygon]  # 确保转换为Python float
                y_coords = [float(pt[1]) for pt in polygon]  # 确保转换为Python float
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                detections.append({
                    "label": label,
                    "confidence": confidence,  # 已经处理为保留两位小数的浮点数
                    "pixel_bbox": {
                        "x_min": int(x_min),  # 转换为整数
                        "y_min": int(y_min),  # 转换为整数
                        "x_max": int(x_max),  # 转换为整数
                        "y_max": int(y_max)  # 转换为整数
                    }
                })

            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "success",
                "detections": detections,
                "image_dimensions_pixels": {
                    "width": img.width,  # 确保转换为Python int
                    "height": img.height  # 确保转换为Python int
                },
                "error_code": SUCCESS,
                "tool_reward": tool_reward+correct_param_content_num/required_keys_num
            }
            return pred_dict
            
        except Exception as e:
            logger.error(f"Error when ocr: {e}")
            logger.error(traceback.format_exc())
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error: {str(e)}\nTraceback:{traceback.format_exc()}\n",
                "error_code": TOOL_RUN_FAILED,
                "tool_reward": tool_reward+(correct_param_content_num/required_keys_num if required_keys_num > 0 else 0)
            }
            return pred_dict


if __name__ == "__main__":
    # Use the new argument parser from transformers
    from transformers import HfArgumentParser
    
    parser = HfArgumentParser(WorkerArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    logger.info(f"args: {args}")

    worker = OCRToolWorker(worker_arguments=args)
    worker.run()