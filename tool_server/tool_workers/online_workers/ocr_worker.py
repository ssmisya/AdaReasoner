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
                            "description": "The identifier of the image in which to locate the object, e.g., 'img_1'."
                        }
                    },
                    "required": ["image"]
                }
            }
        }
        
    def init_model(self):
        logger.info(f"Initializing model {self.model_name}...")
        try:
            self.ocr_model = easyocr.Reader(['ch_sim','en'])
        except Exception as e:
            logger.error(f"Error initializing OCR model: {e}")
            raise
        
    def get_tool_instruction(self):
        return self.instruction
        
    def generate(self, params):
        try:
            try:
                image = params["image"]
            except KeyError as e:
                message = f"Invalid parameters: expected keys: image. Please reference the tool instruction: {self.get_tool_instruction()}"
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": message,
                    "error_code": INVALID_PARAMETERS
                }
                return pred_dict
            
            # If params are ok, continue
            try:
                img = base64_to_pil(image).convert("RGB")
                width, height = img.size

                result = self.ocr_model.readtext(np.array(img))
                detections = []

                for polygon, label, confidence in result:
                    # Extract polygon coordinates min/max values
                    x_coords = [float(pt[0]) for pt in polygon]  # 确保转换为Python float
                    y_coords = [float(pt[1]) for pt in polygon]  # 确保转换为Python float
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    detections.append({
                        "label": label,
                        "confidence": float(confidence),  # 确保转换为Python float
                        "pixel_bbox": {
                            "x_min": float(x_min),  # 确保转换为Python float
                            "y_min": float(y_min),  # 确保转换为Python float
                            "x_max": float(x_max),  # 确保转换为Python float
                            "y_max": float(y_max)  # 确保转换为Python float
                        },
                        "normalized_bbox": {
                            "x_min": float(x_min / width),  # 确保转换为Python float
                            "y_min": float(y_min / height),  # 确保转换为Python float
                            "x_max": float(x_max / width),  # 确保转换为Python float
                            "y_max": float(y_max / height)  # 确保转换为Python float
                        }
                    })

                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "success",
                    "detections": detections,
                    "image_dimensions_pixels": {
                        "width": int(img.width),  # 确保转换为Python int
                        "height": int(img.height)  # 确保转换为Python int
                    },
                    "error_code": SUCCESS
                }
                return pred_dict
                
            except Exception as e:
                logger.error(f"Error when processing OCR: {e}")
                logger.error(traceback.format_exc())
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"OCR processing failed: {str(e)}",
                    "error_code": TOOL_RUN_FAILED
                }
                return pred_dict
                
        except Exception as e:
            logger.error(f"Unexpected error in OCR worker: {e}")
            logger.error(traceback.format_exc())
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error: {str(e)}\n Traceback:{traceback.format_exc()}\n",
                "error_code": TOOL_RUN_FAILED
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