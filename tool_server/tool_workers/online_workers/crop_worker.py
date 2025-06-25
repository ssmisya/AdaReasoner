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
                "description": "Crop an image using specified bounding box coordinates. This tool returns the cropped image in base64 format along with its dimensions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier of the image to crop, e.g., 'img_1'."
                        },
                        "description": {
                            "type": "string",
                            "description": "Coordinates in format '[x_min, y_min, x_max, y_max]', eg., '[100, 100, 200, 200]'. Only absolute pixel values (integers) are supported."
                        }
                    },
                    "required": ["image", "description"]
                }
            }
        }

    def init_model(self):
        logger.info(f"No need to initialize model {self.model_name}.")
        self.model = None
        
    def generate(self, params):
        try:
            # Extract input parameters
            try:
                image_data = params["image"]
                crop_param = params.get("description", "")
                
                if not crop_param:
                    raise KeyError("'description' not found in params")
            except Exception as e:
                message = f"Invalid parameters: expected keys: image, description. Error: {str(e)}"
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": message,
                    "error_code": INVALID_PARAMETERS
                }
                return pred_dict
            
            # Load image
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
            
            # Parse crop parameters
            match = re.match(r'\[\s*([\d\.,\s]+)\s*\]', crop_param)
            if match:
                try:
                    # Extract coordinates
                    coords_str = match.group(1).split(',')
                    # Try to convert to float
                    crop_coords = [float(c) for c in coords_str]
                    
                    if len(crop_coords) != 4:
                        raise ValueError(f"Invalid number of coordinates: {crop_coords}. Expected 4 values [x_min, y_min, x_max, y_max].")
                    
                    # Ensure all coordinates are absolute values (no normalized coordinates)
                    width, height = image.size
                    if any(0 <= c <= 1 for c in crop_coords) and all(c <= 1 for c in crop_coords):
                        pred_dict = {
                            "tool_response_from": self.model_name,
                            "status": "failed",
                            "message": "Normalized coordinates (0-1) are not supported. Please use absolute pixel values.",
                            "error_code": INVALID_PARAMETERS
                        }
                        return pred_dict
                    
                    # Convert to integers for PIL crop
                    absolute_coords = [int(c) for c in crop_coords]
                    
                    # Convert to PIL crop format [left, upper, right, lower]
                    x_min, y_min, x_max, y_max = absolute_coords
                    pil_crop_coords = [x_min, y_min, x_max, y_max]
                    
                    # Crop image
                    cropped_image = image.crop(pil_crop_coords)
                    
                    # Convert cropped image to base64
                    buffered = BytesIO()
                    image_format = image.format if image.format else 'PNG'
                    cropped_image.save(buffered, format=image_format)
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    pred_dict = {
                        "tool_response_from": self.model_name,
                        "status": "success",
                        "edited_image": img_str,
                        "message": f"Image cropped successfully using absolute coordinates.",
                        "image_dimensions_pixels": {
                            "width": cropped_image.width,
                            "height": cropped_image.height
                        },
                        "error_code":SUCCESS
                    }
                    
                    return pred_dict
                
                except ValueError as e:
                    pred_dict = {
                        "tool_response_from": self.model_name,
                        "status": "failed",
                        "message": f"Error processing crop parameters '{crop_param}': {str(e)}",
                        "error_code": TOOL_RUN_FAILED
                    }
                    return pred_dict
            else:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Parameter format mismatch: {crop_param}. Expected format '[x_min, y_min, x_max, y_max]' with absolute pixel values.",
                    "error_code": INVALID_PARAMETERS
                }
                return pred_dict
                
        except Exception as e:
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error: {str(e)}\n Traceback:{traceback.format_exc()}\n",
                "error_code": TOOL_RUN_FAILED
            }
            logger.error(f"Error during crop operation: {e}")
            logger.error(traceback.format_exc())
            return pred_dict
    
    def get_tool_instruction(self):
        return self.instruction

if __name__ == "__main__":
    parser = HfArgumentParser((CropArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")

    worker = CropToolWorker(
        worker_arguments=args
    )
    worker.run()