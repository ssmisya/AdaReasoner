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
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from tool_server.utils.worker_arguments import WorkerArguments
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
        super().__init__(worker_arguments)
        if self.model_name is None:
            self.model_name = "ocr"
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
                            "description": "The identifier or base64-encoded image content in which to detect text, e.g., 'img_1' or base64 string."
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
        try:
            image = params["image"]
        except:
            message = f"Invalid parameters: expected keys: image. Please reference the tool instruction: {self.get_tool_instruction()}"
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": message,
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
                x_coords = [pt[0] for pt in polygon]
                y_coords = [pt[1] for pt in polygon]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "pixel_bbox": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max
                    },
                    "normalized_bbox": {
                        "x_min": x_min / width,
                        "y_min": y_min / height,
                        "x_max": x_max / width,
                        "y_max": y_max / height
                    }
                })

            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "success",
                "detections": detections,
                "image_dimensions_pixels": {
                    "width": img.width,
                    "height": img.height
                },
            }
            return pred_dict
            
        except Exception as e:
            logger.error(f"Error when ocr: {e}")
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": str(e),
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
