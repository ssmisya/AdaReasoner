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
                "description": (
                    "Draw geometric shapes (rectangle, ellipse, or circle) with red borders at specified bounding box locations on the image. "
                    "Returns the edited image in base64 format."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier of the image to edit."
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
        raise NotImplementedError()
        
    def generate(self, params):
        raise NotImplementedError()
    
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