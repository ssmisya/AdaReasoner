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
import matplotlib.pyplot as plt

from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

from tool_server.utils.server_utils import build_logger
from tool_server.utils.utils import load_image, pil_to_base64, base64_to_pil
from tool_server.utils.worker_arguments import WorkerArguments
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"DrawLine_worker_{worker_id}.log")

@dataclass
class DrawLineArguments(WorkerArguments):
    pass

def extract_points(generate_param, image_w, image_h):
    all_points = []
    
    # Regular expression to match x and y values separately or together, with or without quotes
    pattern = re.compile(r'(x\d*)?=\s*"?([0-9]+(?:\.[0-9]+)?)"?|'
                         r'(y\d*)?=\s*"?([0-9]+(?:\.[0-9]+)?)"?')
    
    # Initialize default x and y
    points = {}
    for match in pattern.finditer(generate_param):
        attr, x_val, _, y_val = match.groups()
        
        if attr and 'x' in attr:
            points[attr] = float(x_val)
        elif _ and 'y' in _:
            points[_] = float(y_val)
    
    # Process matched pairs
    indices = sorted(set(int(key[1:]) for key in points.keys() if key[1:].isdigit()))
    if not indices:
        indices = [0] if 'x' in points or 'y' in points else []
    
    raw_points = []
    for i in indices:
        x_key = f'x{i}' if f'x{i}' in points else 'x'
        y_key = f'y{i}' if f'y{i}' in points else 'y'
        
        x_value = points.get(x_key, 0.0)
        y_value = points.get(y_key, 0.0)
        
        raw_points.append([x_value, y_value])
    
    # 自动判断是否为归一化坐标
    is_normalized = True
    for point in raw_points:
        # 如果任何坐标值大于1，则认为是绝对坐标
        if any(coord > 1.0 for coord in point):
            is_normalized = False
            break
            
    logger.info(f"坐标类型自动检测: {'归一化(0-1)' if is_normalized else '绝对像素值'}")
    
    # 根据坐标类型进行转换
    for point in raw_points:
        if is_normalized:
            # 归一化坐标，转换为绝对坐标
            point = np.array(point) * np.array([image_w, image_h])
        else:
            # 已经是绝对坐标，不需要额外处理
            point = np.array(point)
        
        all_points.append(point)
    
    return all_points
    
def DrawLine(image, point_coords=None, line_type="horizontal"):
    image_format = image.format.lower() if image.format else 'png'
    if image_format not in ['png', 'jpeg', 'jpg']:
        image_format = 'png'
        
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    
    for point in point_coords:
        if line_type.lower() == "horizontal":
            y = point[1]
            if 0 <= y < image.height:
                ax.axhline(y=y, color='#e6194b', linewidth=2, linestyle='dashed')
        else:  # vertical
            x = point[0]
            if 0 <= x < image.width:
                ax.axvline(x=x, color='#e6194b', linewidth=2, linestyle='dashed')
    
    buf = BytesIO()
    fig.savefig(buf, format=image_format, bbox_inches='tight', pad_inches=0)  
    plt.close(fig) 
    buf.seek(0)
    
    edited_image = Image.open(buf).convert("RGB")
    
    return edited_image

class DrawLineToolWorker(BaseToolWorker):
    def __init__(self, worker_arguments: DrawLineArguments = None):
        # 在调用父类初始化前先设置模型名称
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "DrawLine"
        super().__init__(worker_arguments)
            
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Draw horizontal or vertical lines on an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier or path of the image to draw on, or base64 encoded image data."
                        },
                        "line_type": {
                            "type": "string",
                            "description": "Type of line to draw: 'horizontal' or 'vertical'.",
                            "enum": ["horizontal", "vertical"]
                        },
                        "param": {
                            "type": "string",
                            "description": "Coordinates in format 'x=\"value\"' or 'y=\"value\"'. Values can be absolute pixel values or normalized between 0-1 (automatically detected)."
                        }
                    },
                    "required": ["image", "line_type", "param"]
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
                line_type = params.get("line_type", "horizontal").lower()
                generate_param = params.get("param", "")
                
                if not generate_param:
                    raise KeyError("'param' not found in params")
                    
                if line_type not in ["horizontal", "vertical"]:
                    line_type = "horizontal"  # 默认水平线
                    
            except Exception as e:
                message = f"Invalid parameters: expected keys: image, line_type, param. Error: {str(e)}"
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": message,
                }
                return pred_dict
            
            # 加载图像
            try:
                if os.path.exists(image_data):
                    img = Image.open(image_data).convert("RGB")
                else:
                    img = base64_to_pil(image_data).convert("RGB")
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Failed to load image: {str(e)}",
                }
                return pred_dict
            
            width, height = img.size
            
            # 提取点坐标
            points = extract_points(generate_param, width, height)
            
            if not points:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": "No valid points extracted from parameters.",
                }
                return pred_dict
            
            # 绘制线条
            edited_img = DrawLine(img, point_coords=points, line_type=line_type)
            
            # 将编辑后的图像转换为base64
            img_str = pil_to_base64(edited_img)
            
            line_type_str = "horizontal" if line_type == "horizontal" else "vertical"
            
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "success",
                "edited_image": img_str,
                "message": f"{line_type_str.capitalize()} lines drawn successfully at {len(points)} positions.",
                "image_dimensions_pixels": {
                    "width": edited_img.width,
                    "height": edited_img.height
                }
            }
            
            return pred_dict
                
        except Exception as e:
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            logger.error(f"Error during drawing operation: {e}")
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
    from transformers import HfArgumentParser
    
    parser = HfArgumentParser((DrawLineArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")

    worker = DrawLineToolWorker(
        worker_arguments=args
    )
    worker.run()
