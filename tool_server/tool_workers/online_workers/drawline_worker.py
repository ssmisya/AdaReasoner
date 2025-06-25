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

def extract_points(generate_param, image_w, image_h, line_type):
    all_points = []
    
    # 根据线条类型确定需要提取的坐标类型
    if line_type.lower() == "horizontal":
        coordinate_type = "y"
    else:  # vertical
        coordinate_type = "x"
    
    # 提取坐标值，支持逗号分隔的格式
    # 匹配 x1=100, y2=200 这样的格式
    pattern = re.compile(r'([xy]\d*)=\s*"?([0-9]+(?:\.[0-9]+)?)"?')
    
    coords = {}
    wrong_coord_type_found = False
    wrong_coord_name = None
    
    for match in pattern.finditer(generate_param):
        attr, value = match.groups()
        
        # 检查坐标类型是否正确
        if attr and attr[0].lower() != coordinate_type:
            wrong_coord_type_found = True
            wrong_coord_name = attr
            continue
        
        if attr:
            # 有数字索引的坐标 (y1, y2, ...)
            index = attr[1:] if len(attr) > 1 and attr[1:].isdigit() else "0"
            coords[index] = float(value)
    
    # 如果发现了错误类型的坐标
    if wrong_coord_type_found:
        logger.info(f"Found wrong coordinate type: {wrong_coord_name} for line_type: {line_type}")
        raise ValueError(f"For {line_type} lines, only {coordinate_type} coordinates should be provided, but found {wrong_coord_name}.")
    
    # 如果没有找到坐标
    if not coords:
        raise ValueError(f"No valid {coordinate_type} coordinates found. For {line_type} lines, please provide {coordinate_type} coordinates in format '{coordinate_type}1=value1, {coordinate_type}2=value2'.")
    
    # 检查是否可能是归一化坐标
    for index, value in coords.items():
        # 检测可能的归一化坐标
        is_small_decimal = 0 < value < 1 and value != int(value)
        
        # 检查坐标是否明显小于图像尺寸
        max_dimension = image_h if coordinate_type == "y" else image_w
        too_small_for_pixels = (0 < value < 10 and 
                               value != int(value) and
                               max_dimension > 50)
        
        if is_small_decimal or too_small_for_pixels:
            logger.info(f"Detected potential normalized coordinate: {value}")
            raise ValueError("Normalized coordinates (0-1) are not supported. Please use absolute pixel coordinates instead.")
    
    # 构建点坐标
    for index, value in coords.items():
        if line_type.lower() == "horizontal":
            # 水平线 (x值不重要，设为0)
            point = np.array([0, value])
        else:  # vertical
            # 垂直线 (y值不重要，设为0)
            point = np.array([value, 0])
        
        all_points.append(point)
    
    logger.info(f"Extracted {len(all_points)} {line_type} lines with {coordinate_type} coordinates")
    
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
            y = point[1]  # 只使用y坐标
            if 0 <= y < image.height:
                ax.axhline(y=y, color='#e6194b', linewidth=2, linestyle='dashed')
        else:  # vertical
            x = point[0]  # 只使用x坐标
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
                "description": "Draw horizontal or vertical lines on an image. This tool supports drawing multiple lines of the same type simultaneously. Only accepts absolute pixel coordinates (not normalized values). Returns base64 encoded image with lines drawn.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier of the image in which to locate the object, e.g., 'img_1'."
                        },
                        "line_type": {
                            "type": "string",
                            "description": "Type of line to draw: 'horizontal' (requires y coordinates) or 'vertical' (requires x coordinates).",
                            "enum": ["horizontal", "vertical"]
                        },
                        "description": {
                            "type": "string",
                            "description": "For horizontal lines, provide y coordinates in format 'y1=100, y2=200, y3=300'. For vertical lines, provide x coordinates in format 'x1=100, x2=200, x3=300'. Multiple coordinates should be separated by commas. Only absolute pixel values are supported."
                        }
                    },
                    "required": ["image", "line_type", "description"]
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
                generate_param = params.get("description", "")
                
                if not generate_param:
                    raise KeyError("'description' not found in params")
                    
                if line_type not in ["horizontal", "vertical"]:
                    line_type = "horizontal"  # 默认水平线
                    
            except Exception as e:
                message = f"Invalid parameters: expected keys: image, line_type, description. Error: {str(e)}"
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
            try:
                points = extract_points(generate_param, width, height, line_type)
            except ValueError as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": str(e),
                }
                return pred_dict
            
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
