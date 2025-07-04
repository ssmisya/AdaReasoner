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
from tool_server.utils.error_codes import *
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
    
    for match in pattern.finditer(generate_param):
        attr, value = match.groups()
        
        if attr:
            # 有数字索引的坐标 (y1, y2, ...)
            index = attr[1:] if len(attr) > 1 and attr[1:].isdigit() else "0"
            coords[index] = float(value)
    
    # 如果没有找到坐标
    if not coords:
        raise ValueError(f"No valid {coordinate_type} coordinates found. For {line_type} lines, please provide {coordinate_type} coordinates in format '{coordinate_type}1=value1, {coordinate_type}2=value2'.")
    
    # 检查是否可能是归一化坐标
    for index, value in coords.items():
        # 检测可能的归一化坐标
        is_small_decimal = 0 < value < 1 and value != int(value)
        
        if is_small_decimal:
            logger.info(f"Detected potential normalized coordinate: {value}")
            raise ValueError("Normalized coordinates (0-1) are not supported. Please use absolute pixel coordinates instead.")
        
        # 检查坐标是否超出图像范围
        if line_type.lower() == "horizontal" and (value < 0 or value >= image_h):
            logger.info(f"Coordinate y={value} is out of image bounds) (height={image_h})")
            raise ValueError(f"Coordinate y={value} is out of image bounds")
        elif line_type.lower() == "vertical" and (value < 0 or value >= image_w):
            logger.info(f"Coordinate x={value} is out of image bounds (width={image_w})")
            raise ValueError(f"Coordinate x={value} is out of image bounds")
    
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
    
    # 保存原始图像尺寸
    original_width, original_height = image.size
        
    # 创建图形时指定尺寸和DPI以匹配原始图像
    dpi = 100
    figsize = (original_width / dpi, original_height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 设置图像显示范围
    ax.imshow(image)
    ax.axis("off")
    
    # 设置坐标轴范围以匹配图像尺寸
    ax.set_xlim(0, original_width)
    ax.set_ylim(original_height, 0)  # 注意y轴是反的
    
    for point in point_coords:
        if line_type.lower() == "horizontal":
            y = point[1]  # 只使用y坐标
            ax.axhline(y=y, color='#e6194b', linewidth=2, linestyle='dashed')
        else:  # vertical
            x = point[0]  # 只使用x坐标
            ax.axvline(x=x, color='#e6194b', linewidth=2, linestyle='dashed')
    
    # 确保图像边界与原始图像完全一致
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    buf = BytesIO()
    # 移除bbox_inches='tight'参数，使用固定尺寸
    fig.savefig(buf, format=image_format, pad_inches=0, dpi=dpi)  
    plt.close(fig) 
    buf.seek(0)
    
    edited_image = Image.open(buf).convert("RGB")
    
    # 确保输出图像与输入图像尺寸完全一致
    if edited_image.size != (original_width, original_height):
        edited_image = edited_image.resize((original_width, original_height), Image.LANCZOS)
    
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
                            "description": "The identifier of the image to edit, e.g., 'img_1'"
                        },
                        "line_type": {
                            "type": "string",
                            "description": "Type of line to draw: 'horizontal' (requires y coordinates) or 'vertical' (requires x coordinates).",
                            "enum": ["horizontal", "vertical"]
                        },
                        "coordinates": {
                            "type": "string",
                            "description": "For horizontal lines, provide y coordinates in format 'y1=100, y2=200, y3=300'. For vertical lines, provide x coordinates in format 'x1=100, x2=200, x3=300'. Multiple coordinates should be separated by commas. Only absolute pixel values are supported."
                        }
                    },
                    "required": ["image", "line_type", "coordinates"]
                }
            }
        }

    def init_model(self):
        logger.info(f"No need to initialize model {self.model_name}.")
        self.model = None
        
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
                "message": "Invalid parameters: expected keys: image, line_type, coordinates.",
                "error_code": INVALID_PARAMETERS,
                "tool_reward": tool_reward
            }

        required_keys_num = len(required_keys)
        message =""
        encounter_error = False
        # 初始化参数合规计数器
        correct_param_content_num = 0
        try:
            image_data = params["image"]
            line_type = params["line_type"]
            generate_param = params["coordinates"]
                
            # 检查line_type参数是否合规
            if line_type in ["horizontal", "vertical"]:
                correct_param_content_num += 1
            else:
                # line_type参数不合规
                message = "Invalid line_type: expected 'horizontal' or 'vertical'."
                encounter_error = True
            
            # 加载图像
            try:
                img = base64_to_pil(image_data).convert("RGB")
            except Exception as e:
                if encounter_error:
                    message = message + f"Cannot load image: {str(e)}"
                else:
                    message = f"Cannot load image: {str(e)}"
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": message,
                    "error_code": CANNOT_LOAD_IMAGE,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num # 添加合规的参数个数
                }
                return pred_dict
            correct_param_content_num += 1 
            width, height = img.size
            
            # 提取点坐标
            try:
                points = extract_points(generate_param, width, height, line_type)
            except ValueError as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": str(e),
                    "error_code": INVALID_PARAMETERS,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num # 添加合规的参数个数
                }
                return pred_dict
            
            if not points:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": "No valid points extracted from parameters.",
                    "error_code": INVALID_PARAMETERS,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num # 添加合规的参数个数
                }
                return pred_dict

            correct_param_content_num += 1  # 坐标提取成功且在图像范围内，参数合规+1
            
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
                    "width": int(edited_img.width),
                    "height": int(edited_img.height)
                },
                "error_code": SUCCESS,
                "tool_reward": tool_reward+correct_param_content_num/required_keys_num,
            }
            
            return pred_dict
                
        except Exception as e:
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error: {str(e)}\nTraceback:{traceback.format_exc()}\n",
                "error_code": TOOL_RUN_FAILED,
                "tool_reward": tool_reward+correct_param_content_num/required_keys_num,
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
