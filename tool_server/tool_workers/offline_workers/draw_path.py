# draw_path.py
import io
import re
from PIL import Image, ImageDraw
import ast


from tool_server.utils.server_utils import build_logger
from tool_server.utils.utils import load_image, pil_to_base64
from tool_server.tool_workers.offline_workers.base_offline_worker import BaseOfflineWorker
from tool_server.utils.error_codes import *

logger = build_logger("draw_path_worker")

class Draw2DPath(BaseOfflineWorker):
    """
    在图片上根据起点和方向序列绘制路径
    """
    
    def __init__(self):
        super().__init__(model_name="Draw2DPath")
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Draw a path on an image following a sequence of directional commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The image to draw on (image identifier)"
                        },
                        "start_point": {
                            "type": "array",
                            "description": "Starting point coordinates [x, y]"
                        },
                        "directions": {
                            "type": "string",
                            "description": "Direction sequence with 'u'=up, 'd'=down, 'l'=left, 'r'=right. Accepts both 'rruldd' format and comma-separated 'R,R,U,L,D,D' format."
                        },
                        "step": {
                            "type": "integer",
                            "description": "Step size in pixels for each direction (default: 64)"
                        },
                        "pixel_coordinate": {
                            "type": "boolean",
                            "description": "If true, start_point is in pixel coordinates; if false, in grid coordinates (default: true)"
                        },
                        "line_width": {
                            "type": "integer",
                            "description": "Width of the drawn line (default: 3)"
                        },
                        "line_color": {
                            "type": "string",
                            "description": "Color of the line (default: 'red')"
                        }
                    },
                    "required": ["image", "start_point", "directions"]
                }
            }
        }

    def _normalize_directions(self, directions):
        """
        将方向序列标准化为内部格式 (小写无逗号)
        支持两种输入格式:
        1. "ludr" - 传统格式
        2. "L,R,U,D" - 新格式，逗号分隔的大写字母
        """
        if not directions:
            return ""
        
        # 如果是字符串，尝试处理
        if isinstance(directions, str):
            # 移除空格
            directions = directions.replace(" ", "")
            
            # 检查是否为新格式（包含逗号）
            if "," in directions:
                # 分割并转换为小写
                dir_list = directions.lower().split(",")
                # 仅保留有效字符
                dir_list = [d for d in dir_list if d in ['l', 'r', 'u', 'd']]
                return "".join(dir_list)
            else:
                # 传统格式，仅保留有效字符
                return "".join(c for c in directions.lower() if c in "lrud")
        
        # 如果不是字符串，返回空字符串
        return ""
    
    def _execute(self, params):
        """执行路径绘制"""
        try:
            
            # 提取必要参数
            image = params["image"]
            start_point = params["start_point"]
            directions = params["directions"]
            
            directions = self._normalize_directions(directions)
        
            
            # 提取可选参数
            step = params.get("step", 64)
            pixel_coordinate = params.get("pixel_coordinate", True)
            line_width = params.get("line_width", 3)
            line_color = params.get("line_color", "red")
            
            # 加载图片
            img = image
            
            # 处理起始点
            if isinstance(start_point, list) and len(start_point) == 2:
                start_coords = start_point
            else:
                return {
                    "status": "failed",
                    "message": "start_point must be a list of two coordinates [x, y]"
                }
            
            # 将格子坐标转换为像素坐标（如果需要）
            if not pixel_coordinate:
                half_step = step / 2
                start_coords = [(coord - 1) * step + half_step for coord in start_coords]
            
            # 绘制路径
            edited_image = self.draw_direction_sequence(
                img, 
                tuple(start_coords), 
                directions, 
                step=step, 
                line_width=line_width,
                line_color=line_color
            )
            
            # 转换为base64并返回结果
            image_base64 = pil_to_base64(edited_image)
            return {
                "status": "success",
                "edited_image": image_base64,
                "message": "Path drawn successfully",
                "error_code": SUCCESS  # 添加这一行
            }
            
        except Exception as e:
            logger.error(f"Error drawing path: {str(e)}")
            return {
                "status": "failed",
                "message": f"Error drawing path: {str(e)}",
                "error_code": TOOL_RUN_FAILED  # 添加这一行
            }
    
    def draw_direction_sequence(self, image, start, directions, step=64, line_width=3, line_color="red"):
        """
        在图片上从起点沿方向序列画带箭头的线段。
        
        Args:
            image: PIL.Image对象或图片路径
            start (tuple): 起点坐标 (x, y)
            directions (str): 方向序列, 由 'u','d','l','r' 组成
            step (int): 每个方向移动的像素数
            line_width (int): 线条宽度
            line_color (str): 线条颜色
        
        Returns:
            PIL.Image: 带有绘制路径的图像
        """
        # 确保image是PIL.Image对象
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.copy().convert("RGB")
        else:
            raise ValueError("image must be a file path or a PIL Image object")
        
        draw = ImageDraw.Draw(img)
        
        # 当前坐标
        x, y = start
        
        # 遍历方向
        for dir in directions:
            old_x, old_y = x, y
            if dir == 'u':
                y -= step
            elif dir == 'd':
                y += step
            elif dir == 'l':
                x -= step
            elif dir == 'r':
                x += step
            else:
                raise ValueError(f"Unknown direction: {dir}")
            
            # 画从(old_x, old_y)到(x, y)的线
            draw.line([(old_x, old_y), (x, y)], fill=line_color, width=line_width)
            
            # 添加箭头头部
            arrow_size = line_width * 2
            
            # 根据方向计算箭头头部的点
            if dir == 'u':
                # 箭头指向上方
                arrow_points = [
                    (x, y),
                    (x - arrow_size, y + arrow_size * 2),
                    (x + arrow_size, y + arrow_size * 2)
                ]
            elif dir == 'd':
                # 箭头指向下方
                arrow_points = [
                    (x, y),
                    (x - arrow_size, y - arrow_size * 2),
                    (x + arrow_size, y - arrow_size * 2)
                ]
            elif dir == 'l':
                # 箭头指向左方
                arrow_points = [
                    (x, y),
                    (x + arrow_size * 2, y - arrow_size),
                    (x + arrow_size * 2, y + arrow_size)
                ]
            elif dir == 'r':
                # 箭头指向右方
                arrow_points = [
                    (x, y),
                    (x - arrow_size * 2, y - arrow_size),
                    (x - arrow_size * 2, y + arrow_size)
                ]
            
            # 绘制箭头头部
            draw.polygon(arrow_points, fill=line_color)

        return img
    
    def verify_tool_parameter(self,params):
        # 提取必要参数
        try:
            
            image = params["image"]
            image = load_image(image)
            
            start_point = params["start_point"]
            start_point = ast.literal_eval(start_point) if isinstance(start_point, str) else start_point
                            
            directions = params["directions"]
            if isinstance(directions, str):
                # 统一处理方向格式
                normalized_directions = self._normalize_directions(directions)
                if not normalized_directions:
                    raise ValueError("Invalid directions format. Use 'ludr' or 'L,R,U,D'")
            
            # 提取可选参数
            step = int(params.get("step", 64))
            
            pixel_coordinate = params.get("pixel_coordinate", True)
            if isinstance(pixel_coordinate, str):
                pixel_coordinate = pixel_coordinate.lower()
                if pixel_coordinate == "true":
                    pixel_coordinate = True
                elif pixel_coordinate == "false":
                    pixel_coordinate = False
                else:
                    raise ValueError("pixel_coordinate must be 'true' or 'false'")
            if not isinstance(pixel_coordinate, bool):
                raise ValueError("pixel_coordinate must be a boolean value")
            
            
            line_width = int(params.get("line_width", 3))
            line_color = params.get("line_color", "red")
            
            new_params = {
                "image": image,
                "start_point": start_point,
                "directions": directions,
                "step": step,
                "pixel_coordinate": pixel_coordinate,
                "line_width": line_width,
                "line_color": line_color
            }
            res = {
                "params_qualified_reward": 1,
                "params_qualified": True,
                "new_params": new_params
            }
            return res
        except Exception as e:
            error_info = str(e)
            res = {
                "params_qualified_reward": 0,
                "params_qualified": False,
                "error_info": error_info,
                "new_params": None,
            }
            return res