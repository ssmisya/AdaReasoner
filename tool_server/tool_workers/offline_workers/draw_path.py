# draw_path.py
import io
import re
from PIL import Image, ImageDraw
import ast


from tool_server.utils.server_utils import build_logger
from tool_server.utils.utils import load_image, pil_to_base64
from tool_server.tool_workers.offline_workers.base_offline_worker import BaseOfflineWorker

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
                            "description": "The image to draw on (base64 or path)"
                        },
                        "start_point": {
                            "type": "array",
                            "description": "Starting point coordinates [x, y]"
                        },
                        "directions": {
                            "type": "string",
                            "description": "Direction sequence string with 'u'=up, 'd'=down, 'l'=left, 'r'=right, e.g. 'rruldd'"
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

    def _execute(self, params):
        """执行路径绘制"""
        try:
            
            # 提取必要参数
            image = params["image"]
            start_point = params["start_point"]
            directions = params["directions"]
            
            # 提取可选参数
            step = params.get("step", 64)
            pixel_coordinate = params.get("pixel_coordinate", True)
            line_width = params.get("line_width", 3)
            line_color = params.get("line_color", "red")
            
            # 加载图片
            img = load_image(image)
            
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
                "message": "Path drawn successfully"
            }
            
        except Exception as e:
            logger.error(f"Error drawing path: {str(e)}")
            return {
                "status": "failed",
                "message": f"Error drawing path: {str(e)}"
            }
    
    def draw_direction_sequence(self, image, start, directions, step=64, line_width=3, line_color="red"):
        """
        在图片上从起点沿方向序列画线段。
        
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
                directions = re.sub(r"[^udlr]", "", directions.lower())  # 只保留有效方向字符
            
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