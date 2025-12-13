# draw_dash_line_path.py
import io
from PIL import Image, ImageDraw
import ast

from tool_server.utils.server_utils import build_logger
from tool_server.utils.utils import load_image, pil_to_base64
from tool_server.tool_workers.offline_workers.base_offline_worker import BaseOfflineWorker
from tool_server.utils.error_codes import *

logger = build_logger("draw_dash_line_path_worker")

class DrawDashLinePath(BaseOfflineWorker):
    """
    在图片上根据起点和方向序列绘制虚线路径
    """
    
    def __init__(self):
        super().__init__(model_name="DrawDashLinePath")
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Draw a dashed path on an image following a sequence of directional commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The image to draw on (image identifier)"
                        },
                        "start_point": {
                            "type": "array",
                            "description": "Starting point coordinates [x, y] in pixels"
                        },
                        "directions": {
                            "type": "string",
                            "description": "Direction sequence with 'u'=up, 'd'=down, 'l'=left, 'r'=right. Accepts both 'rruldd' format and comma-separated 'R,R,U,L,D,D' format."
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
        
        if isinstance(directions, str):
            directions = directions.replace(" ", "")
            
            if "," in directions:
                dir_list = directions.lower().split(",")
                dir_list = [d for d in dir_list if d in ['l', 'r', 'u', 'd']]
                return "".join(dir_list)
            else:
                return "".join(c for c in directions.lower() if c in "lrud")
        
        return ""
    
    def _execute(self, params):
        """执行虚线路径绘制"""
        try:
            # 提取必要参数
            image = params["image"]
            start_point = params["start_point"]
            directions = params["directions"]
            
            directions = self._normalize_directions(directions)
            
            # 使用默认参数
            step = 64
            line_width = 3
            line_color = "blue"
            dash_length = 10
            gap_length = 5
            
            # 处理起始点
            if isinstance(start_point, list) and len(start_point) == 2:
                start_coords = tuple(start_point)
            else:
                return {
                    "status": "failed",
                    "message": "start_point must be a list of two coordinates [x, y]",
                    "error_code": INVALID_PARAMETERS
                }
            
            # 绘制虚线路径
            edited_image = self.draw_dashed_direction_sequence(
                image, 
                start_coords, 
                directions, 
                step=step, 
                line_width=line_width,
                line_color=line_color,
                dash_length=dash_length,
                gap_length=gap_length
            )
            
            # 转换为base64并返回结果
            image_base64 = pil_to_base64(edited_image)
            return {
                "status": "success",
                "edited_image": image_base64,
                "message": "Dashed path drawn successfully",
                "error_code": SUCCESS
            }
            
        except Exception as e:
            logger.error(f"Error drawing dashed path: {str(e)}")
            return {
                "status": "failed",
                "message": f"Error drawing dashed path: {str(e)}",
                "error_code": TOOL_RUN_FAILED
            }
    
    def draw_dashed_line(self, draw, start, end, width, color, dash_length, gap_length):
        """
        在两点之间绘制虚线
        
        Args:
            draw: ImageDraw对象
            start: 起点 (x, y)
            end: 终点 (x, y)
            width: 线宽
            color: 颜色
            dash_length: 虚线段长度
            gap_length: 间隔长度
        """
        x1, y1 = start
        x2, y2 = end
        
        # 计算线段的总长度
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        
        if length == 0:
            return
        
        # 计算单位向量
        ux = dx / length
        uy = dy / length
        
        # 绘制虚线段
        current_length = 0
        dash_pattern_length = dash_length + gap_length
        
        while current_length < length:
            # 计算当前虚线段的起点
            seg_start_x = x1 + ux * current_length
            seg_start_y = y1 + uy * current_length
            
            # 计算当前虚线段的终点
            seg_end_length = min(current_length + dash_length, length)
            seg_end_x = x1 + ux * seg_end_length
            seg_end_y = y1 + uy * seg_end_length
            
            # 绘制虚线段
            draw.line(
                [(seg_start_x, seg_start_y), (seg_end_x, seg_end_y)],
                fill=color,
                width=width
            )
            
            # 移动到下一个虚线段
            current_length += dash_pattern_length
    
    def draw_dashed_direction_sequence(self, image, start, directions, step=64, 
                                      line_width=3, line_color="blue", 
                                      dash_length=10, gap_length=5):
        """
        在图片上从起点沿方向序列画虚线路径。
        
        Args:
            image: PIL.Image对象
            start (tuple): 起点坐标 (x, y)
            directions (str): 方向序列, 由 'u','d','l','r' 组成
            step (int): 每个方向移动的像素数
            line_width (int): 线条宽度
            line_color (str): 线条颜色
            dash_length (int): 虚线段长度
            gap_length (int): 虚线间隔长度
        
        Returns:
            PIL.Image: 带有绘制路径的图像
        """
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
            
            # 画虚线从(old_x, old_y)到(x, y)
            self.draw_dashed_line(
                draw, 
                (old_x, old_y), 
                (x, y), 
                line_width, 
                line_color,
                dash_length,
                gap_length
            )
            
            # 添加箭头头部（实心）
            arrow_size = line_width * 2
            
            if dir == 'u':
                arrow_points = [
                    (x, y),
                    (x - arrow_size, y + arrow_size * 2),
                    (x + arrow_size, y + arrow_size * 2)
                ]
            elif dir == 'd':
                arrow_points = [
                    (x, y),
                    (x - arrow_size, y - arrow_size * 2),
                    (x + arrow_size, y - arrow_size * 2)
                ]
            elif dir == 'l':
                arrow_points = [
                    (x, y),
                    (x + arrow_size * 2, y - arrow_size),
                    (x + arrow_size * 2, y + arrow_size)
                ]
            elif dir == 'r':
                arrow_points = [
                    (x, y),
                    (x - arrow_size * 2, y - arrow_size),
                    (x - arrow_size * 2, y + arrow_size)
                ]
            
            draw.polygon(arrow_points, fill=line_color)

        return img
    
    def verify_tool_parameter(self, params):
        """验证工具的输入参数"""
        try:
            # 加载图像
            image = params["image"]
            image = load_image(image)
            
            # 验证起点
            start_point = params["start_point"]
            start_point = ast.literal_eval(start_point) if isinstance(start_point, str) else start_point
            
            if not isinstance(start_point, list) or len(start_point) != 2:
                raise ValueError("start_point must be a list of two coordinates [x, y]")
            
            # 验证方向
            directions = params["directions"]
            if isinstance(directions, str):
                normalized_directions = self._normalize_directions(directions)
                if not normalized_directions:
                    raise ValueError("Invalid directions format. Use 'ludr' or 'L,R,U,D'")
            
            new_params = {
                "image": image,
                "start_point": start_point,
                "directions": directions
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