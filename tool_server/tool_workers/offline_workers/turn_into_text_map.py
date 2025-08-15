# turn_into_text_map.py
import numpy as np
from tool_server.utils.server_utils import build_logger
from tool_server.tool_workers.offline_workers.base_offline_worker import BaseOfflineWorker
from tool_server.utils.error_codes import *

logger = build_logger("text_map_worker")

class TurnCoordinateIntoTextMap(BaseOfflineWorker):
    """
    将像素坐标转换为文本地图表示
    起点用@表示，终点用*表示，障碍物用#表示，空白区域用_表示
    """
    
    def __init__(self):
        super().__init__(model_name="TurnCoordinateIntoTextMap")
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Convert pixel coordinates into a text-based map representation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start": {
                            "type": "array",
                            "description": "Starting point coordinates [x, y] in pixels, e.g., [100, 200]"
                        },
                        "goal": {
                            "type": "array",
                            "description": "Goal point coordinates [x, y] in pixels, e.g., [300, 400]"
                        },
                        "obstacles": {
                            "type": "array",
                            "description": "Array of obstacle coordinates [[x1, y1], [x2, y2], ...] in pixels, e.g., [[150, 150], [200, 250], [300, 300]]"
                        },
                        "cell_size": {
                            "type": "integer",
                            "description": "Size of each cell in pixels (default: 64)"
                        }
                    },
                    "required": ["start", "goal", "obstacles"]
                }
            }
        }

    def _execute(self, params):
        """执行坐标转文本地图的转换"""
        try:
            # 提取参数
            start = tuple(params["start"])
            goal = tuple(params["goal"])
            obstacles = [tuple(obs) for obs in params["obstacles"]]
            cell_size = params.get("cell_size", 64)
            
            # 将像素坐标转换为网格坐标
            start_grid = self.pixel_to_grid(start, cell_size)
            goal_grid = self.pixel_to_grid(goal, cell_size)
            obstacles_grid = [self.pixel_to_grid(obs, cell_size) for obs in obstacles]
            
            # 计算网格尺寸
            grid_size = self.calculate_grid_size(start_grid, goal_grid, obstacles_grid)
            
            # 生成文本地图
            text_map = self.generate_text_map(grid_size, start_grid, goal_grid, obstacles_grid)
            
            return {
                "status": "success",
                "text_map": text_map,
                "grid_width": grid_size[0],
                "grid_height": grid_size[1],
                "message": "Text map generated successfully",
                "error_code": SUCCESS,
            }
            
        except Exception as e:
            logger.error(f"Error generating text map: {str(e)}")
            return {
                "status": "failed",
                "message": f"Error generating text map: {str(e)}",
                "error_code": TOOL_RUN_FAILED,
            }
    
    def pixel_to_grid(self, pixel_coord, cell_size):
        """将像素坐标转换为网格坐标（从0开始）"""
        x, y = pixel_coord
        grid_x = int(x // cell_size)
        grid_y = int(y // cell_size)
        return (grid_x, grid_y)
    
    def calculate_grid_size(self, start, goal, obstacles):
        """计算包含所有点的最小网格尺寸"""
        all_points = [start, goal] + obstacles
        max_x = max(point[0] for point in all_points) + 1  # +1以确保包含最右边的点
        max_y = max(point[1] for point in all_points) + 1  # +1以确保包含最下边的点
        
        return (int(max_x), int(max_y))
    
    def generate_text_map(self, grid_size, start, goal, obstacles):
        """
        生成文本地图表示
        @ 表示起点
        * 表示终点
        # 表示障碍物
        _ 表示空白区域
        """
        width, height = grid_size
        
        # 创建表头行
        header_row = "| |"
        for col in range(1, width + 1):
            header_row += f" Col {col} |"
        
        # 创建网格
        grid_rows = []
        grid_rows.append(header_row)
        
        # 构建障碍物集合以加速查找
        obstacles_set = set(obstacles)
        
        for y in range(height):
            row = f"| Row {y+1} |"
            for x in range(width):
                pos = (x, y)
                if pos == start:
                    row += " @ |"
                elif pos == goal:
                    row += " * |"
                elif pos in obstacles_set:
                    row += " # |"
                else:
                    row += " _ |"
            grid_rows.append(row)
        
        # 合并为完整文本地图
        text_map = "\n".join(grid_rows)
        return text_map
    
    def verify_tool_parameter(self, params):
        """验证工具的输入参数"""
        try:
            # 检查必要参数是否存在
            required_params = ["start", "goal", "obstacles"]
            for param in required_params:
                if param not in params:
                    raise ValueError(f"Missing required parameter: {param}")
            
            # 验证并转换 start 参数
            start = params["start"]
            if isinstance(start, str):
                try:
                    # 尝试解析字符串形式的坐标
                    import ast
                    start = ast.literal_eval(start)
                except:
                    raise ValueError("start must be a valid coordinate array [x, y]")
            
            # 检查start是否为合法的坐标
            if not isinstance(start, list) or len(start) != 2 or not all(isinstance(x, (int, float)) for x in start):
                raise ValueError("start must be a valid coordinate array [x, y]")
                
            # 验证并转换 goal 参数
            goal = params["goal"]
            if isinstance(goal, str):
                try:
                    import ast
                    goal = ast.literal_eval(goal)
                except:
                    raise ValueError("goal must be a valid coordinate array [x, y]")
                    
            # 检查goal是否为合法的坐标
            if not isinstance(goal, list) or len(goal) != 2 or not all(isinstance(x, (int, float)) for x in goal):
                raise ValueError("goal must be a valid coordinate array [x, y]")
                
            # 验证并转换 obstacles 参数
            obstacles = params["obstacles"]
            if isinstance(obstacles, str):
                try:
                    import ast
                    obstacles = ast.literal_eval(obstacles)
                except:
                    raise ValueError("obstacles must be a valid array of coordinates")
                    
            # 检查obstacles是否为数组
            if not isinstance(obstacles, list):
                raise ValueError("obstacles must be an array")
                
            # 检查obstacles中的每个元素是否为合法的坐标
            for obs in obstacles:
                if not isinstance(obs, list) or len(obs) != 2 or not all(isinstance(x, (int, float)) for x in obs):
                    raise ValueError("Each obstacle must be a valid coordinate array [x, y]")
            
            # 验证cell_size参数（如果提供的话）
            cell_size = params.get("cell_size", 64)
            if isinstance(cell_size, str):
                try:
                    cell_size = int(cell_size)
                except:
                    raise ValueError("cell_size must be an integer")
            
            if not isinstance(cell_size, (int, float)) or cell_size <= 0:
                raise ValueError("cell_size must be a positive number")
            
            # 创建新的参数字典，确保所有值的类型正确
            new_params = {
                "start": [float(x) for x in start],
                "goal": [float(x) for x in goal],
                "obstacles": [[float(x) for x in obs] for obs in obstacles],
                "cell_size": int(cell_size)
            }
            
            # 确保所有坐标为非负数
            for coord in [start, goal] + obstacles:
                if any(x < 0 for x in coord):
                    raise ValueError("Coordinates must be non-negative")
            
            # 验证通过，返回成功结果
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