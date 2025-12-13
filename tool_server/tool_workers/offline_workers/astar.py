# astar.py
import heapq
import json
import numpy as np

from tool_server.utils.server_utils import build_logger
from tool_server.utils.error_codes import *
from tool_server.tool_workers.offline_workers.base_offline_worker import BaseOfflineWorker

logger = build_logger("astar_worker")

class AStarWithPixelCoordinate(BaseOfflineWorker):
    """
    使用A*算法在给定起点和终点之间寻找最短路径，避开障碍物
    """
    
    def __init__(self):
        super().__init__(model_name="AStarWithPixelCoordinate")
        self.instruction = {
        "type": "function",
        "function": {
            "name": self.model_name,
            "description": "Find the shortest path from start to goal while avoiding obstacles using A* algorithm",
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
                        "description": "Size of each grid cell in pixels (default is 64)",
                    }
                },
                "required": ["start", "goal", "obstacles"]
            },
            "returns": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Status of the pathfinding operation ('success' or 'failed')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Control sequence as comma-separated uppercase directions (L,R,U,D), e.g., 'R,R,U,L,D,D'"
                    },
                    "error_code": {
                        "type": "integer",
                        "description": "Error code (0 for success)"
                    }
                }
            }
        }
    }
        
    def _format_path_string(self, path_string):
        """将路径字符串从 'ludr' 格式转换为 'L,R,U,D' 格式"""
        if not path_string:
            return ""
        
        # 创建映射字典
        direction_map = {'l': 'L', 'r': 'R', 'u': 'U', 'd': 'D'}
        
        # 转换并加入逗号
        formatted = [direction_map.get(c, c.upper()) for c in path_string]
        return ",".join(formatted)
        
    def _execute(self, params):
        """执行A*寻路算法"""
        try:
            # 提取参数
            start = tuple(params["start"])
            goal = tuple(params["goal"])
            obstacles = [tuple(obs) for obs in params["obstacles"]]
            cell_size = params.get("cell_size", 64)
            
            # 执行A*算法，固定cell_size为64
            path_string, path_coords = self.astar_search(
                start, goal, obstacles, cell_size
            )
            
            formatted_path = self._format_path_string(path_string)
        
            result = {
                "status": "success",
                "path": formatted_path,
                "error_code": SUCCESS
            }
            
            return result
            
        except KeyError as e:
            return {
                "status": "failed",
                "message": f"Missing required parameter: {str(e)}",
                "error_code": INVALID_PARAMETERS
            }
        except ValueError as e:
            return {
                "status": "failed",
                "message": f"Invalid parameter value: {str(e)}",
                "error_code": INVALID_PARAMETERS
            }
        except Exception as e:
            logger.error(f"Error in A* algorithm: {str(e)}")
            return {
                "status": "failed",
                "message": f"Error during path finding: {str(e)}",
                "error_code": TOOL_RUN_FAILED
            }
    
    def astar_search(self, start, goal, obstacles, cell_size):
        """
        A* 搜索算法查找从起点到终点的最短路径，同时避开障碍物
        
        Args:
            start (tuple): 起点位置（像素坐标）
            goal (tuple): 终点位置（像素坐标）
            obstacles (list): 障碍物位置列表（像素坐标）
        
        Returns:
            tuple: (路径字符串（'l','r','u','d'）, 路径坐标列表)
        """
        
        # 将像素坐标转换为网格坐标（从0开始）
        def pixel_to_grid(pixel_coord):
            x, y = pixel_coord
            grid_x = int(x // cell_size)
            grid_y = int(y // cell_size)
            return (grid_x, grid_y)
        
        # 将网格坐标转换回像素坐标（取网格中心点）
        def grid_to_pixel(grid_coord):
            grid_x, grid_y = grid_coord
            pixel_x = int((grid_x + 0.5) * cell_size)
            pixel_y = int((grid_y + 0.5) * cell_size)
            return (pixel_x, pixel_y)
        
        # 计算网格大小
        all_points = [start, goal] + obstacles
        max_x = max(point[0] for point in all_points) / cell_size
        max_y = max(point[1] for point in all_points) / cell_size
        grid_width = int(max_x) + 1
        grid_height = int(max_y) + 1
        
        start_grid = pixel_to_grid(start)
        goal_grid = pixel_to_grid(goal)
        obstacles_grid = [pixel_to_grid(obs) for obs in obstacles]
        obstacles_set = set(obstacles_grid)  # 使用集合加速查找
        
        # 定义可能的移动：左、右、上、下
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        direction_chars = ['l', 'r', 'u', 'd']
        
        # 初始化A*算法的数据结构
        open_set = []
        closed_set = set()
        g_score = {start_grid: 0}
        f_score = {start_grid: self.manhattan_distance(start_grid, goal_grid)}
        came_from = {}
        
        # 将起点添加到优先队列
        heapq.heappush(open_set, (f_score[start_grid], start_grid))
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_grid:
                path_string, path_coords = self.reconstruct_path(came_from, current, direction_chars, grid_to_pixel)
                return path_string, path_coords
                
            closed_set.add(current)
            
            for i, (dx, dy) in enumerate(directions):
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 检查邻居是否有效
                if (not (0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height) or
                    neighbor in obstacles_set or neighbor in closed_set):
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = (current, i)
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.manhattan_distance(neighbor, goal_grid)
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return "", []  # 未找到路径
    
    def manhattan_distance(self, a, b):
        """计算两点之间的曼哈顿距离"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def reconstruct_path(self, came_from, current, direction_chars, grid_to_pixel):
        """
        重建从起点到终点的路径
        
        Returns:
            tuple: (路径方向字符串, 路径坐标列表)
        """
        path_chars = []
        path_coords = [grid_to_pixel(current)]  # 从终点开始
        
        while current in came_from:
            current, direction_idx = came_from[current]
            path_chars.append(direction_chars[direction_idx])
            path_coords.append(grid_to_pixel(current))
        
        # 反转路径，使其从起点开始
        path_chars.reverse()
        path_coords.reverse()
        
        return ''.join(path_chars), path_coords
    
    
    def verify_tool_parameter(self, params):
        """验证A*算法的输入参数"""
        try:
            # 检查必要参数是否存在
            required_params = ["start", "goal", "obstacles"]
            for param in required_params:
                if param not in params:
                    raise ValueError(f"Missing required parameter: {param}")
            
            if "cell_size" in params:
                cell_size = int(params["cell_size"])
            else:
                cell_size = 64
            
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
            
            # 确保所有坐标为正数
            for coord in [start, goal] + obstacles:
                if any(x < 0 for x in coord):
                    raise ValueError("Coordinates must be non-negative")
            
            # 创建新的参数字典，确保所有值的类型正确
            new_params = {
                "start": [float(x) for x in start],
                "goal": [float(x) for x in goal],
                "obstacles": [[float(x) for x in obs] for obs in obstacles],
                "cell_size": cell_size
            }
            
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