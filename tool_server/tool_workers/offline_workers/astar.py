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
                            "description": "Starting point coordinates [x, y] in pixels"
                        },
                        "goal": {
                            "type": "array",
                            "description": "Goal point coordinates [x, y] in pixels"
                        },
                        "obstacles": {
                            "type": "array",
                            "description": "Array of obstacle coordinates [[x1, y1], [x2, y2], ...] in pixels"
                        }
                    },
                    "required": ["start", "goal", "obstacles"]
                }
            }
        }
    
    def _execute(self, params):
        """执行A*寻路算法"""
        try:
            # 提取参数
            start = tuple(params["start"])
            goal = tuple(params["goal"])
            obstacles = [tuple(obs) for obs in params["obstacles"]]
            
            # 执行A*算法，固定cell_size为64
            path_string, path_coords = self.astar_search(
                start, goal, obstacles
            )
            
            result = {
                "status": "success",
                "path": path_string,
                "path_coordinates": path_coords,
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
    
    def astar_search(self, start, goal, obstacles):
        """
        A* 搜索算法查找从起点到终点的最短路径，同时避开障碍物
        
        Args:
            start (tuple): 起点位置（像素坐标）
            goal (tuple): 终点位置（像素坐标）
            obstacles (list): 障碍物位置列表（像素坐标）
        
        Returns:
            tuple: (路径字符串（'l','r','u','d'）, 路径坐标列表)
        """
        cell_size = 64  # 固定cell尺寸为64
        
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
