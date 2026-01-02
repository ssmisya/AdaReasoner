# data_curation_point_random.py
import os
import sys
import re
import json
import random
import heapq
import base64
import io
import uuid
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# 导入工具管理器
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from tool_server.utils.utils import *

# 确保能导入相关工具函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import List, Tuple, Dict, Any, Optional

# 设置常量
CELL_SIZE = 64
POINT_MARKER_SIZE = 5
LINE_WIDTH = 3
LINE_COLOR = "red"  # 默认线条颜色

# 初始化工具管理器
tool_manager = ToolManager()

def image_to_base64(image):
    """将PIL图像转换为Base64字符串"""
    if isinstance(image, str) and os.path.exists(image):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    else:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def base64_to_image(base64_string):
    """将Base64字符串转换为PIL图像"""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def extract_coordinates(text_map, cell_size=64):
    """
    从文本地图中提取起点、终点和障碍物的坐标
    
    Args:
        text_map (list): 2D地图，S=起点, G=终点, H=障碍物
        cell_size (int): 每个格子的像素大小
            
    Returns:
        dict: 包含起点、终点和障碍物的像素坐标
    """
    half_cell = cell_size / 2
    start_point = None
    holes = []
    goal_point = None
    
    for i, row in enumerate(text_map):
        for j, cell in enumerate(row):
            # 计算像素坐标（格子中心）
            pixel_x = j * cell_size + half_cell
            pixel_y = i * cell_size + half_cell
            
            cell_str = cell.decode('utf-8') if isinstance(cell, bytes) else cell
            
            if cell_str == 'S':
                start_point = (pixel_x, pixel_y)
            elif cell_str == 'H':
                holes.append((pixel_x, pixel_y))
            elif cell_str == 'G':
                goal_point = (pixel_x, pixel_y)
    
    return {
        'start': start_point,
        'holes': holes,
        'goal': goal_point
    }

def verify_safe_path(text_map, path_string, verbose=False):
    """
    验证给定的动作序列是否安全（不会落入冰洞），无论是否到达目标点
    更简单的实现方法：直接检查每一步是否会落入冰洞
    
    Args:
        text_map (list): 表示FrozenLake地图的2D列表
        path_string (str): 逗号分隔的动作字符串 (R, L, U, D)
        verbose (bool): 是否打印详细信息
    
    Returns:
        bool: 如果整个路径安全（不会落入冰洞）则为True，否则为False
    """
    # 找到起点位置
    start_row, start_col = None, None
    
    for i, row in enumerate(text_map):
        for j, cell in enumerate(row):
            cell_str = cell.decode('utf-8') if isinstance(cell, bytes) else cell
            if cell_str == 'S':
                start_row, start_col = i, j
                break
        if start_row is not None:
            break
    
    if start_row is None:
        if verbose:
            print("起点未找到")
        return False
    
    # 解析路径字符串
    directions = path_string.split(',')
    
    # 字典，将方向映射到位置变化
    direction_to_move = {
        'L': (0, -1),
        'R': (0, 1),
        'U': (-1, 0),
        'D': (1, 0),
    }
    
    # 跟踪当前位置
    curr_row, curr_col = start_row, start_col
    map_size = len(text_map)
    
    # 逐步执行路径
    for idx, direction in enumerate(directions):
        direction = direction.strip()
        
        if direction not in direction_to_move:
            if verbose:
                print(f"无效方向: {direction}")
            return False
        
        # 获取移动的方向
        dr, dc = direction_to_move[direction]
        
        # 计算新位置
        new_row = curr_row + dr
        new_col = curr_col + dc
        
        # 检查边界
        if new_row < 0 or new_row >= map_size or new_col < 0 or new_col >= map_size:
            # 如果超出边界，保持原位
            if verbose:
                print(f"步骤 {idx+1}: {direction} -> 超出边界，保持原位")
            continue
        
        # 更新当前位置
        curr_row, curr_col = new_row, new_col
        
        # 获取当前格子的类型
        cell = text_map[curr_row][curr_col]
        cell_str = cell.decode('utf-8') if isinstance(cell, bytes) else cell
        
        # 检查是否掉入冰洞
        if cell_str == 'H':
            if verbose:
                print(f"步骤 {idx+1}: {direction} -> 掉入冰洞!")
            return False
        
        # 对于目标点，只需要记录，不影响路径安全性
        if cell_str == 'G' and verbose:
            print(f"步骤 {idx+1}: {direction} -> 到达目标!")
    
    # 如果整个路径执行完毕没有掉入冰洞，则安全
    return True

def verify_path(text_map, path_string, verbose=False):
    """
    验证给定的动作序列是否安全（不会落入冰窟窿）
    
    Args:
        text_map (list): 表示FrozenLake地图的2D列表
        path_string (str): 逗号分隔的动作字符串 (R, L, U, D)
        verbose (bool): 是否打印详细信息
    
    Returns:
        bool: 如果路径安全（不会落入冰窟窿）则为True，否则为False
    """
    # 创建环境
    env = gym.make('FrozenLake-v1', desc=text_map, render_mode="rgb_array" if verbose else None, is_slippery=False)
    obs, _ = env.reset()
    
    # 字典，将方向映射到FrozenLake预期的动作
    direction_to_action = {
        'L': 0,  # LEFT
        'D': 1,  # DOWN
        'R': 2,  # RIGHT
        'U': 3,  # UP
    }
    
    # 解析路径字符串（由逗号分隔的方向）
    directions = path_string.split(',')
    
    # 跟随路径，检查是否安全
    is_safe = True
    step_count = 0
    
    for direction in directions:
        direction = direction.strip()  # 移除可能的空格
        action = direction_to_action.get(direction)
        if action is None:
            if verbose:
                print(f"Invalid direction: {direction}")
            is_safe = False
            break
            
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        if verbose:
            print(f"Step {step_count}: {direction} -> Terminated: {terminated}, Reward: {reward}")
        
        # 如果reward是0且terminated为True，说明掉进了冰窟窿
        if terminated and reward == 0:
            if verbose:
                print(f"落入冰窟窿! 步骤: {step_count}, 方向: {direction}")
            is_safe = False
            break
    
    env.close()
    return is_safe

def generate_random_path(text_map, min_length=1, max_length=20):
    """
    为FrozenLake环境生成随机路径。
    允许重复走过的路线，但确保所有步骤都不会超出地图边界。
    
    Args:
        text_map (list): 表示FrozenLake地图的2D列表
        min_length (int): 路径的最小长度
        max_length (int): 路径的最大长度
    
    Returns:
        str: 表示随机路径的方向字符串，格式为"L,R,U,D,..."
    """
    # 定义可能的移动
    directions = ['L', 'R', 'U', 'D']
    
    # 确定路径长度
    map_size = len(text_map)
    if max_length < min_length:
        max_length = min_length
    
    # 根据需要根据地图大小调整max_length
    suggested_max = map_size * 4  # 基于地图大小的合理上限
    if max_length > suggested_max:
        max_length = suggested_max
    
    # 生成随机路径长度
    path_length = random.randint(min_length, max_length)
    
    # 找到起点位置（S）
    start_row, start_col = None, None
    for i, row in enumerate(text_map):
        for j, cell in enumerate(row):
            cell_str = cell.decode('utf-8') if isinstance(cell, bytes) else cell
            if cell_str == 'S':
                start_row, start_col = i, j
                break
        if start_row is not None:
            break
    
    if start_row is None:
        # 如果找不到起点，假设在左上角
        start_row, start_col = 0, 0
    
    # 跟踪当前位置，用于确保所有步骤不会出界
    current_row, current_col = start_row, start_col
    
    path_directions = []
    
    # 逐步生成路径，确保不会出界
    for _ in range(path_length):
        # 找出不会导致出界的方向
        valid_directions = []
        
        # 检查每个方向是否会导致出界
        if current_col > 0:  # 可以向左移动
            valid_directions.append('L')
        if current_col < map_size - 1:  # 可以向右移动
            valid_directions.append('R')
        if current_row > 0:  # 可以向上移动
            valid_directions.append('U')
        if current_row < map_size - 1:  # 可以向下移动
            valid_directions.append('D')
        
        # 如果没有有效方向（应该不会发生，因为至少一个方向是有效的，除非在角落）
        if not valid_directions:
            # 这种情况下，选择一个方向，但移动后仍停留在原地（模拟撞墙不动）
            chosen_direction = random.choice(directions)
        else:
            # 从有效方向中随机选择
            chosen_direction = random.choice(valid_directions)
        
        path_directions.append(chosen_direction)
        
        # 根据选择的方向更新当前位置
        if chosen_direction == 'L' and current_col > 0:
            current_col -= 1
        elif chosen_direction == 'R' and current_col < map_size - 1:
            current_col += 1
        elif chosen_direction == 'U' and current_row > 0:
            current_row -= 1
        elif chosen_direction == 'D' and current_row < map_size - 1:
            current_row += 1
        # 如果方向会导致出界，当前位置保持不变（不应该发生，因为我们只选择有效方向）
    
    # 将方向列表转换为逗号分隔的字符串
    path = ','.join(path_directions)
    
    return path



def create_draw_path_data(env_image, start_coords, path, img_id, is_random=False, img_ref="img_1"):
    """
    创建路径绘制工具的输入输出数据，实际调用工具
    
    Args:
        env_image: 环境图像
        start_coords: 起点坐标
        path: 路径字符串 (格式为"L,R,U,D,...")
        img_id: 图像ID
        is_random: 是否为随机路径
        img_ref: 图像引用名称
        
    Returns:
        dict: 输入、输出数据和图像路径
    """
    if not path:
        # 如果没有路径，返回空结果
        return None
    
    # 直接使用逗号分隔的大写方向格式，无需转换
    tool_path_format = path  # 直接使用"L,R,U,D"格式
    
    # 生成图像ID
    path_img_id = f"{img_id}_path_{'random' if is_random else 'astar'}"
    
    # 保存临时图像文件
    temp_path = f"temp_{uuid.uuid4().hex[:8]}.png"
    env_image.save(temp_path)
    
    # Draw2DPath工具的输入参数 - 用于保存在数据集中
    params = {
        "image": img_ref,  # 使用图像引用
        "start_point": list(start_coords),
        "directions": tool_path_format,  # 直接使用"L,R,U,D"格式
        "pixel_coordinate": True,
    }
    
    # 用于实际调用的参数
    call_params = {
        "image": temp_path,
        "start_point": list(start_coords),
        "directions": tool_path_format,  # 直接使用"L,R,U,D"格式
        "pixel_coordinate": True,
    }
    
    # 实际调用Draw2DPath工具
    if "Draw2DPath" in tool_manager.available_tools:
        result = tool_manager.call_tool("Draw2DPath", call_params)
        # 移除一些不需要的字段
        if "raw_response" in result:
            result.pop("raw_response", None)
        if "tool_reward" in result:
            result.pop("tool_reward", None)
        
        output = {
            "tool_response_from": "Draw2DPath",
            **result
        }
        
        # 将base64图像转换回PIL对象
        path_image = None
        if "edited_image" in output:
            path_image = base64_to_pil(output["edited_image"])
            # 从输出中移除base64图像
            output.pop("edited_image", None)
        else:
            # 如果工具未返回编辑后的图像，使用原始图像
            path_image = env_image.copy()
    else:
        raise ValueError("Draw2DPath tool is not available!")
    
    # 删除临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # DrawPath工具的输入
    draw_path_input = {
        "name": "Draw2DPath",
        "parameters": params
    }
    
    return {
        "input": draw_path_input,
        "output": output,
        "image_id": path_img_id,
        "image": path_image
    }


def create_point_data(env_image, point_type, coords, img_id, img_ref="img_1"):
    """
    创建Point工具的输入输出数据，实际调用Point工具
    
    Args:
        env_image: 环境图像
        point_type: 点类型 (Elf, Ice Hole, Gift)
        coords: 点坐标（用于仿真）
        img_id: 图像ID
        img_ref: 图像引用名称
        
    Returns:
        dict: 输入、输出数据和图像路径
    """
    # 生成图像ID
    point_img_id = f"{img_id}_point_{point_type.lower()}"
    
    # 生成描述
    description = point_type
    
    # Point工具的输入参数
    params = {
        "image": img_ref,  # 使用图像引用而非直接包含base64
        "description": description
    }
    
    # 原始参数用于实际调用工具
    call_params = {
        "image": pil_to_base64(env_image),
        "description": description
    }
    
    # 实际调用Point工具
    if "Point" in tool_manager.available_tools:
        result = tool_manager.call_tool("Point", call_params)
        # 移除一些不需要的字段
        if "raw_response" in result:
            result.pop("raw_response", None)
        if "tool_reward" in result:
            result.pop("tool_reward", None)
        
        # 如果工具调用失败或未返回点，使用已知坐标
        if result.get('status') != "success" or len(result.get('points', [])) == 0:
            print(f"Point tool failed for {point_type}, using known coordinates")
            result = {
                "status": "success",
                "points": [{"x": coords[0], "y": coords[1]}],
            }
        
        output = {
            "tool_response_from": "Point",
            **result
        }
        
        # 将base64图像转换回PIL对象并保存
        marked_image = None
        if "edited_image" in output:
            marked_image = base64_to_pil(output["edited_image"])
            # 移除输出中的base64图像
            output.pop("edited_image", None)
        else:
            raise ValueError("Point tool did not return edited image!")
    else:
        raise ValueError("Point tool is not available!")
    
    # Point工具的输入
    point_input = {
        "name": "Point",
        "parameters": params
    }
    
    return {
        "input": point_input,
        "output": output,
        "image_id": point_img_id,
        "image": marked_image
    }

def create_ice_holes_data(env_image, holes, img_id, img_ref="img_1"):
    """
    创建标记所有冰洞的Point工具数据，实际调用Point工具
    
    Args:
        env_image: 环境图像
        holes: 冰洞坐标列表
        img_id: 图像ID
        img_ref: 图像引用名称
        
    Returns:
        dict: 输入、输出数据和图像路径
    """
    # 生成图像ID
    ice_holes_img_id = f"{img_id}_point_ice_holes"
    
    # 生成描述 - 请求标记所有冰洞
    description = "all ice holes in the frozen lake"
    
    # Point工具的输入参数
    params = {
        "image": img_ref,  # 使用图像引用而非直接包含base64
        "description": description
    }
    
    # 原始参数用于实际调用工具
    call_params = {
        "image": pil_to_base64(env_image),
        "description": description
    }
    
    # 实际调用Point工具
    if "Point" in tool_manager.available_tools:
        result = tool_manager.call_tool("Point", call_params)
        # 移除一些不需要的字段
        if "raw_response" in result:
            result.pop("raw_response", None)
        if "tool_reward" in result:
            result.pop("tool_reward", None)
        
        # 如果工具调用失败或未返回足够的点，使用已知坐标
        detected_holes = result.get('points', [])
        if result.get('status') != "success" or len(detected_holes) < len(holes) / 2:  # 至少检测到一半的洞
            print(f"Point tool failed for ice holes, using known coordinates")
            result = {
                "status": "success",
                "points": [{"x": hole[0], "y": hole[1]} for hole in holes],
            }
        
        output = {
            "tool_response_from": "Point",
            **result
        }
        
        # 将base64图像转换回PIL对象并保存
        marked_image = None
        if "edited_image" in output:
            marked_image = base64_to_pil(output["edited_image"])
            # 移除输出中的base64图像
            output.pop("edited_image", None)
        else:
            # 如果工具未返回编辑后的图像，手动创建一个
            marked_image = env_image.copy()
            draw = ImageDraw.Draw(marked_image)
            for point in output.get('points', []):
                x, y = point['x'], point['y']
                draw.ellipse((x-POINT_MARKER_SIZE, y-POINT_MARKER_SIZE, 
                            x+POINT_MARKER_SIZE, y+POINT_MARKER_SIZE), 
                            fill="red", outline="white")
    else:
        raise ValueError("Point tool is not available!")
    
    # Point工具的输入
    point_input = {
        "name": "Point",
        "parameters": params
    }
    
    return {
        "input": point_input,
        "output": output,
        "image_id": ice_holes_img_id,
        "image": marked_image
    }

    """
    创建标记所有冰洞的Point工具数据，但不实际调用工具，而是直接使用提供的坐标
    
    Args:
        env_image: 环境图像
        holes: 冰洞坐标列表
        img_id: 图像ID
        img_ref: 图像引用名称
        
    Returns:
        dict: 输入、输出数据和图像路径
    """
    # 生成图像ID
    ice_holes_img_id = f"{img_id}_point_ice_holes"
    
    # 生成描述 - 请求标记所有冰洞
    description = "all ice holes in the frozen lake"
    
    # Point工具的输入参数
    params = {
        "image": img_ref,  # 使用图像引用而非直接包含base64
        "description": description
    }
    
    # 直接使用提供的坐标创建结果
    result = {
        "status": "success",
        "points": [{"x": hole[0], "y": hole[1]} for hole in holes],
    }
    
    output = {
        "tool_response_from": "Point",
        **result
    }
    
    # 手动创建标记图像
    marked_image = env_image.copy()
    draw = ImageDraw.Draw(marked_image)
    
    for point in output.get('points', []):
        x, y = point['x'], point['y']
        draw.ellipse((x-POINT_MARKER_SIZE, y-POINT_MARKER_SIZE, 
                    x+POINT_MARKER_SIZE, y+POINT_MARKER_SIZE), 
                    fill="red", outline="white")
    
    # Point工具的输入
    point_input = {
        "name": "Point",
        "parameters": params
    }
    
    return {
        "input": point_input,
        "output": output,
        "image_id": ice_holes_img_id,
        "image": marked_image
    }

def create_astar_data(start_coords, goal_coords, obstacle_coords, img_id):
    """
    创建A*路径规划工具的输入输出数据，实际调用工具
    
    Args:
        start_coords: 起点坐标
        goal_coords: 终点坐标
        obstacle_coords: 障碍物坐标列表
        img_id: 图像ID
        
    Returns:
        dict: 输入、输出数据
    """
    # A*工具的输入参数
    params = {
        "start": list(start_coords),
        "goal": list(goal_coords),
        "obstacles": [list(obs) for obs in obstacle_coords]
    }
    
    # 实际调用A*工具
    if "AStarWithPixelCoordinate" in tool_manager.available_tools:
        result = tool_manager.call_tool("AStarWithPixelCoordinate", params)
        # 移除一些不需要的字段
        if "raw_response" in result:
            result.pop("raw_response", None)
        if "tool_reward" in result:
            result.pop("tool_reward", None)
        
        # 如果工具调用失败，使用内置的A*实现
        if result.get('status') != "success" or not result.get('path'):
            result["path"] = ""  # 如果没有路径，设置为空字符串
        
        # 无需转换格式，直接使用工具返回的路径
        # 假设工具已经支持返回"L,R,U,D"格式
        
        output = {
            "tool_response_from": "AStarWithPixelCoordinate",
            **result
        }
    else:
        raise ValueError("AStarWithPixelCoordinate tool is not available!")
    
    # A*工具的输入
    astar_input = {
        "name": "AStarWithPixelCoordinate",
        "parameters": params
    }
    
    return {
        "input": astar_input,
        "output": output,
        "path": output.get("path", "")
    }
    
def create_point_data_no_tool(env_image, point_type, coords, img_id, img_ref="img_1"):
    """
    创建Point工具的输入输出数据，但不实际调用工具，而是直接使用提供的坐标
    输出格式与MolmoPointWorker保持一致
    
    Args:
        env_image: 环境图像
        point_type: 点类型 (Elf, Ice Hole, Gift)
        coords: 点坐标
        img_id: 图像ID
        img_ref: 图像引用名称
        
    Returns:
        dict: 输入、输出数据和图像路径
    """
    # 生成图像ID
    point_img_id = f"{img_id}_point_{point_type.lower()}"
    
    description = point_type
    # Point工具的输入参数
    params = {
        "image": img_ref,  # 使用图像引用
        "description": description
    }
    
    # 生成请求ID (与MolmoPointWorker保持一致)
    request_id = str(uuid.uuid4())
    
    # 创建与MolmoPointWorker相同格式的输出结果
    point_data = [{
        "x": float(coords[0]),
        "y": float(coords[1])
    }]
    
    output = {
        "tool_response_from": "Point",
        "status": "success",
        "points": point_data,
        "image_dimensions_pixels": {
            "width": env_image.width,
            "height": env_image.height
        },
        "error_code": 0,
    }
    
    # 手动创建标记图像，使用与MolmoPointWorker相同的标记方式
    import matplotlib.pyplot as plt
    from io import BytesIO
    
    # 创建包含标记点的图像
    fig, ax = plt.subplots(figsize=(env_image.width / 100, env_image.height / 100), dpi=100)
    ax.imshow(env_image)
    
    # 将点坐标转换为numpy数组格式
    points = np.array([[coords[0], coords[1]]])
    
    # 使用与MolmoPointWorker相同的标记风格
    # 绿色星形标记，白色边缘（除了冰洞使用红色）
    marker_size = 375  # MolmoPointWorker默认使用的标记大小
    marker_color =  "green"
    
    ax.scatter(
        points[:, 0], points[:, 1], 
        color=marker_color, marker='*', 
        s=marker_size, edgecolor='white', 
        linewidth=1.25
    )
    
    plt.axis('off')  # 关闭坐标轴
    
    # 转换图像格式
    buffered = BytesIO()
    plt.savefig(buffered, format="PNG", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buffered.seek(0)
    marked_image = Image.open(buffered)
    
    # Point工具的输入
    point_input = {
        "name": "Point",
        "parameters": params
    }
    
    return {
        "input": point_input,
        "output": output,
        "image_id": point_img_id,
        "image": marked_image
    }

def create_ice_holes_data_no_tool(env_image, holes, img_id, img_ref="img_1"):
    """
    创建标记所有冰洞的Point工具数据，但不实际调用工具，而是直接使用提供的坐标
    输出格式与MolmoPointWorker保持一致
    
    Args:
        env_image: 环境图像
        holes: 冰洞坐标列表
        img_id: 图像ID
        img_ref: 图像引用名称
        
    Returns:
        dict: 输入、输出数据和图像路径
    """
    # 生成图像ID
    ice_holes_img_id = f"{img_id}_point_ice_holes"
    
    # 生成描述
    description = "Ice Holes"
    
    # Point工具的输入参数
    params = {
        "image": img_ref,  # 使用图像引用
        "description": description
    }
    
    # 生成请求ID (与MolmoPointWorker保持一致)
    request_id = str(uuid.uuid4())
    
    # 创建与MolmoPointWorker相同格式的输出结果
    point_data = []
    for hole in holes:
        point_data.append({
            "x": float(hole[0]),
            "y": float(hole[1])
        })
    
    output = {
        "tool_response_from": "Point",
        "status": "success",
        "points": point_data,
        "image_dimensions_pixels": {
            "width": env_image.width,
            "height": env_image.height
        },
        "error_code": 0,
    }
    
    # 手动创建标记图像，使用与MolmoPointWorker相同的标记方式
    import matplotlib.pyplot as plt
    from io import BytesIO
    
    # 创建包含标记点的图像
    fig, ax = plt.subplots(figsize=(env_image.width / 100, env_image.height / 100), dpi=100)
    ax.imshow(env_image)
    
    if holes:  # 确保有坐标才进行绘制
        # 将所有冰洞坐标转换为numpy数组
        points = np.array([[float(hole[0]), float(hole[1])] for hole in holes])
        
        # 使用红色星形标记，白色边缘
        marker_size = 375  # MolmoPointWorker默认使用的标记大小
        ax.scatter(
            points[:, 0], points[:, 1], 
            color="green", marker='*', 
            s=marker_size, edgecolor='white', 
            linewidth=1.25
        )
    
    plt.axis('off')  # 关闭坐标轴
    
    # 转换图像格式
    buffered = BytesIO()
    plt.savefig(buffered, format="PNG", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buffered.seek(0)
    marked_image = Image.open(buffered)
    
    # Point工具的输入
    point_input = {
        "name": "Point",
        "parameters": params
    }
    
    return {
        "input": point_input,
        "output": output,
        "image_id": ice_holes_img_id,
        "image": marked_image
    }
    
def create_text_map_data(start_coords, goal_coords, obstacle_coords, img_id):
    """
    创建文本地图生成工具的输入输出数据，实际调用工具
    
    Args:
        start_coords: 起点坐标
        goal_coords: 终点坐标
        obstacle_coords: 障碍物坐标列表
        img_id: 图像ID
        
    Returns:
        dict: 输入、输出数据
    """
    # 文本地图工具的输入参数
    params = {
        "start": list(start_coords),
        "goal": list(goal_coords),
        "obstacles": [list(obs) for obs in obstacle_coords],
    }
    
    # 实际调用文本地图工具
    if "TurnCoordinateIntoTextMap" in tool_manager.available_tools:
        result = tool_manager.call_tool("TurnCoordinateIntoTextMap", params)
        # 移除一些不需要的字段
        if "raw_response" in result:
            result.pop("raw_response", None)
        if "tool_reward" in result:
            result.pop("tool_reward", None)
        
        output = result
    else:
        raise ValueError("TurnCoordinateIntoTextMap tool is not available!")
    
    # 文本地图工具的输入
    text_map_input = {
        "name": "TurnCoordinateIntoTextMap",
        "parameters": params
    }
    
    return {
        "input": text_map_input,
        "output": output
    }

def create_env_description(text_map, size):
    """创建环境的文本描述"""
    # 计算环境中的元素
    flat_map = [cell for row in text_map for cell in row]
    
    if isinstance(flat_map[0], bytes):
        flat_map = [cell.decode('utf-8') for cell in flat_map]
        
    hole_count = flat_map.count('H')
    
    # 生成环境描述
    description = (
        f"This is a {size}x{size} FrozenLake environment. "
        f"It contains {hole_count} holes (H) that should be avoided. "
        "The starting position (S) is marked as an Elf, "
        "and the goal position (G) is marked as a Gift. "
        "The objective is to navigate from the Elf to the Gift without "
        "falling into any holes."
    )
    
    return description

def generate_random_map(size: int = 8, p: float = 0.8, seed: Optional[int] = None) -> List[str]:
    """
    生成随机有效地图（即从起点到终点有可行路径）
    
    Args:
        size: 地图的边长
        p: 格子是安全的概率（非障碍物）
        seed: 随机数种子，用于生成可重复的地图
    
    Returns:
        List[str]: 表示地图的字符串列表
    """
    valid = False
    board = []  # 初始化以满足类型检查
    
    # 设置随机数生成器
    if seed is not None:
        np_random = np.random.RandomState(seed)
    else:
        np_random = np.random.RandomState()
    
    while not valid:
        p = min(1, p)
        # 生成初始地图，只有安全格子(F)和危险格子(H)
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        
        # 随机放置起点(S)
        start_row = np_random.randint(0, size)
        start_col = np_random.randint(0, size)
        board[start_row][start_col] = "S"
        
        # 随机放置终点(G)，确保不与起点重叠
        goal_row, goal_col = start_row, start_col
        while (goal_row, goal_col) == (start_row, start_col):
            goal_row = np_random.randint(0, size)
            goal_col = np_random.randint(0, size)
        board[goal_row][goal_col] = "G"
        
        # 检查地图是否有效（能从起点到达终点）
        valid = check_valid_map(board)
    
    # 将二维数组转换为字符串列表
    return ["".join(x) for x in board]

def generate_balanced_random_paths(text_map, size, img_id, start_coords, env_image):
    """
    生成平衡的随机路径集合，确保不同长度的路径都有均衡的正确/错误比例
    
    Args:
        text_map: 表示FrozenLake地图的2D列表
        size: 地图大小
        img_id: 图像ID
        start_coords: 起点坐标
        
    Returns:
        dict: 包含随机路径数据的字典
    """
    # 根据地图大小确定要生成的路径长度
    path_lengths = []
    if size <= 4:
        path_lengths = [1, 3, 5]
    elif size <= 6:
        path_lengths = [1, 3, 5, 7]
    else:
        path_lengths = [1, 3, 5, 7, 9, 11]
    
    # 为每种长度生成一条路径，并保持正确/错误比例
    random_paths = []
    
    # 首先尝试为每个长度生成一条正确路径和一条错误路径
    for length in path_lengths:
        # 尝试生成一条正确的路径
        safe_path = generate_path_with_safety(text_map, length, True, max_attempts=50)
        if safe_path:
            random_paths.append({
                "path": safe_path["path"],
                "is_safe": True,
                "length": length
            })
        
        # 尝试生成一条错误的路径
        unsafe_path = generate_path_with_safety(text_map, length, False, max_attempts=50)
        if unsafe_path:
            random_paths.append({
                "path": unsafe_path["path"],
                "is_safe": False,
                "length": length
            })
    
    # 如果生成的路径不足，再补充一些随机路径
    if not random_paths:
        # 备用方案：生成一条完全随机的路径
        random_path = generate_random_path(text_map, min_length=1, max_length=size*2)
        is_safe = verify_safe_path(text_map, random_path, verbose=False)
        random_paths.append({
            "path": random_path,
            "is_safe": is_safe,
            "length": len(random_path.split(","))
        })
    
    # 从生成的路径中随机选择一条
    selected_path_data = random.choice(random_paths)
    
    # 创建随机路径绘制数据
    try:
        random_draw_data = create_draw_path_data(
            env_image,  # 稍后填充图像
            start_coords,
            selected_path_data["path"],
            img_id,
            is_random=True
        )
        
        if random_draw_data:
            # 添加安全性标记和路径数据
            random_draw_data["path"] = selected_path_data["path"]
            random_draw_data["is_safe"] = selected_path_data["is_safe"]
            random_draw_data["is_valid"] = verify_path(text_map, selected_path_data["path"], verbose=False)
            random_draw_data["path_length"] = selected_path_data["length"]
            return random_draw_data
    except Exception as e:
        print(f"Error creating random path drawing data: {e}")
    
    return None

def generate_path_with_safety(text_map, target_length, should_be_safe, max_attempts=100):
    """
    生成指定长度和安全性的路径
    
    Args:
        text_map: 表示FrozenLake地图的2D列表
        target_length: 目标路径长度
        should_be_safe: 是否应该是安全路径
        max_attempts: 最大尝试次数
        
    Returns:
        dict: 包含路径和安全性的字典，如果无法生成则返回None
    """
    for _ in range(max_attempts):
        # 生成指定长度的随机路径
        path = generate_random_path_with_length(text_map, target_length)
        
        # 验证路径安全性
        is_safe = verify_safe_path(text_map, path, verbose=False)
        
        # 如果安全性符合要求，返回路径
        if is_safe == should_be_safe:
            return {
                "path": path,
                "is_safe": is_safe
            }
    
    # 达到最大尝试次数仍未找到符合条件的路径
    return None

def generate_random_path_with_length(text_map, target_length):
    """
    生成指定长度的随机路径
    
    Args:
        text_map: 表示FrozenLake地图的2D列表
        target_length: 目标路径长度
        
    Returns:
        str: 表示随机路径的方向字符串，格式为"L,R,U,D,..."
    """
    # 定义可能的移动
    directions = ['L', 'R', 'U', 'D']
    
    # 找到起点位置
    start_row, start_col = None, None
    for i, row in enumerate(text_map):
        for j, cell in enumerate(row):
            cell_str = cell.decode('utf-8') if isinstance(cell, bytes) else cell
            if cell_str == 'S':
                start_row, start_col = i, j
                break
        if start_row is not None:
            break
    
    if start_row is None:
        # 如果找不到起点，假设在左上角
        start_row, start_col = 0, 0
    
    # 跟踪当前位置
    current_row, current_col = start_row, start_col
    map_size = len(text_map)
    
    path_directions = []
    
    # 生成指定长度的路径
    for _ in range(target_length):
        # 找出不会导致出界的方向
        valid_directions = []
        
        # 检查每个方向是否会导致出界
        if current_col > 0:  # 可以向左移动
            valid_directions.append('L')
        if current_col < map_size - 1:  # 可以向右移动
            valid_directions.append('R')
        if current_row > 0:  # 可以向上移动
            valid_directions.append('U')
        if current_row < map_size - 1:  # 可以向下移动
            valid_directions.append('D')
        
        # 如果没有有效方向（应该不会发生，因为至少一个方向是有效的，除非在角落）
        if not valid_directions:
            # 这种情况下，选择一个方向，但移动后仍停留在原地（模拟撞墙不动）
            chosen_direction = random.choice(directions)
        else:
            # 从有效方向中随机选择
            chosen_direction = random.choice(valid_directions)
        
        path_directions.append(chosen_direction)
        
        # 根据选择的方向更新当前位置
        if chosen_direction == 'L' and current_col > 0:
            current_col -= 1
        elif chosen_direction == 'R' and current_col < map_size - 1:
            current_col += 1
        elif chosen_direction == 'U' and current_row > 0:
            current_row -= 1
        elif chosen_direction == 'D' and current_row < map_size - 1:
            current_row += 1
    
    # 将方向列表转换为逗号分隔的字符串
    path = ','.join(path_directions)
    return path

def check_valid_map(text_map):
    """
    检查地图是否有效：有起点和终点，且存在从起点到终点的路径
    
    Args:
        text_map: 表示地图的2D列表
        
    Returns:
        bool: 地图是否有效
    """
    # 找到起点和终点的位置
    start_pos = None
    goal_pos = None
    
    for i, row in enumerate(text_map):
        for j, cell in enumerate(row):
            if cell == 'S':
                start_pos = (i, j)
            elif cell == 'G':
                goal_pos = (i, j)
    
    # 如果缺少起点或终点，地图无效
    if start_pos is None or goal_pos is None:
        return False
    
    # 如果起点和终点重叠，地图无效
    if start_pos == goal_pos:
        return False
    
    # 如果起点或终点是冰洞，地图无效
    if text_map[start_pos[0]][start_pos[1]] == 'H' or text_map[goal_pos[0]][goal_pos[1]] == 'H':
        return False
    
    # 使用BFS检查是否有路径从起点到达终点
    size = len(text_map)
    visited = set([start_pos])
    queue = [start_pos]
    
    while queue:
        current = queue.pop(0)
        
        # 如果到达终点，返回有效
        if current == goal_pos:
            return True
        
        # 探索四个方向
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_r, new_c = current[0] + dr, current[1] + dc
            
            # 检查是否在地图范围内
            if 0 <= new_r < size and 0 <= new_c < size:
                new_pos = (new_r, new_c)
                
                # 如果是未访问过的安全格子或终点，加入队列
                if new_pos not in visited and (text_map[new_r][new_c] == 'F' or text_map[new_r][new_c] == 'G'):
                    visited.add(new_pos)
                    queue.append(new_pos)
    
    # 无法到达终点
    return False



def generate_frozen_lake_dataset(save_dir, size_range=(4, 8), samples_per_size_default=50, samples_per_size_dict= {}):
    """
    生成并保存具有不同尺寸的FrozenLake环境。
    每个环境实例的所有信息将被整合到一个JSON条目中。
    只保存包含有效A*路径的环境实例。
    
    Args:
        save_dir (str): 保存生成图像的目录
        size_range (tuple): 大小范围(最小值, 最大值)（包含）
        samples_per_size (int): 为每个大小生成的样本数
    """
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建图像的子目录
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    min_size, max_size = size_range
    
    # 创建数据集的JSONL文件
    jsonl_path = os.path.join(save_dir, "dataset.jsonl")
    
    # 检查可用工具
    available_tools = set(tool_manager.available_tools)
    print(f"Available tools: {available_tools}")
    required_tools = {"Point", "AStarWithPixelCoordinate", "Draw2DPath", "TurnCoordinateIntoTextMap"}
    missing_tools = required_tools - available_tools
    
    if missing_tools:
        raise ValueError(f"Required tools are missing: {missing_tools}")
    
    # 跟踪每个大小已成功生成的样本数
    successful_samples = {size: 0 for size in range(min_size, max_size + 1)}
    
    # 为每个大小生成环境，直到达到目标样本数
    # with open(jsonl_path, 'w') as jsonl_file:
        
    for size in range(min_size, max_size + 1):
        print(f"Generating environments of size {size}x{size}...")
        
        samples_per_size = samples_per_size_dict.get(size, samples_per_size_default) if samples_per_size_dict else samples_per_size_default
        # 继续生成样本直到达到目标数量
        sample_idx = 0
        attempts = 0
        max_attempts = samples_per_size * 3  # 允许的最大尝试次数，避免无限循环
        
        
        # 添加对重复地图的检查
        generated_maps = set()
        # 计算此尺寸理论上最大的唯一地图数量（保守估计）
        total_cells = size * size
        # 选择S和G的位置的组合: C(total_cells, 2)
        sg_combinations = total_cells * (total_cells - 1)  # S和G的有序对数量
        # 剩余格子可以是F或H: 2^(total_cells - 2)
        remaining_combinations = 2 ** (total_cells - 2)
        max_unique_maps = sg_combinations * remaining_combinations
        
        # 对于非常大的尺寸，这个数值会天文数字般大，设置一个合理的上限
        reasonable_max = min(max_unique_maps, 1000000)  # 上限为100万个唯一地图
        samples_per_size = min(samples_per_size, reasonable_max)
        pbar = tqdm(total=samples_per_size, desc=f"Size {size}x{size}")
        num_continued_skips = 0  # 跳过的计数器
        
        while successful_samples[size] < samples_per_size and attempts < max_attempts:
            attempts += 1
            
            # 生成随机地图
            p_value = 0.8  # 80%的概率是安全的，20%的概率是障碍物
            text_map = generate_random_map(size=size, p=p_value)  # p=0.8表示80%是安全格子，20%是冰洞
            # 将地图序列化为字符串，检查唯一性
            text_map_str = '\n'.join(text_map)
            if text_map_str in generated_maps:
                print(f"Map already generated, skipping...")
                num_continued_skips += 1
                if num_continued_skips > 100:
                    print(f"Too many continued skips ({num_continued_skips}), breaking out of the loop.")
                    break
                continue
            else:
                num_continued_skips = 0
                generated_maps.add(text_map_str)
                
                
            # 创建环境并渲染图像
            env = gym.make('FrozenLake-v1', desc=text_map, render_mode="rgb_array", is_slippery=False)
            env.reset()
            env_image = Image.fromarray(env.render())
            
            # 生成唯一ID
            img_id = f"lake_s{size}_{sample_idx}_{uuid.uuid4().hex[:8]}"
            
            # 临时保存原始环境图像
            temp_env_path = os.path.join(img_dir, f"temp_{img_id}.png")
            env_image.save(temp_env_path)
            
            # 提取坐标（仅用于初始数据和备份）
            coords = extract_coordinates(text_map, CELL_SIZE)
            if not coords['start'] or not coords['goal']:
                print(f"Warning: Missing start or goal in map, skipping...")
                env.close()
                if os.path.exists(temp_env_path):
                    os.remove(temp_env_path)
                continue
            
            # 创建一个包含所有信息的综合数据对象
            instance_data = {
                "id": img_id,
                "image_path": "",  # 稍后填充
                "size": size,
                "generation_map": text_map,
                "start_coords": list(coords['start']),
                "goal_coords": list(coords['goal']),
                "obstacle_coords": [list(hole) for hole in coords['holes']],
                # 将为这些字段填充后续工具调用的结果
                "point_tools": {},
                "text_map": {},
                "astar_path": {},
                "path_drawings": {}
            }
            
            # 用于存储Point工具实际检测的坐标
            detected_points = {}
            
            # 2. 创建Point工具数据 - 起点(Elf)
            try:
                elf_data = create_point_data_no_tool(
                    env_image, 
                    "Elf", 
                    coords['start'], 
                    img_id
                )
                
                # 保存Point工具检测的坐标
                if elf_data['output'].get('points') and len(elf_data['output']['points']) > 0:
                    detected_points['start'] = [
                        elf_data['output']['points'][0]['x'],
                        elf_data['output']['points'][0]['y']
                    ]
                else:
                    detected_points['start'] = list(coords['start'])  # 备用方案
                
                # 先不保存图像，等确认有效路径后再保存
                instance_data["point_tools"]["elf"] = {
                    "image_path": "",  # 稍后填充
                    "input": elf_data['input'],
                    "output": elf_data['output'],
                    "point_type": "Elf"
                }
            except Exception as e:
                print(f"Error creating Elf point data: {e}")
                detected_points['start'] = list(coords['start'])  # 备用方案
            
            # 3. 创建Point工具数据 - 终点(Gift)
            try:
                gift_data = create_point_data_no_tool(
                    env_image, 
                    "Gift", 
                    coords['goal'], 
                    img_id
                )
                
                # 保存Point工具检测的坐标
                if gift_data['output'].get('points') and len(gift_data['output']['points']) > 0:
                    detected_points['goal'] = [
                        gift_data['output']['points'][0]['x'],
                        gift_data['output']['points'][0]['y']
                    ]
                else:
                    detected_points['goal'] = list(coords['goal'])  # 备用方案
                
                # 先不保存图像，等确认有效路径后再保存
                instance_data["point_tools"]["gift"] = {
                    "image_path": "",  # 稍后填充
                    "input": gift_data['input'],
                    "output": gift_data['output'],
                    "point_type": "Gift"
                }
            except Exception as e:
                print(f"Error creating Gift point data: {e}")
                detected_points['goal'] = list(coords['goal'])  # 备用方案
            
            # 4. 创建Point工具数据 - 所有冰洞(Ice Holes)
            detected_points['holes'] = []
            if coords['holes']:  # 只在有冰洞时创建
                try:
                    holes_data = create_ice_holes_data_no_tool(
                        env_image, 
                        coords['holes'], 
                        img_id
                    )
                    
                    # 保存Point工具检测的障碍物坐标
                    if holes_data['output'].get('points'):
                        for point in holes_data['output']['points']:
                            detected_points['holes'].append([point['x'], point['y']])
                    
                    if not detected_points['holes']:  # 如果未检测到任何障碍物
                        detected_points['holes'] = [list(hole) for hole in coords['holes']]  # 备用方案
                    
                    # 先不保存图像，等确认有效路径后再保存
                    instance_data["point_tools"]["ice_holes"] = {
                        "image_path": "",  # 稍后填充
                        "input": holes_data['input'],
                        "output": holes_data['output'],
                        "point_type": "Ice Holes"
                    }
                except Exception as e:
                    print(f"Error creating Ice Holes point data: {e}")
                    detected_points['holes'] = [list(hole) for hole in coords['holes']]  # 备用方案
            
            # 5. 创建文本地图数据 - 使用Point工具检测的坐标
            try:
                text_map_data = create_text_map_data(
                    detected_points['start'],
                    detected_points['goal'],
                    detected_points['holes'],
                    img_id
                )
                
                # 将数据添加到综合对象中
                instance_data["text_map"] = {
                    "input": text_map_data['input'],
                    "output": text_map_data['output']
                }
            except Exception as e:
                print(f"Error creating text map data: {e}")
            
            # 6. 创建A*路径数据 - 使用Point工具检测的坐标
            valid_astar_path = False  # 标记A*路径是否有效
            try:
                astar_data = create_astar_data(
                    detected_points['start'],
                    detected_points['goal'],
                    detected_points['holes'],
                    img_id
                )
                
                # 检查A*算法是否找到了路径
                if astar_data['path']:
                    # 验证A*路径是否有效
                    is_valid = verify_path(text_map, astar_data['path'], verbose=False)
                    if is_valid:
                        valid_astar_path = True
                    else:
                        print(f"A* path is invalid for {img_id}, skipping...")
                else:
                    print(f"No A* path found for {img_id}, skipping...")
                
                # 将数据添加到综合对象中
                instance_data["astar_path"] = {
                    "input": astar_data['input'],
                    "output": astar_data['output'],
                    "path": astar_data['path'],
                    "is_valid": valid_astar_path
                }
            except Exception as e:
                print(f"Error creating A* path data: {e}")
            
            # 如果A*路径无效，则跳过此样本
            if not valid_astar_path:
                env.close()
                if os.path.exists(temp_env_path):
                    os.remove(temp_env_path)
                continue
            
            # 7. 创建绘制A*路径数据 - 使用Point工具检测的起点
            try:
                draw_path_data = create_draw_path_data(
                    env_image,
                    detected_points['start'],
                    astar_data['path'],
                    img_id,
                    is_random=False
                )
                
                if draw_path_data:
                    # 先不保存图像，等确认有效路径后再保存
                    instance_data["path_drawings"]["astar"] = {
                        "image_path": "",  # 稍后填充
                        "input": draw_path_data['input'],
                        "output": draw_path_data['output'],
                        "is_random": False
                    }
            except Exception as e:
                print(f"Error creating A* path drawing data: {e}")
            
            # 8. 创建随机路径数据和绘制 - 使用平衡的路径生成逻辑
            try:
                # 使用平衡路径生成逻辑
                random_draw_data = generate_balanced_random_paths(
                    text_map,
                    size,
                    img_id,
                    detected_points['start'],
                    env_image,
                )
                
                if random_draw_data:
                    # 为random_draw_data添加环境图像
                    # random_draw_data['image'] = env_image
                    
                    # 先不保存图像，等确认A*路径有效后再保存
                    instance_data["path_drawings"]["random"] = {
                        "image_path": "",  # 稍后填充
                        "input": random_draw_data['input'],
                        "output": random_draw_data['output'],
                        "path": random_draw_data['path'],
                        "is_safe": random_draw_data['is_safe'],
                        "is_valid": random_draw_data['is_valid']
                    }
            except Exception as e:
                print(f"Error creating balanced random path data: {e}")
                # 随机路径失败不影响整体数据保存
                random_draw_data = None

            # 只有当A*路径有效时，才保存此实例
            if valid_astar_path:
                # 正式保存所有图像
                env_path = os.path.join(img_dir, f"{img_id}.png")
                os.rename(temp_env_path, env_path)
                instance_data["image_path"] = env_path
                
                # 保存Elf图像
                elf_path = os.path.join(img_dir, f"{img_id}_point_elf.png")
                elf_data['image'].save(elf_path)
                instance_data["point_tools"]["elf"]["image_path"] = elf_path
                
                # 保存Gift图像
                gift_path = os.path.join(img_dir, f"{img_id}_point_gift.png")
                gift_data['image'].save(gift_path)
                instance_data["point_tools"]["gift"]["image_path"] = gift_path
                
                # 如果有冰洞，保存冰洞图像
                if "ice_holes" in instance_data["point_tools"]:
                    holes_path = os.path.join(img_dir, f"{img_id}_point_ice_holes.png")
                    holes_data['image'].save(holes_path)
                    instance_data["point_tools"]["ice_holes"]["image_path"] = holes_path
                
                # 保存A*路径图像
                path_path = os.path.join(img_dir, f"{img_id}_path_astar.png")
                draw_path_data['image'].save(path_path)
                instance_data["path_drawings"]["astar"]["image_path"] = path_path
                
                # 如果有随机路径数据，保存随机路径图像
                if random_draw_data:
                    random_path_path = os.path.join(img_dir, f"{img_id}_path_random.png")
                    random_draw_data['image'].save(random_path_path)
                    instance_data["path_drawings"]["random"]["image_path"] = random_path_path

                append_jsonl(instance_data, jsonl_path)
                successful_samples[size] += 1
                sample_idx += 1
                pbar.update(1)
            else:
                # 如果A*路径无效，清理临时文件
                if os.path.exists(temp_env_path):
                    os.remove(temp_env_path)
            
            # 释放环境资源
            env.close()
        
        pbar.close()
        
        if successful_samples[size] < samples_per_size:
            print(f"Warning: Could only generate {successful_samples[size]}/{samples_per_size} valid samples for size {size}x{size} after {max_attempts} attempts.")
    
    print(f"Dataset generation complete. Data stored in {jsonl_path}")
    # 输出每个尺寸成功生成的样本数
    for size, count in successful_samples.items():
        print(f"Size {size}x{size}: {count} valid samples")

# 示例用法
if __name__ == "__main__":
    # 检测是否存在工具
    import time
    print(f"Available tools: {tool_manager.available_tools}")
    start_time = time.time()
    # 使用较小的值进行测试
    samples_per_size_dict = {
        4: 3000, 
        5: 300, 
        6: 3000,  
        7: 300,  
        8: 3000,
        9: 300
    }
    generate_frozen_lake_dataset(
        save_dir="./frozen_lake_metadata_v3",
        size_range=(4, 9),  # 从4x4到8x8的环境
        samples_per_size_default=5,
        samples_per_size_dict=samples_per_size_dict
    )
    end_time = time.time()
    consumed_time = end_time - start_time
    print(f"Dataset generation took {consumed_time:.2f} seconds.")