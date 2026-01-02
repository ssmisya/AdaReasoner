# curate_minigrid_data.py
import os
import sys
import json
import uuid
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

# 导入自定义环境
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv

# 设置常量
CELL_SIZE = 32  # MiniGrid默认单元格大小


class CustomEmptyEnv(MiniGridEnv):
    """自定义空环境，可指定起点和终点位置"""
    
    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        goal_pos=(8, 8),
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height):
        # 创建空网格
        self.grid = Grid(width, height)

        # 生成外围墙壁
        self.grid.wall_rect(0, 0, width, height)

        # 放置目标
        if self.goal_pos is not None:
            self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

        # 设置agent位置和方向
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "reach the goal"


def append_jsonl(data, filepath):
    """追加数据到JSONL文件"""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def generate_random_positions(size, min_distance=2):
    """
    随机生成不重叠的起点和终点位置
    
    Args:
        size: 网格大小
        min_distance: 起点和终点之间的最小曼哈顿距离
        
    Returns:
        tuple: (agent_pos, agent_dir, goal_pos)
    """
    max_attempts = 100
    
    for _ in range(max_attempts):
        # 随机生成agent位置 (避开墙壁，范围是1到size-2)
        agent_x = random.randint(1, size - 2)
        agent_y = random.randint(1, size - 2)
        agent_pos = (agent_x, agent_y)
        
        # 随机生成agent方向
        agent_dir = random.randint(0, 3)
        
        # 随机生成goal位置
        goal_x = random.randint(1, size - 2)
        goal_y = random.randint(1, size - 2)
        goal_pos = (goal_x, goal_y)
        
        # 确保起点和终点不重叠，且有一定距离
        manhattan_distance = abs(agent_x - goal_x) + abs(agent_y - goal_y)
        
        if agent_pos != goal_pos and manhattan_distance >= min_distance:
            return agent_pos, agent_dir, goal_pos
    
    # 如果多次尝试失败，使用默认位置
    return (1, 1), 0, (size - 2, size - 2)


def grid_to_pixel_coords(grid_pos, cell_size=CELL_SIZE):
    """
    将网格坐标转换为像素坐标（格子中心）
    
    Args:
        grid_pos: (x, y) 网格坐标
        cell_size: 单元格像素大小
        
    Returns:
        tuple: (pixel_x, pixel_y) 像素坐标
    """
    x, y = grid_pos
    pixel_x = x * cell_size + cell_size / 2
    pixel_y = y * cell_size + cell_size / 2
    return (pixel_x, pixel_y)


def direction_to_action(env, direction_char):
    """将方向字符(L/R/U/D)转换为MiniGrid的动作序列"""
    current_direction = env.unwrapped.agent_dir
    
    direction_map = {'R': 0, 'D': 1, 'L': 2, 'U': 3}
    
    target_direction = direction_map.get(direction_char.upper())
    if target_direction is None:
        raise ValueError(f"Invalid direction: {direction_char}")
    
    turn_diff = (target_direction - current_direction) % 4
    
    actions = []
    
    if turn_diff == 1:
        actions.append(1)  # turn right
    elif turn_diff == 2:
        actions.append(1)
        actions.append(1)
    elif turn_diff == 3:
        actions.append(0)  # turn left
    
    actions.append(2)  # forward
    
    return actions


def verify_minigrid_path(env, path_string, verbose=False):
    """
    验证MiniGrid路径是否到达目标
    
    Args:
        env: MiniGrid环境
        path_string: 逗号分隔的方向字符串 (L,R,U,D)
        verbose: 是否打印详细信息
        
    Returns:
        dict: 包含验证结果的字典
    """
    env.reset()
    
    directions = [d.strip() for d in path_string.split(',')]
    
    total_reward = 0
    step_count = 0
    
    for i, direction in enumerate(directions):
        try:
            actions = direction_to_action(env, direction)
        except ValueError as e:
            if verbose:
                print(f"Invalid direction: {e}")
            return {
                "success": False,
                "total_reward": 0,
                "steps": step_count,
                "reason": "Invalid direction"
            }
        
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if terminated:
                return {
                    "success": reward > 0,
                    "total_reward": total_reward,
                    "steps": step_count,
                    "reached_goal": True
                }
            
            if truncated:
                return {
                    "success": False,
                    "total_reward": total_reward,
                    "steps": step_count,
                    "reason": "Truncated"
                }
    
    return {
        "success": False,
        "total_reward": total_reward,
        "steps": step_count,
        "reason": "Goal not reached"
    }


def generate_random_minigrid_path(env, min_length=3, max_length=15):
    """
    为MiniGrid生成随机路径
    
    Args:
        env: MiniGrid环境
        min_length: 最小路径长度
        max_length: 最大路径长度
        
    Returns:
        str: 路径字符串 "L,R,U,D,..."
    """
    directions = ['L', 'R', 'U', 'D']
    path_length = random.randint(min_length, max_length)
    
    path_directions = [random.choice(directions) for _ in range(path_length)]
    
    return ','.join(path_directions)


def generate_minigrid_dataset(save_dir, size_range=(7, 10), samples_per_size=50, 
                              num_random_paths=3, min_distance=2):
    """
    生成MiniGrid-Empty数据集（使用随机起点和终点）
    
    Args:
        save_dir: 保存目录
        size_range: 尺寸范围 (min, max)
        samples_per_size: 每个尺寸的样本数
        num_random_paths: 每个环境生成的随机路径数量
        min_distance: 起点终点最小距离
    """
    os.makedirs(save_dir, exist_ok=True)
    
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    jsonl_path = os.path.join(save_dir, "dataset.jsonl")
    
    # 如果文件已存在，删除它
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)
    
    min_size, max_size = size_range
    
    total_samples = 0
    successful_paths = 0
    
    for size in range(min_size, max_size + 1):
        print(f"\n{'='*60}")
        print(f"Generating MiniGrid-Empty-{size}x{size} environments...")
        print(f"{'='*60}")
        
        pbar = tqdm(total=samples_per_size, desc=f"Size {size}x{size}")
        
        for sample_idx in range(samples_per_size):
            # 随机生成起点、方向和终点位置
            agent_pos, agent_dir, goal_pos = generate_random_positions(size, min_distance)
            
            # 使用CustomEmptyEnv创建环境
            env = CustomEmptyEnv(
                size=size,
                agent_start_pos=agent_pos,
                agent_start_dir=agent_dir,
                goal_pos=goal_pos,
                render_mode='rgb_array'
            )
            
            # 重置环境
            obs, info = env.reset()
            
            # 渲染并保存环境图像
            env_image = Image.fromarray(env.render())
            
            # 生成唯一ID
            img_id = f"minigrid_custom_s{size}_{sample_idx}_{uuid.uuid4().hex[:8]}"
            
            # 保存环境图像
            env_path = os.path.join(img_dir, f"{img_id}.png")
            env_image.save(env_path)
            
            # 转换坐标为像素坐标
            agent_pixel = grid_to_pixel_coords(agent_pos)
            goal_pixel = grid_to_pixel_coords(goal_pos)
            
            # 生成多条随机路径
            random_paths = []
            for path_idx in range(num_random_paths):
                random_path = generate_random_minigrid_path(
                    env, 
                    min_length=3, 
                    max_length=size * 2
                )
                
                # 验证路径
                path_result = verify_minigrid_path(env, random_path, verbose=False)
                
                random_paths.append({
                    "path": random_path,
                    "is_valid": path_result["success"],
                    "total_reward": path_result["total_reward"],
                    "steps": path_result["steps"],
                    "length": len(random_path.split(','))
                })
                
                if path_result["success"]:
                    successful_paths += 1
            
            # 创建数据条目
            instance_data = {
                "id": img_id,
                "image_path": env_path,
                "env_type": "CustomEmptyEnv",
                "size": size,
                "grid_state": {
                    "width": size,
                    "height": size,
                    "agent_pos": list(agent_pos),
                    "agent_dir": agent_dir,
                    "goal_pos": list(goal_pos)
                },
                "pixel_coords": {
                    "agent": list(agent_pixel),
                    "goal": list(goal_pixel)
                },
                "manhattan_distance": abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1]),
                "random_paths": random_paths
            }
            
            # 保存到JSONL
            append_jsonl(instance_data, jsonl_path)
            
            env.close()
            total_samples += 1
            pbar.update(1)
        
        pbar.close()
    
    print(f"\n{'='*60}")
    print(f"Dataset Generation Summary")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"Total paths generated: {total_samples * num_random_paths}")
    print(f"Successful paths: {successful_paths}")
    print(f"Success rate: {successful_paths / (total_samples * num_random_paths) * 100:.2f}%")
    print(f"\nDataset stored in: {jsonl_path}")
    print(f"Images stored in: {img_dir}")


if __name__ == "__main__":
    generate_minigrid_dataset(
        save_dir="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/others/grid_bench/minigrid_empty_dataset",
        size_range=(7, 10),
        samples_per_size=50,
        num_random_paths=3,  # 每个环境生成3条随机路径
        min_distance=2  # 起点终点最小距离
    )