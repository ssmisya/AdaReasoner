# tf_eval/tasks/minigrid_nav/task.py
from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os
import sys
import json
import re
from PIL import Image

# 导入 MiniGrid 相关模块
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

fs = """
<im_start>user:\nYou are a maze navigator. Your goal is to guide an agent from the start position (Red triangle) to the goal position (green square) on a grid map. The agent can move one square at a time in the directions: Left (L), Right (R), Up (U), or Down (D).

Important rules:
- Moving into a wall has no effect (the agent stays in place)
- The agent must reach the exact goal position to succeed
- Use the shortest path when possible

Please generate an action sequence to navigate from the start to the goal. Your final answer should be formatted as \\boxed{L,R,U,D,...} with directions separated by commas.

Example: \\boxed{R,R,D,D,L}<im_end>
<im_start>\nassistant:\n<think>I think I should first use the Point tool to locate the start point (Red triangle).</think> 
<tool_call>
{
  "name": "Point",
  "parameters": {
    "image": "img_1",
    "description": "Red triangle"
  }
}
</tool_call>
<im_end>
<im_start> user: \n{'tool_response_from': 'Point', 'status': 'success', 'points': [{'x': 48.28, 'y': 48.84}], 'image_dimensions_pixels': {'width': 256, 'height': 256}, 'error_code': 0}
<im_end>
<im_start>\nassistant:\n<think>I have located the start point. Next, I should use the Point tool to locate the goal point (green square).</think>
<tool_call>
{
    "name": "Point",
    "parameters": {
        "image": "img_1",
        "description": "green square"
    }
}
</tool_call>
<im_end>
<im_start> user: \n{'tool_response_from': 'Point', 'status': 'success', 'points': [{'x': 144.12, 'y': 144.76}], 'image_dimensions_pixels': {'width': 256, 'height': 256}, 'error_code': 0}
<im_end>
<im_start>\nassistant:\n<think>I have located both the start and goal points. Now, I will use the AStarWithPixelCoordinate tool to compute the optimal path from the start to the goal.</think>
<tool_call>
{
    "name": "AStarWithPixelCoordinate",
    "parameters": {
        "image": "img_1",
        "start": [48.28,48.84],
        "goal": [144.12,144.76],
        "obstacles":[],
        "cell_size":32
    }
}
</tool_call>
<im_end>
<im_start> user: {'status': 'success',
 'path': 'D,D,D,R,R,R',
 'error_code': 0,
 'tool_response_from': 'AStarWithPixelCoordinate',
 'execution_time': 0.0001308917999267578,
 'tool_reward': 4.0}<im_end>
<im_start>\nassistant:\n<think>I have obtained the optimal path. Now, I will format my final answer accordingly.</think><final_answer>\boxed{D,D,D,R,R,R}</final_answer>\n<im_end>
"""

# 任务提示
NAVIGATION_INSTRUCTION = """You are a maze navigator. Your goal is to guide an agent from the start position (Red triangle) to the goal position (green square) on a grid map. The agent can move one square at a time in the directions: Left (L), Right (R), Up (U), or Down (D).

Important rules:
- Moving into a wall has no effect (the agent stays in place)
- The agent must reach the exact goal position to succeed
- Use the shortest path when possible

Please generate an action sequence to navigate from the start to the goal. Your final answer should be formatted as \\boxed{L,R,U,D,...} with directions separated by commas.

Example: \\boxed{R,R,D,D,L}
"""


class CustomEmptyEnv(MiniGridEnv):
    """自定义空环境，可从保存的状态重建"""
    
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


def direction_to_action(env, direction_char):
    """将方向字符(L/R/U/D)转换为MiniGrid的动作序列"""
    current_direction = env.unwrapped.agent_dir
    
    # 方向映射: 右=0, 下=1, 左=2, 上=3
    direction_map = {'R': 0, 'D': 1, 'L': 2, 'U': 3}
    
    target_direction = direction_map.get(direction_char.upper())
    if target_direction is None:
        raise ValueError(f"Invalid direction: {direction_char}")
    
    turn_diff = (target_direction - current_direction) % 4
    
    actions = []
    
    if turn_diff == 1:  # 右转90度
        actions.append(1)  # turn right
    elif turn_diff == 2:  # 转180度
        actions.append(1)
        actions.append(1)
    elif turn_diff == 3:  # 左转90度
        actions.append(0)  # turn left
    
    # 添加前进动作
    actions.append(2)  # forward
    
    return actions


def load_data_function():
    """加载MiniGrid导航数据集"""
    dataset_path = task_config.get("dataset_path")
    num_samples = task_config.get("num_sample")
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return []
    
    meta_data = []
    
    # 读取JSONL文件
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if num_samples and idx >= num_samples:
                break
            
            data = json.loads(line.strip())
            
            # 加载图像
            image_path = data["image_path"]
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            image = Image.open(image_path)
            
            # 提取网格状态
            grid_state = data["grid_state"]
            
            # 构建问题文本
            question_text = NAVIGATION_INSTRUCTION
            
            # 创建元数据条目
            meta_item = {
                "idx": data["id"],
                "image": image,
                "text": question_text,
                "grid_state": grid_state,
                "size": data["size"],
                "manhattan_distance": data["manhattan_distance"],
                "task_type": "navigation",
                "tool_few_shot": fs
            }
            
            meta_data.append(meta_item)
    
    # 统计信息
    logger.info(f"Total data loaded: {len(meta_data)}")
    
    # 按size统计
    size_counts = {}
    for item in meta_data:
        size = item["size"]
        size_counts[size] = size_counts.get(size, 0) + 1
    
    for size, count in sorted(size_counts.items()):
        logger.info(f"Size {size}x{size}: {count} samples")
    
    return meta_data


def evaluate_function(results, meta_data):
    """评估导航任务结果"""
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    
    # 统计结果
    overall_correct = 0
    overall_total = 0
    
    # 按size分类统计
    size_results = {}
    
    # 按曼哈顿距离分类统计
    distance_results = {}
    
    compare_logs = []
    
    for idx, meta in meta_dict.items():
        size = meta["size"]
        distance = meta["manhattan_distance"]
        
        # 初始化size统计
        if size not in size_results:
            size_results[size] = {"correct": 0, "total": 0}
        
        # 初始化distance统计
        if distance not in distance_results:
            distance_results[distance] = {"correct": 0, "total": 0}
        
        # 获取预测结果
        if idx in results_dict:
            prediction = results_dict[idx]["results"].get("final_answer", "")
        else:
            prediction = ""
        
        # 评估路径
        score, message = evaluate_navigation_path(prediction, meta)
        
        # 更新统计
        overall_correct += score
        overall_total += 1
        
        size_results[size]["correct"] += score
        size_results[size]["total"] += 1
        
        distance_results[distance]["correct"] += score
        distance_results[distance]["total"] += 1
        
        # 记录日志
        compare_logs.append({
            "idx": idx,
            "size": size,
            "distance": distance,
            "pred": prediction,
            "score": score,
            "message": message,
            "grid_state": meta["grid_state"]
        })
    
    # 计算准确率
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    
    # 计算各size的准确率
    for size in size_results:
        if size_results[size]["total"] > 0:
            size_results[size]["accuracy"] = size_results[size]["correct"] / size_results[size]["total"]
        else:
            size_results[size]["accuracy"] = 0.0
    
    # 计算各distance的准确率
    for distance in distance_results:
        if distance_results[distance]["total"] > 0:
            distance_results[distance]["accuracy"] = distance_results[distance]["correct"] / distance_results[distance]["total"]
        else:
            distance_results[distance]["accuracy"] = 0.0
    
    result = {
        "overall_accuracy": overall_accuracy,
        "size_results": size_results,
        "distance_results": distance_results,
        "compare_logs": compare_logs,
        "results": results
    }
    
    # 打印结果摘要
    logger.info(f"Overall accuracy: {overall_accuracy:.4f}")
    logger.info("\nAccuracy by grid size:")
    for size in sorted(size_results.keys()):
        acc = size_results[size]["accuracy"]
        correct = size_results[size]["correct"]
        total = size_results[size]["total"]
        logger.info(f"  Size {size}x{size}: {acc:.4f} ({correct}/{total})")
    
    logger.info("\nAccuracy by Manhattan distance:")
    for distance in sorted(distance_results.keys()):
        acc = distance_results[distance]["accuracy"]
        correct = distance_results[distance]["correct"]
        total = distance_results[distance]["total"]
        logger.info(f"  Distance {distance}: {acc:.4f} ({correct}/{total})")
    
    return result


def evaluate_navigation_path(prediction, meta):
    """
    评估导航路径是否正确
    
    Args:
        prediction: 模型预测的路径字符串
        meta: 元数据，包含grid_state等信息
        
    Returns:
        tuple: (score, message) score为1.0表示成功，0.0表示失败
    """
    if not prediction or prediction.strip() == "":
        return 0.0, "Empty prediction"
    
    # 提取路径序列
    path_string = extract_path_from_prediction(prediction)
    
    if not path_string:
        return 0.0, "Failed to extract path from prediction"
    
    # 从grid_state重建环境
    grid_state = meta["grid_state"]
    
    try:
        env = CustomEmptyEnv(
            size=grid_state["width"],
            agent_start_pos=tuple(grid_state["agent_pos"]),
            agent_start_dir=grid_state["agent_dir"],
            goal_pos=tuple(grid_state["goal_pos"]),
            render_mode=None
        )
        env.reset()
        
        # 验证路径
        result = verify_path(env, path_string)
        
        env.close()
        
        if result["success"]:
            return 1.0, f"Success! Reward: {result['total_reward']:.2f}, Steps: {result['steps']}"
        else:
            return 0.0, f"Failed: {result.get('reason', 'Unknown')}"
    
    except Exception as e:
        return 0.0, f"Error during evaluation: {str(e)}"


def extract_path_from_prediction(prediction):
    """
    从模型预测中提取路径序列
    
    支持的格式:
    - \\boxed{L,R,U,D}
    - [L,R,U,D]
    - L,R,U,D
    - LRUD (连续字符)
    """
    # 尝试匹配 \boxed{...} 格式
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', prediction, re.IGNORECASE)
    if boxed_match:
        path_content = boxed_match.group(1).strip()
        return normalize_path(path_content)
    
    # 尝试匹配 [...] 格式
    bracket_match = re.search(r'\[([^\]]+)\]', prediction)
    if bracket_match:
        path_content = bracket_match.group(1).strip()
        return normalize_path(path_content)
    
    # 尝试匹配逗号分隔的路径
    comma_match = re.search(r'([LRUD](?:\s*,\s*[LRUD])+)', prediction, re.IGNORECASE)
    if comma_match:
        path_content = comma_match.group(1).strip()
        return normalize_path(path_content)
    
    # 尝试匹配连续的方向字符
    continuous_match = re.search(r'([LRUD]{2,})', prediction, re.IGNORECASE)
    if continuous_match:
        path_content = continuous_match.group(1).strip()
        # 将连续字符转换为逗号分隔
        return ','.join(list(path_content.upper()))
    
    # 尝试直接使用整个预测（如果它看起来像一个路径）
    cleaned = re.sub(r'[^LRUD,\s]', '', prediction, flags=re.IGNORECASE)
    if cleaned.strip():
        return normalize_path(cleaned)
    
    return None


def normalize_path(path_string):
    """
    标准化路径字符串为逗号分隔的大写格式
    
    Args:
        path_string: 原始路径字符串
        
    Returns:
        str: 标准化后的路径，如 "L,R,U,D"
    """
    # 移除多余的空格和引号
    path_string = path_string.strip().strip('"').strip("'")
    
    # 如果包含逗号，按逗号分割
    if ',' in path_string:
        directions = [d.strip().upper() for d in path_string.split(',')]
    else:
        # 否则按字符分割
        directions = [d.upper() for d in path_string if d.upper() in ['L', 'R', 'U', 'D']]
    
    # 过滤掉无效方向
    valid_directions = [d for d in directions if d in ['L', 'R', 'U', 'D']]
    
    if not valid_directions:
        return None
    
    return ','.join(valid_directions)


def verify_path(env, path_string):
    """
    在环境中验证路径是否到达目标
    
    Args:
        env: MiniGrid环境
        path_string: 逗号分隔的方向字符串
        
    Returns:
        dict: 包含验证结果的字典
    """
    env.reset()
    
    directions = [d.strip() for d in path_string.split(',')]
    
    total_reward = 0
    step_count = 0
    
    for direction in directions:
        try:
            actions = direction_to_action(env, direction)
        except ValueError as e:
            return {
                "success": False,
                "total_reward": 0,
                "steps": step_count,
                "reason": f"Invalid direction: {direction}"
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
                    "reason": "Episode truncated (timeout)"
                }
    
    return {
        "success": False,
        "total_reward": total_reward,
        "steps": step_count,
        "reason": "Goal not reached"
    }