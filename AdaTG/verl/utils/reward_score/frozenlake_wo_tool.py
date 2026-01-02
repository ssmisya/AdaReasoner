from openai import OpenAI
import requests
import random
import re
import os
import ast
import gymnasium as gym
import numpy as np



def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None, tool_rewards=None, **kwargs):
    # 每个的形状都是batch_size*n，包括模型的回答solution_strs
    scores = [0] * len(data_sources)
    reward_details = []  # 新增：存储详细的奖励信息
    
    for i in range(len(data_sources)):
        solution_str = solution_strs[i]
        ground_truth = ground_truths[i]
        data_source = data_sources[i]
        
        # 计算format奖励
        format_reward = calculate_format_reward(solution_str)
        accuracy_reward = 0
        
        if format_reward:
            # 对于不同的data_source进行不同的计算
            if data_source == "path_nav":
                accuracy_reward = calculate_path_nav_accuracy(solution_str, ground_truth)
            elif data_source == "path_ver":
                accuracy_reward = calculate_path_ver_accuracy(solution_str, ground_truth)
            
            total_score = accuracy_reward
            scores[i] = total_score
        
        else:
            total_score = 0.0
            scores[i] = 0.0
        
        # 存储详细的奖励信息
        reward_details.append({
            "score": total_score,
            "format_reward": format_reward,
            "accuracy_reward": accuracy_reward,
            "tool_reward": 0.0, # 为了统一返回值
            "data_source": data_source
        })
    
    return reward_details  # 修改：返回详细信息而不是简单的分数列表


def calculate_format_reward(solution_str):
    pattern = re.compile(r"<think>.*?</think>\s*<response>.*?</response>", re.DOTALL)
    format_match = re.fullmatch(pattern, solution_str)
    return 1.0 if format_match else 0.0


def calculate_path_ver_accuracy(solution_str, ground_truth):
    """
    计算path_ver任务的准确性奖励
    
    参数:
        solution_str (str): 模型的回复字符串
        ground_truth (bool): 真实答案，True表示安全，False表示不安全
        
    返回:
        int: 1表示回答正确，0表示回答错误
    """
    try:
        # 先提取<response>标签中的内容
        response_pattern = r'<response>(.*?)</response>'
        response_match = re.search(response_pattern, solution_str, re.DOTALL)
        
        if response_match:
            response_content = response_match.group(1).strip()
        else:
            # 如果没有找到response标签，返回0
            return 0
        
        # 方法1：尝试从response内容中提取\\boxed{}
        model_answer = extract_boxed_answer(response_content)
        
        # 方法2：如果没有找到boxed，则直接查找yes/no
        if model_answer is None:
            model_answer = extract_yes_no_answer(response_content)
        
        # 如果仍然没有找到有效答案，返回0
        if model_answer is None:
            return 0
        
        # 比较模型答案和真实答案
        if ground_truth == "yes" and model_answer == "yes":
            return 1
        elif ground_truth == "no" and model_answer == "no":
            return 1
        else:
            return 0
            
    except Exception as e:
        print(f"Error in path_ver accuracy calculation: {e}")
        return 0


def calculate_path_nav_accuracy(solution_str, ground_truth):
    """
    计算path_nav任务的准确性奖励
    
    参数:
        solution_str (str): 模型的回复字符串
        ground_truth (list): gym地图格式，如[["F", "H", "S", "H", "F"], ...]
        
    返回:
        int: 1表示路径有效且到达目标，0表示路径无效或未到达目标
    """
    try:
        # 先提取<response>标签中的内容
        response_pattern = r'<response>(.*?)</response>'
        response_match = re.search(response_pattern, solution_str, re.DOTALL)
        
        if response_match:
            response_content = response_match.group(1).strip()
        else:
            # 如果没有找到response标签，返回0
            return 0
        
        # 提取路径序列
        action_sequence = extract_path_sequence(response_content)
        
        if not action_sequence:
            return 0
        
        # 在gym环境中验证路径
        return validate_path_in_gym(action_sequence, ground_truth)
            
    except Exception as e:
        print(f"Error in path_nav accuracy calculation: {e}")
        return 0


def extract_boxed_answer(text):
    """
    从文本中提取\\boxed{}中的答案
    
    参数:
        text (str): 要提取的文本
        
    返回:
        str or None: "yes"、"no" 或 None
    """
    # 匹配boxed{...}模式
    boxed_pattern = r'boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, text, re.IGNORECASE)
    
    if boxed_match:
        boxed_content = boxed_match.group(1).strip().lower()
        if "yes" in boxed_content:
            return "yes"
        elif "no" in boxed_content:
            return "no"
    
    return None


def extract_yes_no_answer(text):
    """
    从文本中直接查找yes/no
    
    参数:
        text (str): 要查找的文本
        
    返回:
        str or None: "yes"、"no" 或 None
    """
    text_lower = text.lower()
    
    # 统计yes和no的出现次数
    yes_count = text_lower.count("yes")
    no_count = text_lower.count("no")
    
    # 如果只有yes，没有no
    if yes_count > 0 and no_count == 0:
        return "yes"
    # 如果只有no，没有yes
    elif no_count > 0 and yes_count == 0:
        return "no"
    # 如果都有或者都没有，返回None
    else:
        return None


def extract_path_sequence(text):
    """
    从文本中提取路径序列
    
    参数:
        text (str): 要提取的文本
        
    返回:
        list: 动作序列，如['L', 'R', 'U', 'D']，如果提取失败返回空列表
    """
    # 方法1：尝试从\\boxed{}中提取
    boxed_pattern = r'boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, text, re.IGNORECASE)
    
    if boxed_match:
        boxed_content = boxed_match.group(1).strip()
        # 从boxed内容中提取路径
        actions = extract_actions_from_string(boxed_content)
        if actions:
            return actions
    
    # 方法2：尝试从整个文本中提取路径模式
    # 匹配类似 "L,R,U,D" 或 "L, R, U, D" 的模式
    path_pattern = r'([UDLR](?:\s*,\s*[UDLR])*)'
    path_matches = re.findall(path_pattern, text, re.IGNORECASE)
    
    for match in path_matches:
        actions = extract_actions_from_string(match)
        if actions:
            return actions
    
    # 方法3：提取所有UDLR字符作为连续序列
    actions = []
    for char in text.upper():
        if char in ['U', 'D', 'L', 'R']:
            actions.append(char)
    
    # 如果找到的动作太少，可能不是有效路径
    if len(actions) >= 1:
        return actions
    
    return []


def extract_actions_from_string(action_str):
    """
    从字符串中提取动作列表
    
    参数:
        action_str (str): 包含动作的字符串
        
    返回:
        list: 动作列表
    """
    actions = []
    
    # 按逗号分割
    if ',' in action_str:
        for action in action_str.split(','):
            action = action.strip().upper()
            if action in ['L', 'R', 'U', 'D']:
                actions.append(action)
    else:
        # 没有逗号，提取所有UDLR字符
        for char in action_str.upper():
            if char in ['U', 'D', 'L', 'R']:
                actions.append(char)
    
    return actions


def validate_path_in_gym(actions, gym_map):
    """
    在gym环境中验证路径的有效性
    
    参数:
        actions (list): 动作序列，如['L', 'R', 'U', 'D']
        gym_map (list): gym地图格式
        
    返回:
        int: 1表示路径有效且到达目标，0表示路径无效或未到达目标
    """
    try:
        # 创建FrozenLake环境
        gym_map = ast.literal_eval(gym_map) # 将字符串转换为列表
        env = gym.make('FrozenLake-v1', desc=gym_map, is_slippery=False)
        env.reset()
        
        # 动作映射
        action_mapping = {'L': 0, 'D': 1, 'R': 2, 'U': 3}
        
        # 执行动作序列
        for action in actions:
            if action not in action_mapping:
                env.close()
                return 0  # 无效动作
            
            observation, reward, terminated, truncated, info = env.step(action_mapping[action])
            
            if terminated:
                env.close()
                # 如果到达目标 (reward > 0)
                if reward > 0:
                    return 1
                # 如果掉入洞中 (reward = 0)
                else:
                    return 0
        
        env.close()
        # 如果完成所有动作但没有terminated，说明没有到达目标
        return 0
    
    except Exception as e:
        print(f"Error validating path in gym: {e}")
        return 0
