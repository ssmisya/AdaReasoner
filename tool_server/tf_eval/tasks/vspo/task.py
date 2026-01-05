# task.py
from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import json
import os
import re
import sys
import random
import numpy as np
from PIL import Image
import io
import base64
from collections import defaultdict
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

# 路径验证任务说明
PATH_VERIFY_TASK_INSTRUCTION = """
You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success. 

Now please determine if the action sequence is safe for the given maze. Your final answer should be formatted as \\boxed{Yes} or \\boxed{No}.

The action sequence is:

<ACTION-SEQ>
"""

# 路径导航任务说明
PATH_NAVIGATION_INSTRUCTION = """
You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success. Your final answer should be formatted as \\boxed{L,R,U,D}.

Please generate action plan for the input maze image.
"""

def load_data_function():
    """从Hugging Face Hub加载数据集"""
    dataset_type = task_config.get("dataset_type", "unique")
    
    if dataset_type == "huggingface":
        return load_data_from_huggingface()
    else:
        # 保留原有的加载逻辑作为备份
        return load_data_from_local()

def load_data_from_huggingface():
    """从Hugging Face Hub加载数据集"""
    dataset_repo = task_config.get("dataset_repo")
    tasks = task_config.get("tasks", ["navigation-test", "verify-test"])
    use_auth_token = task_config.get("use_auth_token", False)
    hf_token = task_config.get("hf_token") or os.environ.get("HF_TOKEN")
    
    if not dataset_repo:
        raise ValueError("dataset_repo must be specified in config when using huggingface dataset_type")
    
    logger.info(f"Loading dataset from Hugging Face Hub: {dataset_repo}")
    
    # 构建需要加载的split列表
    splits_to_load = []
    for task in tasks:
        task_prefix, task_suffix = task.split("-")
        split_name = f"{task_prefix}_{task_suffix}"
        splits_to_load.append(split_name)
    
    meta_data = []
    
    # 加载每个split
    for split_name in splits_to_load:
        try:
            logger.info(f"Loading split: {split_name}")
            
            # 从Hugging Face加载数据集
            dataset = load_dataset(
                dataset_repo,
                split=split_name,
                token=hf_token if use_auth_token else None
            )
            
            logger.info(f"Loaded {len(dataset)} samples from split {split_name}")
            
            # 转换为meta_data格式
            for item in dataset:
                # 解析gym_map（从JSON字符串转回列表）
                gym_map = json.loads(item['gym_map']) if isinstance(item['gym_map'], str) else item['gym_map']
                
                # 构建数据项
                data_item = {
                    'idx': item['idx'],
                    'original_id': item['original_id'],
                    'image': item['image'],  # PIL Image对象
                    'text': item['text'],
                    'answer': item['answer'],
                    'task_type': item['task_type'],
                    'split': item['split'],
                    'size': item['size'],
                    'gym_map': gym_map
                }
                
                # 根据任务类型添加特定字段
                if item['task_type'] == 'verify':
                    if item['path_length'] is not None:
                        data_item['path_length'] = item['path_length']
                    if item['path'] is not None:
                        data_item['path'] = item['path']
                elif item['task_type'] == 'navigation':
                    if item['start_coords'] is not None:
                        data_item['start_coords'] = json.loads(item['start_coords']) if isinstance(item['start_coords'], str) else item['start_coords']
                    if item['goal_coords'] is not None:
                        data_item['goal_coords'] = json.loads(item['goal_coords']) if isinstance(item['goal_coords'], str) else item['goal_coords']
                    if item['obstacle_coords'] is not None:
                        data_item['obstacle_coords'] = json.loads(item['obstacle_coords']) if isinstance(item['obstacle_coords'], str) else item['obstacle_coords']
                    if item['astar_path'] is not None:
                        data_item['astar_path'] = item['astar_path']
                
                meta_data.append(data_item)
                
        except Exception as e:
            logger.error(f"Error loading split {split_name}: {e}")
            continue
    
    # 数据集统计信息
    logger.info(f"Total data loaded from Hugging Face: {len(meta_data)}")
    
    # 统计各任务的数据量
    task_counts = defaultdict(int)
    for item in meta_data:
        task_type = item.get("task_type", "unknown")
        task_counts[task_type] += 1
    
    for task_type, count in task_counts.items():
        logger.info(f"Task type: {task_type}, count: {count}")
    
    return meta_data

def load_data_from_local():
    """从本地加载数据集（原有逻辑，作为备份）"""
    dataset_path = task_config.get("dataset_path")
    tasks = task_config.get("tasks", ["navigation-test", "verify-test"])
    
    # 从配置文件获取数据路径
    data_dir = task_config.get("data_dir", "./metadata_split")
    data_dir = os.path.join(dataset_path, data_dir) if dataset_path else data_dir
    img_dir = task_config.get("image_dir", "./images")
    
    verify_dir = os.path.join(data_dir, "path_verify")
    navigation_dir = os.path.join(data_dir, "path_navigation")
    
    meta_data = []
    
    # 加载不同任务的数据
    for task in tasks:
        task_prefix, task_suffix = task.split("-")
        if task_prefix == "verify":  # 路径验证任务
            # 加载验证数据
            data_path = os.path.join(verify_dir, f"{task_suffix}.jsonl")
            if os.path.exists(data_path):
                meta_data.extend(load_path_verify_data(data_path, task_prefix, task_suffix, img_dir))
        
        elif task_prefix == "navigation":  # 路径导航任务
            data_path = os.path.join(navigation_dir, f"{task_suffix}.jsonl")
            if os.path.exists(data_path):
                meta_data.extend(load_path_navigation_data(data_path, task_prefix, task_suffix, img_dir))
    
    # 数据集统计信息
    logger.info(f"Total data loaded from local: {len(meta_data)}")
    
    # 统计各任务的数据量
    task_counts = defaultdict(int)
    for item in meta_data:
        task_type = item.get("task_type", "unknown")
        task_counts[task_type] += 1
    
    for task_type, count in task_counts.items():
        logger.info(f"Task type: {task_type}, count: {count}")
    
    return meta_data

def read_jsonl(file_path):
    """读取JSONL文件"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        logger.info(f"Successfully loaded {len(data)} records from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return []

def convert_to_gym_map(item):
    """
    将数据项转换为gym环境可用的地图格式
    
    Args:
        item: 数据项，包含地图信息
        
    Returns:
        list: gym环境可用的地图格式
    """
    # 首先从text_map中提取文本地图
    if "text_map" in item and "output" in item["text_map"] and "text_map" in item["text_map"]["output"]:
        text_map_str = item["text_map"]["output"]["text_map"]
        # 解析文本地图
        rows = []
        for line in text_map_str.split('\n'):
            if '|' in line and ('Row' in line or 'Col' in line):
                continue  # 跳过表头行
            
            row_cells = []
            cells = line.split('|')
            for cell in cells[1:]:  # 跳过第一个空元素
                if not cell.strip():
                    continue
                
                cell_value = cell.strip()
                if cell_value == '_':  # 空格表示安全区域
                    row_cells.append('F')
                elif cell_value == '#':  # '#' 表示冰洞
                    row_cells.append('H')
                elif cell_value == '@':  # '@' 表示起点
                    row_cells.append('S')
                elif cell_value == '*':  # '*' 表示终点
                    row_cells.append('G')
            
            if row_cells:  # 如果行不为空
                rows.append(row_cells)
        
        # 检查地图是否有效
        if rows and all(len(row) == len(rows[0]) for row in rows):
            return rows
    
    # 如果无法从text_map中提取，则从坐标信息中构建
    size = item["size"]
    cell_size = 64  # 假设每个单元格是64像素
    
    # 创建一个全是安全区域的地图
    map_data = [['F' for _ in range(size)] for _ in range(size)]
    
    # 设置起点
    start_x = int(item["start_coords"][0] / cell_size)
    start_y = int(item["start_coords"][1] / cell_size)
    map_data[start_y][start_x] = 'S'
    
    # 设置终点
    goal_x = int(item["goal_coords"][0] / cell_size)
    goal_y = int(item["goal_coords"][1] / cell_size)
    map_data[goal_y][goal_x] = 'G'
    
    # 设置障碍物
    for obs in item["obstacle_coords"]:
        obs_x = int(obs[0] / cell_size)
        obs_y = int(obs[1] / cell_size)
        # 确保坐标在有效范围内
        if 0 <= obs_y < size and 0 <= obs_x < size:
            map_data[obs_y][obs_x] = 'H'
    
    return map_data

def load_path_verify_data(data_path, task_type, split, img_dir):
    """加载路径验证任务数据"""
    meta_data = []
    
    # 读取JSONL文件
    data = read_jsonl(data_path)
    
    for item in data:
        # 检查item是否包含必要的字段
        if not all(key in item for key in ["id", "image_path", "path_drawings",]):
            raise ValueError(f"Missing required keys in item: {item.keys()}")
        
        # 获取随机路径数据
        random_path = item["path_drawings"]["random"]
        if "path" not in random_path or "is_safe" not in random_path:
            raise ValueError(f"Missing 'path' or 'is_safe' in random path: {random_path}")
        
        path_string = random_path["path"]
        is_safe = random_path["is_safe"]
        
        # 构建提示
        text_prompt = PATH_VERIFY_TASK_INSTRUCTION.replace("<ACTION-SEQ>", path_string)
        
        # 转换gym地图
        gym_map = convert_to_gym_map(item)
        
        image_path = os.path.join(img_dir, item["image_path"]) if img_dir else item["image_path"]
        meta_data.append({
            "idx": f"{item['id']}_verify_{split}",
            "original_id": item["id"],
            "image_path": image_path,
            "text": text_prompt,
            "answer": "Yes" if is_safe else "No",
            "task_type": task_type,
            "split": split,
            "size": item["size"],
            "path_length": len(path_string.split(",")),
            "path": path_string,
            "gym_map": gym_map
        })
    
    logger.info(f"Loaded {len(meta_data)} records for {task_type} from {data_path}")
    return meta_data

def load_path_navigation_data(data_path, task_type, split, img_dir):
    """加载路径导航任务数据"""
    meta_data = []
    
    # 读取JSONL文件
    data = read_jsonl(data_path)
    
    for item in data:
        # 检查item是否包含必要的字段
        if not all(key in item for key in ["id", "image_path", "start_coords", "goal_coords", "obstacle_coords"]):
            raise ValueError(f"Missing required keys in item: {item.keys()}")
        
        # 构建提示
        text_prompt = PATH_NAVIGATION_INSTRUCTION
        
        # 获取A*路径（如果有）
        astar_path = item.get("astar_path", {}).get("path", "") if "astar_path" in item else ""
        
        # 转换gym地图
        gym_map = convert_to_gym_map(item)
        
        image_path = os.path.join(img_dir, item["image_path"]) if img_dir else item["image_path"]
        meta_data.append({
            "idx": f"{item['id']}_navigation_{split}",
            "original_id": item["id"],
            "image_path": image_path,
            "text": text_prompt,
            "answer": "DYNAMIC_EVAL",  # 这个任务需要动态评估
            "task_type": task_type,
            "split": split,
            "size": item["size"],
            "start_coords": item["start_coords"],
            "goal_coords": item["goal_coords"],
            "obstacle_coords": item["obstacle_coords"],
            "astar_path": astar_path,
            "gym_map": gym_map
        })
    
    logger.info(f"Loaded {len(meta_data)} records for {task_type} from {data_path}")
    return meta_data

def evaluate_function(results, meta_data):
    """评估函数，根据任务类型对结果进行评估"""
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    
    # 按任务类型统计结果
    task_results = {}
    # 按任务类型和尺寸统计结果
    size_results = {}
    # 按任务类型、尺寸和路径长度统计结果
    path_length_results = {}
    
    overall_correct = 0
    overall_total = 0
    
    compare_logs = []
    
    for idx, meta in meta_dict.items():
        task_type = meta["task_type"]
        size = meta.get("size", "unknown")
        path_length = meta.get("path_length", "unknown")
        
        # 初始化任务类型统计
        if task_type not in task_results:
            task_results[task_type] = {"correct": 0, "total": 0}
        
        # 初始化任务类型+尺寸统计
        task_size_key = f"{task_type}_size{size}"
        if task_size_key not in size_results:
            size_results[task_size_key] = {"correct": 0, "total": 0, "task_type": task_type, "size": size}
        
        # 初始化任务类型+尺寸+路径长度统计
        if task_type == "verify":  # 只为路径验证任务统计路径长度
            task_size_length_key = f"{task_type}_size{size}_length{path_length}"
            if task_size_length_key not in path_length_results:
                path_length_results[task_size_length_key] = {
                    "correct": 0, "total": 0, "task_type": task_type, "size": size, "path_length": path_length
                }
        
        if idx in results_dict:
            prediction = results_dict[idx]["results"]["final_answer"]
            meta["prediction"] = prediction
        else:
            prediction = None
            meta["prediction"] = None
        
        # 根据任务类型评估结果
        if task_type == "verify":
            score, message = evaluate_path_validation(prediction, meta)
        elif task_type == "navigation":
            score, message = evaluate_path_navigation(prediction, meta)
        else:
            score, message = 0, "Unknown task type"
        
        # 记录任务类型结果
        task_results[task_type]["correct"] += score
        task_results[task_type]["total"] += 1
        
        # 记录任务类型+尺寸结果
        size_results[task_size_key]["correct"] += score
        size_results[task_size_key]["total"] += 1
        
        # 记录任务类型+尺寸+路径长度结果
        if task_type == "verify" and task_size_length_key in path_length_results:
            path_length_results[task_size_length_key]["correct"] += score
            path_length_results[task_size_length_key]["total"] += 1
        
        # 记录总体结果
        overall_correct += score
        overall_total += 1
        
        # 日志记录
        compare_logs.append({
            "idx": idx,
            "task_type": task_type,
            "size": size,
            "path_length": path_length if task_type == "verify" else None,
            "gold": meta["answer"],
            "pred": prediction,
            "score": score,
            "message": message
        })
    
    # 计算每个任务类型的准确率
    for task_type in task_results:
        if task_results[task_type]["total"] > 0:
            task_results[task_type]["accuracy"] = task_results[task_type]["correct"] / task_results[task_type]["total"]
        else:
            task_results[task_type]["accuracy"] = 0
    
    # 计算每个任务类型+尺寸的准确率
    for key in size_results:
        if size_results[key]["total"] > 0:
            size_results[key]["accuracy"] = size_results[key]["correct"] / size_results[key]["total"]
        else:
            size_results[key]["accuracy"] = 0
    
    # 计算每个任务类型+尺寸+路径长度的准确率
    for key in path_length_results:
        if path_length_results[key]["total"] > 0:
            path_length_results[key]["accuracy"] = path_length_results[key]["correct"] / path_length_results[key]["total"]
        else:
            path_length_results[key]["accuracy"] = 0
    
    # 将结果转换为列表，便于按任务类型、尺寸和路径长度排序
    size_results_list = list(size_results.values())
    size_results_list.sort(key=lambda x: (x["task_type"], x["size"]))
    
    path_length_results_list = list(path_length_results.values())
    path_length_results_list.sort(key=lambda x: (x["task_type"], x["size"], x["path_length"]))
    
    # 计算总体准确率
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    
    # 按任务类型计算每个尺寸的平均准确率
    size_summary = {}
    for item in size_results_list:
        task_type = item["task_type"]
        size = item["size"]
        
        if task_type not in size_summary:
            size_summary[task_type] = {}
        
        size_summary[task_type][size] = item["accuracy"]
    
    # 按任务类型和尺寸计算每个路径长度的平均准确率
    length_summary = {}
    for item in path_length_results_list:
        task_type = item["task_type"]
        size = item["size"]
        path_length = item["path_length"]
        
        if task_type not in length_summary:
            length_summary[task_type] = {}
        
        if size not in length_summary[task_type]:
            length_summary[task_type][size] = {}
        
        length_summary[task_type][size][path_length] = item["accuracy"]
    
    result = {
        "overall_accuracy": overall_accuracy,
        "task_results": task_results,
        "size_results": size_results_list,       
        "path_length_results": path_length_results_list,  
        "size_summary": size_summary,           
        "length_summary": length_summary,       
        "compare_logs": compare_logs,
        "results": results
    }
    
    # 打印结果摘要
    logger.info(f"Overall accuracy: {overall_accuracy:.4f}")
    
    for task_type in task_results:
        logger.info(f"Task type {task_type}: {task_results[task_type]['accuracy']:.4f} "
                   f"({task_results[task_type]['correct']}/{task_results[task_type]['total']})")
    
    # 打印按尺寸的结果
    logger.info("Size breakdown:")
    for task_type in sorted(size_summary.keys()):
        logger.info(f"  {task_type}:")
        for size in sorted(size_summary[task_type].keys()):
            accuracy = size_summary[task_type][size]
            # 查找对应的正确数和总数
            key = f"{task_type}_size{size}"
            correct = size_results[key]["correct"] if key in size_results else 0
            total = size_results[key]["total"] if key in size_results else 0
            logger.info(f"    Size {size}: {accuracy:.4f} ({correct}/{total})")
    
    # 打印按路径长度的结果（只对路径验证任务）
    if "verify" in length_summary:
        logger.info("Path length breakdown for path verification task:")
        for size in sorted(length_summary["verify"].keys()):
            logger.info(f"  Size {size}:")
            for path_length in sorted(length_summary["verify"][size].keys()):
                accuracy = length_summary["verify"][size][path_length]
                key = f"verify_size{size}_length{path_length}"
                correct = path_length_results[key]["correct"] if key in path_length_results else 0
                total = path_length_results[key]["total"] if key in path_length_results else 0
                logger.info(f"    Length {path_length}: {accuracy:.4f} ({correct}/{total})")
    
    return result

def evaluate_path_validation(prediction, meta):
    """评估路径验证任务"""
    if prediction is None:
        return 0, "No prediction"
    
    # 预处理预测结果
    yes_pattern = r'\\boxed\s*{\s*yes\s*}|boxed\s*{\s*yes\s*}|boxed{yes}|boxed\(\s*yes\s*\)|yes'
    no_pattern = r'\\boxed\s*{\s*no\s*}|boxed\s*{\s*no\s*}|boxed{no}|boxed\(\s*no\s*\)|no'
    
    prediction = prediction.lower().strip()
    if prediction in ["yes", "no"]:
        pred_answer = prediction
    elif re.search(yes_pattern, prediction, re.IGNORECASE) and not re.search(no_pattern, prediction, re.IGNORECASE):
        pred_answer = "yes"
    elif re.search(no_pattern, prediction, re.IGNORECASE) and not re.search(yes_pattern, prediction, re.IGNORECASE):
        pred_answer = "no"
    else:
        # 尝试从最后一段文本中提取答案
        last_paragraph = prediction.split('\n')[-1].strip()
        if "yes" in last_paragraph.lower() and "no" not in last_paragraph.lower():
            pred_answer = "yes"
        elif "no" in last_paragraph.lower() and "yes" not in last_paragraph.lower():
            pred_answer = "no"
        else:
            return 0, "Invalid prediction format: cannot determine yes/no"
    
    # 获取正确答案
    gold_answer = meta["answer"].lower().strip()
    
    # 比较答案
    if pred_answer == gold_answer.lower():
        return 1, "Correct"
    else:
        return 0, f"Incorrect. Expected: {gold_answer}, Got: {pred_answer}"

def evaluate_path_navigation(prediction, meta):
    """评估路径导航任务，使用gym环境验证路径"""
    if prediction is None:
        return 0, "No prediction"
    
    # 预处理预测结果
    prediction = prediction.strip()
    
    # 尝试提取路径序列
    path_pattern = r'([UDLR](,[UDLR])*)'
    path_match = re.search(path_pattern, prediction, re.IGNORECASE)
    
    if path_match:
        action_sequence = path_match.group(1).upper()
    else:
        # 尝试从任何位置提取UDLR字符
        actions = []
        for char in prediction:
            if char.upper() in ['U', 'D', 'L', 'R']:
                actions.append(char.upper())
        if actions:
            action_sequence = ",".join(actions)
        else:
            return 0, "No valid path found"
    
    # 将动作序列转换为列表
    actions = []
    for action in action_sequence.split(','):
        action = action.strip().upper()
        if action in ['L', 'R', 'U', 'D']:
            actions.append(action)
    
    if not actions:
        return 0, "No valid actions found"
    
    # 获取地图
    gym_map = meta["gym_map"]
    try:
        # 创建FrozenLake环境
        env = gym.make('FrozenLake-v1', desc=gym_map, is_slippery=False)
        env.reset()
        
        # 执行动作序列
        action_mapping = {'L': 0, 'D': 1, 'R': 2, 'U': 3}
        
        for action in actions:
            if action in action_mapping:
                observation, reward, terminated, truncated, info = env.step(action_mapping[action])
                
                if terminated:
                    # 如果到达目标
                    if reward > 0:
                        env.close()
                        return 1, "Path leads to goal"
                    # 如果掉入洞中
                    else:
                        env.close()
                        return 0, "Path leads to hole"
        
        env.close()
        # 如果完成所有动作但没有到达目标
        return 0, "Path does not reach goal"
    
    except Exception as e:
        return 0, f"Error validating path: {str(e)}"