# task.py
from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import gymnasium as gym
from datasets import load_dataset
import json
import os
import re
import sys
import random
import numpy as np
from PIL import Image
import io
import base64

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)


def load_data_function():
    """从HuggingFace Hub加载VSP数据集"""
    dataset_path = task_config.get("dataset_path", "hitsmy/AdaEval-VSP")
    tasks = task_config.get("tasks", ["verify_test", "navigation_test"])
    num_samples = task_config.get("num_sample")
    
    logger.info(f"Loading dataset from HuggingFace Hub: {dataset_path}")
    
    meta_data = []
    
    for task_split in tasks:
        try:
            # 从HuggingFace加载数据集
            logger.info(f"Loading split: {task_split}")
            dataset = load_dataset(dataset_path, split=task_split)
            
            # 限制样本数量
            if num_samples and len(dataset) > num_samples:
                # 每个任务类型取相同数量
                max_per_task = num_samples // len(tasks)
                dataset = dataset.select(range(min(len(dataset), max_per_task)))
            
            # 转换为meta_data格式
            for item in dataset:
                # 将HF数据集格式转换为原始meta_data格式
                meta_item = {
                    "idx": item["idx"],
                    "original_id": item["original_id"],
                    "image": item["image"],  # PIL Image对象
                    "text": item["text"],
                    "answer": item["answer"],
                    "task_type": item["task_type"],
                    "split": item["split"],
                    "size": item["size"],
                    "level": item["level"],
                }
                
                # 解析gym_map
                if item["gym_map"]:
                    meta_item["map_text_list"] = json.loads(item["gym_map"])
                else:
                    meta_item["map_text_list"] = []
                
                # 添加任务特定字段
                if item["task_type"] == "verify":
                    meta_item["path_length"] = item["path_length"]
                    meta_item["path"] = item["path"]
                elif item["task_type"] == "navigation":
                    if item["start_coords"]:
                        meta_item["start_coords"] = json.loads(item["start_coords"])
                    if item["goal_coords"]:
                        meta_item["goal_coords"] = json.loads(item["goal_coords"])
                    if item["obstacle_coords"]:
                        meta_item["obstacle_coords"] = json.loads(item["obstacle_coords"])
                    meta_item["astar_path"] = item["astar_path"]
                
                meta_data.append(meta_item)
            
            logger.info(f"Loaded {len(dataset)} samples from {task_split}")
            
        except Exception as e:
            logger.error(f"Error loading split {task_split}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 数据集统计信息
    logger.info(f"Total data loaded: {len(meta_data)}")
    
    task_counts = {}
    for item in meta_data:
        task_type = item.get("task_type", "unknown")
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    for task_type, count in task_counts.items():
        logger.info(f"Task type: {task_type}, count: {count}")
    
    return meta_data


def evaluate_function(results, meta_data):
    """评估函数，根据任务类型和级别对结果进行评估"""
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    
    # 按任务类型统计结果
    task_results = {}
    # 按任务类型和级别统计结果
    level_results = {}
    overall_correct = 0
    overall_total = 0
    
    compare_logs = []
    
    for idx, meta in meta_dict.items():
        task_type = meta["task_type"]
        level = meta.get("level", "unknown")
        
        # 初始化任务类型统计
        if task_type not in task_results:
            task_results[task_type] = {"correct": 0, "total": 0}
        
        # 初始化任务类型+级别统计
        task_level_key = f"{task_type}_level{level}"
        if task_level_key not in level_results:
            level_results[task_level_key] = {
                "correct": 0, 
                "total": 0, 
                "task_type": task_type, 
                "level": level
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
            score, message = evaluate_planning(prediction, meta)
        else:
            score, message = 0, "Unknown task type"
        
        # 记录任务类型结果
        task_results[task_type]["correct"] += score
        task_results[task_type]["total"] += 1
        
        # 记录任务类型+级别结果
        level_results[task_level_key]["correct"] += score
        level_results[task_level_key]["total"] += 1
        
        # 记录总体结果
        overall_correct += score
        overall_total += 1
        
        # 日志记录
        compare_logs.append({
            "idx": idx,
            "task_type": task_type,
            "level": level,
            "gold": meta["answer"],
            "pred": prediction,
            "score": score,
            "message": message
        })
    
    # 计算每个任务类型的准确率
    for task_type in task_results:
        if task_results[task_type]["total"] > 0:
            task_results[task_type]["accuracy"] = (
                task_results[task_type]["correct"] / task_results[task_type]["total"]
            )
        else:
            task_results[task_type]["accuracy"] = 0
    
    # 计算每个任务类型+级别的准确率
    for key in level_results:
        if level_results[key]["total"] > 0:
            level_results[key]["accuracy"] = (
                level_results[key]["correct"] / level_results[key]["total"]
            )
        else:
            level_results[key]["accuracy"] = 0
    
    # 将level_results转换为列表，便于按任务类型和级别排序
    level_results_list = list(level_results.values())
    level_results_list.sort(key=lambda x: (x["task_type"], x["level"]))
    
    # 计算总体准确率
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    
    # 按任务类型计算每个级别的平均准确率
    level_summary = {}
    for item in level_results_list:
        task_type = item["task_type"]
        level = item["level"]
        
        if task_type not in level_summary:
            level_summary[task_type] = {}
        
        level_summary[task_type][level] = item["accuracy"]
    
    result = {
        "overall_accuracy": overall_accuracy,
        "task_results": task_results,
        "level_results": level_results_list,
        "level_summary": level_summary,
        "compare_logs": compare_logs,
        "results": results
    }
    
    # 打印结果摘要
    logger.info(f"Overall accuracy: {overall_accuracy:.4f}")
    
    for task_type in task_results:
        logger.info(
            f"Task type {task_type}: {task_results[task_type]['accuracy']:.4f} "
            f"({task_results[task_type]['correct']}/{task_results[task_type]['total']})"
        )
    
    logger.info("Level breakdown:")
    for task_type in sorted(level_summary.keys()):
        logger.info(f"  {task_type}:")
        for level in sorted(level_summary[task_type].keys()):
            accuracy = level_summary[task_type][level]
            key = f"{task_type}_level{level}"
            correct = level_results[key]["correct"] if key in level_results else 0
            total = level_results[key]["total"] if key in level_results else 0
            logger.info(f"    Level {level}: {accuracy:.4f} ({correct}/{total})")
    
    return result


def evaluate_path_validation(prediction, meta):
    """评估路径验证任务"""
    if prediction is None:
        return 0, "No prediction"
    
    # 预处理预测结果
    prediction = prediction.lower().strip()
    
    # 提取Yes/No答案
    # 尝试多种格式
    pred_answer = None
    
    # 格式1: \boxed{Yes} 或 \boxed{No}
    boxed_match = re.search(r"\\boxed\{(yes|no)\}", prediction, re.IGNORECASE)
    if boxed_match:
        pred_answer = boxed_match.group(1).lower()
    # 格式2: <output>Yes</output> 或 <output>No</output>
    elif "<output>" in prediction:
        output_match = re.search(r"<output>\s*(yes|no)", prediction, re.IGNORECASE)
        if output_match:
            pred_answer = output_match.group(1).lower()
    # 格式3: 直接是Yes或No
    elif prediction.strip() in ["yes", "no"]:
        pred_answer = prediction.strip()
    # 格式4: 包含yes或no关键词
    elif "yes" in prediction and "no" not in prediction:
        pred_answer = "yes"
    elif "no" in prediction and "yes" not in prediction:
        pred_answer = "no"
    
    if pred_answer is None:
        return 0, "Invalid prediction format"
    
    # 获取正确答案
    gold_answer = meta["answer"].lower().strip()
    if gold_answer == "y":
        gold_answer = "yes"
    elif gold_answer == "n":
        gold_answer = "no"
    
    # 比较答案
    if pred_answer == gold_answer:
        return 1, "Correct"
    else:
        return 0, f"Incorrect. Expected: {gold_answer}, Got: {pred_answer}"


def evaluate_planning(prediction, meta):
    """评估路径规划任务，使用gym环境验证路径"""
    if prediction is None:
        return 0, "No prediction"
    
    # 预处理预测结果，提取动作序列
    prediction_clean = prediction.strip()
    
    # 尝试提取\boxed{}中的内容
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", prediction_clean, re.IGNORECASE)
    if boxed_match:
        action_sequence = boxed_match.group(1)
    else:
        # 尝试提取最后一行包含L,R,U,D的内容
        lines = prediction_clean.split('\n')
        action_sequence = None
        for line in reversed(lines):
            if any(action in line.upper() for action in ['L', 'R', 'U', 'D']):
                action_sequence = line
                break
        
        if action_sequence is None:
            action_sequence = prediction_clean
    
    # 将动作序列转换为列表
    actions = []
    action_sequence = action_sequence.upper().replace(" ", "")
    
    # 处理逗号分隔的格式
    if ',' in action_sequence:
        for action in action_sequence.split(','):
            action = action.strip()
            if action in ['L', 'R', 'U', 'D']:
                actions.append(action)
    else:
        # 处理连续字符格式
        for char in action_sequence:
            if char in ['L', 'R', 'U', 'D']:
                actions.append(char)
    
    if not actions:
        return 0, "No valid actions found"
    
    # 获取地图
    if "map_text_list" not in meta or not meta["map_text_list"]:
        return 0, "No map information available"
    
    rows = meta["map_text_list"]
    
    try:
        # 创建FrozenLake环境
        env = gym.make('FrozenLake-v1', desc=rows, is_slippery=False, render_mode=None)
        env.reset()
        
        # 执行动作序列
        action_mapping = {'L': 0, 'D': 1, 'R': 2, 'U': 3}
        
        for action in actions:
            if action in action_mapping:
                observation, reward, terminated, truncated, info = env.step(action_mapping[action])
                
                if terminated:
                    env.close()
                    # 如果到达目标
                    if reward > 0:
                        return 1, f"Path leads to goal (length: {len(actions)})"
                    # 如果掉入洞中
                    else:
                        return 0, f"Path leads to hole (at step {actions.index(action) + 1})"
        
        env.close()
        # 如果完成所有动作但没有到达目标
        return 0, f"Path does not reach goal (completed {len(actions)} actions)"
    
    except Exception as e:
        logger.error(f"Error validating path: {str(e)}")
        return 0, f"Error validating path: {str(e)}"


# 保留一些辅助函数供可能的扩展使用
def convert_markdown_table_to_map(table_str):
    """将Markdown格式的表格转换为FrozenLake风格的地图表示"""
    rows = [row.strip() for row in table_str.strip().split('\n') if '|' in row]
    
    if len(rows) >= 2 and all('-' in cell for cell in rows[1].split('|')):
        rows = [rows[0]] + rows[2:]
    
    map_rows = []
    for row in rows:
        cells = [cell.strip() for cell in row.split('|')]
        cells = [cell for cell in cells if cell and not cell.startswith('Row')]
        map_rows.append(cells)
    
    frozen_lake_map = []
    for row in map_rows:
        map_row = ""
        for cell in row:
            if cell == '@':
                map_row += 'S'
            elif cell == '#':
                map_row += 'H'
            elif cell == '*':
                map_row += 'G'
            elif cell == '_':
                map_row += 'F'
            else:
                continue
        if map_row:
            frozen_lake_map.append(map_row)
    
    return frozen_lake_map