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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)
PATH_VERIFY_TASK_INSTRUCTION_SHORT = """

You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success. 

Now please determine if the action sequence is safe for the given maze. Your final answer should be formatted as \\boxed{Yes} or \\boxed{No}.

The action sequence is:

<ACTION-SEQ>
"""

def convert_markdown_table_to_map(table_str):
    """
    将Markdown格式的表格转换为FrozenLake风格的地图表示
    
    Args:
        table_str (str): Markdown格式的表格字符串
    
    Returns:
        list: FrozenLake风格的地图，由字符串组成的列表
    """
    # 解析表格
    rows = [row.strip() for row in table_str.strip().split('\n') if '|' in row]
    
    # 忽略表头和分隔符行
    if len(rows) >= 2 and all('-' in cell for cell in rows[1].split('|')):
        rows = [rows[0]] + rows[2:]
    
    # 提取实际的地图内容
    map_rows = []
    for row in rows:
        cells = [cell.strip() for cell in row.split('|')]
        # 过滤空单元格和行标签
        cells = [cell for cell in cells if cell and not cell.startswith('Row')]
        map_rows.append(cells)
    
    # 构建FrozenLake风格的地图
    frozen_lake_map = []
    for row in map_rows:
        map_row = ""
        for cell in row:
            if cell == '@':
                map_row += 'S'  # 起点
            elif cell == '#':
                map_row += 'H'  # 障碍物/洞
            elif cell == '*':
                map_row += 'G'  # 目标
            elif cell == '_':
                map_row += 'F'  # 安全地板
            else:
                # 忽略列标题或其他内容
                continue
        if map_row:  # 只添加非空行
            frozen_lake_map.append(map_row)
    
    return frozen_lake_map


def load_data_function():
    """加载VSP数据集的函数"""
    dataset_path = task_config.get("dataset_path")
    tasks = task_config.get("tasks", ["task-main"])
    num_samples = task_config.get("num_sample")
    
    meta_data = []
    
    for task in tasks:
        task_dir = os.path.join(dataset_path, "maze", task)
        if not os.path.exists(task_dir):
            logger.warning(f"Task directory not found: {task_dir}")
            continue
            
        # 根据任务类型选择适当的数据加载逻辑
        if task == "task4":
            # 路径验证任务
            meta_data.extend(load_path_validation_data(task_dir, task, num_samples))
        elif task == "task3":
            # 地图感知任务
            meta_data.extend(load_perception_data(task_dir, task, num_samples))
        elif task == "task2":
            # 空间关系任务
            meta_data.extend(load_relation_data(task_dir, task, num_samples))
        elif task == "task1":
            # 障碍物感知任务
            meta_data.extend(load_hole_perception_data(task_dir, task, num_samples))
        elif task == "task-main":
            # 路径规划任务
            meta_data.extend(load_planning_data(task_dir, task, num_samples))
    
    # 数据集统计信息
    logger.info(f"Total data loaded: {len(meta_data)}")
    task_counts = {}
    for item in meta_data:
        task_type = item.get("task_type", "unknown")
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    for task_type, count in task_counts.items():
        logger.info(f"Task type: {task_type}, count: {count}")
    
    return meta_data


def load_path_validation_data(task_dir, task_type, num_samples=None):
    """加载路径验证任务数据"""
    meta_data = []
    
    # 读取提示文本
    # with open(os.path.join(task_dir,"prompt-text", "prompt-text.txt"), "r") as f:
    #     prompt_text = f.read()
    prompt_text = PATH_VERIFY_TASK_INSTRUCTION_SHORT
    # 获取视觉提示图像
    prompt_visual_images = []
    visual_dir = os.path.join(task_dir, "prompt-visual-images")
    if os.path.exists(visual_dir):
        for img_file in sorted(os.listdir(visual_dir)):
            if img_file.endswith(".png"):
                img_path = os.path.join(visual_dir, img_file)
                prompt_visual_images.append(img_path)
    
    levels = [1,3,5,7,9]
    
    for level in levels:
        map_dir = os.path.join(task_dir, "maps", f"level_step{level}")
        img_dir = os.path.join(map_dir, "img")
        question_dir = os.path.join(map_dir, "question")
        answer_dir = os.path.join(map_dir, "answer")
        
        if not os.path.exists(img_dir) or not os.path.exists(question_dir):
            continue
            
        files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        # 如果指定了样本数量，限制每个级别的样本数
        if num_samples:
            max_per_level = max(1, num_samples // len(levels))
            files = files[:min(len(files), max_per_level)]
        
        for file in files:
            idx = file.split('.')[0]
            img_path = os.path.join(img_dir, file)
            
            # 读取问题和答案
            question_path = os.path.join(question_dir, f"{idx}.txt")
            answer_path = os.path.join(answer_dir, f"{idx}.txt")
            
            if os.path.exists(question_path) and os.path.exists(answer_path):
                with open(question_path, "r") as f:
                    question = f.read().strip()
                with open(answer_path, "r") as f:
                    answer = f.read().strip()
                
                # 构建提示
                text_prompt = prompt_text.replace("<ACTION-SEQ>", question)
                
                meta_data.append({
                    "idx": f"vsp_{task_type}_level{level}_{idx}",
                    "image_path": img_path,
                    "text": text_prompt,
                    "answer": answer,
                    "level": level,
                    "task_type": task_type,
                    "prompt_images": prompt_visual_images,
                    "map_text": extract_map_from_image(img_path)
                })
    
    return meta_data


def load_perception_data(task_dir, task_type, num_samples=None):
    """加载地图感知任务数据"""
    meta_data = []
    
    # 读取提示文本
    with open(os.path.join(task_dir, "prompt-text.txt"), "r") as f:
        prompt_text = f.read()
    
    # 获取视觉提示图像
    prompt_visual_images = []
    visual_dir = os.path.join(task_dir, "prompt-visual-images")
    if os.path.exists(visual_dir):
        for img_file in sorted(os.listdir(visual_dir)):
            if img_file.endswith(".png"):
                img_path = os.path.join(visual_dir, img_file)
                prompt_visual_images.append(img_path)
    
    levels = [3, 4, 5, 6, 7, 8]
    
    for level in levels:
        map_dir = os.path.join(task_dir, "maps", f"level{level}")
        img_dir = os.path.join(map_dir, "img")
        question_dir = os.path.join(map_dir, "question")
        answer_dir = os.path.join(map_dir, "answer")
        
        if not os.path.exists(img_dir) or not os.path.exists(question_dir):
            continue
            
        files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        # 如果指定了样本数量，限制每个级别的样本数
        if num_samples:
            max_per_level = max(1, num_samples // len(levels))
            files = files[:min(len(files), max_per_level)]
        
        for file in files:
            idx = file.split('.')[0]
            img_path = os.path.join(img_dir, file)
            
            # 读取问题和答案
            question_path = os.path.join(question_dir, f"{idx}.txt")
            answer_path = os.path.join(answer_dir, f"{idx}.txt")
            
            if os.path.exists(question_path) and os.path.exists(answer_path):
                with open(question_path, "r") as f:
                    candidates = f.read().strip()
                with open(answer_path, "r") as f:
                    answer_lines = f.read().strip().split('\n')
                    answer_idx = int(answer_lines[-1])
                    answer = chr(65 + answer_idx)  # 将数字转换为A,B,C,D
                
                # 构建提示
                text_prompt = prompt_text.replace("<CANDIDATES>", candidates)
                
                meta_data.append({
                    "idx": f"vsp_{task_type}_level{level}_{idx}",
                    "image_path": img_path,
                    "text": text_prompt,
                    "answer": answer,
                    "level": level,
                    "task_type": task_type,
                    "prompt_images": prompt_visual_images,
                    "map_text": extract_map_from_image(img_path)
                })
    
    return meta_data


def load_relation_data(task_dir, task_type, num_samples=None):
    """加载空间关系任务数据"""
    meta_data = []
    
    # 读取提示文本
    with open(os.path.join(task_dir, "prompt-text.txt"), "r") as f:
        prompt_text = f.read()
    
    # 获取视觉提示图像
    prompt_visual_images = []
    visual_dir = os.path.join(task_dir, "prompt-visual-images")
    if os.path.exists(visual_dir):
        for img_file in sorted(os.listdir(visual_dir)):
            if img_file.endswith(".png"):
                img_path = os.path.join(visual_dir, img_file)
                prompt_visual_images.append(img_path)
    
    levels = [3, 4, 5, 6, 7, 8]
    
    for level in levels:
        map_dir = os.path.join(task_dir, "maps", f"level{level}")
        img_dir = map_dir
        text_map_dir = os.path.join(task_dir, "maps", f"level{level}_text")
        
        if not os.path.exists(img_dir) or not os.path.exists(text_map_dir):
            continue
            
        files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        # 如果指定了样本数量，限制每个级别的样本数
        if num_samples:
            max_per_level = max(1, num_samples // len(levels))
            files = files[:min(len(files), max_per_level)]
        
        for file in files:
            idx = file.split('.')[0]
            img_path = os.path.join(img_dir, f"{idx}.png")
            map_text_path = os.path.join(text_map_dir, f"{idx}.txt")
            
            if os.path.exists(img_path) and os.path.exists(map_text_path):
                with open(map_text_path, "r") as f:
                    map_text = f.read()
                
                # 从地图文本分析正确答案
                answer = analyze_spatial_relation(map_text)
                
                meta_data.append({
                    "idx": f"vsp_{task_type}_level{level}_{idx}",
                    "image_path": img_path,
                    "text": prompt_text,
                    "answer": answer,
                    "level": level,
                    "task_type": task_type,
                    "prompt_images": prompt_visual_images,
                    "map_text": map_text
                })
    
    return meta_data


def load_hole_perception_data(task_dir, task_type, num_samples=None):
    """加载障碍物感知任务数据"""
    meta_data = []
    
    # 读取提示文本
    with open(os.path.join(task_dir, "prompt-text.txt"), "r") as f:
        prompt_text = f.read()
    
    # 获取视觉提示图像
    prompt_visual_images = []
    visual_dir = os.path.join(task_dir, "prompt-visual-images")
    if os.path.exists(visual_dir):
        for img_file in sorted(os.listdir(visual_dir)):
            if img_file.endswith(".png"):
                img_path = os.path.join(visual_dir, img_file)
                prompt_visual_images.append(img_path)
    
    levels = [3, 4, 5, 6, 7, 8]
    
    for level in levels:
        map_dir = os.path.join(task_dir, "maps", f"level{level}")
        img_dir = os.path.join(map_dir, "img")
        question_dir = os.path.join(map_dir, "question")
        answer_dir = os.path.join(map_dir, "answer")
        
        if not os.path.exists(img_dir) or not os.path.exists(question_dir):
            continue
            
        files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        # 如果指定了样本数量，限制每个级别的样本数
        if num_samples:
            max_per_level = max(1, num_samples // len(levels))
            files = files[:min(len(files), max_per_level)]
        
        for file in files:
            idx = file.split('.')[0]
            img_path = os.path.join(img_dir, file)
            
            # 读取问题和答案
            question_path = os.path.join(question_dir, f"{idx}.txt")
            answer_path = os.path.join(answer_dir, f"{idx}.txt")
            
            if os.path.exists(question_path) and os.path.exists(answer_path):
                with open(question_path, "r") as f:
                    question = f.read().strip()
                with open(answer_path, "r") as f:
                    answer_code = f.read().strip()
                    answer = "Yes" if answer_code == "Y" else "No"
                
                # 构建提示
                text_prompt = prompt_text.replace("<QUESTION>", question)
                
                meta_data.append({
                    "idx": f"vsp_{task_type}_level{level}_{idx}",
                    "image_path": img_path,
                    "text": text_prompt,
                    "answer": answer,
                    "level": level,
                    "task_type": task_type,
                    "prompt_images": prompt_visual_images,
                    "map_text": extract_map_from_image(img_path)
                })
    
    return meta_data


def load_planning_data(task_dir, task_type, num_samples=None):
    """加载路径规划任务数据"""
    meta_data = []
    
    # 读取提示文本
    with open(os.path.join(task_dir, "prompt-text" ,"prompt-text.txt"), "r") as f:
        prompt_text = f.read()
    
    # 获取视觉提示图像
    prompt_visual_images = []
    visual_dir = os.path.join(task_dir, "prompt-visual-images")
    if os.path.exists(visual_dir):
        for img_file in sorted(os.listdir(visual_dir)):
            if img_file.endswith(".png"):
                img_path = os.path.join(visual_dir, img_file)
                prompt_visual_images.append(img_path)
    
    levels = [3, 4, 5, 6, 7, 8]
    
    for level in levels:
        map_dir = os.path.join(task_dir, "maps", f"level{level}")
        img_dir = os.path.join(map_dir, "img")
        text_map_dir = os.path.join(map_dir,"table")
        
        if not os.path.exists(img_dir) or not os.path.exists(text_map_dir):
            continue
            
        files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        # 如果指定了样本数量，限制每个级别的样本数
        if num_samples:
            max_per_level = max(1, num_samples // len(levels))
            files = files[:min(len(files), max_per_level)]
        
        for file in files:
            idx = file.split('.')[0]
            img_path = os.path.join(img_dir, f"{idx}.png")
            map_text_path = os.path.join(text_map_dir, f"{idx}.txt")
            
            if os.path.exists(img_path) and os.path.exists(map_text_path):
                with open(map_text_path, "r") as f:
                    map_text = f.read()
                map_list = convert_markdown_table_to_map(map_text)
                meta_data.append({
                    "idx": f"vsp_{task_type}_level{level}_{idx}",
                    "image_path": img_path,
                    "text": prompt_text,
                    "answer": "DYNAMIC_EVAL",  # 这个任务需要动态评估
                    "level": level,
                    "task_type": task_type,
                    "prompt_images": prompt_visual_images,
                    "map_text": map_text,
                    "map_text_list": map_list,
                })
    
    return meta_data


def extract_map_from_image(img_path):
    """从图像中提取地图表示（这里是一个简化的实现）"""
    # 实际应用中可以使用计算机视觉技术提取地图
    # 这里简单返回一个占位符
    return "MAP_EXTRACTION_NOT_IMPLEMENTED"


def analyze_spatial_relation(map_text):
    """分析地图文本中玩家和目标的空间关系"""
    rows = map_text.strip().split('\n')
    player_pos = None
    goal_pos = None
    
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if cell == 'S':  # 玩家位置
                player_pos = (i, j)
            elif cell == 'G':  # 目标位置
                goal_pos = (i, j)
    
    if not player_pos or not goal_pos:
        return "Unknown"
        
    relations = []
    
    # 检查垂直关系
    if player_pos[0] < goal_pos[0]:
        relations.append("Above")
    elif player_pos[0] > goal_pos[0]:
        relations.append("Below")
    
    # 检查水平关系
    if player_pos[1] < goal_pos[1]:
        relations.append("Left")
    elif player_pos[1] > goal_pos[1]:
        relations.append("Right")
        
    return ",".join(relations) if relations else "Same"


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
            level_results[task_level_key] = {"correct": 0, "total": 0, "task_type": task_type, "level": level}
        
        if idx in results_dict:
            prediction = results_dict[idx]["results"]["final_answer"]
            meta["prediction"] = prediction
        else:
            prediction = None
            meta["prediction"] = None
        
        # 根据任务类型评估结果
        if task_type == "task4":
            score, message = evaluate_path_validation(prediction, meta)
        elif task_type == "task3":
            score, message = evaluate_perception(prediction, meta)
        elif task_type == "task2":
            score, message = evaluate_relation(prediction, meta)
        elif task_type == "task1":
            score, message = evaluate_hole_perception(prediction, meta)
        elif task_type == "task-main":
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
            task_results[task_type]["accuracy"] = task_results[task_type]["correct"] / task_results[task_type]["total"]
        else:
            task_results[task_type]["accuracy"] = 0
    
    # 计算每个任务类型+级别的准确率
    for key in level_results:
        if level_results[key]["total"] > 0:
            level_results[key]["accuracy"] = level_results[key]["correct"] / level_results[key]["total"]
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
        "level_results": level_results_list,  # 每个任务类型+级别的详细结果
        "level_summary": level_summary,       # 按任务类型汇总的级别结果
        "compare_logs": compare_logs,
        "meta_data": meta_data,
        "results": results
    }
    
    # 打印结果摘要
    logger.info(f"Overall accuracy: {overall_accuracy:.4f}")
    
    for task_type in task_results:
        logger.info(f"Task type {task_type}: {task_results[task_type]['accuracy']:.4f} "
                   f"({task_results[task_type]['correct']}/{task_results[task_type]['total']})")
    
    logger.info("Level breakdown:")
    for task_type in sorted(level_summary.keys()):
        logger.info(f"  {task_type}:")
        for level in sorted(level_summary[task_type].keys()):
            accuracy = level_summary[task_type][level]
            # 查找对应的正确数和总数
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
    output_match = re.search(r"<output>\s*(yes|no)", prediction, re.IGNORECASE)
    if output_match:
        pred_answer = output_match.group(1).lower()
    elif prediction == "yes":
        pred_answer = "yes"
    elif prediction == "no":
        pred_answer = "no"
    else:
        # 尝试其他可能的格式
        if "yes" in prediction and "no" not in prediction:
            pred_answer = "yes"
        elif "no" in prediction and "yes" not in prediction:
            pred_answer = "no"
        else:
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


def evaluate_perception(prediction, meta):
    """评估地图感知任务"""
    if prediction is None:
        return 0, "No prediction"
    
    # 预处理预测结果
    prediction = prediction.upper().strip()
    
    # 提取选项 (A/B/C/D)
    answer_match = re.search(r"<answer>\s*([A-D])", prediction, re.IGNORECASE)
    if not answer_match:
        # 尝试其他格式
        answer_match = re.search(r"\b([A-D])\b", prediction)
    
    if answer_match:
        pred_answer = answer_match.group(1).upper()
    else:
        return 0, "Invalid prediction format"
    
    # 获取正确答案
    gold_answer = meta["answer"].upper().strip()
    
    # 比较答案
    if pred_answer == gold_answer:
        return 1, "Correct"
    else:
        return 0, f"Incorrect. Expected: {gold_answer}, Got: {pred_answer}"


def evaluate_relation(prediction, meta):
    """评估空间关系任务"""
    if prediction is None:
        return 0, "No prediction"
    
    # 预处理预测结果
    prediction = prediction.lower().strip()
    
    # 提取空间关系答案
    output_match = re.search(r"<output>\s*(.+)", prediction, re.IGNORECASE)
    if output_match:
        pred_relations = output_match.group(1).lower()
    else:
        # 尝试识别"Above", "Below", "Left", "Right"关键词
        relations = []
        if "above" in prediction:
            relations.append("above")
        if "below" in prediction:
            relations.append("below")
        if "left" in prediction:
            relations.append("left")
        if "right" in prediction:
            relations.append("right")
        
        if relations:
            pred_relations = ",".join(relations)
        else:
            return 0, "Invalid prediction format"
    
    # 移除引号和空格
    pred_relations = pred_relations.replace('"', '').replace("'", '').strip()
    
    # 获取正确答案
    gold_relations = meta["answer"].lower().strip()
    
    # 将答案标准化为集合进行比较
    pred_set = set(r.strip() for r in pred_relations.split(','))
    gold_set = set(r.strip() for r in gold_relations.split(','))
    
    # 计算准确率（部分匹配）
    common = len(pred_set.intersection(gold_set))
    total = len(gold_set)
    
    if common == total and len(pred_set) == len(gold_set):
        return 1, "Correct"
    elif common > 0:
        score = common / total
        return score, f"Partially correct. Expected: {gold_relations}, Got: {pred_relations}"
    else:
        return 0, f"Incorrect. Expected: {gold_relations}, Got: {pred_relations}"


def evaluate_hole_perception(prediction, meta):
    """评估障碍物感知任务"""
    if prediction is None:
        return 0, "No prediction"
    
    # 预处理预测结果
    prediction = prediction.lower().strip()
    
    # 提取Yes/No答案
    output_match = re.search(r"<output>\s*(yes|no)", prediction, re.IGNORECASE)
    if output_match:
        pred_answer = output_match.group(1).lower()
    else:
        # 尝试其他可能的格式
        if "yes" in prediction and "no" not in prediction:
            pred_answer = "yes"
        elif "no" in prediction and "yes" not in prediction:
            pred_answer = "no"
        else:
            return 0, "Invalid prediction format"
    
    # 获取正确答案
    gold_answer = meta["answer"].lower().strip()
    
    # 比较答案
    if pred_answer == gold_answer:
        return 1, "Correct"
    else:
        return 0, f"Incorrect. Expected: {gold_answer}, Got: {pred_answer}"


def evaluate_planning(prediction, meta):
    """评估路径规划任务，使用gym环境验证路径"""
    if prediction is None:
        return 0, "No prediction"
    
    # 预处理预测结果
    prediction = prediction.lower().strip()
    action_sequence = prediction
    
    # 将动作序列转换为列表
    actions = []
    for action in action_sequence.split(','):
        action = action.strip().upper()
        if action in ['L', 'R', 'U', 'D']:
            actions.append(action)
    
    if not actions:
        return 0, "No valid actions found"
    
    # 获取地图文本
    rows = meta["map_text_list"]
    try:
        # 创建FrozenLake环境
        env = gym.make('FrozenLake-v1', desc=rows, is_slippery=False)
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


def find_symbol(map_text, symbol):
    """在地图文本中查找特定符号的坐标"""
    rows = map_text.strip().split('\n')
    results = []
    
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if cell == symbol:
                results.append([i, j])
    
    return results