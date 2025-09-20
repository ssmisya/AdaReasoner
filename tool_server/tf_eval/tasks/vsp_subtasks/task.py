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
import math
from tool_server.utils.debug import remote_breakpoint

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

# 路径验证任务说明（使用画有路径的图像）
PATH_VERIFY_WITH_PATH_INSTRUCTION = """
You are a helpful visual assistant.

Please determine whether the red line in the figure passes through or enters any Blue Ice Holes.

Please note that the Ice Holes are represented by blue water pits, while the white areas are snow-covered ground, which are safe and not Ice Holes.

Your final answer should be formatted as \\boxed{Yes} or \\boxed{No}.
"""

# 起点定位任务说明
LOCATE_START_POINT_INSTRUCTION = """
You are looking at a frozen lake maze. In this maze:
- The elf (player) is positioned at the starting point
- The gift is positioned at the goal
- The dark spots are ice holes that must be avoided

Please locate the coordinates of the starting point (where the elf is positioned) in the image.
Your answer should be formatted as \\boxed{x,y} where x and y are coordinates.

You can either return pixel coordinates or absolute coordinates.
For pixel coordinates, the origin is set at the top-left corner (0,0). The x-axis increases to the right, and the y-axis increases downward. You should return the pixel coordinates of the target object. For example, if the target is in the top-left grid cell, its pixel coordinates are (32, 32).

For absolute coordinates, each grid cell is counted as one unit, with the same axis directions. For example, if the target is in the top-left grid cell, its absolute coordinates are (1, 1).

The conversion between pixel coordinates and absolute coordinates is usually:
Pixel coordinate = (absolute_x * 64 - 32, absolute_y * 64 - 32)

You must return either pixel coordinates or absolute coordinates, but never both at the same time.
"""

# 新增：带有坐标信息的路径验证任务
PATH_VERIFY_WITH_COORDS_INSTRUCTION = """
You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success. 

You have the following coordinate information:
- Starting point (elf position): {start_coords_pixel} in pixel coordinates, or {start_coords_abs} in grid coordinates
- Ice hole locations: {holes_pixel} in pixel coordinates, or {holes_abs} in grid coordinates

The pixel coordinates use the top-left corner as origin (0,0), with x-axis increasing to the right and y-axis increasing downward.
The grid coordinates count each cell as one unit with the top-left cell being (1,1).

Now please determine if the action sequence is safe for the given maze. The action sequence is:

<ACTION-SEQ>

Your final answer should be formatted as \\boxed{{Yes}} or \\boxed{{No}}.
"""

def load_data_function():
    """加载自定义数据集的函数"""
    tasks = task_config.get("tasks", ["verify_with_line", "locate_start", "verify_with_coords"])
    # remote_breakpoint(port=7119)
    # 从配置文件获取数据路径字典
    dataset_path_dict = task_config.get("dataset_path_dict", {})
    if not dataset_path_dict:
        logger.error("No dataset_path_dict found in config")
        return []
    
    img_dir = task_config.get("image_dir", "")
    
    meta_data = []
    
    # 加载不同任务的数据
    for task in tasks:
        if task == "verify_with_coords" and "verify_with_line" in dataset_path_dict:
            # 对于新增的带坐标信息的路径验证任务，复用verify_with_line的数据
            task_file_path = dataset_path_dict["verify_with_line"]
        elif task not in dataset_path_dict:
            logger.warning(f"No path specified for task {task} in dataset_path_dict")
            continue
        else:
            task_file_path = dataset_path_dict[task]
        
        # 根据文件扩展名选择加载方法
        if task_file_path.endswith(".jsonl"):
            raw_data = process_jsonl(task_file_path)
        elif task_file_path.endswith(".json"):
            raw_data = load_json_file(task_file_path)
        else:
            logger.warning(f"Unsupported file format: {task_file_path}")
            continue
        
        logger.info(f"Loaded {len(raw_data)} raw records for task {task} from {task_file_path}")
        
        # 根据任务类型处理数据
        if task == "verify_with_line":
            task_data = process_verify_with_line_data(raw_data, task, img_dir)
        elif task == "locate_start":
            task_data = process_locate_start_data(raw_data, task, img_dir)
        elif task == "verify_with_coords":
            task_data = process_verify_with_coords_data(raw_data, task, img_dir)
        else:
            logger.warning(f"Unknown task type: {task}")
            continue
        
        meta_data.extend(task_data)
    
    # 数据集统计信息
    logger.info(f"Total processed data: {len(meta_data)}")
    
    # 统计各任务的数据量
    task_counts = defaultdict(int)
    for item in meta_data:
        task_type = item.get("task_type", "unknown")
        task_counts[task_type] += 1
    
    for task_type, count in task_counts.items():
        logger.info(f"Task type: {task_type}, count: {count}")
    
    return meta_data

def process_verify_with_line_data(raw_data, task_type, img_dir):
    """处理带有路径的验证任务数据"""
    processed_data = []
    
    for item in raw_data:
        # 检查item是否包含必要的字段
        if not all(key in item for key in ["id", "image_path", "path_drawings"]):
            logger.warning(f"Missing required keys in item: {item.get('id', 'unknown')}")
            continue
        
        # 获取随机路径数据
        random_path = item["path_drawings"]["random"]
        
        path_string = random_path["path"]
        is_safe = random_path["is_safe"]
        path_image = random_path["image_path"]
        
        # 构建提示
        text_prompt = PATH_VERIFY_WITH_PATH_INSTRUCTION
        # 转换gym地图
        gym_map = convert_to_gym_map(item)
        
        path_image_full = os.path.join(img_dir, path_image) if img_dir else path_image
        path_image_pil = Image.open(path_image_full).convert("RGB")
        
        
        processed_data.append({
            "idx": f"{item['id']}_verify_with_line",
            "original_id": item["id"],
            "images":[path_image_pil],
            "text": text_prompt,
            "answer": "no" if is_safe else "yes", # 注意答案与is_safe相反
            "task_type": task_type,
            "path_length": len(path_string.split(",")) if "," in path_string else 1,
            "path": path_string,
            "gym_map": gym_map
        })
    
    logger.info(f"Processed {len(processed_data)} records for {task_type}")
    return processed_data

def process_locate_start_data(raw_data, task_type, img_dir):
    """处理起点定位任务数据"""
    processed_data = []
    
    for item in raw_data:
        # 检查item是否包含必要的字段
        if not all(key in item for key in ["id", "image_path", "start_coords"]):
            logger.warning(f"Missing required keys in item: {item.get('id', 'unknown')}")
            continue
        
        # 构建提示
        text_prompt = LOCATE_START_POINT_INSTRUCTION
        
        # 获取起点坐标
        start_x, start_y = item["start_coords"]
        
        image_path = item["image_path"]
        image_pil = Image.open(os.path.join(img_dir, image_path) if img_dir else image_path).convert("RGB")
        
        processed_data.append({
            "idx": f"{item['id']}_locate_start",
            "original_id": item["id"],
            "images": [image_pil],
            "text": text_prompt,
            "answer": f"{int(start_x)},{int(start_y)}",  # 标准答案为整数坐标
            "task_type": task_type,
            "start_coords": item["start_coords"]
        })
    
    logger.info(f"Processed {len(processed_data)} records for {task_type}")
    return processed_data

def process_verify_with_coords_data(raw_data, task_type, img_dir):
    """处理带有坐标信息的路径验证任务数据"""
    processed_data = []
    
    for item in raw_data:
        # 检查item是否包含必要的字段
        if not all(key in item for key in ["id", "image_path", "path_drawings", "start_coords", "obstacle_coords"]):
            logger.warning(f"Missing required keys in item: {item.get('id', 'unknown')}")
            continue
        
        # 获取随机路径数据
        random_path = item["path_drawings"]["random"]
        
        path_string = random_path["path"]
        is_safe = random_path["is_safe"]
        
        
        
        
        # 提取坐标信息
        start_coords_pixel = item["start_coords"]
        start_x_pixel, start_y_pixel = int(start_coords_pixel[0]), int(start_coords_pixel[1])
        
        # 转换为绝对坐标（网格坐标）
        start_x_abs = int(start_x_pixel / 64) + 1
        start_y_abs = int(start_y_pixel / 64) + 1
        
        # 提取冰洞坐标
        holes_pixel = item["obstacle_coords"]
        holes_abs = []
        
        for hole_x, hole_y in holes_pixel:
            hole_x_abs = int(hole_x / 64) + 1
            hole_y_abs = int(hole_y / 64) + 1
            holes_abs.append((hole_x_abs, hole_y_abs))
        
        # 构建提示
        text_prompt = PATH_VERIFY_WITH_COORDS_INSTRUCTION.format(
            start_coords_pixel=f"({start_x_pixel}, {start_y_pixel})",
            start_coords_abs=f"({start_x_abs}, {start_y_abs})",
            holes_pixel=", ".join([f"({int(x)}, {int(y)})" for x, y in holes_pixel]),
            holes_abs=", ".join([f"({x}, {y})" for x, y in holes_abs])
        )
        text_prompt = text_prompt.replace("<ACTION-SEQ>", path_string)
        # 转换gym地图
        gym_map = convert_to_gym_map(item)
        
        origin_image_path = item["image_path"]
        origin_image_path_full = os.path.join(img_dir, origin_image_path) if img_dir else origin_image_path
        origin_image_pil = Image.open(origin_image_path_full).convert("RGB")
        

        processed_data.append({
            "idx": f"{item['id']}_verify_with_coords",
            "original_id": item["id"],
            "images": [origin_image_pil],
            "text": text_prompt,
            "answer": "yes" if is_safe else "no",  # 注意答案与is_safe相反
            "task_type": task_type,
            "path_length": len(path_string.split(",")) if "," in path_string else 1,
            "path": path_string,
            "gym_map": gym_map,
            "start_coords": start_coords_pixel,
            "obstacle_coords": holes_pixel
        })
    
    logger.info(f"Processed {len(processed_data)} records for {task_type}")
    return processed_data

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
    size = item.get("size", 5)  # 默认大小为5
    cell_size = 64  # 假设每个单元格是64像素
    
    # 创建一个全是安全区域的地图
    map_data = [['F' for _ in range(size)] for _ in range(size)]
    
    # 设置起点
    if "start_coords" in item:
        start_x = int(item["start_coords"][0] / cell_size)
        start_y = int(item["start_coords"][1] / cell_size)
        if 0 <= start_y < size and 0 <= start_x < size:
            map_data[start_y][start_x] = 'S'
    
    # 设置终点
    if "goal_coords" in item:
        goal_x = int(item["goal_coords"][0] / cell_size)
        goal_y = int(item["goal_coords"][1] / cell_size)
        if 0 <= goal_y < size and 0 <= goal_x < size:
            map_data[goal_y][goal_x] = 'G'
    
    # 设置障碍物
    if "obstacle_coords" in item:
        for obs in item["obstacle_coords"]:
            obs_x = int(obs[0] / cell_size)
            obs_y = int(obs[1] / cell_size)
            # 确保坐标在有效范围内
            if 0 <= obs_y < size and 0 <= obs_x < size:
                map_data[obs_y][obs_x] = 'H'
    
    return map_data

def evaluate_function(results, meta_data):
    """评估函数，根据任务类型对结果进行评估"""
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    
    # 按任务类型统计结果
    task_results = {}
    
    # 按任务类型和路径长度统计结果（仅对路径验证任务）
    path_length_results = {}
    
    # 添加合规输出统计
    compliance_stats = {
        "overall": {"compliant": 0, "total": 0, "correct_if_compliant": 0},
        "verify_with_line": {"compliant": 0, "total": 0, "correct_if_compliant": 0},
        "verify_with_coords": {"compliant": 0, "total": 0, "correct_if_compliant": 0},
        "locate_start": {"compliant": 0, "total": 0, "correct_if_compliant": 0}
    }
    
    overall_correct = 0
    overall_total = 0
    
    compare_logs = []
    
    for idx, meta in meta_dict.items():
        task_type = meta["task_type"]
        
        # 初始化任务类型统计
        if task_type not in task_results:
            task_results[task_type] = {"correct": 0, "total": 0}
        
        # 对于路径验证相关任务，初始化路径长度统计
        if task_type in ["verify_with_line", "verify_with_coords"]:
            path_length = meta.get("path_length", "unknown")
            task_path_key = f"{task_type}_length{path_length}"
            if task_path_key not in path_length_results:
                path_length_results[task_path_key] = {
                    "correct": 0, "total": 0, "task_type": task_type, "path_length": path_length
                }
        
        if idx in results_dict:
            prediction = results_dict[idx]["results"]["final_answer"]
            meta["prediction"] = prediction
        else:
            prediction = None
            meta["prediction"] = None
        
        # 根据任务类型评估结果
        if task_type in ["verify_with_line", "verify_with_coords"]:
            score, message, is_compliant = evaluate_path_validation(prediction, meta)
            
            # 记录路径长度结果
            if task_path_key in path_length_results:
                path_length_results[task_path_key]["correct"] += score
                path_length_results[task_path_key]["total"] += 1
                
            # 更新合规性统计
            compliance_stats[task_type]["total"] += 1
            compliance_stats["overall"]["total"] += 1
            if is_compliant:
                compliance_stats[task_type]["compliant"] += 1
                compliance_stats["overall"]["compliant"] += 1
                if score == 1:  # 如果答案正确
                    compliance_stats[task_type]["correct_if_compliant"] += 1
                    compliance_stats["overall"]["correct_if_compliant"] += 1
        
        elif task_type == "locate_start":
            score, message, is_compliant = evaluate_locate_start(prediction, meta)
            
            # 更新合规性统计
            compliance_stats[task_type]["total"] += 1
            compliance_stats["overall"]["total"] += 1
            if is_compliant:
                compliance_stats[task_type]["compliant"] += 1
                compliance_stats["overall"]["compliant"] += 1
                if score == 1:  # 如果答案正确
                    compliance_stats[task_type]["correct_if_compliant"] += 1
                    compliance_stats["overall"]["correct_if_compliant"] += 1
        else:
            score, message = 0, "Unknown task type"
            is_compliant = False
        
        # 记录任务类型结果
        task_results[task_type]["correct"] += score
        task_results[task_type]["total"] += 1
        
        # 记录总体结果
        overall_correct += score
        overall_total += 1
        
        # 日志记录
        log_entry = {
            "idx": idx,
            "task_type": task_type,
            "gold": meta["answer"],
            "pred": prediction,
            "score": score,
            "message": message,
            "is_compliant": is_compliant
        }
        
        # 为不同任务类型添加特定字段
        if task_type in ["verify_with_line", "verify_with_coords"]:
            log_entry["path_length"] = meta.get("path_length", "unknown")
            log_entry["path"] = meta.get("path", "")
            
            # 为带坐标的验证任务添加坐标信息
            if task_type == "verify_with_coords":
                log_entry["start_coords"] = meta.get("start_coords", [])
                log_entry["obstacle_coords"] = meta.get("obstacle_coords", [])
                
        elif task_type == "locate_start":
            log_entry["start_coords"] = meta.get("start_coords", [])
        
        compare_logs.append(log_entry)
    
    # 计算每个任务类型的准确率
    for task_type in task_results:
        if task_results[task_type]["total"] > 0:
            task_results[task_type]["accuracy"] = task_results[task_type]["correct"] / task_results[task_type]["total"]
        else:
            task_results[task_type]["accuracy"] = 0
    
    # 计算每个路径长度的准确率
    for key in path_length_results:
        if path_length_results[key]["total"] > 0:
            path_length_results[key]["accuracy"] = path_length_results[key]["correct"] / path_length_results[key]["total"]
        else:
            path_length_results[key]["accuracy"] = 0
    
    # 计算合规率和合规答案的准确率
    compliance_rates = {}
    for task_type, stats in compliance_stats.items():
        if stats["total"] > 0:
            compliance_rate = stats["compliant"] / stats["total"]
        else:
            compliance_rate = 0
            
        if stats["compliant"] > 0:
            compliant_accuracy = stats["correct_if_compliant"] / stats["compliant"]
        else:
            compliant_accuracy = 0
            
        compliance_rates[task_type] = {
            "compliance_rate": compliance_rate,
            "compliant_accuracy": compliant_accuracy,
            "compliant_count": stats["compliant"],
            "total_count": stats["total"],
            "correct_if_compliant": stats["correct_if_compliant"]
        }
    
    # 将结果转换为列表，便于按路径长度排序
    path_length_results_list = list(path_length_results.values())
    path_length_results_list.sort(key=lambda x: (x["task_type"], x["path_length"]))
    
    # 计算总体准确率
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    
    result = {
        "overall_accuracy": overall_accuracy,
        "task_results": task_results,
        "path_length_results": path_length_results_list,
        "compliance_stats": compliance_rates,
        "compare_logs": compare_logs
    }
    
    # 打印结果摘要
    logger.info(f"Overall accuracy: {overall_accuracy:.4f}")
    
    for task_type in task_results:
        logger.info(f"Task type {task_type}: {task_results[task_type]['accuracy']:.4f} "
                   f"({task_results[task_type]['correct']}/{task_results[task_type]['total']})")
    
    # 打印按路径长度的结果（仅对路径验证任务）
    for task_type in ["verify_with_line", "verify_with_coords"]:
        if task_type in task_results:
            logger.info(f"Path length breakdown for {task_type} task:")
            for item in path_length_results_list:
                if item["task_type"] == task_type:
                    logger.info(f"  Length {item['path_length']}: {item['accuracy']:.4f} "
                               f"({item['correct']}/{item['total']})")
    
    # 打印合规性统计
    logger.info("\nCompliance statistics:")
    for task_type, stats in compliance_rates.items():
        logger.info(f"{task_type}:")
        logger.info(f"  Compliance rate: {stats['compliance_rate']:.4f} ({stats['compliant_count']}/{stats['total_count']})")
        logger.info(f"  Accuracy if compliant: {stats['compliant_accuracy']:.4f} ({stats['correct_if_compliant']}/{stats['compliant_count']})")
    
    return result

def evaluate_path_validation(prediction, meta):
    """评估路径验证任务"""
    if prediction is None:
        return 0, "No prediction", False
    
    # 预处理预测结果
    yes_pattern = r'\\boxed\s*{\s*yes\s*}|boxed\s*{\s*yes\s*}|boxed{yes}|boxed\(\s*yes\s*\)|yes'
    no_pattern = r'\\boxed\s*{\s*no\s*}|boxed\s*{\s*no\s*}|boxed{no}|boxed\(\s*no\s*\)|no'
    
    prediction = prediction.lower().strip()
    
    # 判断是否合规
    is_compliant = False
    pred_answer = None
    
    # 检查是否只包含yes或no答案
    if prediction in ["yes", "no", "yes.", "no."]:
        is_compliant = True
        pred_answer = "yes" if prediction.startswith("yes") else "no"
    elif re.search(yes_pattern, prediction, re.IGNORECASE) and not re.search(no_pattern, prediction, re.IGNORECASE):
        is_compliant = True
        pred_answer = "yes"
    elif re.search(no_pattern, prediction, re.IGNORECASE) and not re.search(yes_pattern, prediction, re.IGNORECASE):
        is_compliant = True
        pred_answer = "no"
    else:
        # 尝试从最后一段文本中提取答案
        last_paragraph = prediction.split('\n')[-1].strip()
        if "yes" in last_paragraph.lower() and "no" not in last_paragraph.lower():
            pred_answer = "yes"
        elif "no" in last_paragraph.lower() and "yes" not in last_paragraph.lower():
            pred_answer = "no"
        else:
            return 0, "Invalid prediction format: cannot determine yes/no", False
    
    # 获取正确答案
    gold_answer = meta["answer"].lower().strip()
    
    # 比较答案
    if pred_answer == gold_answer.lower():
        return 1, "Correct", is_compliant
    else:
        return 0, f"Incorrect. Expected: {gold_answer}, Got: {pred_answer}", is_compliant

def evaluate_locate_start(prediction, meta):
    """评估起点定位任务"""
    if prediction is None:
        return 0, "No prediction", False
    
    # 尝试从预测中提取坐标
    coord_pattern = r'\\boxed\s*{\s*(\d+)\s*,\s*(\d+)\s*}|boxed\s*{\s*(\d+)\s*,\s*(\d+)\s*}|(\d+)\s*,\s*(\d+)'
    matches = re.search(coord_pattern, prediction)
    
    # 判断是否合规（是否有有效的坐标格式）
    is_compliant = matches is not None
    
    if not is_compliant:
        return 0, "No valid coordinates found in prediction", False
    
    # 提取预测的x,y坐标
    groups = matches.groups()
    for i in range(0, len(groups), 2):
        if groups[i] is not None and groups[i+1] is not None:
            pred_x = int(groups[i])
            pred_y = int(groups[i+1])
            break
    else:
        return 0, "Could not parse coordinates from prediction", False
    
    # 获取正确答案
    true_x, true_y = meta["start_coords"]
    true_x, true_y = int(true_x), int(true_y)
    
    # 允许±32像素的误差
    tolerance = 32
    

    distance_x = math.sqrt((pred_x - true_x)**2)
    distance_y = math.sqrt((pred_y - true_y)**2)
    
    if distance_x <= tolerance and distance_y <= tolerance:
        qualify = True
    else:
        pred_x = pred_x*64 - 32
        pred_y = pred_y*64 - 32
        distance_x = math.sqrt((pred_x - true_x)**2)
        distance_y = math.sqrt((pred_y - true_y)**2)
        if distance_x <= tolerance and distance_y <= tolerance:
            qualify = True
        else:
            qualify = False
    
    if qualify:
        return 1, f"Correct within tolerance. Distance X: {distance_x:.2f}px Distance Y: {distance_y:.2f}px", True
    else:
        return 0, f"Incorrect. Expected: {true_x},{true_y}, Got: {pred_x},{pred_y}, Distance X: {distance_x:.2f}px Distance Y: {distance_y:.2f}px", True