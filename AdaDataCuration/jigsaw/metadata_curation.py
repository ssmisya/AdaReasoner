# metadata_curation.py
import os
import sys
import json
import uuid
import random
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil
import io
import base64
import ast
import copy
import string

# 导入工具管理器
from tool_server.tool_workers.tool_manager.base_manager import ToolManager

# 初始化工具管理器
tool_manager = ToolManager()

def image_to_base64(image):
    """将PIL图像或图像路径转换为Base64字符串"""
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

def split_image_into_grid(image, rows=3, cols=3):
    """
    将图像分割成网格
    
    Args:
        image: PIL图像对象
        rows: 行数
        cols: 列数
        
    Returns:
        list: 包含所有子图的二维列表 [row][col]
    """
    width, height = image.size
    cell_width = width // cols
    cell_height = height // rows
    
    grid = []
    for i in range(rows):
        row = []
        for j in range(cols):
            left = j * cell_width
            upper = i * cell_height
            right = left + cell_width
            lower = upper + cell_height
            
            # 提取子图
            cell = image.crop((left, upper, right, lower))
            row.append(cell)
        grid.append(row)
    
    return grid

def select_random_2x2_region(grid, rows=3, cols=3):
    """
    从网格中随机选择一个2x2区域
    
    Args:
        grid: 图像网格
        rows: 网格的行数
        cols: 网格的列数
        
    Returns:
        tuple: (起始行, 起始列, 2x2子网格)
    """
    # 随机选择起始点（确保有足够空间选择2x2区域）
    start_row = random.randint(0, rows - 2)
    start_col = random.randint(0, cols - 2)
    
    # 提取2x2区域
    subgrid = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append(grid[start_row + i][start_col + j])
        subgrid.append(row)
    
    return start_row, start_col, subgrid

def extract_cutout_and_piece(subgrid, split_type="sft"):
    """
    从2x2区域中选择位置进行裁剪，根据分割类型选择不同的裁剪位置
    
    Args:
        subgrid: 2x2子网格
        split_type: 数据集分割类型，"sft"、"rl"或"test"
        
    Returns:
        tuple: (裁剪后的图像, 被裁剪的部分, 裁剪位置坐标)
    """
    if split_type == "test":
        # 测试集只在右下角裁剪
        row, col = 1, 1  # 右下角
    else:
        # SFT和RL集在左上、左下或右上裁剪
        positions = [(0, 0), (1, 0), (0, 1)]  # 左上、左下、右上
        row, col = random.choice(positions)
    
    # 创建一个新的2x2图像，从原始子网格拼接
    base_width = subgrid[0][0].width
    base_height = subgrid[0][0].height
    base_image = Image.new('RGB', (base_width * 2, base_height * 2))
    
    # 添加所有部分到基础图像
    for i in range(2):
        for j in range(2):
            base_image.paste(subgrid[i][j], (j * base_width, i * base_height))
    
    # 保存被裁剪的部分
    cutout_piece = subgrid[row][col]
    
    # 创建裁剪后的图像（将裁剪位置设为黑色）
    cutout_image = base_image.copy()
    # 用黑色填充裁剪区域
    black_block = Image.new('RGB', (base_width, base_height), (0, 0, 0))
    cutout_image.paste(black_block, (col * base_width, row * base_height))
    
    # 裁剪位置的坐标
    cutout_coords = [col * base_width, row * base_height, (col+1) * base_width, (row+1) * base_height]
    
    return cutout_image, cutout_piece, cutout_coords

def select_distractors(grid, start_row, start_col, rows=3, cols=3, num_distractors=2):
    """
    从网格中选择指定数量不在选定2x2区域内的子图作为干扰项
    
    Args:
        grid: 图像网格
        start_row: 选定2x2区域的起始行
        start_col: 选定2x2区域的起始列
        rows: 网格的行数
        cols: 网格的列数
        num_distractors: 要选择的干扰项数量
        
    Returns:
        list: 包含干扰子图的列表
    """
    # 列出不在2x2区域内的所有子图
    outside_cells = []
    for i in range(rows):
        for j in range(cols):
            if not (start_row <= i < start_row + 2 and start_col <= j < start_col + 2):
                outside_cells.append((i, j))
    
    # 随机选择指定数量的不同子图
    selected_coords = random.sample(outside_cells, min(num_distractors, len(outside_cells)))
    distractors = [grid[i][j] for i, j in selected_coords]
    
    return distractors

def call_detect_black_area(image_path, threshold=10, min_area=100):
    """
    调用DetectBlackArea工具检测图像中的黑色区域，并记录输入参数
    
    Args:
        image_path: 图像路径
        threshold: 亮度阈值
        min_area: 最小区域大小
        
    Returns:
        tuple: (工具输入参数, 工具响应结果)
    """
    params = {
        "image": image_path,
        "threshold": threshold,
        "min_area": min_area
    }
    
    # 记录工具输入
    tool_input = copy.deepcopy(params)
    
    # 调用工具
    result = tool_manager.call_tool("DetectBlackArea", params)
    
    # 移除不需要的字段
    if "edited_image" in result:
        del result["edited_image"]
    if "tool_reward" in result:
        del result["tool_reward"]
    
    return tool_input, result

def compare_bounding_boxes(gt_bbox, detected_bbox, tolerance=10):
    """
    比较真实边界框和检测到的边界框是否在容差范围内
    
    Args:
        gt_bbox: 真实边界框 [x_min, y_min, x_max, y_max]
        detected_bbox: 检测到的边界框 [x_min, y_min, x_max, y_max]
        tolerance: 允许的最大像素误差
        
    Returns:
        tuple: (是否一致, 差异值)
    """
    if isinstance(gt_bbox, str):
        gt_bbox = ast.literal_eval(gt_bbox)
    if isinstance(detected_bbox, str):
        detected_bbox = ast.literal_eval(detected_bbox)
    
    # 计算各边界点的差异
    diff_x_min = abs(gt_bbox[0] - detected_bbox[0])
    diff_y_min = abs(gt_bbox[1] - detected_bbox[1])
    diff_x_max = abs(gt_bbox[2] - detected_bbox[2])
    diff_y_max = abs(gt_bbox[3] - detected_bbox[3])
    
    # 计算最大差异
    max_diff = max(diff_x_min, diff_y_min, diff_x_max, diff_y_max)
    
    # 如果所有差异都在容差范围内，认为一致
    is_consistent = max_diff <= tolerance
    
    return is_consistent, max_diff

def call_insert_image(base_image, insert_image_path, coordinates, resize=True):
    """
    调用InsertImage工具将图像插入到另一个图像中，并记录输入参数
    
    Args:
        base_image: 基础图像（可以是路径或base64字符串）
        insert_image_path: 要插入的图像路径
        coordinates: 插入坐标
        resize: 是否调整大小
        
    Returns:
        tuple: (工具输入参数, 工具响应结果, 生成的图像)
    """
    params = {
        "base_image": base_image,
        "image_to_insert": insert_image_path,
        "coordinates": coordinates,
        "resize": resize
    }
    
    # 记录工具输入（创建深拷贝以避免修改原始参数）
    tool_input = copy.deepcopy(params)
    # 避免在输入记录中保存大型base64字符串
    if isinstance(tool_input["base_image"], str) and len(tool_input["base_image"]) > 200:
        tool_input["base_image"] = "base64_image_data"
    
    # 调用工具
    result = tool_manager.call_tool("InsertImage", params)
    
    # 获取生成的图像（如果存在）
    generated_image = None
    if result["status"] == "success" and "edited_image" in result:
        generated_image = base64_to_image(result["edited_image"])
    
    # 创建结果的副本以便在返回前修改
    result_copy = copy.deepcopy(result)
    
    # 移除不需要的字段
    if "edited_image" in result_copy:
        del result_copy["edited_image"]
    if "tool_reward" in result_copy:
        del result_copy["tool_reward"]
    
    return tool_input, result_copy, generated_image

def generate_question_text(corner_position="lower right", choices_count=2):
    """
    生成问题文本
    
    Args:
        corner_position: 缺失的角落位置描述
        choices_count: 选项数量
        
    Returns:
        str: 问题文本
    """
    number_num = ["zero","first", "second", "third", "fourth","fifth"]
    question = f"Given the first image (img_1) with one part missing, can you tell which one of the "
    # Your final answer should be formatted as \\boxed{Yes} or \\boxed{No}.
    
    if choices_count == 2:
        question += "second image (img_2) or the third image (img_3) is the missing part? "
    else:
        options = ["second (img_2)", "third (img_3)", "fourth (img_4)"][:choices_count]
        options_text = ", ".join(options[:-1]) + " or the " + options[-1] if len(options) > 1 else options[0]
        question += f"{options_text} images is the missing part? "
    
    question += "Imagine which image would be more appropriate to place in the missing spot. "
    question += "You can also carefully observe and compare the edges of the images.\n\n"
    question += "Select from the following choices.\n\n"
    
    # 添加选项
    for i in range(choices_count):
        letter = string.ascii_uppercase[i]
        img_num = i + 2
        question += f"({letter}) The {number_num[img_num]} image (img_{img_num})\n"
    
    question += "Your final answer should be formatted as \\boxed{Your Choice}, for example, \\boxed{A} or \\boxed{B} or \\boxed{C}."
    return question

def create_choices_and_answer(correct_piece, distractors):
    """
    创建选项和答案
    
    Args:
        correct_piece: 正确的图像片段
        distractors: 干扰项列表
        
    Returns:
        tuple: (所有选项列表, 正确答案索引, 正确答案字母)
    """
    # 创建选项列表，包含正确答案和干扰项
    all_choices = [correct_piece] + distractors
    
    # 随机打乱选项顺序，确保答案位置随机
    correct_index = 0  # 初始正确答案在索引0
    random.shuffle(all_choices)
    
    # 找出正确答案现在的位置
    for i, choice in enumerate(all_choices):
        if choice is correct_piece:
            correct_index = i
            break
    
    # 正确答案对应的字母
    correct_letter = string.ascii_uppercase[correct_index]
    
    return all_choices, correct_index, correct_letter

def process_image(image_path, output_dir, prefix, img_id, split_type="sft", use_two_choices=False, tolerance=10):
    """
    处理单个图像并生成相应的数据
    
    Args:
        image_path: 图像路径
        output_dir: 输出目录
        prefix: ID前缀
        img_id: 图像ID
        split_type: 数据集分割类型，"sft"、"rl"或"test"
        use_two_choices: 是否只使用两个选项
        tolerance: 边界框比较的容差（像素）
        
    Returns:
        tuple: (元数据字典, 生成的图像路径字典, 工具是否失败)
    """
    # 加载图像
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"无法打开图像 {image_path}: {e}")
        return None, None, False
    
    # 分割图像为3x3网格
    grid = split_image_into_grid(image)
    
    # 随机选择2x2区域
    start_row, start_col, subgrid = select_random_2x2_region(grid)
    
    # 提取裁剪后的图像和被裁剪的部分，根据数据集类型决定裁剪位置
    cutout_image, cutout_piece, cutout_coords = extract_cutout_and_piece(subgrid, split_type)
    
    # 确定干扰项数量
    num_distractors = 1 if use_two_choices else 2
    
    # 选择干扰项
    distractors = select_distractors(grid, start_row, start_col, num_distractors=num_distractors)
    
    # 确保有足够的干扰项
    while len(distractors) < num_distractors:
        # 如果没有足够的干扰项，创建一个随机颜色的图像作为填充
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        random_distractor = Image.new('RGB', cutout_piece.size, random_color)
        distractors.append(random_distractor)
    
    # 创建此条目的子目录 - 修改为新的目录结构
    item_id = f"{prefix}_{split_type}_{img_id}"
    item_dir = os.path.join(output_dir, "images", split_type, item_id)
    os.makedirs(item_dir, exist_ok=True)
    
    # 保存所有生成的图像文件的路径
    image_paths = {}
    
    # 保存问题图像
    question_path = os.path.join(item_dir, f"{item_id}_question.jpg")
    cutout_image.save(question_path)
    image_paths["question"] = question_path
    
    # 创建选项和确定答案
    all_choices, correct_index, correct_letter = create_choices_and_answer(cutout_piece, distractors)
    
    # 保存所有选项图像
    choice_paths = []
    for i, choice in enumerate(all_choices):
        choice_path = os.path.join(item_dir, f"{item_id}_choice_{i+1}.jpg")
        choice.save(choice_path)
        choice_paths.append(choice_path)
    
    image_paths["choices"] = choice_paths
    
    # 确定缺失角落的位置描述
    corner_position = "lower right" if split_type == "test" else "corner"
    
    # 生成问题文本
    question_text = generate_question_text(corner_position, len(all_choices))
    
    # 调用DetectBlackArea工具检测黑色区域
    detect_input, detect_result = call_detect_black_area(question_path)
    
    # 初始化工具失败标志
    tool_failed = False
    
    # 提取检测到的边界框
    detected_bbox = None
    if detect_result["status"] == "success" and detect_result.get("bounding_boxes"):
        detected_bbox = detect_result["bounding_boxes"][0]
        
        # 比较真实边界框和检测到的边界框
        is_consistent, max_diff = compare_bounding_boxes(cutout_coords, detected_bbox, tolerance)
        
        # 设置工具失败标志
        tool_failed = not is_consistent
        

    else:
        # 如果检测完全失败（没有返回边界框），也标记为失败
        tool_failed = True
        detected_bbox = str(cutout_coords)  # 使用真实坐标作为后备
    
    # 使用base64格式的问题图像，这样可以在多次调用InsertImage时重用
    question_base64 = image_to_base64(question_path)
    
    # 对每个选项调用InsertImage工具
    insert_inputs = []
    insert_results = []
    inserted_images = []
    
    # 使用实际检测到的边界框作为插入坐标
    bbox_for_insert = detected_bbox
    
    for choice_path in choice_paths:
        insert_input, insert_result, inserted_image = call_insert_image(question_path, choice_path, bbox_for_insert)
        insert_inputs.append(insert_input)
        insert_results.append(insert_result)
        inserted_images.append(inserted_image)
    
    # 保存插入结果图像
    image_paths["inserted_images"] = []
    for i, inserted_image in enumerate(inserted_images):
        if inserted_image:
            inserted_path = os.path.join(item_dir, f"{item_id}_choice_{i+1}_inserted.jpg")
            inserted_image.save(inserted_path)
            image_paths["inserted_images"].append(inserted_path)
    
    # 创建元数据
    metadata = {
        "id": item_id,
        "split": split_type,
        "original_image": str(image_path),
        "question_image": question_path,
        "question_text": question_text,
        "choices": [
            {"image": path} for path in choice_paths
        ],
        "correct_answer": {
            "index": correct_index,
            "letter": correct_letter
        },
        "inserted_images": [
            os.path.join(item_dir, f"{item_id}_choice_{i+1}_inserted.jpg") 
            for i in range(len(choice_paths)) if inserted_images[i]
        ],
        "tools": {
            "detect_black_area": {
                "input": detect_input,
                "output": detect_result
            },
            "insert_images": [
                {
                    "input": insert_inputs[i],
                    "output": result
                }
                for i, result in enumerate(insert_results)
            ],
            "tool_failed": tool_failed  # 添加工具失败标志
        }
    }
    
    return metadata, image_paths, tool_failed

def generate_dataset(input_dir, output_dir, prefix="puzzle", sft_count=50, rl_count=50, test_count=20, tolerance=10):
    """
    为输入目录中的图像生成数据集，分为SFT、RL和TEST三个分割
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出目录
        prefix: ID前缀
        sft_count: SFT分割的样本数量
        rl_count: RL分割的样本数量
        test_count: TEST分割的样本数量
        tolerance: 边界框比较的容差（像素）
        
    Returns:
        dict: 包含各分割数据集的字典
    """
    # 确保输出目录结构存在
    for split in ["sft", "rl", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "splits", split), exist_ok=True)
    
    # 图像文件扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    # 获取所有图像文件路径
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(input_dir).glob(f"*{ext}")))
    
    # 计算需要的总图像数
    total_needed = sft_count + rl_count + test_count
    
    # 如果图像文件不够，则报错
    if len(image_files) < total_needed:
        raise ValueError(f"输入目录中只有{len(image_files)}个图像文件，但需要{total_needed}个")
    
    # 随机打乱并选择所需数量的图像
    random.shuffle(image_files)
    selected_images = image_files[:total_needed]
    
    # 分割图像集
    sft_images = selected_images[:sft_count]
    rl_images = selected_images[sft_count:sft_count+rl_count]
    test_images = selected_images[sft_count+rl_count:total_needed]
    
    # 检查必要的工具是否可用
    required_tools = ["DetectBlackArea", "InsertImage"]
    for tool in required_tools:
        if tool not in tool_manager.available_tools:
            raise ValueError(f"所需工具 '{tool}' 不可用，请确保工具已正确加载")
    
    # 创建各分割的数据集
    datasets = {
        "sft": [],
        "rl": [],
        "test": []
    }
    
    # 工具失败统计
    failure_stats = {
        "sft": {"count": 0, "total": 0},
        "rl": {"count": 0, "total": 0},
        "test": {"count": 0, "total": 0}
    }
    
    # 处理SFT分割的图像
    print(f"处理SFT分割 ({sft_count}个样本)...")
    for idx, image_path in enumerate(tqdm(sft_images)):
        # 一半数据使用2个选项，一半使用3个选项
        use_two_choices = idx < sft_count // 2
        img_id = f"{idx:04d}_{image_path.stem}"
        metadata, _, tool_failed = process_image(
            image_path, output_dir, prefix, img_id, "sft", use_two_choices, tolerance
        )
        if metadata:
            datasets["sft"].append(metadata)
            failure_stats["sft"]["total"] += 1
            if tool_failed:
                failure_stats["sft"]["count"] += 1
    
    # 处理RL分割的图像
    print(f"处理RL分割 ({rl_count}个样本)...")
    for idx, image_path in enumerate(tqdm(rl_images)):
        # 一半数据使用2个选项，一半使用3个选项
        use_two_choices = idx < rl_count // 2
        img_id = f"{idx:04d}_{image_path.stem}"
        metadata, _, tool_failed = process_image(
            image_path, output_dir, prefix, img_id, "rl", use_two_choices, tolerance
        )
        if metadata:
            datasets["rl"].append(metadata)
            failure_stats["rl"]["total"] += 1
            if tool_failed:
                failure_stats["rl"]["count"] += 1
    
    # 处理TEST分割的图像
    print(f"处理TEST分割 ({test_count}个样本)...")
    for idx, image_path in enumerate(tqdm(test_images)):
        # 一半数据使用2个选项，一半使用3个选项
        use_two_choices = idx < test_count // 2
        img_id = f"{idx:04d}_{image_path.stem}"
        metadata, _, tool_failed = process_image(
            image_path, output_dir, prefix, img_id, "test", use_two_choices, tolerance
        )
        if metadata:
            datasets["test"].append(metadata)
            failure_stats["test"]["total"] += 1
            if tool_failed:
                failure_stats["test"]["count"] += 1
    
    # 计算总失败率
    total_failures = sum(stat["count"] for stat in failure_stats.values())
    total_samples = sum(stat["total"] for stat in failure_stats.values())
    failure_rate = total_failures / total_samples if total_samples > 0 else 0
    
    # 创建统计信息
    stats = {
        "failure_stats": {
            "overall": {
                "count": total_failures,
                "total": total_samples,
                "rate": failure_rate
            }
        }
    }
    
    # 为每个分割添加失败率
    for split, stat in failure_stats.items():
        rate = stat["count"] / stat["total"] if stat["total"] > 0 else 0
        stats["failure_stats"][split] = {
            "count": stat["count"],
            "total": stat["total"],
            "rate": rate
        }
    
    # 保存统计信息
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 保存各分割的数据集到新的目录结构
    for split, dataset in datasets.items():
        # 保存数据集元数据到JSON文件
        metadata_path = os.path.join(output_dir, "splits", split, f"dataset_{split}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        # 同时保存为JSONL格式，便于大数据集处理
        jsonl_path = os.path.join(output_dir, "splits", split, f"dataset_{split}.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 创建示例数据集（随机选择10个样本或全部如果少于10个）
        example_size = min(10, len(dataset))
        if example_size > 0:
            example_dataset = random.sample(dataset, example_size)
            example_path = os.path.join(output_dir, "splits", split, f"example_dataset_{split}.json")
            with open(example_path, 'w', encoding='utf-8') as f:
                json.dump(example_dataset, f, indent=2, ensure_ascii=False)
    
    # 创建完整数据集合并文件，保存在最外层
    all_dataset = datasets["sft"] + datasets["rl"] + datasets["test"]
    complete_path = os.path.join(output_dir, "dataset_complete.json")
    with open(complete_path, 'w', encoding='utf-8') as f:
        json.dump(all_dataset, f, indent=2, ensure_ascii=False)
    
    complete_jsonl_path = os.path.join(output_dir, "dataset_complete.jsonl")
    with open(complete_jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return datasets, stats

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成图像拼图数据集")
    parser.add_argument("--input_dir", type=str, default="/mnt/petrelfs/share_data/songmingyang/data/mm/imgs/coco/train2017", help="输入图像目录")
    parser.add_argument("--output_dir", type=str, default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/jigsaw/jigsaw_metadata_v1", help="输出目录")
    parser.add_argument("--prefix", type=str, default="COCO", help="ID前缀")
    parser.add_argument("--sft_count", type=int, default=1000, help="SFT分割的样本数量")
    parser.add_argument("--rl_count", type=int, default=5000, help="RL分割的样本数量")
    parser.add_argument("--test_count", type=int, default=1000, help="TEST分割的样本数量")
    parser.add_argument("--tolerance", type=int, default=10, help="边界框比较的容差（像素）")
    
    args = parser.parse_args()
    
    # 生成数据集
    datasets, stats = generate_dataset(
        args.input_dir,
        args.output_dir,
        args.prefix,
        args.sft_count,
        args.rl_count,
        args.test_count,
        args.tolerance
    )
    
    # 显示统计信息
    overall_stats = stats["failure_stats"]["overall"]
    print("\n===== 工具检测统计 =====")
    print(f"总样本数: {overall_stats['total']}")
    print(f"工具失败数: {overall_stats['count']}")
    print(f"失败率: {overall_stats['rate']*100:.2f}%")
    print("\n分割统计:")
    for split, split_stats in stats["failure_stats"].items():
        if split != "overall":
            print(f"- {split.upper()}: {split_stats['count']}/{split_stats['total']} 失败 ({split_stats['rate']*100:.2f}%)")
    
    # 统计各分割的样本数
    counts = {split: len(dataset) for split, dataset in datasets.items()}
    total_count = sum(counts.values())
    
    print(f"\n数据集生成完成，共 {total_count} 个条目")
    for split, count in counts.items():
        print(f"- {split.upper()} 分割: {count} 个样本")
    print(f"数据保存在: {args.output_dir}")
    print(f"目录结构:")
    print(f"- 图像文件: {os.path.join(args.output_dir, 'images/[split]')}")
    print(f"- 分割数据: {os.path.join(args.output_dir, 'splits/[split]')}")
    print(f"- 完整数据集: {os.path.join(args.output_dir, 'dataset_complete.json')}")
    print(f"- 统计信息: {os.path.join(args.output_dir, 'stats.json')}")

if __name__ == "__main__":
    main()