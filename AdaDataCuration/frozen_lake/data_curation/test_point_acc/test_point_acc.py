import os
import json
import math
import time
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import base64
import io
from pathlib import Path
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# 初始化工具管理器
tool_manager = ToolManager()

image_output_dir = "/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/test_point_acc/res/images"

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

def process_jsonl(file_path):
    """读取JSONL文件并返回列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    print(f"Error parsing line: {line[:50]}...")
    return data

def call_point_tool(image_path, description="Elf",):
    """
    调用Point工具定位图像中的特定元素
    
    Args:
        image_path: 图像路径
        description: 要定位的元素描述
    
    Returns:
        dict: 包含成功状态和点坐标的结果
    """
    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image file not found: {image_path}", "points": []}
    
    # 加载图像
    try:
        img = Image.open(image_path).convert("RGB")
        img_base64 = image_to_base64(img)
    except Exception as e:
        return {"success": False, "error": f"Error loading image {image_path}: {e}", "points": []}
    
    # 调用Point工具的参数
    params = {
        "image": img_base64,
        "description": description
    }
    
    # 确保Point工具可用
    if "Point" not in tool_manager.available_tools:
        return {"success": False, "error": "Point tool is not available!", "points": []}
    
    # 调用Point工具
    try:
        result = tool_manager.call_tool("Point", params)
        
        # 检查调用是否成功
        if result.get('status') != "success":
            return {"success": False, "error": f"Point tool call failed: {result.get('message', 'Unknown error')}", "points": []}
        
        # 提取坐标
        points = result.get('points', [])
        edited_image = result["edited_image"]
        edited_image = base64_to_image(edited_image)
        
        new_image_path = os.path.join(image_output_dir, os.path.basename(image_path))
        edited_image.save(new_image_path)
        return {"success": True, "points": points, "error": None}
    except Exception as e:
        return {"success": False, "error": f"Error calling Point tool: {e}", "points": []}

def process_sample(item, img_base_dir, tolerance):
    """
    处理单个样本（用于并发处理）
    
    Args:
        item: 数据项
        img_base_dir: 图像基础目录
        tolerance: 坐标误差容忍度（像素）
    
    Returns:
        dict: 处理结果
    """
    sample_id = item.get("id", "unknown")
    
    # 获取图像路径
    image_path = item.get("image_path")
    if not image_path or not os.path.isabs(image_path):
        image_path = os.path.join(img_base_dir, image_path)
    
    if not os.path.exists(image_path):
        return {
            "id": sample_id,
            "success": False,
            "error": f"Image file not found: {image_path}",
            "map_size": item.get("size", "unknown")
        }
    
    # 获取地图大小
    map_size = item.get("size", "unknown")
    
    # 获取真实起点坐标
    true_coords = item.get("start_coords")
    if not true_coords:
        return {
            "id": sample_id,
            "success": False,
            "error": f"No start coordinates found for sample {sample_id}",
            "map_size": map_size
        }
    
    true_x, true_y = true_coords
    
    # 调用Point工具定位Elf
    tool_result = call_point_tool(image_path, "Elf")
    
    if not tool_result["success"] or not tool_result["points"]:
        return {
            "id": sample_id,
            "success": False,
            "error": tool_result.get("error", "No points detected"),
            "map_size": map_size
        }
    
    if map_size == 9:
        true_x = true_x *8 /9
        true_y = true_y *8 /9
    
    detected_points = tool_result["points"]
    detected_point = detected_points[0]
    detected_x = detected_point.get("x", 0)
    detected_y = detected_point.get("y", 0)
    
    # 计算距离
    distance_x = abs(detected_x - true_x)
    distance_y = abs(detected_y - true_y)
    
    # 检查是否在容忍度范围内
    is_correct = distance_x <= tolerance and distance_y <= tolerance
    
    return {
        "id": sample_id,
        "success": True,
        "map_size": map_size,
        "true_coords": [true_x, true_y],
        "detected_coords": [detected_x, detected_y],
        "distance_x": distance_x,
        "distance_y": distance_y,
        "is_correct": is_correct
    }

def evaluate_point_accuracy(data_file, img_base_dir=".", tolerance=32, max_samples=None, verbose=True, num_threads=32,output_image_dir=None):
    """
    评估Point工具的准确性（使用多线程并发）
    
    Args:
        data_file: JSONL数据文件路径
        img_base_dir: 图像基础目录
        tolerance: 坐标误差容忍度（像素）
        max_samples: 最大测试样本数，None表示全部测试
        verbose: 是否输出详细信息
        num_threads: 并发线程数
    
    Returns:
        dict: 评估结果
    """
    # 读取数据
    print(f"Reading data from {data_file}...")
    data = process_jsonl(data_file)
    data = [item for item in data if item["size"] == 9]
    if max_samples and max_samples < len(data):
        print(f"Limiting evaluation to {max_samples} samples (from total {len(data)})")
        # 随机选择样本
        np.random.seed(42)  # 固定随机种子以确保可重复性
        data = np.random.choice(data, max_samples, replace=False).tolist()
    
    total_samples = len(data)
    print(f"Total samples: {total_samples}")
    print(f"Using {num_threads} threads for concurrent processing")
    
    # 初始化统计
    stats = {
        "total": total_samples,
        "successful_calls": 0,
        "correct": 0,
        "errors": []
    }
    
    # 按地图大小统计
    size_stats = {}
    
    # 详细日志
    logs = []
    
    # 使用ThreadPoolExecutor进行并发处理
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(process_sample, item, img_base_dir, tolerance): item
            for item in data
        }
        
        # 使用tqdm创建进度条
        for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Processing samples"):
            item = future_to_item[future]
            
            try:
                result = future.result()
                
                # 获取地图大小并确保它在统计中
                map_size = result["map_size"]
                if map_size not in size_stats:
                    size_stats[map_size] = {"total": 0, "successful_calls": 0, "correct": 0}
                
                size_stats[map_size]["total"] += 1
                
                if not result["success"]:
                    # 处理失败的情况
                    stats["errors"].append({
                        "id": result["id"],
                        "error": result.get("error", "Unknown error")
                    })
                    if verbose:
                        print(f"Error processing {result['id']}: {result.get('error')}")
                    continue
                
                # 处理成功的情况
                stats["successful_calls"] += 1
                size_stats[map_size]["successful_calls"] += 1
                
                # 记录检测准确性
                if result["is_correct"]:
                    stats["correct"] += 1
                    size_stats[map_size]["correct"] += 1
                elif verbose:
                    print(f"Incorrect detection for {result['id']}:")
                    print(f"  True: ({result['true_coords'][0]}, {result['true_coords'][1]}), " +
                          f"Detected: ({result['detected_coords'][0]}, {result['detected_coords'][1]})")
                    print(f"  Distance: X={result['distance_x']:.2f}, Y={result['distance_y']:.2f}")
                
                # 添加到日志
                logs.append(result)
                
            except Exception as e:
                # 处理异常情况
                stats["errors"].append({
                    "id": item.get("id", "unknown"),
                    "error": str(e)
                })
                if verbose:
                    print(f"Exception processing {item.get('id', 'unknown')}: {str(e)}")
    
    # 计算处理时间
    processing_time = time.time() - start_time
    stats["processing_time"] = processing_time
    stats["avg_time_per_sample"] = processing_time / total_samples if total_samples > 0 else 0
    
    # 计算准确率
    if stats["successful_calls"] > 0:
        stats["accuracy"] = stats["correct"] / stats["successful_calls"]
        stats["call_success_rate"] = stats["successful_calls"] / stats["total"]
    else:
        stats["accuracy"] = 0
        stats["call_success_rate"] = 0
    
    # 计算每个地图大小的准确率
    for size in size_stats:
        if size_stats[size]["successful_calls"] > 0:
            size_stats[size]["accuracy"] = size_stats[size]["correct"] / size_stats[size]["successful_calls"]
        else:
            size_stats[size]["accuracy"] = 0
    
    stats["size_stats"] = size_stats
    
    # 输出结果
    print("\n===== Point Tool Accuracy Evaluation =====")
    print(f"Total samples: {stats['total']}")
    print(f"Successful tool calls: {stats['successful_calls']} ({stats['call_success_rate']:.2%})")
    print(f"Correct detections: {stats['correct']} ({stats['accuracy']:.2%})")
    print(f"Processing time: {processing_time:.2f} seconds (avg: {stats['avg_time_per_sample']:.4f} sec/sample)")
    print("\nAccuracy by map size:")
    
    for size in sorted(size_stats.keys()):
        size_data = size_stats[size]
        accuracy = size_data.get("accuracy", 0)
        print(f"  Size {size}: {accuracy:.2%} ({size_data['correct']}/{size_data['successful_calls']})")
    
    # 保存结果
    output_dir = "point_accuracy_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 保存完整结果（包括所有日志）
    full_results = {
        "stats": stats,
        "logs": logs
    }
    
    results_file = os.path.join(output_dir, f"point_accuracy_{timestamp}.json")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nDetailed results saved to {results_file}")
    
    # 绘制准确率图表
    plt.figure(figsize=(12, 10))
    
    # 按地图大小的准确率
    sizes = sorted(size_stats.keys())
    accuracies = [size_stats[size]["accuracy"] for size in sizes]
    
    plt.subplot(2, 2, 1)
    plt.bar(sizes, accuracies)
    plt.xlabel('Map Size')
    plt.ylabel('Accuracy')
    plt.title('Point Tool Accuracy by Map Size')
    plt.ylim(0, 1)
    
    # 绘制距离分布直方图
    distances_x = [log["distance_x"] for log in logs]
    distances_y = [log["distance_y"] for log in logs]
    
    plt.subplot(2, 2, 2)
    plt.hist(distances_x, alpha=0.5, bins=20, label='X Distance')
    plt.hist(distances_y, alpha=0.5, bins=20, label='Y Distance')
    plt.axvline(x=tolerance, color='r', linestyle='--', label=f'Tolerance ({tolerance}px)')
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Count')
    plt.title('Distance Distribution')
    plt.legend()
    
    # 绘制坐标差异散点图
    plt.subplot(2, 2, 3)
    x_diffs = [log["detected_coords"][0] - log["true_coords"][0] for log in logs]
    y_diffs = [log["detected_coords"][1] - log["true_coords"][1] for log in logs]
    
    plt.scatter(x_diffs, y_diffs, alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=tolerance, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=-tolerance, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=tolerance, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=-tolerance, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('X Coordinate Difference')
    plt.ylabel('Y Coordinate Difference')
    plt.title('Coordinate Difference Distribution')
    
    # 绘制成功率
    plt.subplot(2, 2, 4)
    success_rates = [size_stats[size]["successful_calls"] / size_stats[size]["total"] for size in sizes]
    
    plt.bar(sizes, success_rates)
    plt.xlabel('Map Size')
    plt.ylabel('Success Rate')
    plt.title('Point Tool Success Rate by Map Size')
    plt.ylim(0, 1)
    
    # 保存图表
    chart_file = os.path.join(output_dir, f"point_accuracy_chart_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(chart_file)
    print(f"Accuracy charts saved to {chart_file}")
    
    return stats, logs

def main():
    parser = argparse.ArgumentParser(description="Evaluate Point tool accuracy for detecting Elfs in FrozenLake environments")
    parser.add_argument("--data", type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/metadata_split/path_verify/test.jsonl", help="Path to JSONL data file")
    parser.add_argument("--img_dir", type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation", help="Base directory for images")
    parser.add_argument("--tolerance", type=int, default=32, help="Coordinate error tolerance in pixels")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--threads", type=int, default=32, help="Number of threads to use for concurrent processing")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--output", type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/test_point_acc/res", help="Output directory for results")
    parser.add_argument("--output_image_dir", type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/test_point_acc/res/images", help="Output directory for results")
    
    args = parser.parse_args()
    
    # 检查Point工具是否可用
    if "Point" not in tool_manager.available_tools:
        print("Error: Point tool is not available in the tool manager!")
        print(f"Available tools: {tool_manager.available_tools}")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 评估Point工具的准确性
    stats, logs = evaluate_point_accuracy(
        data_file=args.data,
        img_base_dir=args.img_dir,
        tolerance=args.tolerance,
        max_samples=args.max_samples,
        verbose=args.verbose,
        num_threads=args.threads,
        output_image_dir = args.output_image_dir
    )
    
    # 将结果保存为单独的JSON文件（方便直接查看统计数据）
    stats_file = os.path.join(args.output, "point_accuracy_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"Summary statistics saved to {stats_file}")

if __name__ == "__main__":
    main()