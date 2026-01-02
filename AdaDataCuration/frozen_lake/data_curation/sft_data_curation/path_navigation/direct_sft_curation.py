# direct_navigation_sft_curation.py
import os
import json
import random
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Set
from copy import deepcopy

from tool_server.utils.utils import process_jsonl, write_json_file, load_json_file
from frozen_lake.data_curation.sft_data_curation.prompts import PATH_FINDING_TASK_INSTRUCTION_SHORT, TOOL_PROMPTS

def check_image_exists(image_path):
    """验证图像文件是否存在"""
    return os.path.isfile(image_path) and os.path.getsize(image_path) > 0

def create_direct_navigation_conversation(item, image_dir):
    """
    为路径导航任务创建直接的SFT对话数据
    
    Args:
        item: 原始元数据项
        image_dir: 图像目录路径
        
    Returns:
        dict: ShareGPT格式的对话数据
    """
    # 获取最优路径信息
    astar_path_data = item["astar_path"]
    path = astar_path_data["path"]
    is_valid = astar_path_data.get("is_valid", True)
    
    # 准备系统提示
    # system_message = {
    #     "from": "system",
    #     "value": TOOL_PROMPTS
    # }
    
    # 准备用户问题
    instruction_text = PATH_FINDING_TASK_INSTRUCTION_SHORT
    user_message = {
        "from": "human",
        "value": instruction_text + "\n<image>"
    }
    
    # 准备助手答案 - 直接给出最优路径
    assistant_message = {
        "from": "gpt",
        "value": f"After analyzing the maze, I've determined the optimal path from the Elf to the Gift.\n\n"
                 f"This path avoids all ice holes and uses the fewest moves possible to reach the goal.\n\n"
                 f"\\boxed{{{path}}}"
    }
    
    # 组装对话
    conversation = [user_message, assistant_message]
    
    # 准备图像路径
    image_path = os.path.join(image_dir, item["image_path"])
    assert check_image_exists(image_path), f"Image file does not exist: {image_path}"
    
    # 创建ShareGPT格式数据
    sharegpt_item = {
        "qid": item["id"],
        "conversations": conversation,
        "images": [image_path]
    }
    
    return sharegpt_item

def generate_direct_navigation_dataset(input_path, output_path, image_dir="./frozen_lake_metadata_v2", 
                               max_samples=1000, seed=42, reference_json=None):
    """
    生成直接SFT导航数据集
    
    Args:
        input_path: 输入元数据文件路径
        output_path: 输出目录
        image_dir: 图像目录路径
        max_samples: 最大样本数
        seed: 随机种子
        reference_json: 参考JSON文件，用于ID对齐
    """
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"从 {input_path} 加载数据...")
    dataset = process_jsonl(input_path)
    print(f"加载了 {len(dataset)} 条数据")
    
    # 如果提供了参考JSON文件，加载参考ID
    reference_ids = set()
    if reference_json and os.path.exists(reference_json):
        print(f"从参考文件 {reference_json} 加载ID...")
        try:
            reference_data = load_json_file(reference_json)
            for item in reference_data:
                if "qid" in item:
                    reference_ids.add(item["qid"])
                elif "id" in item:
                    reference_ids.add(item["id"])
            print(f"从参考文件加载了 {len(reference_ids)} 个ID")
        except Exception as e:
            print(f"加载参考文件失败: {e}")
            reference_ids = set()
    
    # 首先尝试从参考ID中筛选数据
    selected_dataset = []
    remaining_dataset = []
    
    if reference_ids:
        # 将数据集分为匹配参考ID的和不匹配的两部分
        for item in dataset:
            if item["id"] in reference_ids:
                selected_dataset.append(item)
            else:
                remaining_dataset.append(item)
        
        print(f"根据参考ID筛选，找到了 {len(selected_dataset)} 条匹配数据")
        
        # 如果匹配的数据量不足max_samples，从剩余数据中随机选择补充
        if len(selected_dataset) < max_samples and remaining_dataset:
            needed = max_samples - len(selected_dataset)
            if len(remaining_dataset) > needed:
                additional_samples = random.sample(remaining_dataset, needed)
                print(f"从剩余数据中随机选择了额外的 {len(additional_samples)} 条数据补充")
                selected_dataset.extend(additional_samples)
            else:
                print(f"剩余数据不足，全部添加了 {len(remaining_dataset)} 条数据补充")
                selected_dataset.extend(remaining_dataset)
    else:
        # 如果没有参考ID或参考ID为空，直接从整个数据集选择
        if len(dataset) > max_samples:
            selected_dataset = random.sample(dataset, max_samples)
            print(f"随机选择了 {max_samples} 条数据")
        else:
            selected_dataset = dataset
            print(f"使用全部 {len(dataset)} 条数据")
    
    # 生成ShareGPT格式数据
    sharegpt_data = []
    
    # 统计路径长度
    path_length_stats = {}
    
    with tqdm(total=len(selected_dataset), desc="生成直接导航SFT数据") as pbar:
        for item in selected_dataset:
            # 检查是否有A*路径数据
            if "astar_path" not in item or "path" not in item["astar_path"]:
                print(f"跳过无A*路径数据的项: {item.get('id', 'unknown')}")
                continue
                
            # 创建直接SFT对话
            sharegpt_item = create_direct_navigation_conversation(item, image_dir)
            sharegpt_data.append(sharegpt_item)
            
            # 统计路径长度
            path = item["astar_path"]["path"]
            path_length = len(path.split(","))
            path_length_stats[path_length] = path_length_stats.get(path_length, 0) + 1
                
            pbar.update(1)
    
    # 保存ShareGPT格式数据
    output_file = os.path.join(output_path, "direct_navigation_sharegpt_data.json")
    write_json_file(sharegpt_data, output_file)
    
    print(f"数据集生成完成，共生成 {len(sharegpt_data)} 个样本")
    print(f"结果保存至 {output_file}")
    
    # 输出参考ID和随机选择的比例
    if reference_ids:
        ref_count = sum(1 for item in sharegpt_data if item["qid"] in reference_ids)
        random_count = len(sharegpt_data) - ref_count
        print(f"参考ID匹配数据: {ref_count} ({ref_count/len(sharegpt_data)*100:.1f}%)")
        print(f"随机选择补充数据: {random_count} ({random_count/len(sharegpt_data)*100:.1f}%)")
    
    # 输出路径长度统计
    print("路径长度统计:")
    for length, count in sorted(path_length_stats.items()):
        print(f"  长度 {length}: {count} 条路径 ({count/len(sharegpt_data)*100:.1f}%)")
    
    # 统计平均路径长度
    total_steps = sum([length * count for length, count in path_length_stats.items()])
    avg_length = total_steps / len(sharegpt_data) if sharegpt_data else 0
    print(f"平均路径长度: {avg_length:.2f} 步")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成FrozenLake路径导航任务的直接SFT数据集')
    parser.add_argument('--input_path', type=str, 
                        default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/metadata_split/path_navigation/sft.jsonl", 
                        help='输入数据集路径')
    parser.add_argument('--output_path', type=str, 
                        default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/path_navigation", 
                        help='输出目录')
    parser.add_argument('--image_dir', type=str, 
                        default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation", 
                        help='图像目录')
    parser.add_argument('--max_samples', type=int, default=550, 
                        help='最大样本数')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--reference_json', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/path_navigation/stage2/navigation_stage2_sharegpt_formatted.json",
                        help='参考JSON文件，用于ID对齐')
    
    args = parser.parse_args()
    
    # 生成数据集
    generate_direct_navigation_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        image_dir=args.image_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        reference_json=args.reference_json
    )

if __name__ == "__main__":
    main()