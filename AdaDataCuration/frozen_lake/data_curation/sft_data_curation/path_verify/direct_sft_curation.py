# direct_sft_curation.py
import os
import json
import random
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Set
from copy import deepcopy

from tool_server.utils.utils import process_jsonl, write_json_file, load_json_file
from frozen_lake.data_curation.sft_data_curation.prompts import PATH_VERIFY_TASK_INSTRUCTION_SHORT, TOOL_PROMPTS

def check_image_exists(image_path):
    """验证图像文件是否存在"""
    return os.path.isfile(image_path) and os.path.getsize(image_path) > 0

def create_direct_sft_conversation(item, image_dir):
    """
    为路径验证任务创建直接的SFT对话数据
    
    Args:
        item: 原始元数据项
        image_dir: 图像目录路径
        
    Returns:
        dict: ShareGPT格式的对话数据
    """
    # 获取随机路径信息
    random_path_data = item["path_drawings"]["random"]
    path = random_path_data["path"]
    is_safe = random_path_data.get("is_safe", False)
    
    # # 准备系统提示
    # system_message = {
    #     "from": "system",
    #     "value": TOOL_PROMPTS
    # }
    
    # 准备用户问题
    instruction_text = PATH_VERIFY_TASK_INSTRUCTION_SHORT.replace("<ACTION-SEQ>", path)
    user_message = {
        "from": "human",
        "value": instruction_text + "\n<image>"
    }
    
    # 准备助手答案
    valid_text = "Yes" if is_safe else "No"
    assistant_message = {
        "from": "gpt",
        "value": f"After analyzing the path {path}, I can determine that this path is {'safe' if is_safe else 'not safe'}.\n\n"
                 f"{'This path safely navigates from the starting point without falling into any ice holes.' if is_safe else 'This path would cause the player to fall into an ice hole.'}\n\n"
                 f"\\boxed{{{valid_text}}}"
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

def generate_direct_sft_dataset(input_path, output_path, image_dir="./frozen_lake_metadata_v2", 
                               max_samples=1000, seed=42, reference_json=None):
    """
    生成直接SFT数据集
    
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
    
    # 筛选数据集，只保留参考ID中存在的数据
    if reference_ids:
        filtered_dataset = []
        for item in dataset:
            if item["id"] in reference_ids:
                filtered_dataset.append(item)
        
        print(f"根据参考ID筛选后，保留了 {len(filtered_dataset)} 条数据")
        dataset = filtered_dataset
    
    # 如果数据超过最大样本数且没有使用参考ID，随机选择
    if len(dataset) > max_samples and not reference_ids:
        dataset = random.sample(dataset, max_samples)
        print(f"随机选择了 {max_samples} 条数据")
    
    # 生成ShareGPT格式数据
    sharegpt_data = []
    
    # 统计安全和不安全路径数量
    safe_count = 0
    unsafe_count = 0
    
    with tqdm(total=len(dataset), desc="生成直接SFT数据") as pbar:
        for item in dataset:
            # 检查是否有随机路径数据
            if "path_drawings" not in item or "random" not in item["path_drawings"]:
                print(f"跳过无随机路径数据的项: {item.get('id', 'unknown')}")
                continue
                
            # 创建直接SFT对话
            sharegpt_item = create_direct_sft_conversation(item, image_dir)
            sharegpt_data.append(sharegpt_item)
            
            # 统计安全和不安全路径
            is_safe = item["path_drawings"]["random"].get("is_safe", False)
            if is_safe:
                safe_count += 1
            else:
                unsafe_count += 1
                
            pbar.update(1)
    
    # 保存ShareGPT格式数据
    output_file = os.path.join(output_path, "direct_verify_sharegpt_data.json")
    write_json_file(sharegpt_data, output_file)
    
    print(f"数据集生成完成，共生成 {len(sharegpt_data)} 个样本")
    print(f"结果保存至 {output_file}")
    print(f"安全路径数量: {safe_count} ({safe_count/len(sharegpt_data)*100:.1f}%)")
    print(f"不安全路径数量: {unsafe_count} ({unsafe_count/len(sharegpt_data)*100:.1f}%)")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成FrozenLake路径验证任务的直接SFT数据集')
    parser.add_argument('--input_path', type=str, 
                        default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/metadata_split/path_verify/sft.jsonl", 
                        help='输入数据集路径')
    parser.add_argument('--output_path', type=str, 
                        default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/path_verify", 
                        help='输出目录')
    parser.add_argument('--image_dir', type=str, 
                        default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation", 
                        help='图像目录')
    parser.add_argument('--max_samples', type=int, default=1000, 
                        help='最大样本数')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--reference_json', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/path_verify/verify_sharegpt_data.json",
                        help='参考JSON文件，用于ID对齐')
    
    args = parser.parse_args()
    
    # 生成数据集
    generate_direct_sft_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        image_dir=args.image_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        reference_json=args.reference_json
    )

if __name__ == "__main__":
    main()