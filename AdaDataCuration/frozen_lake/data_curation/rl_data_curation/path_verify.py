# rl/path_verify.py
import os
import json
import random
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from PIL import Image
from huggingface_hub import login
from datasets import Dataset, Features, Value, Image as DsImage, DatasetDict
from frozen_lake.data_curation.sft_data_curation.prompts import PATH_VERIFY_TASK_INSTRUCTION_SHORT
import re

def check_image_exists(image_path):
    """验证图像文件是否存在"""
    return os.path.isfile(image_path) and os.path.getsize(image_path) > 0

def process_jsonl(file_path):
    """读取 JSONL 文件并返回 JSON 对象列表"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading JSONL file {file_path}: {e}")
    return data

def write_jsonl(data, file_path):
    """将数据写入 JSONL 文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        return True
    except Exception as e:
        print(f"Error writing to JSONL file {file_path}: {e}")
        return False

def build_rl_data(
    full_dataset_path: str, 
    sft_ids_path: str, 
    output_dir: str,
    source_image_dir: str,
    max_samples: int = 2000,
    val_size: int = 1000,
    seed: int = 42
) -> Tuple[DatasetDict, str]:
    """
    构建RL训练数据，从元数据中排除已用于SFT的数据，并划分为训练集和验证集

    Args:
        full_dataset_path: 完整元数据的路径
        sft_ids_path: 用于SFT的ID文件路径
        output_dir: 输出目录
        source_image_dir: 源图像目录的基础路径
        max_samples: 最大样本数量
        val_size: 验证集大小
        seed: 随机种子

    Returns:
        Tuple[DatasetDict, str]: 包含训练集和验证集的数据集字典，以及数据集保存路径
    """
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"加载完整数据集: {full_dataset_path}")
    full_dataset = process_jsonl(full_dataset_path)
    print(f"加载完成，共 {len(full_dataset)} 条记录")
    
    # 加载已用于SFT的ID
    print(f"加载SFT数据ID: {sft_ids_path}")
    sft_data = json.load(open(sft_ids_path, 'r'))
    sft_ids = set(item["id"] for item in sft_data)
    print(f"加载完成，共 {len(sft_ids)} 个SFT ID")
    
    # 过滤掉已用于SFT的数据，并且只选择有随机路径的数据
    candidate_data = []
    for item in full_dataset:
        if (item["id"] not in sft_ids and 
            item.get("path_drawings") and 
            item["path_drawings"].get("random") and 
            item["path_drawings"]["random"].get("path")):
            candidate_data.append(item)
    
    print(f"可用于RL的数据: {len(candidate_data)} 条")
    
    # 按安全性分类
    safe_paths = []
    unsafe_paths = []
    
    for item in candidate_data:
        is_safe = item["path_drawings"]["random"].get("is_safe", False)
        if is_safe:
            safe_paths.append(item)
        else:
            unsafe_paths.append(item)
    
    print(f"安全路径: {len(safe_paths)} 条")
    print(f"不安全路径: {len(unsafe_paths)} 条")
    
    # 验证集大小不能超过总样本数的一半
    val_size = min(val_size, max_samples // 2)
    
    # 确定训练集和验证集的每类样本数
    train_per_group = min(len(safe_paths), len(unsafe_paths), (max_samples - val_size) // 2)
    val_per_group = min(len(safe_paths) - train_per_group, len(unsafe_paths) - train_per_group, val_size // 2)
    
    print(f"训练集每类样本数: {train_per_group}")
    print(f"验证集每类样本数: {val_per_group}")
    
    # 随机采样
    random.shuffle(safe_paths)
    random.shuffle(unsafe_paths)
    
    train_safe = safe_paths[:train_per_group]
    train_unsafe = unsafe_paths[:train_per_group]
    val_safe = safe_paths[train_per_group:train_per_group+val_per_group]
    val_unsafe = unsafe_paths[train_per_group:train_per_group+val_per_group]
    
    # 合并并打乱
    train_data = train_safe + train_unsafe
    val_data = val_safe + val_unsafe
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    print(f"训练集样本数: {len(train_data)} (安全: {len(train_safe)}, 不安全: {len(train_unsafe)})")
    print(f"验证集样本数: {len(val_data)} (安全: {len(val_safe)}, 不安全: {len(val_unsafe)})")
    
    # 构建数据集函数
    def create_dataset_from_items(items, desc):
        # 构建数据结构以保存原始图像路径（用于JSONL）和实际图像数据（用于HF数据集）
        dataset_dict = {
            "id": [],
            "image": [],         # 这里将存储图像对象，而不是路径
            "question": [],
            "answer": [],
            "metadata": []
        }
        
        # 用于JSONL的备份数据结构
        jsonl_items = []
        
        # 验证图像
        invalid_images = 0
        
        for item in tqdm(items, desc=f"构建{desc}"):
            # 获取随机路径数据
            random_path_data = item["path_drawings"]["random"]
            path = random_path_data["path"]
            is_safe = random_path_data.get("is_safe", False)
            question = PATH_VERIFY_TASK_INSTRUCTION_SHORT.replace("<ACTION-SEQ>", path) if PATH_VERIFY_TASK_INSTRUCTION_SHORT else f"Is the path {path} safe for the player to follow?"
            
            # 使用原始图像路径
            original_image_path = os.path.join(source_image_dir, item["image_path"])
            
            # 验证图像是否存在
            if not check_image_exists(original_image_path):
                print(f"警告: 图像文件不存在: {original_image_path}")
                invalid_images += 1
                continue
            
            # 尝试加载图像（提前验证图像是否可读取）
            try:
                # 直接加载图像到内存
                image_data = Image.open(original_image_path)
                if image_data.mode != "RGB":
                    image_data = image_data.convert("RGB")
            except Exception as e:
                print(f"加载图像失败 {original_image_path}: {e}")
                invalid_images += 1
                continue
                
            # 元数据
            metadata = {
                "size": item["size"],
                "start_coords": item["start_coords"],
                "goal_coords": item["goal_coords"],
                "obstacle_coords": item["obstacle_coords"],
                "path": path,
                "is_safe": is_safe
            }
            
            # 添加到数据集
            dataset_dict["id"].append(item['id'])
            dataset_dict["image"].append(image_data)  # 直接存储图像对象
            dataset_dict["question"].append(question)
            dataset_dict["answer"].append("Yes" if is_safe else "No")
            dataset_dict["metadata"].append(json.dumps(metadata))
            
            # 同时为JSONL构建记录（使用图像路径）
            jsonl_items.append({
                "id": item['id'],
                "image": original_image_path,  # 使用路径
                "question": question,
                "answer": "Yes" if is_safe else "No",
                "metadata": json.dumps(metadata)
            })
        
        print(f"{desc}跳过了 {invalid_images} 个无效图像")
        print(f"{desc}成功加载 {len(dataset_dict['id'])} 个样本")
        
        return dataset_dict, jsonl_items
    
    # 创建训练集和验证集
    train_dict, train_jsonl = create_dataset_from_items(train_data, "训练集")
    val_dict, val_jsonl = create_dataset_from_items(val_data, "验证集")
    
    # 创建Hugging Face数据集 - 使用实际图像而非路径
    features = Features({
        "id": Value("string"),
        "image": DsImage(),
        "question": Value("string"),
        "answer": Value("string"),
        "metadata": Value("string")
    })
    
    # 创建包含实际图像的数据集
    train_dataset = Dataset.from_dict(train_dict, features=features)
    val_dataset = Dataset.from_dict(val_dict, features=features)
    
    # 合并为DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    # 保存数据集到本地
    dataset_path = os.path.join(output_dir, "path_verify_rl_dataset")
    dataset_dict.save_to_disk(dataset_path)
    
    # 保存JSONL版本（使用图像路径而非图像对象）
    train_jsonl_path = os.path.join(output_dir, "path_verify_rl_train.jsonl")
    val_jsonl_path = os.path.join(output_dir, "path_verify_rl_val.jsonl")
    write_jsonl(train_jsonl, train_jsonl_path)
    write_jsonl(val_jsonl, val_jsonl_path)
    
    # 保存RL数据的ID列表
    train_ids_path = os.path.join(output_dir, "path_verify_rl_train_ids.json")
    val_ids_path = os.path.join(output_dir, "path_verify_rl_val_ids.json")
    with open(train_ids_path, 'w') as f:
        json.dump([item["id"] for item in train_jsonl], f)
    with open(val_ids_path, 'w') as f:
        json.dump([item["id"] for item in val_jsonl], f)
    
    print(f"RL数据集已保存到: {dataset_path}")
    print(f"训练集备份为JSONL格式: {train_jsonl_path}")
    print(f"验证集备份为JSONL格式: {val_jsonl_path}")
    
    # 生成统计信息
    stats = {
        "total_samples": len(train_dataset) + len(val_dataset),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_safe_paths": sum(1 for item in train_dataset if item["answer"] == "Yes"),
        "train_unsafe_paths": sum(1 for item in train_dataset if item["answer"] == "No"),
        "val_safe_paths": sum(1 for item in val_dataset if item["answer"] == "Yes"),
        "val_unsafe_paths": sum(1 for item in val_dataset if item["answer"] == "No"),
        "average_path_length": {
            "train": sum(len(json.loads(item["metadata"])["path"].split(",")) for item in train_dataset) / len(train_dataset),
            "val": sum(len(json.loads(item["metadata"])["path"].split(",")) for item in val_dataset) / len(val_dataset)
        },
        "map_sizes": {
            "train": {
                "min": min(json.loads(item["metadata"])["size"] for item in train_dataset),
                "max": max(json.loads(item["metadata"])["size"] for item in train_dataset),
                "distribution": {
                    str(size): sum(1 for item in train_dataset if json.loads(item["metadata"])["size"] == size)
                    for size in set(json.loads(item["metadata"])["size"] for item in train_dataset)
                }
            },
            "val": {
                "min": min(json.loads(item["metadata"])["size"] for item in val_dataset),
                "max": max(json.loads(item["metadata"])["size"] for item in val_dataset),
                "distribution": {
                    str(size): sum(1 for item in val_dataset if json.loads(item["metadata"])["size"] == size)
                    for size in set(json.loads(item["metadata"])["size"] for item in val_dataset)
                }
            }
        }
    }
    
    # 保存统计信息
    stats_path = os.path.join(output_dir, "path_verify_rl_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"统计信息已保存至 {stats_path}")
    
    return dataset_dict, dataset_path

def upload_to_hf(dataset_path, repo_id, token=None):
    """
    将数据集上传到Hugging Face Hub
    
    Args:
        dataset_path: 数据集本地路径
        repo_id: Hugging Face仓库ID
        token: Hugging Face API令牌
    """
    try:
        from datasets import load_from_disk
        
        # 加载数据集
        dataset = load_from_disk(dataset_path)
        
        # 登录Hugging Face
        if token:
            login(token=token)
        else:
            print("没有提供HF token，尝试使用已保存的token登录")
        
        # 上传数据集
        dataset.push_to_hub(
            repo_id=repo_id, 
            token=token,
            private=False
        )
        
        print(f"数据集已成功上传到 {repo_id}")
        return True
    
    except Exception as e:
        print(f"上传到Hugging Face时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='为路径验证任务构建RL数据')
    parser.add_argument('--full_dataset', type=str, default='/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata/dataset.jsonl', 
                        help='完整元数据文件路径')
    parser.add_argument('--sft_ids', type=str, default='/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata/sft_output/selected_ids.json', 
                        help='SFT ID文件路径')
    parser.add_argument('--output_dir', type=str, default='/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata/rl_output', 
                        help='RL输出目录')
    parser.add_argument('--source_image_dir', type=str, default='/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation', 
                        help='源图像目录基础路径')
    parser.add_argument('--max_samples', type=int, default=6000, 
                        help='最大样本数量')
    parser.add_argument('--val_size', type=int, default=1000, 
                        help='验证集大小')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--upload_to_hf', action='store_true',
                        help='是否上传到Hugging Face')
    parser.add_argument('--hf_repo_id', type=str, default='hitsmy/tool-pathverify-rl-v0',
                        help='Hugging Face仓库ID')
    parser.add_argument('--hf_token', type=str, default="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",
                        help='Hugging Face API令牌')
    
    args = parser.parse_args()
    
    dataset_dict, dataset_path = build_rl_data(
        full_dataset_path=args.full_dataset,
        sft_ids_path=args.sft_ids,
        output_dir=args.output_dir,
        source_image_dir=args.source_image_dir,
        max_samples=args.max_samples,
        val_size=args.val_size,
        seed=args.seed
    )
    
    # 如果需要上传到Hugging Face
    if args.upload_to_hf:
        print(f"上传数据集到Hugging Face仓库: {args.hf_repo_id}")
        upload_to_hf(
            dataset_path=dataset_path,
            repo_id=args.hf_repo_id,
            token=args.hf_token
        )

if __name__ == "__main__":
    main()