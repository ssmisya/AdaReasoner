# data_split.py
import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """从JSONL文件加载数据"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        print(f"成功加载 {len(data)} 条记录")
        return data
    except Exception as e:
        print(f"加载文件错误: {e}")
        return []

def write_jsonl(data, file_path):
    """将数据写入JSONL文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        return True
    except Exception as e:
        print(f"写入JSONL文件错误: {e}")
        return False
        
def write_json(data, file_path):
    """将数据写入JSON文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"写入JSON文件错误: {e}")
        return False

def analyze_path_balance(data: List[Dict[str, Any]]) -> Dict:
    """分析不同长度路径的安全性比例"""
    # 按尺寸和长度统计
    stats_by_size_length = defaultdict(lambda: defaultdict(lambda: {"safe": 0, "unsafe": 0}))
    
    # 按长度统计(所有尺寸)
    stats_by_length = defaultdict(lambda: {"safe": 0, "unsafe": 0})
    
    # 总体统计
    total_stats = {"safe": 0, "unsafe": 0}
    
    for item in data:
        if "path_drawings" in item and "random" in item["path_drawings"]:
            random_path = item["path_drawings"]["random"]
            
            if "path" in random_path and "is_safe" in random_path:
                path = random_path["path"]
                is_safe = random_path["is_safe"]
                size = item["size"]
                
                # 计算路径长度
                path_length = len(path.split(','))
                
                # 更新统计
                if is_safe:
                    stats_by_size_length[size][path_length]["safe"] += 1
                    stats_by_length[path_length]["safe"] += 1
                    total_stats["safe"] += 1
                else:
                    stats_by_size_length[size][path_length]["unsafe"] += 1
                    stats_by_length[path_length]["unsafe"] += 1
                    total_stats["unsafe"] += 1
    
    # 计算百分比
    length_percentages = {}
    for length, stats in stats_by_length.items():
        total = stats["safe"] + stats["unsafe"]
        if total > 0:
            safe_percent = (stats["safe"] / total) * 100
            unsafe_percent = (stats["unsafe"] / total) * 100
            length_percentages[length] = {
                "safe_percent": safe_percent,
                "unsafe_percent": unsafe_percent,
                "safe_count": stats["safe"],
                "unsafe_count": stats["unsafe"],
                "total": total
            }
    
    # 计算每个尺寸下不同长度的百分比
    size_length_percentages = {}
    for size, lengths in stats_by_size_length.items():
        size_length_percentages[size] = {}
        for length, stats in lengths.items():
            total = stats["safe"] + stats["unsafe"]
            if total > 0:
                safe_percent = (stats["safe"] / total) * 100
                unsafe_percent = (stats["unsafe"] / total) * 100
                size_length_percentages[size][length] = {
                    "safe_percent": safe_percent,
                    "unsafe_percent": unsafe_percent,
                    "safe_count": stats["safe"],
                    "unsafe_count": stats["unsafe"],
                    "total": total
                }
    
    # 总体百分比
    total = total_stats["safe"] + total_stats["unsafe"]
    if total > 0:
        total_safe_percent = (total_stats["safe"] / total) * 100
        total_unsafe_percent = (total_stats["unsafe"] / total) * 100
    else:
        total_safe_percent = 0
        total_unsafe_percent = 0
    
    return {
        "by_length": length_percentages,
        "by_size_length": size_length_percentages,
        "overall": {
            "safe_percent": total_safe_percent,
            "unsafe_percent": total_unsafe_percent,
            "safe_count": total_stats["safe"],
            "unsafe_count": total_stats["unsafe"],
            "total": total
        }
    }

def print_balance_statistics(stats: Dict, title: str = ""):
    """打印平衡统计信息"""
    if title:
        print(f"\n===== {title} =====")
    
    print(f"总样本数: {stats['overall']['total']}")
    print(f"安全路径: {stats['overall']['safe_count']} ({stats['overall']['safe_percent']:.2f}%)")
    print(f"不安全路径: {stats['overall']['unsafe_count']} ({stats['overall']['unsafe_percent']:.2f}%)")
    
    print("\n按路径长度统计:")
    for length in sorted(stats["by_length"].keys()):
        data = stats["by_length"][length]
        print(f"  长度 {length}: 总数={data['total']}, 安全={data['safe_count']}({data['safe_percent']:.1f}%), 不安全={data['unsafe_count']}({data['unsafe_percent']:.1f}%)")
    
    print("\n按地图尺寸和路径长度统计:")
    for size in sorted(stats["by_size_length"].keys()):
        print(f"\n  地图尺寸 {size}x{size}:")
        for length in sorted(stats["by_size_length"][size].keys()):
            data = stats["by_size_length"][size][length]
            print(f"    长度 {length}: 总数={data['total']}, 安全={data['safe_count']}({data['safe_percent']:.1f}%), 不安全={data['unsafe_count']}({data['unsafe_percent']:.1f}%)")

def filter_path_length_one(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """过滤掉路径长度为1的数据"""
    # 不再过滤步长=1的数据，直接返回原数据
    print("保留所有步长的数据（包括步长=1）")
    return data

def balance_groups_strict(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    严格平衡每个尺寸每个步长的安全/不安全比例，确保接近1:1
    
    Args:
        candidates: 候选数据列表
        
    Returns:
        List[Dict[str, Any]]: 平衡后的数据子集
    """
    # 按地图尺寸和路径长度分组
    safe_groups = defaultdict(list)
    unsafe_groups = defaultdict(list)
    
    for item in candidates:
        if "path_drawings" in item and "random" in item["path_drawings"]:
            random_path = item["path_drawings"]["random"]
            
            if "path" in random_path and "is_safe" in random_path:
                size = item["size"]
                path = random_path["path"]
                is_safe = random_path["is_safe"]
                path_length = len(path.split(','))
                
                key = (size, path_length)
                
                if is_safe:
                    safe_groups[key].append(item)
                else:
                    unsafe_groups[key].append(item)
    
    balanced_data = []
    total_filtered = 0
    
    # 处理每个组合，确保1:1平衡
    all_keys = set(safe_groups.keys()) | set(unsafe_groups.keys())
    
    for key in sorted(all_keys):
        size, path_length = key
        safe_items = safe_groups.get(key, [])
        unsafe_items = unsafe_groups.get(key, [])
        
        safe_count = len(safe_items)
        unsafe_count = len(unsafe_items)
        
        if safe_count == 0 and unsafe_count == 0:
            continue
        
        # 计算平衡后的数量（取较小值）
        balanced_count = min(safe_count, unsafe_count)
        
        if balanced_count > 0:
            # 随机选择平衡数量的样本
            if safe_count > 0:
                selected_safe = random.sample(safe_items, balanced_count)
                balanced_data.extend(selected_safe)
            
            if unsafe_count > 0:
                selected_unsafe = random.sample(unsafe_items, balanced_count)
                balanced_data.extend(selected_unsafe)
            
            filtered_count = (safe_count - balanced_count) + (unsafe_count - balanced_count)
            total_filtered += filtered_count
            
            print(f"尺寸{size}x{size} 步长{path_length}: 安全={safe_count} 不安全={unsafe_count} -> 平衡后各{balanced_count} (过滤{filtered_count})")
        
        else:
            # 如果某一类为0，则保留另一类的少量样本以维持多样性
            if safe_count > 0:
                keep_count = min(safe_count, 10)  # 最多保留10个
                selected_safe = random.sample(safe_items, keep_count)
                balanced_data.extend(selected_safe)
                filtered_count = safe_count - keep_count
                total_filtered += filtered_count
                print(f"尺寸{size}x{size} 步长{path_length}: 只有安全路径{safe_count} -> 保留{keep_count} (过滤{filtered_count})")
            
            if unsafe_count > 0:
                keep_count = min(unsafe_count, 10)  # 最多保留10个
                selected_unsafe = random.sample(unsafe_items, keep_count)
                balanced_data.extend(selected_unsafe)
                filtered_count = unsafe_count - keep_count
                total_filtered += filtered_count
                print(f"尺寸{size}x{size} 步长{path_length}: 只有不安全路径{unsafe_count} -> 保留{keep_count} (过滤{filtered_count})")
    
    print(f"\n总共过滤了 {total_filtered} 个不平衡的样本")
    print(f"平衡后剩余 {len(balanced_data)} 个样本")
    
    return balanced_data

def balance_selection_strict(candidates: List[Dict[str, Any]], target_count: int = None) -> List[Dict[str, Any]]:
    """
    严格平衡选择：先平衡每个组合，再选择目标数量
    
    Args:
        candidates: 候选数据列表
        target_count: 目标样本数
        
    Returns:
        List[Dict[str, Any]]: 平衡的数据子集
    """
    # 首先对所有候选数据进行组内平衡
    balanced_candidates = balance_groups_strict(candidates)
    if target_count is None:
        print("目标数量未指定，直接返回平衡后的数据")
        return balanced_candidates
    
    if len(balanced_candidates) <= target_count:
        print(f"平衡后的数据量({len(balanced_candidates)})不超过目标数量({target_count})，直接返回全部")
        return balanced_candidates
    
    # 如果平衡后的数据仍然超过目标数量，需要进一步采样
    print(f"需要从平衡后的{len(balanced_candidates)}个样本中选择{target_count}个")
    
    # 按尺寸和步长重新分组
    safe_groups = defaultdict(list)
    unsafe_groups = defaultdict(list)
    
    for item in balanced_candidates:
        if "path_drawings" in item and "random" in item["path_drawings"]:
            random_path = item["path_drawings"]["random"]
            
            if "path" in random_path and "is_safe" in random_path:
                size = item["size"]
                path = random_path["path"]
                is_safe = random_path["is_safe"]
                path_length = len(path.split(','))
                
                key = (size, path_length)
                
                if is_safe:
                    safe_groups[key].append(item)
                else:
                    unsafe_groups[key].append(item)
    
    # 找出同时有安全和不安全路径的组合
    common_keys = set(safe_groups.keys()) & set(unsafe_groups.keys())
    
    if not common_keys:
        print("警告: 平衡后没有找到同时包含安全和不安全路径的组合")
        return random.sample(balanced_candidates, target_count)
    
    # 计算每个组合的配额
    target_pairs = target_count // 2  # 一半安全，一半不安全
    pairs_per_group = max(1, target_pairs // len(common_keys))
    
    selected = []
    
    for key in sorted(common_keys):
        safe_available = len(safe_groups[key])
        unsafe_available = len(unsafe_groups[key])
        
        # 每组选择的对数受限于较小的那个组和配额
        actual_pairs = min(pairs_per_group, safe_available, unsafe_available)
        
        if actual_pairs > 0:
            # 随机选择
            selected_safe = random.sample(safe_groups[key], actual_pairs)
            selected_unsafe = random.sample(unsafe_groups[key], actual_pairs)
            
            selected.extend(selected_safe)
            selected.extend(selected_unsafe)
            
            print(f"组合 {key}: 选择了 {actual_pairs} 对 (安全={actual_pairs}, 不安全={actual_pairs})")
    
    # 如果选择的样本不足目标数量，从剩余的平衡数据中补充
    if len(selected) < target_count:
        remaining_candidates = [item for item in balanced_candidates if item not in selected]
        additional_needed = target_count - len(selected)
        
        if remaining_candidates:
            # 尝试保持平衡地补充
            remaining_safe = []
            remaining_unsafe = []
            
            for item in remaining_candidates:
                if "path_drawings" in item and "random" in item["path_drawings"]:
                    random_path = item["path_drawings"]["random"]
                    if "is_safe" in random_path:
                        if random_path["is_safe"]:
                            remaining_safe.append(item)
                        else:
                            remaining_unsafe.append(item)
            
            # 平衡地补充
            safe_needed = additional_needed // 2
            unsafe_needed = additional_needed - safe_needed
            
            safe_to_add = min(safe_needed, len(remaining_safe))
            unsafe_to_add = min(unsafe_needed, len(remaining_unsafe))
            
            if safe_to_add > 0:
                selected.extend(random.sample(remaining_safe, safe_to_add))
            if unsafe_to_add > 0:
                selected.extend(random.sample(remaining_unsafe, unsafe_to_add))
            
            print(f"补充样本: 安全={safe_to_add}, 不安全={unsafe_to_add}")
    
    # 如果仍然不足，随机补充
    if len(selected) < target_count:
        remaining = [item for item in balanced_candidates if item not in selected]
        still_needed = target_count - len(selected)
        if remaining and still_needed > 0:
            selected.extend(random.sample(remaining, min(still_needed, len(remaining))))
    
    # 如果过多，随机移除
    if len(selected) > target_count:
        selected = random.sample(selected, target_count)
    
    print(f"最终选择了 {len(selected)} 个样本")
    return selected

def split_datasets(data, test_sizes, sft_size_verify, rl_size_verify, output_dir="./data_split", seed=42):
    """
    将数据集分为训练、验证和测试集，并确保平衡性
    """
    # 设置随机种子
    random.seed(seed)
    
    # 按任务分组
    path_verify_items = []
    path_navigation_items = []
    
    # 检查每个数据项是否适合特定任务
    for item in data:
        # Path Verification任务：需要随机路径并标明安全性
        if ("path_drawings" in item and 
            "random" in item["path_drawings"] and 
            "path" in item["path_drawings"]["random"] and 
            "is_safe" in item["path_drawings"]["random"]):
            path_verify_items.append(item)
        
        # Path Navigation任务：需要A*路径
        if ("astar_path" in item and 
            "path" in item["astar_path"] and 
            item["astar_path"]["path"]):
            path_navigation_items.append(item)
    
    print(f"Path Verification任务可用数据: {len(path_verify_items)}")
    print(f"Path Navigation任务可用数据: {len(path_navigation_items)}")
    
    # 过滤掉路径长度为1的数据
    path_verify_items = filter_path_length_one(path_verify_items)
    print(f"过滤后Path Verification任务可用数据: {len(path_verify_items)}")
    
    # 1. 先分离测试集数据池（根据地图尺寸）
    test_verify_pool = [item for item in path_verify_items if item["size"] in test_sizes]
    test_navigation_pool = [item for item in path_navigation_items if item["size"] in test_sizes]
    
    # 对测试集数据池进行平衡处理
    print(f"\n开始处理测试集...")
    print(f"Path Verification测试集数据池: {len(test_verify_pool)}")
    test_verify = balance_groups_strict(test_verify_pool)
    test_navigation = test_navigation_pool  # Navigation任务测试集暂不需要特殊平衡
    
    # 2. 剩余的作为训练数据池
    train_verify = [item for item in path_verify_items if item["size"] not in test_sizes]
    train_navigation = [item for item in path_navigation_items if item["size"] not in test_sizes]
    
    print(f"Path Verification测试集: {len(test_verify)} (平衡后)")
    print(f"Path Navigation测试集: {len(test_navigation)}")
    print(f"Path Verification训练池: {len(train_verify)}")
    print(f"Path Navigation训练池: {len(train_navigation)}")
    
    # 3. 从训练池中选择SFT数据集（严格平衡）
    print("\n开始选择SFT数据集...")
    sft_verify = balance_selection_strict(train_verify, target_count=sft_size_verify)
    
    # 4. 从剩余数据中选择RL数据集
    remaining_verify = [item for item in train_verify if item not in sft_verify]
    print(f"\n剩余数据: {len(remaining_verify)} 条")
    print("开始选择RL数据集...")
    rl_verify = balance_selection_strict(remaining_verify, target_count=None)
    
    # 为Navigation任务计算合适的SFT大小
    sft_size_navigation = min(1000, len(train_navigation) // 2)  # 确保有足够的RL数据
    sft_navigation = random.sample(train_navigation, sft_size_navigation)
    
    # 剩余的作为RL数据集
    rl_navigation_ids = {item["id"] for item in sft_navigation}
    rl_navigation = [item for item in train_navigation if item["id"] not in rl_navigation_ids]
    
    print(f"\nPath Verification SFT集: {len(sft_verify)}")
    print(f"Path Verification RL集: {len(rl_verify)}")
    print(f"Path Navigation SFT集: {len(sft_navigation)}")
    print(f"Path Navigation RL集: {len(rl_navigation)}")
    
    # 5. 保存分割后的数据集
    os.makedirs(output_dir, exist_ok=True)
    
    # Path Verification任务
    verify_dir = os.path.join(output_dir, "path_verify")
    os.makedirs(verify_dir, exist_ok=True)
    
    write_jsonl(sft_verify, os.path.join(verify_dir, "sft.jsonl"))
    write_jsonl(rl_verify, os.path.join(verify_dir, "rl.jsonl"))
    write_jsonl(test_verify, os.path.join(verify_dir, "test.jsonl"))
    
    # Path Navigation任务
    navigation_dir = os.path.join(output_dir, "path_navigation")
    os.makedirs(navigation_dir, exist_ok=True)
    
    write_jsonl(sft_navigation, os.path.join(navigation_dir, "sft.jsonl"))
    write_jsonl(rl_navigation, os.path.join(navigation_dir, "rl.jsonl"))
    write_jsonl(test_navigation, os.path.join(navigation_dir, "test.jsonl"))
    
    # 6. 分析每个集合的平衡性
    verify_stats = {
        "sft": analyze_path_balance(sft_verify),
        "rl": analyze_path_balance(rl_verify),
        "test": analyze_path_balance(test_verify)
    }
    
    # 保存统计信息
    write_json(verify_stats, os.path.join(verify_dir, "stats.json"))
    
    # 7. 打印统计信息
    print_balance_statistics(verify_stats["sft"], "Path Verification SFT 集平衡性")
    print_balance_statistics(verify_stats["rl"], "Path Verification RL 集平衡性") 
    print_balance_statistics(verify_stats["test"], "Path Verification 测试集平衡性")
    
    return {
        "path_verify": {
            "sft": sft_verify,
            "rl": rl_verify,
            "test": test_verify
        },
        "path_navigation": {
            "sft": sft_navigation,
            "rl": rl_navigation,
            "test": test_navigation
        }
    }

def main():
    parser = argparse.ArgumentParser(description="切分FrozenLake数据集为训练、RL和测试集")
    parser.add_argument('--data', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/dataset.jsonl", help='原始数据集路径')
    parser.add_argument('--output_dir', type=str, default='/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/metadata_split', help='输出目录')
    parser.add_argument('--sft_size_verify', type=int, default=1000, help='Path Verification任务的SFT数据集大小')
    parser.add_argument('--rl_size_verify', type=int, default=2000, help='Path Verification任务的RL数据集大小') 
    parser.add_argument('--test_sizes', type=str, default='5,7,9', help='用于测试集的地图尺寸，逗号分隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 解析测试集地图尺寸
    test_sizes = [int(size) for size in args.test_sizes.split(',')]
    
    # 加载数据
    data = load_jsonl(args.data)
    if not data:
        return
    
    # 切分数据集
    split_datasets(
        data=data,
        test_sizes=test_sizes,
        sft_size_verify=args.sft_size_verify,
        rl_size_verify=args.rl_size_verify,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    print(f"数据集切分完成，结果已保存到 {args.output_dir}")

if __name__ == "__main__":
    main()