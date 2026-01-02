# path_verify_validation.py
import os
import json
import argparse
from tqdm import tqdm
import re
from typing import List, Dict, Any, Optional, Set
from tool_server.utils.utils import *

def verify_safe_path(text_map, path_string, verbose=False):
    """
    验证给定的动作序列是否安全（不会落入冰洞）
    
    Args:
        text_map: 表示FrozenLake地图的2D列表
        path_string: 逗号分隔的动作字符串 (R,L,U,D)
        verbose: 是否打印详细信息
    
    Returns:
        bool: 如果路径安全则为True，否则为False
    """
    # 找到起点位置
    start_row, start_col = None, None
    for i, row in enumerate(text_map):
        for j, cell in enumerate(row):
            cell_str = cell.decode('utf-8') if isinstance(cell, bytes) else cell
            if cell_str == 'S':
                start_row, start_col = i, j
                break
        if start_row is not None:
            break
    
    if start_row is None:
        if verbose:
            print("未找到起点")
        return False
    
    # 解析路径字符串
    directions = path_string.split(',')
    
    # 字典，将方向映射到位置变化
    direction_to_move = {
        'L': (0, -1),
        'R': (0, 1),
        'U': (-1, 0),
        'D': (1, 0),
    }
    
    # 跟踪当前位置
    curr_row, curr_col = start_row, start_col
    map_size = len(text_map)
    
    # 逐步执行路径
    for idx, direction in enumerate(directions):
        direction = direction.strip().upper()
        
        if direction not in direction_to_move:
            if verbose:
                print(f"无效的方向: {direction}")
            continue
        
        # 获取移动方向
        dr, dc = direction_to_move[direction]
        
        # 计算新位置
        new_row = curr_row + dr
        new_col = curr_col + dc
        
        # 检查边界 - 如果出界则位置不变
        if new_row < 0 or new_row >= map_size or new_col < 0 or new_col >= map_size:
            if verbose:
                print(f"步骤 {idx+1}: {direction} - 位置保持不变（超出边界）")
            continue
        
        # 更新当前位置
        curr_row, curr_col = new_row, new_col
        
        # 获取当前格子的类型
        cell = text_map[curr_row][curr_col]
        cell_str = cell.decode('utf-8') if isinstance(cell, bytes) else cell
        
        # 检查是否掉入冰洞
        if cell_str == 'H':
            if verbose:
                print(f"步骤 {idx+1}: {direction} - 掉入冰洞")
            return False
    
    # 如果整个路径执行完毕没有掉入冰洞，则安全
    return True

def extract_text_map(metadata_json):
    """
    从元数据中提取文本地图
    
    Args:
        metadata_json: 包含元数据的JSON对象
    
    Returns:
        list: 2D文本地图
    """
    text_map_str = metadata_json["text_map"]["output"]["text_map"]
    assert text_map_str
    
    # 解析文本地图
    lines = text_map_str.strip().split('\n')
    grid = []
    
    # 跳过表头行
    for line in lines[1:]:
        parts = line.split('|')
        # 删除第一个元素（行标签）和空元素
        cells = [part.strip() for part in parts[2:] if part.strip()]
        row = []
        for cell in cells:
            if cell == '_':
                row.append('F')  # 空白格转换为F (Frozen)
            elif cell == '@':
                row.append('S')  # @ 符号转换为S (Start)
            elif cell == '*':
                row.append('G')  # * 符号转换为G (Goal)
            elif cell == '#':
                row.append('H')  # # 符号转换为H (Hole)
        if row:
            grid.append(row)
    
    return grid


def extract_boxed_answer(text):
    """
    从文本中提取\boxed{}内的答案
    
    Args:
        text: 包含boxed答案的文本
    
    Returns:
        str: 答案文本，没有找到则返回None
    """
    pattern = r'\\boxed{([^}]*)}'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

def extract_path_from_sft_item(sft_item):
    """
    从SFT项中提取路径
    
    Args:
        sft_item: SFT数据项
        
    Returns:
        str: 路径字符串
    """
    # 尝试从additional_data中获取路径
    if "additional_data" in sft_item and "path" in sft_item["additional_data"]:
        return sft_item["additional_data"]["path"]
    else:
        raise ValueError("SFT项中没有找到路径信息")
    # 尝试从轨迹数据中获取
    if "trajectory_data" in sft_item:
        for message in sft_item["trajectory_data"]:
            if message["role"] == "user":
                for content in message.get("content", []):
                    if content.get("type") == "text":
                        text = content.get("text", "")
                        # 尝试从用户问题中提取路径
                        path_match = re.search(r'The action sequence is:\s*\n\s*([UDLR,]+)', text)
                        if path_match:
                            return path_match.group(1).strip()
    
    # 尝试从sharegpt_instance中获取
    if "sharegpt_instance" in sft_item:
        for message in sft_item["sharegpt_instance"].get("conversations", []):
            if message.get("from") == "human":
                text = message.get("value", "")
                path_match = re.search(r'The action sequence is:\s*\n\s*([UDLR,]+)', text)
                if path_match:
                    return path_match.group(1).strip()
    
    return None

def extract_is_safe_from_sft_item(sft_item):
    """
    从SFT项中提取is_safe标志
    
    Args:
        sft_item: SFT数据项
        
    Returns:
        bool: 是否安全
    """
    # 直接从additional_data获取
    if "additional_data" in sft_item and "is_safe" in sft_item["additional_data"]:
        return sft_item["additional_data"]["is_safe"]
    else:
        raise ValueError("SFT项中没有找到is_safe信息")
    # 从sharegpt_instance中提取boxed答案
    if "sharegpt_instance" in sft_item:
        for message in sft_item["sharegpt_instance"].get("conversations", []):
            if message.get("from") == "gpt":
                answer = extract_boxed_answer(message.get("value", ""))
                if answer:
                    return answer.strip().lower() == "yes"
    
    return None

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"加载JSONL文件时出错: {e}")
    return data

def validate_generated_data(sft_file, verify_sft_file, original_metadata_dir):
    """
    验证生成的SFT数据
    
    Args:
        sft_file: 原始SFT数据文件路径
        verify_sft_file: 验证SFT数据文件路径
        original_metadata_dir: 原始元数据目录
        
    Returns:
        dict: 验证结果
    """
    print(f"加载原始SFT数据: {sft_file}")
    sft_data = load_jsonl(sft_file)
    print(f"加载验证SFT数据: {verify_sft_file}")
    verify_data = load_jsonl(verify_sft_file)
    
    # 将原始SFT数据按ID组织成字典，方便查找
    sft_data_dict = {item["id"]: item for item in sft_data}
    verify_data_dict = {item["id"]: item for item in verify_data}
    
    # 共同的ID集合
    common_ids = set(sft_data_dict.keys()) & set(verify_data_dict.keys())
    print(f"共同ID数量: {len(common_ids)}")
    
    # 验证结果
    results = {
        "total_items": len(verify_data),
        "common_items": len(common_ids),
        "safety_check": {
            "correct": 0,
            "incorrect": 0,
            "unknown": 0
        },
        "label_consistency": {
            "consistent": 0,
            "inconsistent": 0,
            "unknown": 0
        },
        "detailed_errors": []
    }
    
    print("验证数据...")
    for item_id in tqdm(common_ids):
        verify_item = verify_data_dict[item_id]
        sft_item = sft_data_dict[item_id]
        
        # 提取路径和安全标签
        path = extract_path_from_sft_item(verify_item)
        is_safe_label = extract_is_safe_from_sft_item(verify_item)
        
        # 使用原始元数据验证路径安全性
        if path:
            
            text_map = extract_text_map(sft_item)
            # sft_item["text_map"]["output"]["text_map"]
            try:
                actual_is_safe = verify_safe_path(text_map, path)
            except:
                actual_is_safe = None
                print(f"验证路径安全性时出错: {item_id}, 路径: {path}")
            # 记录安全性检查结果
            if actual_is_safe is None:
                results["safety_check"]["unknown"] += 1
            elif actual_is_safe == is_safe_label:
                results["safety_check"]["correct"] += 1
            else:
                results["safety_check"]["incorrect"] += 1
                results["detailed_errors"].append({
                    "id": item_id,
                    "path": path,
                    "expected_is_safe": actual_is_safe,
                    "got_is_safe": is_safe_label,
                    "error_type": "safety_mismatch"
                })
        else:
            results["safety_check"]["unknown"] += 1
        
        # 检查标签一致性
        sft_is_safe = None
        if "additional_data" in sft_item and "is_safe" in sft_item["additional_data"]:
            sft_is_safe = sft_item["additional_data"]["is_safe"]
        
        verify_is_safe = None
        if "additional_data" in verify_item and "is_safe" in verify_item["additional_data"]:
            verify_is_safe = verify_item["additional_data"]["is_safe"]
        
        if sft_is_safe is None or verify_is_safe is None:
            results["label_consistency"]["unknown"] += 1
        elif sft_is_safe == verify_is_safe:
            results["label_consistency"]["consistent"] += 1
        else:
            results["label_consistency"]["inconsistent"] += 1
            results["detailed_errors"].append({
                "id": item_id,
                "sft_is_safe": sft_is_safe,
                "verify_is_safe": verify_is_safe,
                "error_type": "label_inconsistency"
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='验证生成的路径验证SFT数据')
    parser.add_argument('--sft_file', type=str, default='/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/metadata_split/path_verify/sft.jsonl', 
                        help='原始SFT数据文件路径')
    parser.add_argument('--verify_file', type=str, default='/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/path_verify/verify_sft_data.jsonl', 
                        help='验证SFT数据文件路径')
    parser.add_argument('--metadata_dir', type=str, default=None, 
                        help='原始元数据目录（可选）')
    parser.add_argument('--output', type=str, default='/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/metadata_split/path_verify/validation_results.json', 
                        help='验证结果输出文件')
    
    args = parser.parse_args()
    
    results = validate_generated_data(args.sft_file, args.verify_file, args.metadata_dir)
    
    # 打印摘要
    print("\n验证结果摘要:")
    print(f"总项目数: {results['total_items']}")
    print(f"共同ID数: {results['common_items']}")
    
    print("\n安全性检查:")
    total_safety_checks = sum(results["safety_check"].values())
    if total_safety_checks > 0:
        print(f"  正确: {results['safety_check']['correct']} ({results['safety_check']['correct']/total_safety_checks*100:.1f}%)")
        print(f"  不正确: {results['safety_check']['incorrect']} ({results['safety_check']['incorrect']/total_safety_checks*100:.1f}%)")
        print(f"  未知: {results['safety_check']['unknown']} ({results['safety_check']['unknown']/total_safety_checks*100:.1f}%)")
    
    print("\n标签一致性:")
    total_consistency_checks = sum(results["label_consistency"].values())
    if total_consistency_checks > 0:
        print(f"  一致: {results['label_consistency']['consistent']} ({results['label_consistency']['consistent']/total_consistency_checks*100:.1f}%)")
        print(f"  不一致: {results['label_consistency']['inconsistent']} ({results['label_consistency']['inconsistent']/total_consistency_checks*100:.1f}%)")
        print(f"  未知: {results['label_consistency']['unknown']} ({results['label_consistency']['unknown']/total_consistency_checks*100:.1f}%)")
    
    # 保存详细结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n详细结果已保存到 {args.output}")
    
    # 如果有错误，输出一些样例
    if results["detailed_errors"]:
        print("\n错误样例:")
        for i, error in enumerate(results["detailed_errors"][:5]):  # 只显示前5个错误
            print(f"  错误 {i+1}:")
            for k, v in error.items():
                print(f"    {k}: {v}")
            print()

if __name__ == "__main__":
    main()