#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import cv2
from PIL import Image
from tqdm import tqdm
import re
from typing import List, Dict, Any, Optional, Tuple
import tempfile

# 用于验证路径安全性的函数
def extract_text_map(metadata_json):
    """从元数据中提取文本地图"""
    if "text_map" not in metadata_json or "output" not in metadata_json["text_map"] or "text_map" not in metadata_json["text_map"]["output"]:
        return None
        
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

def verify_safe_path(text_map, path_string):
    """验证给定的动作序列是否安全（不会落入冰洞）"""
    if text_map is None:
        return None
        
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
        return None
    
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
    for direction in directions:
        direction = direction.strip().upper()
        
        if direction not in direction_to_move:
            continue
        
        # 获取移动方向
        dr, dc = direction_to_move[direction]
        
        # 计算新位置
        new_row = curr_row + dr
        new_col = curr_col + dc
        
        # 检查边界 - 如果出界则位置不变
        if new_row < 0 or new_row >= map_size or new_col < 0 or new_col >= map_size:
            continue
        
        # 更新当前位置
        curr_row, curr_col = new_row, new_col
        
        # 获取当前格子的类型
        cell = text_map[curr_row][curr_col]
        cell_str = cell.decode('utf-8') if isinstance(cell, bytes) else cell
        
        # 检查是否掉入冰洞
        if cell_str == 'H':
            return False
    
    # 如果整个路径执行完毕没有掉入冰洞，则安全
    return True

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

def load_json(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON文件时出错: {e}")
        return None

def extract_path_from_conversation(conversations):
    """从对话中提取路径"""
    if not conversations:
        return None
    
    # 针对ShareGPT格式，找到提问中的路径
    for conv in conversations:
        if conv.get("from") == "human":
            text = conv.get("value", "")
            # 搜索类似 "R,L,U,U,D,U,D" 的路径格式
            path_match = re.search(r'序列是:?\s*([UDLR,]+)', text)
            if not path_match:
                path_match = re.search(r'sequence is:?\s*([UDLR,]+)', text, re.IGNORECASE)
            if not path_match:
                # 更宽松的搜索，匹配任何逗号分隔的UDLR序列
                path_match = re.search(r'([UDLR],[UDLR](?:,[UDLR])*)', text, re.IGNORECASE)
            
            if path_match:
                return path_match.group(1).strip()
    
    return None

def extract_safety_judgment(conversations):
    """从对话中提取安全性判断"""
    if not conversations:
        return None
    
    for conv in conversations:
        if conv.get("from") == "gpt":
            text = conv.get("value", "")
            # 搜索boxed{Yes}或boxed{No}格式
            yes_match = re.search(r'\\boxed\s*{\s*Yes\s*}', text)
            no_match = re.search(r'\\boxed\s*{\s*No\s*}', text)
            
            if yes_match:
                return True
            if no_match:
                return False
            
            # 如果没有找到boxed格式，则搜索普通文本
            if "安全" in text and "不安全" not in text:
                return True
            if "不安全" in text or "unsafe" in text.lower():
                return False
    
    return None

def extract_image_paths(item):
    """提取三种关键图像路径"""
    images = item.get("images", [])
    
    # 提取三种关键图像
    original_img = None
    elf_img = None
    path_img = None
    
    # 检查是否有足够的图像
    if len(images) >= 1:
        original_img = images[0]
    if len(images) >= 2:
        elf_img = images[1]
    if len(images) >= 3:
        path_img = images[2]
        
    return original_img, elf_img, path_img

def create_grid_visualization(text_map, path_string=None):
    """创建网格可视化"""
    if text_map is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "No Text Map Available", ha='center', va='center')
        ax.axis('off')
        return fig
        
    grid_size = len(text_map)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 创建网格
    for i in range(grid_size + 1):
        ax.axhline(i, color='black', linewidth=1)
        ax.axvline(i, color='black', linewidth=1)
    
    # 填充网格
    for i, row in enumerate(text_map):
        for j, cell in enumerate(row):
            cell_str = cell.decode('utf-8') if isinstance(cell, bytes) else cell
            if cell_str == 'H':  # 冰洞
                ax.add_patch(Rectangle((j, grid_size - i - 1), 1, 1, facecolor='skyblue', alpha=0.7))
                plt.text(j + 0.5, grid_size - i - 0.5, 'H', ha='center', va='center', fontsize=20, color='black')
            elif cell_str == 'S':  # 起点
                ax.add_patch(Rectangle((j, grid_size - i - 1), 1, 1, facecolor='green', alpha=0.3))
                plt.text(j + 0.5, grid_size - i - 0.5, 'S', ha='center', va='center', fontsize=20, color='green')
            elif cell_str == 'G':  # 终点
                ax.add_patch(Rectangle((j, grid_size - i - 1), 1, 1, facecolor='red', alpha=0.3))
                plt.text(j + 0.5, grid_size - i - 0.5, 'G', ha='center', va='center', fontsize=20, color='red')
            else:  # 普通冰面
                ax.add_patch(Rectangle((j, grid_size - i - 1), 1, 1, facecolor='white'))
    
    # 如果提供了路径，绘制路径
    if path_string:
        directions = path_string.split(',')
        direction_to_move = {
            'L': (0, -1),
            'R': (0, 1),
            'U': (-1, 0),
            'D': (1, 0),
        }
        
        # 找到起点
        start_row, start_col = None, None
        for i, row in enumerate(text_map):
            for j, cell in enumerate(row):
                cell_str = cell.decode('utf-8') if isinstance(cell, bytes) else cell
                if cell_str == 'S':
                    start_row, start_col = i, j
                    break
            if start_row is not None:
                break
        
        if start_row is not None:
            curr_row, curr_col = start_row, start_col
            path_points = [(curr_col, grid_size - curr_row - 1)]
            
            for direction in directions:
                direction = direction.strip().upper()
                if direction not in direction_to_move:
                    continue
                
                dr, dc = direction_to_move[direction]
                curr_row += dr
                curr_col += dc
                
                # 确保在边界内
                if 0 <= curr_row < grid_size and 0 <= curr_col < grid_size:
                    path_points.append((curr_col, grid_size - curr_row - 1))
            
            # 绘制路径
            if len(path_points) > 1:
                path_x, path_y = zip(*[(x + 0.5, y + 0.5) for x, y in path_points])
                ax.plot(path_x, path_y, 'r-', linewidth=2, marker='o', markersize=8)
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks([i + 0.5 for i in range(grid_size)])
    ax.set_yticks([i + 0.5 for i in range(grid_size)])
    ax.set_xticklabels([str(i + 1) for i in range(grid_size)])
    ax.set_yticklabels([str(i + 1) for i in range(grid_size)][::-1])
    ax.set_aspect('equal')
    
    path_text = f"Path: {path_string}" if path_string else "No path provided"
    ax.set_title(f'Grid Visualization\n{path_text}', fontsize=12)
    
    plt.tight_layout()
    return fig

def save_figure_to_array(fig):
    """保存图形为numpy数组"""
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        fig.savefig(tmpfile.name, dpi=100)
        img_array = cv2.imread(tmpfile.name)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        os.unlink(tmpfile.name)  # 删除临时文件
    return img_array

def visualize_data(sharegpt_data_path, sft_data_path, output_dir, limit=10):
    """可视化数据并验证路径安全性"""
    print(f"加载ShareGPT数据: {sharegpt_data_path}")
    sharegpt_data = load_json(sharegpt_data_path)
    
    print(f"加载SFT数据: {sft_data_path}")
    sft_data = load_jsonl(sft_data_path)
    
    # 将SFT数据按ID组织成字典，方便查找
    sft_data_dict = {item["id"]: item for item in sft_data}
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 记录统计信息
    stats = {
        "total_processed": 0,
        "path_found": 0,
        "safety_judgment_found": 0,
        "correct_judgments": 0,
        "incorrect_judgments": 0
    }
    
    # 记录验证结果
    verification_results = []
    
    # 处理数据
    print(f"可视化和验证数据 (限制为前{limit}项)...")
    count = 0
    
    for item in tqdm(sharegpt_data):
        if count >= limit:
            break
            
        qid = item.get("qid")
        if not qid or qid not in sft_data_dict:
            continue
            
        conversations = item.get("conversations", [])
        
        # 从SFT数据中获取元数据
        sft_item = sft_data_dict[qid]
        
        # 尝试从对话中提取路径
        path_string = extract_path_from_conversation(conversations)
        if not path_string and "path_drawings" in sft_item and "random" in sft_item["path_drawings"]:
            path_string = sft_item["path_drawings"]["random"].get("path")
        
        if not path_string:
            continue
            
        # 从对话中提取安全性判断
        safety_judgment = extract_safety_judgment(conversations)
        
        # 从SFT数据中提取文本地图并验证路径安全性
        text_map = extract_text_map(sft_item)
        actual_safety = verify_safe_path(text_map, path_string)
        
        # 如果无法验证安全性，跳过此项
        if actual_safety is None:
            continue
        
        # 更新统计信息
        stats["total_processed"] += 1
        stats["path_found"] += 1 if path_string else 0
        stats["safety_judgment_found"] += 1 if safety_judgment is not None else 0
        
        if safety_judgment is not None and actual_safety is not None:
            if safety_judgment == actual_safety:
                stats["correct_judgments"] += 1
            else:
                stats["incorrect_judgments"] += 1
        
        # 创建可视化
        plt.figure(figsize=(18, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
        
        # 提取图像路径
        original_img_path, elf_img_path, path_img_path = extract_image_paths(item)
        
        # 加载原始图像
        if original_img_path and os.path.exists(original_img_path):
            img = cv2.imread(original_img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax1 = plt.subplot(gs[0])
            ax1.imshow(img)
            ax1.set_title("Original Frozen Lake", fontsize=12)
            ax1.axis('off')
        else:
            ax1 = plt.subplot(gs[0])
            ax1.text(0.5, 0.5, "Original Image Not Found", ha='center', va='center')
            ax1.axis('off')
        
        # 加载精灵标记图像
        if elf_img_path and os.path.exists(elf_img_path):
            img = cv2.imread(elf_img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax2 = plt.subplot(gs[1])
            ax2.imshow(img)
            ax2.set_title("Elf Position", fontsize=12)
            ax2.axis('off')
        else:
            ax2 = plt.subplot(gs[1])
            ax2.text(0.5, 0.5, "Elf Image Not Found", ha='center', va='center')
            ax2.axis('off')
        
        # 加载路径可视化图像
        if path_img_path and os.path.exists(path_img_path):
            img = cv2.imread(path_img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax3 = plt.subplot(gs[2])
            ax3.imshow(img)
            ax3.set_title(f"Path: {path_string}", fontsize=12)
            ax3.axis('off')
        else:
            # 如果没有路径图像，生成网格可视化
            fig = create_grid_visualization(text_map, path_string)
            ax3 = plt.subplot(gs[2])
            img_array = save_figure_to_array(fig)
            ax3.imshow(img_array)
            ax3.set_title(f"Grid Visualization", fontsize=12)
            ax3.axis('off')
            plt.close(fig)  # 关闭原始图形
        
        # 添加安全性判断
        judgment_color = "green" if safety_judgment == actual_safety else "red"
        judgment_text = f"Path: {path_string}\nActual Safety: {'Safe' if actual_safety else 'Unsafe'}"
        if safety_judgment is not None:
            judgment_text += f"\nModel Judgment: {'Safe' if safety_judgment else 'Unsafe'}"
            judgment_text += f"\nVerdict: {'Correct' if safety_judgment == actual_safety else 'Incorrect'}"
        
        plt.figtext(0.5, 0.02, judgment_text, ha='center', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=judgment_color))
        
        # 添加对话摘要
        last_human = ""
        last_assistant = ""
        for conv in conversations:
            if conv.get("from") == "human":
                last_human = conv.get("value", "")
            elif conv.get("from") == "gpt":
                last_assistant = conv.get("value", "")
        
        # 从对话中提取关键内容
        human_text = f"Q: {path_string}" if path_string else "Q: (No path found)"
        assistant_text = f"A: {'Safe' if safety_judgment else 'Unsafe'}" if safety_judgment is not None else "A: (No judgment found)"
        
        plt.suptitle(f"ID: {qid}\n{human_text}", fontsize=12)
        
        # 保存可视化结果
        output_file = os.path.join(output_dir, f"{qid}.png")
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 记录验证结果
        verification_results.append({
            "id": qid,
            "path": path_string,
            "actual_safety": actual_safety,
            "judged_safety": safety_judgment,
            "judgment_correct": safety_judgment == actual_safety if safety_judgment is not None else None,
            "output_file": output_file
        })
        
        count += 1
    
    # 保存验证结果
    verification_output = os.path.join(output_dir, "verification_results.json")
    with open(verification_output, 'w', encoding='utf-8') as f:
        json.dump({
            "stats": stats,
            "results": verification_results
        }, f, indent=2)
    
    # 打印统计信息
    print("\n统计信息:")
    print(f"处理的项目总数: {stats['total_processed']}")
    print(f"找到路径的项目数: {stats['path_found']}")
    print(f"找到安全判断的项目数: {stats['safety_judgment_found']}")
    print(f"正确判断的项目数: {stats['correct_judgments']}")
    print(f"错误判断的项目数: {stats['incorrect_judgments']}")
    
    if stats['safety_judgment_found'] > 0:
        accuracy = stats['correct_judgments'] / stats['safety_judgment_found'] * 100
        print(f"判断准确率: {accuracy:.2f}%")
    
    print(f"\n验证结果已保存到 {verification_output}")
    print(f"可视化结果已保存到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='可视化冰湖数据并验证路径安全性')
    parser.add_argument('--sharegpt_data', type=str, 
                        default='/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/path_verify/verify_sharegpt_data.json',
                        help='ShareGPT数据文件路径')
    parser.add_argument('--sft_data', type=str, 
                        default='/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/metadata_split/path_verify/sft.jsonl',
                        help='SFT数据文件路径')
    parser.add_argument('--output_dir', type=str, 
                        default='/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/path_verify/correctness_check',
                        help='输出目录')
    parser.add_argument('--limit', type=int, default=20,
                        help='处理的最大项目数')
    
    args = parser.parse_args()
    
    visualize_data(args.sharegpt_data, args.sft_data, args.output_dir, args.limit)

if __name__ == "__main__":
    main()