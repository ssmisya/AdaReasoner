# calculate_statistics.py
import os
import json
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import shutil
from tqdm import tqdm

def check_path_reaches_goal_then_hole(map_text, path):
    """
    检查路径是否先达到目标点，然后再掉进冰窟窿
    
    Args:
        map_text (str): 地图文本
        path (str): 路径方向序列
        
    Returns:
        tuple: (是否先达目标再掉洞, 掉洞前是否达到目标)
    """
    lines = map_text.strip().split('\n')
    map_size = len(lines)
    
    # 找到起点、终点和洞的位置
    start_pos = None
    goal_pos = None
    holes = []
    
    for i, row in enumerate(lines):
        for j, cell in enumerate(row):
            if cell == 'S':
                start_pos = (i, j)
            elif cell == 'G':
                goal_pos = (i, j)
            elif cell == 'H':
                holes.append((i, j))
    
    if not start_pos or not goal_pos:
        return False, False
    
    # 追踪路径
    current_pos = start_pos
    reached_goal = False
    
    for direction in path:
        # 移动位置
        i, j = current_pos
        if direction.upper() == 'U':
            new_pos = (max(0, i-1), j)
        elif direction.upper() == 'D':
            new_pos = (min(map_size-1, i+1), j)
        elif direction.upper() == 'L':
            new_pos = (i, max(0, j-1))
        elif direction.upper() == 'R':
            new_pos = (i, min(map_size-1, j+1))
        else:
            continue  # 忽略无效方向
        
        current_pos = new_pos
        
        # 检查是否到达目标
        if current_pos == goal_pos:
            reached_goal = True
        
        # 检查是否掉入冰洞
        if current_pos in holes:
            return reached_goal, reached_goal
    
    # 如果没有掉入冰洞，返回False
    return False, reached_goal

def draw_direction_sequence(image, start, directions, step=64, line_width=3):
    """
    在图片上从起点沿方向序列画带箭头的线段。
    
    Args:
        image: PIL.Image对象或图片路径
        start (tuple): 起点坐标 (x, y)
        directions (str): 方向序列, 由 'u','d','l','r' 组成
        step (int): 每个方向移动的像素数
        line_width (int): 线条宽度
    
    Returns:
        PIL.Image: 带有绘制路径的图像
    """
    # 确保image是PIL.Image对象
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.copy().convert("RGB")
    else:
        raise ValueError("image must be a file path or a PIL Image object")
    
    draw = ImageDraw.Draw(img)
    
    # 当前坐标
    x, y = start
    
    # 遍历方向
    for dir in directions:
        old_x, old_y = x, y
        if dir.lower() == 'u':
            y -= step
        elif dir.lower() == 'd':
            y += step
        elif dir.lower() == 'l':
            x -= step
        elif dir.lower() == 'r':
            x += step
        else:
            raise ValueError(f"未知方向: {dir}")
        
        # 画从(old_x, old_y)到(x, y)的线
        draw.line([(old_x, old_y), (x, y)], fill="red", width=line_width)
        
        # 添加箭头头部
        arrow_size = line_width * 2
        
        # 根据方向计算箭头头部的点
        if dir.lower() == 'u':
            # 箭头指向上方
            arrow_points = [
                (x, y),
                (x - arrow_size, y + arrow_size * 2),
                (x + arrow_size, y + arrow_size * 2)
            ]
        elif dir.lower() == 'd':
            # 箭头指向下方
            arrow_points = [
                (x, y),
                (x - arrow_size, y - arrow_size * 2),
                (x + arrow_size, y - arrow_size * 2)
            ]
        elif dir.lower() == 'l':
            # 箭头指向左方
            arrow_points = [
                (x, y),
                (x + arrow_size * 2, y - arrow_size),
                (x + arrow_size * 2, y + arrow_size)
            ]
        elif dir.lower() == 'r':
            # 箭头指向右方
            arrow_points = [
                (x, y),
                (x - arrow_size * 2, y - arrow_size),
                (x - arrow_size * 2, y + arrow_size)
            ]
        
        # 绘制箭头头部
        draw.polygon(arrow_points, fill="red")
    
    return img

def extract_start_coordinates(map_text):
    """
    从地图文本中提取起点坐标
    
    Args:
        map_text (str): 地图文本
        
    Returns:
        tuple: (x, y) 像素坐标
    """
    lines = map_text.strip().split('\n')
    cell_size = 64  # 默认格子大小
    half_cell = cell_size / 2
    
    for i, row in enumerate(lines):
        for j, cell in enumerate(row):
            if cell == 'S':
                # 转换为像素坐标（格子中心）
                return (j * cell_size + half_cell, i * cell_size + half_cell)
    
    return None



def extract_coordinates_from_map(map_text):
    """
    从地图文本中提取起点、终点和障碍物的坐标
    
    Args:
        map_text (str): 地图文本
        
    Returns:
        dict: 包含起点、终点和障碍物的像素坐标
    """
    lines = map_text.strip().split('\n')
    cell_size = 64  # 默认格子大小
    half_cell = cell_size / 2
    
    start_point = None
    goal_point = None
    holes = []
    
    for i, row in enumerate(lines):
        for j, cell in enumerate(row):
            # 计算像素坐标（格子中心）
            pixel_x = j * cell_size + half_cell
            pixel_y = i * cell_size + half_cell
            
            if cell == 'S':
                start_point = (pixel_x, pixel_y)
            elif cell == 'H':
                holes.append((pixel_x, pixel_y))
            elif cell == 'G':
                goal_point = (pixel_x, pixel_y)
    
    return {
        'start': start_point,
        'goal': goal_point,
        'holes': holes
    }

def process_maze_data(base_dir):
    """
    处理迷宫数据，统计分布并绘制路径
    
    Args:
        base_dir (str): 基础目录路径
        
    Returns:
        dict: 统计信息
    """
    # 创建统计字典
    stats = {
        'total_mazes': 0,
        'mazes_by_level': {},
        'safe_paths': 0,
        'unsafe_paths': 0,
        'path_lengths': [],
        'path_distribution': {},
        'levels': []
    }
    
    # 迭代每个级别
    for level_dir in sorted(glob.glob(os.path.join(base_dir, 'task4/maps/level_step*'))):
        level_name = os.path.basename(level_dir)
        level_num = int(re.search(r'level_step(\d+)', level_name).group(1))
        stats['levels'].append(level_num)
        
        # 为该级别创建路径目录
        paths_dir = os.path.join(level_dir, 'paths')
        os.makedirs(paths_dir, exist_ok=True)
        
        # 初始化该级别的统计信息
        stats['mazes_by_level'][level_num] = {
            'total': 0,
            'safe_paths': 0,
            'unsafe_paths': 0,
            'avg_path_length': 0,
            'path_lengths': []
        }
        
        # 获取该级别的所有题目
        questions = sorted(glob.glob(os.path.join(level_dir, 'question/*.txt')))
        answers = sorted(glob.glob(os.path.join(level_dir, 'answer/*.txt')))
        tables = sorted(glob.glob(os.path.join(level_dir, 'text/*.txt')))
        
        # 确保文件数量一致
        assert len(questions) == len(answers) == len(tables), \
            f"文件数量不匹配: questions={len(questions)}, answers={len(answers)}, tables={len(tables)}"
        
        # 处理每个迷宫
        for i, (question_file, answer_file, table_file) in enumerate(zip(questions, answers, tables)):
            maze_id = os.path.basename(question_file).split('.')[0]
            stats['total_mazes'] += 1
            stats['mazes_by_level'][level_num]['total'] += 1
            
            # 读取问题、答案和地图
            with open(question_file, 'r') as f:
                question_text = f.read()
            
            with open(answer_file, 'r') as f:
                answer_text = f.read()
            
            with open(table_file, 'r') as f:
                map_text = f.read()
            
            # 提取路径和判断是否安全
            path = question_text
            is_safe = "y" == answer_text.lower()
            
            # 更新统计信息
            if path:
                path_length = len(path)
                stats['path_lengths'].append(path_length)
                stats['mazes_by_level'][level_num]['path_lengths'].append(path_length)
                
                # 更新路径分布
                if path_length not in stats['path_distribution']:
                    stats['path_distribution'][path_length] = 0
                stats['path_distribution'][path_length] += 1
                
                # 更新安全/不安全路径计数
                if is_safe:
                    stats['safe_paths'] += 1
                    stats['mazes_by_level'][level_num]['safe_paths'] += 1
                else:
                    stats['unsafe_paths'] += 1
                    stats['mazes_by_level'][level_num]['unsafe_paths'] += 1
                
                # 提取坐标并绘制路径
                coords = extract_coordinates_from_map(map_text)
                if coords['start']:
                    # 构建图像路径
                    img_file = os.path.join(level_dir, f'img/{maze_id}.png')
                    if os.path.exists(img_file):
                        # 绘制路径并保存
                        path_img = draw_direction_sequence(img_file, coords['start'], path)
                        path_img_file = os.path.join(paths_dir, f'{maze_id}_{"safe" if is_safe else "unsafe"}.png')
                        path_img.save(path_img_file)
    
    # 计算平均值
    for level, level_stats in stats['mazes_by_level'].items():
        if level_stats['path_lengths']:
            level_stats['avg_path_length'] = sum(level_stats['path_lengths']) / len(level_stats['path_lengths'])
    
    stats['avg_path_length'] = sum(stats['path_lengths']) / len(stats['path_lengths']) if stats['path_lengths'] else 0
    
    return stats

def generate_report(stats, output_dir):
    """
    生成统计报告
    
    Args:
        stats (dict): 统计信息
        output_dir (str): 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建报告文件
    report_file = os.path.join(output_dir, 'maze_statistics.md')
    
    with open(report_file, 'w') as f:
        # 写入标题
        f.write("# 迷宫数据集统计报告\n\n")
        
        # 总体信息
        f.write("## 总体统计\n\n")
        f.write(f"* 总迷宫数量: {stats['total_mazes']}\n")
        f.write(f"* 安全路径数量: {stats['safe_paths']} ({stats['safe_paths']/stats['total_mazes']*100:.1f}%)\n")
        f.write(f"* 不安全路径数量: {stats['unsafe_paths']} ({stats['unsafe_paths']/stats['total_mazes']*100:.1f}%)\n")
        f.write(f"* 平均路径长度: {stats['avg_path_length']:.2f}\n\n")
        
        # 按级别统计
        f.write("## 按级别统计\n\n")
        f.write("| 级别 | 迷宫数量 | 安全路径 | 不安全路径 | 平均路径长度 |\n")
        f.write("| ---- | -------- | -------- | ---------- | ------------ |\n")
        
        for level in sorted(stats['mazes_by_level'].keys()):
            level_stats = stats['mazes_by_level'][level]
            f.write(f"| {level} | {level_stats['total']} | {level_stats['safe_paths']} | {level_stats['unsafe_paths']} | {level_stats['avg_path_length']:.2f} |\n")
        
        # 路径长度分布
        f.write("\n## 路径长度分布\n\n")
        f.write("| 路径长度 | 数量 | 百分比 |\n")
        f.write("| -------- | ---- | ------ |\n")
        
        for length, count in sorted(stats['path_distribution'].items()):
            percentage = count / stats['total_mazes'] * 100
            f.write(f"| {length} | {count} | {percentage:.1f}% |\n")
    
    # 生成图表
    plt.figure(figsize=(12, 8))
    
    # 路径长度分布柱状图
    lengths = sorted(stats['path_distribution'].keys())
    counts = [stats['path_distribution'][length] for length in lengths]
    
    plt.bar(lengths, counts)
    plt.title('路径长度分布')
    plt.xlabel('路径长度')
    plt.ylabel('数量')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'path_length_distribution.png'))
    
    # 级别与安全/不安全路径数量
    plt.figure(figsize=(12, 8))
    
    levels = sorted(stats['mazes_by_level'].keys())
    safe_counts = [stats['mazes_by_level'][level]['safe_paths'] for level in levels]
    unsafe_counts = [stats['mazes_by_level'][level]['unsafe_paths'] for level in levels]
    
    bar_width = 0.35
    x = np.arange(len(levels))
    
    plt.bar(x - bar_width/2, safe_counts, bar_width, label='安全路径')
    plt.bar(x + bar_width/2, unsafe_counts, bar_width, label='不安全路径')
    
    plt.title('各级别安全与不安全路径数量')
    plt.xlabel('级别')
    plt.ylabel('数量')
    plt.xticks(x, levels)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'safety_by_level.png'))
    
    print(f"报告已生成: {report_file}")

def process_maze_data_with_goal_hole_check(base_dir):
    """
    处理迷宫数据，特别检查"先达目标再掉洞"的情况
    
    Args:
        base_dir (str): 基础目录路径
        
    Returns:
        dict: 统计信息
    """
    # 创建统计字典
    stats = {
        'total_mazes': 0,
        'goal_then_hole': {
            'count': 0,
            'safe_count': 0,
            'unsafe_count': 0,
            'examples': []
        },
        'reached_goal': {
            'count': 0,
            'safe_count': 0,
            'unsafe_count': 0
        }
    }
    
    # 迭代每个级别
    for level_dir in sorted(glob.glob(os.path.join(base_dir, 'task4/maps/level_step*'))):
        level_name = os.path.basename(level_dir)
        
        # 获取该级别的所有题目
        questions = sorted(glob.glob(os.path.join(level_dir, 'question/*.txt')))
        answers = sorted(glob.glob(os.path.join(level_dir, 'answer/*.txt')))
        tables = sorted(glob.glob(os.path.join(level_dir, 'text/*.txt')))
        
        # 确保文件数量一致
        if not (len(questions) == len(answers) == len(tables)):
            print(f"警告: 文件数量不匹配: {level_name} - questions={len(questions)}, answers={len(answers)}, tables={len(tables)}")
            continue
        
        print(f"处理 {level_name} - {len(questions)} 个迷宫")
        
        # 处理每个迷宫
        for i, (question_file, answer_file, table_file) in enumerate(tqdm(zip(questions, answers, tables), total=len(questions))):
            maze_id = os.path.basename(question_file).split('.')[0]
            stats['total_mazes'] += 1
            
            # 读取问题、答案和地图
            with open(question_file, 'r') as f:
                question_text = f.read()
            
            with open(answer_file, 'r') as f:
                answer_text = f.read()
            
            with open(table_file, 'r') as f:
                map_text = f.read()
            
            # 提取路径和判断是否安全
            path = question_text
            is_safe = "yes" in answer_text.lower() or "<output> yes" in answer_text.lower()
            
            # 如果成功提取路径
            if path:
                # 检查路径是否先达到目标点再掉入冰洞
                goal_then_hole, reached_goal = check_path_reaches_goal_then_hole(map_text, path)
                
                # 更新统计
                if reached_goal:
                    stats['reached_goal']['count'] += 1
                    if is_safe:
                        stats['reached_goal']['safe_count'] += 1
                    else:
                        stats['reached_goal']['unsafe_count'] += 1
                
                if goal_then_hole:
                    stats['goal_then_hole']['count'] += 1
                    if is_safe:
                        stats['goal_then_hole']['safe_count'] += 1
                    else:
                        stats['goal_then_hole']['unsafe_count'] += 1
                    
                    # 保存示例
                    stats['goal_then_hole']['examples'].append({
                        'level': level_name,
                        'id': maze_id,
                        'is_safe': is_safe,
                        'map': map_text,
                        'path': path,
                        'question_file': question_file,
                        'answer_file': answer_file,
                        'table_file': table_file
                    })
    
    return stats

def main_goal_hole_check():
    """专门用于检查先到达目标再掉入冰洞的路径安全性"""
    # 基础目录
    base_dir = "/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/src/Visual-Spatial-Planning/VSP-main/maze"
    
    print("正在检查'先达到目标再掉入冰洞'的情况...")
    stats = process_maze_data_with_goal_hole_check(base_dir)
    
    # 打印结果
    print("\n====== 检查结果 ======")
    print(f"总共处理迷宫数量: {stats['total_mazes']}")
    print(f"到达目标的路径总数: {stats['reached_goal']['count']} ({stats['reached_goal']['count']/stats['total_mazes']*100:.1f}%)")
    print(f"  - 被标记为安全: {stats['reached_goal']['safe_count']} ({stats['reached_goal']['safe_count']/stats['reached_goal']['count']*100:.1f}% 如果有路径)")
    print(f"  - 被标记为不安全: {stats['reached_goal']['unsafe_count']} ({stats['reached_goal']['unsafe_count']/stats['reached_goal']['count']*100:.1f}% 如果有路径)")
    
    print(f"\n先达到目标再掉入冰洞的路径数: {stats['goal_then_hole']['count']}")
    
    if stats['goal_then_hole']['count'] > 0:
        print(f"  - 被标记为安全: {stats['goal_then_hole']['safe_count']} ({stats['goal_then_hole']['safe_count']/stats['goal_then_hole']['count']*100:.1f}%)")
        print(f"  - 被标记为不安全: {stats['goal_then_hole']['unsafe_count']} ({stats['goal_then_hole']['unsafe_count']/stats['goal_then_hole']['count']*100:.1f}%)")
        
        # 显示示例
        print("\n示例:")
        for i, example in enumerate(stats['goal_then_hole']['examples'][:5]):  # 显示前5个示例
            print(f"示例 {i+1}:")
            print(f"  级别: {example['level']}")
            print(f"  ID: {example['id']}")
            print(f"  安全性: {'安全' if example['is_safe'] else '不安全'}")
            print(f"  路径: {example['path']}")
            print(f"  地图:\n{example['map']}")
            print()
        
        # 结论
        if stats['goal_then_hole']['safe_count'] > stats['goal_then_hole']['unsafe_count']:
            conclusion = "安全（多数情况）"
        elif stats['goal_then_hole']['unsafe_count'] > stats['goal_then_hole']['safe_count']:
            conclusion = "不安全（多数情况）"
        else:
            conclusion = "结果不明确（安全和不安全情况相同）"
    else:
        conclusion = "没有发现此类情况"
    
    print(f"\n结论: {conclusion}")
    
    # 保存结果到文件
    with open("goal_hole_check_results.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    print("\n结果已保存到 goal_hole_check_results.json")

if __name__ == "__main__":
    main_goal_hole_check()

