import json
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
import random
from pathlib import Path
import os
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
# 路径验证任务说明
PATH_VERIFY_TASK_INSTRUCTION = """
You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success. 

Now please determine if the action sequence is safe for the given maze. Your final answer should be formatted as \\boxed{Yes} or \\boxed{No}.

The action sequence is:

<ACTION-SEQ>
"""

# 路径导航任务说明
PATH_NAVIGATION_INSTRUCTION = """
You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success. Your final answer should be formatted as \\boxed{L,R,U,D}.

Please generate action plan for the input maze image.
"""

def convert_to_gym_map(item):
    """将数据项转换为gym环境可用的地图格式"""
    # 首先从text_map中提取文本地图
    # if "text_map" in item and "output" in item["text_map"] and "text_map" in item["text_map"]["output"]:
    #     text_map_str = item["text_map"]["output"]["text_map"]
    #     # 解析文本地图
    #     rows = []
    #     for line in text_map_str.split('\n'):
    #         if '|' in line and ('Row' in line or 'Col' in line):
    #             continue  # 跳过表头行
            
    #         row_cells = []
    #         cells = line.split('|')
    #         for cell in cells[1:]:  # 跳过第一个空元素
    #             if not cell.strip():
    #                 continue
                
    #             cell_value = cell.strip()
    #             if cell_value == '_':  # 空格表示安全区域
    #                 row_cells.append('F')
    #             elif cell_value == '#':  # '#' 表示冰洞
    #                 row_cells.append('H')
    #             elif cell_value == '@':  # '@' 表示起点
    #                 row_cells.append('S')
    #             elif cell_value == '*':  # '*' 表示终点
    #                 row_cells.append('G')
            
    #         if row_cells:  # 如果行不为空
    #             rows.append(row_cells)
        
    #     # 检查地图是否有效
    #     if rows and all(len(row) == len(rows[0]) for row in rows):
    #         return rows
    
    # 如果无法从text_map中提取，则从坐标信息中构建
    size = item["size"]
    cell_size = 64  # 假设每个单元格是64像素
    
    # 创建一个全是安全区域的地图
    map_data = [['F' for _ in range(size)] for _ in range(size)]
    
    # 设置起点
    start_x = int(item["start_coords"][0] / cell_size)
    start_y = int(item["start_coords"][1] / cell_size)
    map_data[start_y][start_x] = 'S'
    
    # 设置终点
    goal_x = int(item["goal_coords"][0] / cell_size)
    goal_y = int(item["goal_coords"][1] / cell_size)
    map_data[goal_y][goal_x] = 'G'
    
    # 设置障碍物
    for obs in item["obstacle_coords"]:
        obs_x = int(obs[0] / cell_size)
        obs_y = int(obs[1] / cell_size)
        # 确保坐标在有效范围内
        if 0 <= obs_y < size and 0 <= obs_x < size:
            map_data[obs_y][obs_x] = 'H'
    
    return map_data

def read_image_to_bytes(image_path):
    """读取图片文件并转换为bytes格式"""
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"读取图片失败: {image_path}, 错误: {e}")
        return None

def process_navigation_data(jsonl_file, image_dir=None, system_prompt=None):
    """处理navigation任务的JSONL数据"""
    data = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            
            # 读取图片
            image_path = record['image_path']
            image_path = os.path.join(image_dir, image_path[2:])
            image_bytes = read_image_to_bytes(image_path)
            if image_bytes is None:
                print(f"读取图片失败，报错，中断构造过程，在navigation")
                print("image_path", image_path)
                raise ValueError("读取图片失败，报错，中断构造过程")
            
            # 构造prompt
            prompt_nav = [
                {
                    "content": system_prompt,
                    "role": "system"
                },
                {
                    "content": f"{PATH_NAVIGATION_INSTRUCTION}\n<image>",
                    "role": "user"
                }
            ]

            gym_map = convert_to_gym_map(record)
            answer = str(gym_map)
            
            # 构造数据项
            item = {
                "data_source": "path_nav",
                "prompt": prompt_nav,
                "images": [{"bytes": image_bytes}],
                "ability": "forzenlake",
                "env_name": "forzenlake",
                "reward_model": {
                    "ground_truth": answer,
                    "style": "model"
                },
                "extra_info": {
                    "answer": answer,
                    "index": record["id"],
                    "split": "test"
                }
            }
            
            data.append(item)
    
    return data

def process_verify_data(jsonl_file, image_dir=None, system_prompt=None):
    """处理verify任务的JSONL数据"""
    data = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            
            # 读取图片
            image_path = record['image_path']
            image_path = os.path.join(image_dir, image_path[2:])
            image_bytes = read_image_to_bytes(image_path)
            if image_bytes is None:
                print(f"读取图片失败，报错，中断构造过程，在verify")
                print("image_path", image_path)
                raise ValueError("读取图片失败，报错，中断构造过程")
            
            # 获取路径序列 - 取最后一个path
            path_sequence = None
            if 'path_drawings' in record and 'random' in record['path_drawings']:
                path_sequence = record['path_drawings']['random'].get('path', '')
            
            if not path_sequence:
                print(f"没有找到路径序列，报错，中断构造过程")
                raise ValueError("没有找到路径序列，报错，中断构造过程")
            
            # 获取is_safe值
            is_safe = None
            if 'path_drawings' in record and 'random' in record['path_drawings']:
                is_safe = record['path_drawings']['random'].get('is_safe', False)
            
            if is_safe is None:
                print("没有找到is_safe，中断构造过程")
                raise ValueError("没有找到is_safe，中断构造过程")
            question_instruction = PATH_VERIFY_TASK_INSTRUCTION.replace("<ACTION-SEQ>", path_sequence)
            # 构造prompt
            prompt_ver = [
                {
                    "content": system_prompt,
                    "role": "system"
                },
                {
                    "content": f"{question_instruction}\n<image>",
                    "role": "user"
                }
            ]
            
            # 构造数据项
            if is_safe:
                answer = "yes"
            else:
                answer = "no"
            item = {
                "data_source": "path_ver",
                "prompt": prompt_ver,
                "images": [{"bytes": image_bytes}],
                "ability": "forzenlake",
                "env_name": "forzenlake",
                "reward_model": {
                    "ground_truth": answer,
                    "style": "model"
                },
                "extra_info": {
                    "answer": answer,
                    "index": record["id"],
                    "split": "test"
                }
            }
            
            data.append(item)
    
    return data

def test_parquet_file(parquet_file):
    """测试构造好的Parquet文件"""
    
    print("=== 测试Parquet文件 ===")
    
    # 读取Parquet文件
    df = pd.read_parquet(parquet_file)
    
    # 基本信息
    print(f"总记录数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    print(f"各数据源统计:")
    print(df['data_source'].value_counts())
    
    print("\n=== 数据样例检查 ===")
    
    # 检查navigation数据样例
    nav_sample = df[df['data_source'] == 'path_nav'].iloc[0]
    print("\nNavigation样例:")
    print(f"data_source: {nav_sample['data_source']}")
    print(f"ability: {nav_sample['ability']}")
    print(f"env_name: {nav_sample['env_name']}")
    print(f"prompt长度: {len(nav_sample['prompt'])}")
    print(f"prompt: {nav_sample['prompt']}")
    print(f"images数量: {len(nav_sample['images'])}")
    print(f"reward_model: {nav_sample['reward_model']}")
    print(f"extra_info: {nav_sample['extra_info']}")
    
    # 检查verify数据样例
    ver_sample = df[df['data_source'] == 'path_ver'].iloc[0]
    print("\nVerify样例:")
    print(f"data_source: {ver_sample['data_source']}")
    print(f"ability: {ver_sample['ability']}")
    print(f"env_name: {ver_sample['env_name']}")
    print(f"prompt长度: {len(ver_sample['prompt'])}")
    print(f"prompt: {ver_sample['prompt']}")
    print(f"images数量: {len(ver_sample['images'])}")
    print(f"reward_model: {ver_sample['reward_model']}")
    print(f"extra_info: {ver_sample['extra_info']}")
    
    # 检查图片数据
    print("\n=== 图片数据检查 ===")
    image_bytes = nav_sample['images'][0]['bytes']
    print(f"第一张图片字节数: {len(image_bytes)}")
    
    # 尝试恢复图片（验证图片数据完整性）
    try:
        img_bytes = bytes(image_bytes)
        img = Image.open(io.BytesIO(img_bytes))
        print(f"图片尺寸: {img.size}")
        print("图片数据验证成功")
    except Exception as e:
        print(f"图片数据验证失败: {e}")
    

    
    print("\n测试完成!")
    
    
def form_data_to_parquet(data, output_file):

    print(f"总共{len(data)}条数据")
    
    # 打乱顺序
    random.shuffle(data)
    print("数据已打乱顺序")
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # 保存为Parquet
    df.to_parquet(output_file, index=False)
    print(f"Parquet文件已保存到: {output_file}")
    

def create_parquet_from_jsonl():
    """主函数：从JSONL文件创建Parquet文件"""
    
    selected_tools = ["Point","Draw2DPath"] # "AStarWithPixelCoordinate",
    image_dir = "/mnt/petrelfs/share_data/songmingyang/data/vl_reasoning/tool_dataset"
    dataset_base_dir = "/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/frozen_lake_metadata_v2"
    seperately = True
    # 文件路径

    # nav_file = f"{dataset_base_dir}/metadata_split/path_navigation/test.jsonl"
    # ver_file = f"{dataset_base_dir}/metadata_split/path_verify/test.jsonl"
    # output_file = f"{dataset_base_dir}/rl_data/2tools/test.parquet"
    
    nav_file = f"{dataset_base_dir}/metadata_split/path_navigation/rl.jsonl"
    ver_file = f"{dataset_base_dir}/metadata_split/path_verify/rl.jsonl"
    output_file = f"{dataset_base_dir}/rl_data/2tools/train.parquet"
    
    
    tool_manager = ToolManager(tools=selected_tools)
    system_prompt = tool_manager.get_tool_prompt()
    
    
    print("开始处理navigation数据...")
    nav_data = process_navigation_data(nav_file,image_dir=image_dir,system_prompt=system_prompt)
    print(f"处理完成，共{len(nav_data)}条navigation数据")
    
    print("开始处理verify数据...")
    ver_data = process_verify_data(ver_file,image_dir=image_dir,system_prompt=system_prompt)
    print(f"处理完成，共{len(ver_data)}条verify数据")
    
    # 合并数据
    all_data = nav_data + ver_data
    form_data_to_parquet(all_data, output_file)
    
    if seperately:
        nav_output_file = output_file.replace(".parquet","_nav.parquet")
        ver_output_file = output_file.replace(".parquet","_ver.parquet")
        form_data_to_parquet(nav_data, nav_output_file)
        form_data_to_parquet(ver_data, ver_output_file)
    
    return output_file

if __name__ == "__main__":
    # 创建Parquet文件
    parquet_file = create_parquet_from_jsonl()
    
    # 测试Parquet文件
    test_parquet_file(parquet_file)