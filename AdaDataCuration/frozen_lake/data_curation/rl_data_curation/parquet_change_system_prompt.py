import pandas as pd
import os
import argparse
from pathlib import Path
import json
import random
import numpy as np
from tqdm import tqdm
import copy
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from datasets import load_dataset
from tool_server.utils.utils import *

def standardize_extra_info(extra_info):
    """
    标准化extra_info字段，只保留answer, index, split三个键，并确保所有值都是字符串
    
    Args:
        extra_info: 原始extra_info对象
        
    Returns:
        dict: 标准化后的extra_info字典
    """
    if extra_info is None:
        return {'answer': '', 'index': '', 'split': ''}
    
    # 如果是字符串，尝试解析为JSON
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except:
            return {'answer': str(extra_info), 'index': '', 'split': ''}
    
    # 如果不是字典，转换为字典
    if not isinstance(extra_info, dict):
        return {'answer': str(extra_info), 'index': '', 'split': ''}
    
    # 创建新的标准化字典（只保留三个键）
    standard_info = {}
    
    # 处理answer键
    if 'answer' in extra_info:
        standard_info['answer'] = str(extra_info['answer'])
    else:
        # 如果没有answer键，尝试从其他可能的键获取
        possible_answer_keys = ['correct_answer', 'prediction', 'result', 'response', 'ground_truth']
        for key in possible_answer_keys:
            if key in extra_info:
                standard_info['answer'] = str(extra_info[key])
                break
        else:
            standard_info['answer'] = ''
    
    # 处理index键
    if 'index' in extra_info:
        standard_info['index'] = str(extra_info['index'])
    else:
        # 如果没有index键，尝试从其他可能的键获取
        possible_index_keys = ['id', 'question_id', 'item_id']
        for key in possible_index_keys:
            if key in extra_info:
                standard_info['index'] = str(extra_info[key])
                break
        else:
            standard_info['index'] = ''
    
    # 处理split键
    if 'split' in extra_info:
        standard_info['split'] = str(extra_info['split'])
    else:
        possible_split_keys = ['dataset_split', 'type']
        for key in possible_split_keys:
            if key in extra_info:
                standard_info['split'] = str(extra_info[key])
                break
        else:
            standard_info['split'] = ''
    
    if 'question' in extra_info:
        standard_info['question'] = str(extra_info['question'])
    else:        
        standard_info['question'] = ""
    
    return standard_info

def update_system_prompt(prompt_data, new_system_prompt):
    """更新对话中的系统提示"""
    if isinstance(prompt_data, (list, np.ndarray)) and len(prompt_data) > 0:
        # 如果是numpy数组，先转换为列表
        prompt_list = prompt_data.tolist() if isinstance(prompt_data, np.ndarray) else prompt_data
        
        if isinstance(prompt_list[0], dict) and prompt_list[0].get('role') == 'system':
            new_prompt = copy.deepcopy(prompt_list)
            new_prompt[0]['content'] = new_system_prompt
            return new_prompt
    return prompt_data

def process_parquet_file(file_path, new_system_prompt=None):
    """
    处理单个Parquet文件
    
    Args:
        file_path: Parquet文件路径
        new_system_prompt: 新的系统提示内容，如果不为None则更新系统提示
        
    Returns:
        list: 处理后的数据项列表
    """
    try:
        # 读取Parquet文件
        df = pd.read_parquet(file_path)
        print(f"读取文件: {file_path}, 包含 {len(df)} 条记录")
        
        # 将DataFrame转换为字典列表
        data_list = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {os.path.basename(file_path)}"):
            item = row.to_dict()
            
            
            item['extra_info'] = standardize_extra_info(item['extra_info'])
            
            images = list(item["images"])
            item["images"] = images
            item["prompt"] = list(item["prompt"])
            for image in images:
                assert isinstance(image,dict)
                assert isinstance(image["bytes"],bytes)
                
            if _ % 1000 == 0:
                new_item = copy.deepcopy(item)
                new_images = new_item["images"]
                update_images = []
                for img in new_images:
                    img["bytes"] = "<bytes>"
                    update_images.append(img)
                new_item["images"] = update_images
                append_jsonl(new_item, "/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/rl_data_curation/debug.jsonl")
            
            # 如果需要，更新系统提示
            if new_system_prompt and 'prompt' in item:
                item['prompt'] = update_system_prompt(item['prompt'], new_system_prompt)

            data_list.append(item)
        
        print(f"文件 {file_path} 处理完成，共 {len(data_list)} 条记录")
        return data_list
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return []

def balance_data_by_source(data_list, max_per_source=5000):
    """
    按数据源平衡数据
    
    Args:
        data_list: 数据项列表
        max_per_source: 每个数据源最大保留的记录数
        
    Returns:
        list: 平衡后的数据项列表
    """
    # 按数据源分组
    source_groups = {}
    for item in data_list:
        source = item.get('data_source', 'unknown')
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(item)
    
    # 统计各数据源数量
    print("\n各数据源原始数量:")
    for source, items in source_groups.items():
        print(f"{source}: {len(items)}")
    
    # 确定每个数据源要保留的记录数
    min_count = min(min([len(items) for items in source_groups.values()]), max_per_source)
    print(f"将每个数据源的记录数限制为: {min_count}")
    
    # 对每个数据源进行采样
    balanced_data = []
    for source, items in source_groups.items():
        if len(items) > min_count:
            # 随机采样
            random.seed(42)  # 设置随机种子以确保可重复性
            sampled_items = random.sample(items, min_count)
            balanced_data.extend(sampled_items)
            print(f"数据源 '{source}' 从 {len(items)} 条记录中采样 {min_count} 条")
        else:
            balanced_data.extend(items)
            print(f"数据源 '{source}' 保留所有 {len(items)} 条记录")
    
    # 随机打乱数据
    random.seed(42)
    random.shuffle(balanced_data)
    
    print(f"平衡后的数据集包含 {len(balanced_data)} 条记录")
    return balanced_data

def form_data_to_parquet(data_list, output_file):
    """
    将数据列表保存为Parquet文件，确保所有复杂对象都被正确序列化
    
    Args:
        data_list: 数据项列表
        output_file: 输出Parquet文件路径
    """
    print(f"总共 {len(data_list)} 条数据")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # 预处理数据，确保所有复杂对象都被序列化
        processed_data = data_list

        # 转换为DataFrame
        df = pd.DataFrame(processed_data)
        
        # ---------- 添加以下代码用于调试 ----------
        if 'images' in df.columns:
            print(f"\nDataFrame 'images' 列的 dtype: {df['images'].dtype}")
            if not df['images'].empty and isinstance(df['images'].iloc[0], list):
                first_list_in_images = df['images'].iloc[0]
                if len(first_list_in_images) > 0:
                    print(f"DataFrame 'images' 列中第一个列表的第一个元素的类型: {type(first_list_in_images[0])}")
                    # print(f"  第一个元素的值（截断）: {repr(first_list_in_images[0])[:100]}")
                else:
                    print("DataFrame 'images' 列中第一个列表是空的。")
            else:
                print("DataFrame 'images' 列不是列表类型或为空。")
        else:
            print("DataFrame 不包含 'images' 列。")
        # ------------------------------------------
        
        # 使用fastparquet引擎保存，它对嵌套结构的处理更友好
        df.to_parquet(output_file, index=False)
        print(f"Parquet文件已保存到: {output_file}")
        
        
        return True
    except Exception as e:
        print(f"保存Parquet文件时出错: {str(e)}")


def merge_parquet_files():
    """
    主函数：合并多个Parquet文件
    """
    parser = argparse.ArgumentParser(description="合并多个Parquet文件并标准化extra_info字段")
    
    parser.add_argument("--input_dir", type=str, default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/unified_tool/before_merge/beforemerge_train", help="输入Parquet文件所在目录")
    # /mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/unified_tool/before_merge/beforemerge_train
    # /mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/unified_tool/before_merge/jigsaw
    parser.add_argument("--output_file", type=str, default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/unified_tool/after_merge/vstar_3tasls/unified_train_4tasks.parquet", help="输出合并后的Parquet文件路径")
    parser.add_argument("--pattern", type=str, default="*.parquet", help="文件匹配模式，默认为'*.parquet'")
    parser.add_argument("--max_per_source", type=int, default=3000, help="每个数据源最大保留的记录数")
    parser.add_argument("--system_prompt_file", type=str, help="可选的自定义系统提示文件")
    parser.add_argument("--no_balance", action="store_true", help="禁用数据平衡功能")
    
    args = parser.parse_args()
    
    # 获取新的系统提示
    new_system_prompt = None
    try:
        # 如果提供了自定义系统提示文件，则使用它
        if args.system_prompt_file and os.path.exists(args.system_prompt_file):
            with open(args.system_prompt_file, 'r', encoding='utf-8') as f:
                new_system_prompt = f.read()
            print(f"从文件加载系统提示成功，长度: {len(new_system_prompt)}")
        else:
            tool_manager = ToolManager()
            new_system_prompt = tool_manager.get_tool_prompt()
            print(f"从ToolManager获取工具提示成功，长度: {len(new_system_prompt)}")
        
        print(f"系统提示前100字符:\n{new_system_prompt[:100]}...")
    except Exception as e:
        print(f"获取工具提示时出错: {str(e)}")
        print("将不更新系统提示")
    
    # 获取所有匹配的Parquet文件
    input_dir = Path(args.input_dir)
    input_files = list(input_dir.glob(args.pattern))
    
    if not input_files:
        print(f"在 {args.input_dir} 中没有找到匹配 {args.pattern} 的文件")
        return
    
    print(f"找到 {len(input_files)} 个匹配的Parquet文件")
    
    # 处理每个文件
    all_data = []
    for file_path in input_files:
        data_list = process_parquet_file(str(file_path), new_system_prompt)
        all_data.extend(data_list)
    
    print(f"所有文件处理完成，共 {len(all_data)} 条记录")
    
    # 平衡数据
    if not args.no_balance and len(all_data) > 0:
        balanced_data = balance_data_by_source(all_data, args.max_per_source)
    else:
        balanced_data = all_data
        print("跳过数据平衡步骤")
    
    # 保存为Parquet文件
    if balanced_data:
        form_data_to_parquet(balanced_data, args.output_file)
        
        # 测试生成的文件
        test_parquet_file(args.output_file)
    else:
        print("没有有效的数据可保存")

def test_parquet_file(parquet_file):
    """测试生成的Parquet文件，安全处理可能的嵌套数据"""
    print("\n=== 测试Parquet文件 ===")
    
    # 先检查文件是否存在
    a = load_dataset("parquet", data_files=parquet_file, split="train")

if __name__ == "__main__":
    merge_parquet_files()