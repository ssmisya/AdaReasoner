# jigsaw_rl_data_curation.py
import json
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
import random
import os
from pathlib import Path
from tqdm import tqdm
import argparse
from tool_server.tool_workers.tool_manager.base_manager import ToolManager

def read_image_to_bytes(image_path):
    """读取图片文件并转换为bytes格式"""
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"读取图片失败: {image_path}, 错误: {e}")
        return None

def get_choice_options(num_choices):
    """生成选项文本"""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = []
    for i in range(num_choices):
        letter = letters[i]
        img_num = i + 2  # 选项图片从img_2开始
        options.append(f"({letter}) The image {img_num} (img_{img_num})")
    return "\n".join(options)

def process_jigsaw_data(json_file, split, system_prompt):
    """处理拼图任务的JSON数据"""
    data = []
    
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        if json_file.endswith('.jsonl'):
            records = [json.loads(line.strip()) for line in f]
        else:
            records = json.load(f)
    
    print(f"处理 {split} 分割中的 {len(records)} 条记录...")
    
    for record in tqdm(records):
        if record['split'] != split:
            continue
        
        # 读取问题图片
        question_image_path = record['question_image']
        question_image_bytes = read_image_to_bytes(question_image_path)
        if question_image_bytes is None:
            print(f"读取问题图片失败: {question_image_path}")
            continue
        
        # 读取选项图片
        choice_images = []
        for choice in record['choices']:
            choice_image_path = choice['image']
            choice_image_bytes = read_image_to_bytes(choice_image_path)
            if choice_image_bytes is None:
                print(f"读取选项图片失败: {choice_image_path}")
                continue
            choice_images.append({"bytes": choice_image_bytes})
        
        # 如果选项图片不完整，跳过
        if len(choice_images) != len(record['choices']):
            print(f"选项图片数量不匹配: {record['id']}")
            continue
        
        image_nums = len(choice_images) + 1  # 包括问题图片
        question_text = record["question_text"]
        for i in range(image_nums):
            question_text += "\n<image>"
        
        # 构造prompt
        prompt = [
            {
                "content": system_prompt,
                "role": "system"
            },
            {
                "content": f"{question_text}",
                "role": "user"
            }
        ]
        
        # 获取正确答案
        correct_letter = record['correct_answer']['letter']
        
        # 构造数据项
        item = {
            "data_source": "jigsaw_coco",
            "prompt": prompt,
            "images": [{"bytes": question_image_bytes}] + choice_images,
            "ability": "visual_reasoning",
            "env_name": "jigsaw",
            "reward_model": {
                "ground_truth": correct_letter.lower(),
                "style": "model"
            },
            "extra_info": {
                "answer": correct_letter,
                "index": record["id"],
                "split": split
            }
        }
        
        data.append(item)
    
    print(f"成功处理 {len(data)} 条数据")
    return data

def test_parquet_file(parquet_file):
    """测试构造好的Parquet文件"""
    
    print("=== 测试Parquet文件 ===")
    
    # 读取Parquet文件
    df = pd.read_parquet(parquet_file)
    
    # 基本信息
    print(f"总记录数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    
    if len(df) > 0:
        # 检查数据样例
        sample = df.iloc[0]
        print("\n数据样例:")
        print(f"data_source: {sample['data_source']}")
        print(f"ability: {sample['ability']}")
        print(f"env_name: {sample['env_name']}")
        print(f"prompt长度: {len(sample['prompt'])}")
        print(f"prompt[0]['content']前100字符: {sample['prompt'][0]['content'][:100]}...")
        print(f"prompt[1]['content']前100字符: {sample['prompt'][1]['content'][:100]}...")
        print(f"images数量: {len(sample['images'])}")
        print(f"reward_model: {sample['reward_model']}")
        print(f"extra_info: {sample['extra_info']}")
        
        # 检查图片数据
        print("\n=== 图片数据检查 ===")
        image_bytes = sample['images'][0]['bytes']
        print(f"问题图片字节数: {len(image_bytes)}")
        
        # 尝试恢复图片（验证图片数据完整性）
        try:
            img_bytes = bytes(image_bytes)
            img = Image.open(io.BytesIO(img_bytes))
            print(f"问题图片尺寸: {img.size}")
            print("问题图片数据验证成功")
        except Exception as e:
            print(f"问题图片数据验证失败: {e}")
        
        if len(sample['images']) > 1:
            option_image_bytes = sample['images'][1]['bytes']
            print(f"选项图片字节数: {len(option_image_bytes)}")
            try:
                option_img_bytes = bytes(option_image_bytes)
                option_img = Image.open(io.BytesIO(option_img_bytes))
                print(f"选项图片尺寸: {option_img.size}")
                print("选项图片数据验证成功")
            except Exception as e:
                print(f"选项图片数据验证失败: {e}")
    
    print("\n测试完成!")

def save_data_to_parquet(data, output_file):
    """保存数据为Parquet格式"""
    print(f"总共{len(data)}条数据")
    
    # 打乱顺序
    random.shuffle(data)
    print("数据已打乱顺序")
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存为Parquet
    df.to_parquet(output_file, index=False)
    print(f"Parquet文件已保存到: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="将拼图数据集转换为RL训练格式")
    parser.add_argument("--input_json", type=str, default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/jigsaw/jigsaw_metadata_v1/dataset_complete.json", 
                        help="输入的JSON数据文件")
    parser.add_argument("--output_dir", type=str, default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/jigsaw/jigsaw_metadata_v1/rl_data", 
                        help="输出目录")
    parser.add_argument("--selected_tools", type=str, default="DetectBlackArea,InsertImage", 
                        help="使用的工具，逗号分隔")
    args = parser.parse_args()
    
    # 初始化工具管理器
    selected_tools = args.selected_tools.split(",")
    tool_manager = ToolManager(tools=selected_tools)
    system_prompt = tool_manager.get_tool_prompt()
    
    # 处理训练集数据
    print("开始处理训练集数据...")
    train_data = process_jigsaw_data(args.input_json, "rl", system_prompt)
    train_output_file = os.path.join(args.output_dir, "train.parquet")
    train_parquet_file = save_data_to_parquet(train_data, train_output_file)
    test_parquet_file(train_parquet_file)
    
    # 处理测试集数据
    print("开始处理测试集数据...")
    test_data = process_jigsaw_data(args.input_json, "test", system_prompt)
    test_output_file = os.path.join(args.output_dir, "test.parquet")
    test_output_parquet_file = save_data_to_parquet(test_data, test_output_file)
    test_parquet_file(test_output_parquet_file)
    
    # # 处理SFT数据（如果需要）
    # print("开始处理SFT数据...")
    # sft_data = process_jigsaw_data(args.input_json, "sft", system_prompt)
    # sft_output_file = os.path.join(args.output_dir, "sft.parquet")
    # sft_parquet_file = save_data_to_parquet(sft_data, sft_output_file)
    # test_parquet_file(sft_parquet_file)
    
    print("全部处理完成!")

if __name__ == "__main__":
    main()