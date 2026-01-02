import json
import pandas as pd
import numpy as np
from PIL import Image
import io
import random
from pathlib import Path
import os
from tool_server.tool_workers.tool_manager.base_manager import ToolManager


def read_image_to_bytes(image_path):
    """读取图片文件并转换为bytes格式"""
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"读取图片失败: {image_path}, 错误: {e}")
        return None

def process_visual_cot_data(json_file, system_prompt=None):
    """处理VisualCoT任务的JSON数据"""
    data = []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    for record in records:
        # 读取原始图片
        image_path = record['image_url']
        image_bytes = read_image_to_bytes(image_path)
        if image_bytes is None:
            print(f"跳过记录: {record['pid']}, 原因: 无法读取原始图片")
            continue
        
        # 读取带标注的图片
        output_image_path = record.get('output_image_url')
        output_image_bytes = None
        if output_image_path:
            output_image_bytes = read_image_to_bytes(output_image_path)
        
        # 构造prompt
        question = record['question']
        response_hint = record.get('response', '')
        
        prompt = [
            {
                "content": system_prompt,
                "role": "system"
            },
            {
                "content": f"{question}\n\n<image>",
                "role": "user"
            }
        ]
        
        # 构造图像列表
        images = [{"bytes": image_bytes}]
        # if output_image_bytes:
        #     images.append({"bytes": output_image_bytes})
        
        # 获取答案
        answer = record.get('answer', '')
        
        # 构造数据项
        item = {
            "data_source": "visual_cot",
            "prompt": prompt,
            "images": images,
            "ability": "visual_search",
            "env_name": "visual_search",
            "reward_model": {
                "ground_truth": answer,
                "style": "model"
            },
            "extra_info": {
                "answer": answer,
                "index": record["pid"],
                "split": "train",
                "question": question,
                "image_cot": record.get('generated_response', {}).get('image_cot', ''),
                "edited_image_analysis": record.get('generated_response', {}).get('edited_image_analysis', ''),
                "bbox_details": record.get('bbox_details', [])
            }
        }
        
        data.append(item)
    
    return data

def form_data_to_parquet(data, output_file):
    """将数据转换为Parquet格式并保存"""
    print(f"总共{len(data)}条数据")
    
    # 打乱顺序
    random.shuffle(data)
    print("数据已打乱顺序")
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存为Parquet
    df.to_parquet(output_file, index=False)
    print(f"Parquet文件已保存到: {output_file}")

def test_parquet_file(parquet_file):
    """测试构造好的Parquet文件"""
    print("\n=== 测试Parquet文件 ===")
    
    # 读取Parquet文件
    df = pd.read_parquet(parquet_file)
    
    # 基本信息
    print(f"总记录数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    print(f"各数据源统计:")
    print(df['data_source'].value_counts())
    
    # 检查样例
    if len(df) > 0:
        sample = df.iloc[0]
        print("\n数据样例:")
        print(f"data_source: {sample['data_source']}")
        print(f"ability: {sample['ability']}")
        print(f"env_name: {sample['env_name']}")
        print(f"prompt长度: {len(sample['prompt'])}")
        print(f"images数量: {len(sample['images'])}")
        print(f"reward_model: {sample['reward_model']}")
        print(f"extra_info keys: {list(sample['extra_info'].keys())}")
        
        # 验证图片数据
        image_bytes = sample['images'][0]['bytes']
        print(f"\n第一张图片字节数: {len(image_bytes)}")
        
        try:
            img = Image.open(io.BytesIO(bytes(image_bytes)))
            print(f"图片尺寸: {img.size}")
            print("图片数据验证成功")
        except Exception as e:
            print(f"图片数据验证失败: {e}")
    
    print("\n测试完成!")

def create_parquet_from_json(
    input_json,
    output_file,
    system_prompt=None
):
    """主函数：从JSON文件创建Parquet文件"""
    
    print(f"开始处理JSON数据: {input_json}")
    data = process_visual_cot_data(input_json, system_prompt=system_prompt)
    print(f"处理完成，共{len(data)}条数据")
    
    # 保存为Parquet
    form_data_to_parquet(data, output_file)
    
    return output_file

if __name__ == "__main__":
    # 配置参数
    input_json = "/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/visual_research/vs_data/raw_data/rl_data.json"
    output_file = "/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/visual_research/vs_data/parquets/visual_cot_rl_data.parquet"
    
    # input_json = "/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/visual_research/vs_data/raw_data/val_data.json"
    # output_file = "/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/visual_research/vs_data/parquets/visual_cot_val_data.parquet"

    tool_manager = ToolManager()
    system_prompt = tool_manager.get_tool_prompt()
    # 创建Parquet文件
    parquet_file = create_parquet_from_json(
        input_json=input_json,
        output_file=output_file,
        system_prompt=system_prompt
    )
    
    # 测试Parquet文件
    test_parquet_file(parquet_file)