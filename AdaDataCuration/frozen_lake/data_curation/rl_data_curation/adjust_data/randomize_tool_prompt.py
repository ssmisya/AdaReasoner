import pandas as pd
import os
import argparse
from pathlib import Path
import json
import random
import numpy as np
from tqdm import tqdm
import copy
from tool_server.tool_workers.tool_manager.base_manager_randomize import ToolManager
from datasets import load_dataset
from tool_server.utils.utils import *


def randomize_system_prompt_and_add_mapping(item, general_config):
    """
    随机化system prompt并添加映射字典
    
    Args:
        item: 单个数据项
        general_config: 配置对象
        
    Returns:
        dict: 更新后的数据项
    """
    # 创建ToolManager实例进行随机化
    tool_manager = ToolManager(
        tools = ['Crop','OCR','Point','AStarWithPixelCoordinate','Draw2DPath','DetectBlackArea','InsertImage'],
        randomize=True,
    )
    
    # 获取随机化后的prompt
    new_system_prompt = tool_manager.get_tool_prompt()
    randomized_to_original = tool_manager.randomized_to_original
    
    # 更新prompt中的system部分（第一个对话）
    if 'prompt' in item and isinstance(item['prompt'], list) and len(item['prompt']) > 0:
        new_prompt = copy.deepcopy(item['prompt'])
        if isinstance(new_prompt[0], dict) and new_prompt[0].get('role') == 'system':
            new_prompt[0]['content'] = new_system_prompt
            item['prompt'] = new_prompt
    
    # 添加映射字典 - 使用深拷贝或字典复制
    randomized_to_original = copy.deepcopy(randomized_to_original)  # 或 copy.deepcopy(randomized_to_original)
    item['randomized_to_original'] = json.dumps(randomized_to_original, ensure_ascii=False)
    
    return item


def process_parquet_with_randomization(input_file, output_file, general_config):
    """
    处理Parquet文件，随机化system prompt并添加映射
    
    Args:
        input_file: 输入Parquet文件路径
        output_file: 输出Parquet文件路径
        general_config: 配置对象
    """
    print(f"开始处理文件: {input_file}")
    
    try:
        # 读取Parquet文件
        df = pd.read_parquet(input_file)
        print(f"读取文件成功，共 {len(df)} 条记录")
        
        # 转换为列表进行处理
        data_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="随机化处理"):
            item = row.to_dict()
            
            # 确保prompt字段是列表格式
            if 'prompt' in item and not isinstance(item['prompt'], list):
                try:
                    item['prompt'] = json.loads(item['prompt']) if isinstance(item['prompt'], str) else list(item['prompt'])
                except:
                    print(f"警告: 记录 {idx} 的prompt字段格式异常")
                    continue
            
            # 随机化并添加映射
            item = randomize_system_prompt_and_add_mapping(item, general_config)
            
            # 确保images字段格式正确
            if 'images' in item:
                images = list(item["images"])
                item["images"] = images
                for image in images:
                    assert isinstance(image, dict)
                    assert isinstance(image["bytes"], bytes)
            
            data_list.append(item)
            
            # 定期保存调试信息
            if idx % 1000 == 0 and idx > 0:
                debug_item = copy.deepcopy(item)
                if "images" in debug_item:
                    new_images = []
                    for img in debug_item["images"]:
                        img_copy = img.copy()
                        img_copy["bytes"] = "<bytes_placeholder>"
                        new_images.append(img_copy)
                    debug_item["images"] = new_images
                
                debug_file = output_file.replace('.parquet', '_debug.jsonl')
                append_jsonl(debug_item, debug_file)
        
        # 保存为新的Parquet文件
        print(f"处理完成，准备保存 {len(data_list)} 条记录")
        df_new = pd.DataFrame(data_list)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存
        df_new.to_parquet(output_file, index=False)
        print(f"成功保存到: {output_file}")
        
        # 验证输出文件
        verify_parquet_file(output_file)
        
        return True
        
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def verify_parquet_file(parquet_file):
    """验证生成的Parquet文件"""
    print("\n=== 验证Parquet文件 ===")
    try:
        dataset = load_dataset("parquet", data_files=parquet_file, split="train")
        print(f"文件验证成功，共 {len(dataset)} 条记录")
        
        # 检查第一条记录
        if len(dataset) > 0:
            first_item = dataset[0]
            print("\n第一条记录的字段:")
            for key in first_item.keys():
                print(f"  - {key}: {type(first_item[key])}")
            
            # 检查是否有randomized_to_original字段
            if 'randomized_to_original' in first_item:
                print("\n✓ randomized_to_original字段存在")
                recover = json.loads(first_item['randomized_to_original'])
                print(f"  映射数量: {len(recover)}")
                if first_item['randomized_to_original']:
                    print(f"  示例映射: {list(recover.items())[:3]}")
            else:
                print("\n✗ randomized_to_original字段缺失")
            
            # 检查prompt是否被更新
            if 'prompt' in first_item and len(first_item['prompt']) > 0:
                system_content = first_item['prompt'][0].get('content', '')
                print(f"\n系统提示前100字符:\n{system_content[:100]}...")
        
        return True
    except Exception as e:
        print(f"验证文件时出错: {str(e)}")
        return False




def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="随机化Parquet文件中的system prompt")
    
    parser.add_argument("--input_file", type=str, default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/unified_tool/after_merge/vstar_3tasls/unified_test_4tasks.parquet",
                       help="输入Parquet文件路径")
    parser.add_argument("--output_file", type=str, default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/unified_tool/before_merge/randomized_rl_data/4tasks/unified_test_4tasks.parquet",
                       help="输出Parquet文件路径")
    args = parser.parse_args()
    
    # 处理文件
    success = process_parquet_with_randomization(
        args.input_file,
        args.output_file,
        None,
    )
    
    if success:
        print("\n处理完成！")
    else:
        print("\n处理失败！")
        exit(1)


if __name__ == "__main__":
    # 使用示例:
    # python randomize_tool_prompt.py \
    #     --input_file /path/to/input.parquet \
    #     --output_file /path/to/output.parquet \
    #     --controller_addr "your_controller_address" \
    #     --tools OCR Crop Point
    
    main()