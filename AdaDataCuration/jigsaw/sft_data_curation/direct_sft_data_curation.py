# direct_sft_data_curation.py
import os
import json
import random
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any

def check_image_exists(image_path):
    """验证图像文件是否存在"""
    return os.path.isfile(image_path) and os.path.getsize(image_path) > 0

def load_json_file(file_path):
    """从JSON文件加载数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json_file(data, file_path):
    """将数据写入JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def create_direct_sft_conversation(item):
    """
    为拼图任务创建直接的SFT对话数据，不含思考过程
    
    Args:
        item: 原始元数据项
        
    Returns:
        dict: ShareGPT格式的对话数据
    """
    # 获取问题文本和图像路径
    question_text = item["question_text"]
    question_image = item["question_image"]
    
    # 获取选项图像
    choice_images = [choice.get("image") for choice in item.get("choices", [])]
    
     # 验证图像路径是否存在
    images = [question_image] + choice_images
    for img_path in images:
        if img_path and not check_image_exists(img_path):
            print(f"警告: 图像文件不存在: {img_path}")
    
    image_tokens = "\n"
    for img in images:
        image_tokens += "<image>\n"
    
    # 获取正确答案
    correct_answer = item.get("correct_answer", {}).get("letter", "A")
    
    # 准备用户问题
    user_message = {
        "from": "human",
        "value": question_text+image_tokens
    }
    
    # 准备助手答案（简洁版本，没有详细思考过程）
    assistant_message = {
        "from": "gpt",
        "value": f"After examining the images carefully, I can see that the missing part that fits perfectly in the black area is option ({correct_answer}).\n\nThe edges align correctly, and the content flows naturally when this piece is inserted into the missing spot.\n\n\\boxed{{{correct_answer}}}"
    }
    
    # 组装对话
    conversation = [user_message, assistant_message]
    
   
    
    # 创建ShareGPT格式数据
    sharegpt_item = {
        "id": item["id"],
        "conversations": conversation,
        "images": images
    }
    
    return sharegpt_item

def generate_direct_sft_dataset(input_path, output_path, target_size=1000, backup_file=None, seed=42):
    """
    生成直接SFT数据集
    
    Args:
        input_path: 输入元数据文件路径
        output_path: 输出目录
        target_size: 目标数据集大小
        backup_file: 备选数据文件路径，当主数据文件不足时使用
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 加载主数据
    print(f"从 {input_path} 加载数据...")
    main_dataset = load_json_file(input_path)
    print(f"从主文件加载了 {len(main_dataset)} 条数据")
    
    # 随机打乱主数据集
    random.shuffle(main_dataset)
    selected_data = main_dataset[:]
    
    # 如果主数据不足目标数量且指定了备选文件，则从备选文件中补充数据
    if len(selected_data) < target_size and backup_file and os.path.exists(backup_file):
        print(f"从备选文件 {backup_file} 加载数据...")
        try:
            backup_dataset = load_json_file(backup_file)
            print(f"从备选文件加载了 {len(backup_dataset)} 条数据")
            
            # 随机打乱备选数据
            random.shuffle(backup_dataset)
            
            # 计算需要补充的数量
            needed_count = target_size - len(selected_data)
            backup_selected = backup_dataset[:needed_count]
            selected_data.extend(backup_selected)
            
            print(f"从主文件选择了 {len(main_dataset)} 条数据，从备选文件补充了 {len(backup_selected)} 条数据")
        except Exception as e:
            print(f"加载备选文件失败: {e}")
    
    # 如果总数据量超过目标大小，进行裁剪
    if len(selected_data) > target_size:
        selected_data = selected_data[:target_size]
        print(f"数据超出目标大小，裁剪得到 {target_size} 条数据")
    
    print(f"最终选择了 {len(selected_data)} 条数据")
    
    # 生成ShareGPT格式数据
    sharegpt_data = []
    
    # 统计选项数量
    two_choice_count = 0
    three_choice_count = 0
    
    # 统计不同分割数据的数量
    split_counts = {}
    
    # 统计数据来源
    source_counts = {
        "main": len(main_dataset) if len(main_dataset) <= target_size else target_size,
        "backup": max(0, min(len(selected_data) - len(main_dataset), needed_count if 'needed_count' in locals() else 0))
    }
    
    with tqdm(total=len(selected_data), desc="生成直接SFT数据") as pbar:
        for item in selected_data:
            # 创建直接SFT对话
            sharegpt_item = create_direct_sft_conversation(item)
            sharegpt_data.append(sharegpt_item)
            
            # 统计选项数量
            choices_count = len(item.get("choices", []))
            if choices_count == 2:
                two_choice_count += 1
            elif choices_count == 3:
                three_choice_count += 1
            
            # 统计分割类型
            split_type = item.get("split", "unknown")
            split_counts[split_type] = split_counts.get(split_type, 0) + 1
                
            pbar.update(1)
    
    # 保存ShareGPT格式数据
    output_file = os.path.join(output_path, "direct_jigsaw_sharegpt_data.json")
    write_json_file(sharegpt_data, output_file)
    
    print(f"数据集生成完成，共生成 {len(sharegpt_data)} 个样本")
    print(f"结果保存至 {output_file}")
    print(f"2选项数量: {two_choice_count} ({two_choice_count/len(sharegpt_data)*100:.1f}%)")
    print(f"3选项数量: {three_choice_count} ({three_choice_count/len(sharegpt_data)*100:.1f}%)")
    print("分割类型统计:")
    for split, count in split_counts.items():
        print(f"  {split}: {count} ({count/len(sharegpt_data)*100:.1f}%)")
    print("数据来源统计:")
    print(f"  主文件: {source_counts['main']} ({source_counts['main']/len(sharegpt_data)*100:.1f}%)")
    if source_counts['backup'] > 0:
        print(f"  备选文件: {source_counts['backup']} ({source_counts['backup']/len(sharegpt_data)*100:.1f}%)")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成拼图任务的直接SFT数据集')
    parser.add_argument('--input_path', type=str, 
                        default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/jigsaw/jigsaw_metadata_v1/splits/sft/dataset_sft.json", 
                        help='输入数据集路径')
    parser.add_argument('--output_path', type=str, 
                        default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/jigsaw/jigsaw_metadata_v1/splits/sft/gemini_enhanced/sft_data/", 
                        help='输出目录')
    parser.add_argument('--target_size', type=int, default=1125, 
                        help='目标数据集大小')
    parser.add_argument('--backup_file', type=str, 
                        default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/jigsaw/jigsaw_metadata_v1/splits/rl/dataset_rl.json", 
                        help='备选数据文件路径，当主数据文件不足时使用')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 生成数据集
    generate_direct_sft_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        target_size=args.target_size,
        backup_file=args.backup_file,
        seed=args.seed
    )

if __name__ == "__main__":
    main()