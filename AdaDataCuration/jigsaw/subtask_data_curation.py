import json
import random
import copy
import os
from pathlib import Path
import random

def convert_dataset(input_file, output_dir, base_filename="dataset_judgment",max_num=500):
    """
    将选择题数据集转换为判断题数据集，并将两种任务类型分别输出到不同文件
    
    Args:
        input_file: 输入JSON文件路径
        output_dir: 输出目录路径
        base_filename: 输出文件的基本名称
    """
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据集包含 {len(data)} 个条目")
    
    # 过滤掉工具失效的数据
    filtered_data = []
    for item in data:
        # 检查工具是否失败
        if "tools" in item and "tool_failed" in item["tools"] and item["tools"]["tool_failed"]:
            continue
        filtered_data.append(item)
    
    print(f"过滤掉工具失效后剩余 {len(filtered_data)} 个条目")
    
    # 创建两个不同的数据集
    direct_comparison_dataset = []  # 不使用插入图像的数据集
    with_insertion_dataset = []     # 使用插入图像的数据集
    
    for item in filtered_data:
        # 基础图像（带有缺失部分的图像）
        base_image = item["question_image"]
        
        # 如果没有choices字段或为空，则跳过
        if "choices" not in item or not item["choices"]:
            continue
            
        # 正确选项的索引
        correct_index = item["correct_answer"]["index"] if "index" in item["correct_answer"] else 0
        
        # 找出正确的选项和一个错误的选项
        choice_images = [choice.get("image") for choice in item["choices"] if "image" in choice]
        
        if not choice_images or len(choice_images) < 2:
            continue
            
        # 确保我们有正确答案的图像
        if correct_index >= len(choice_images):
            continue
            
        correct_choice = choice_images[correct_index]
        
        # 选择一个错误选项
        incorrect_indices = [i for i in range(len(choice_images)) if i != correct_index]
        if not incorrect_indices:
            continue
            
        incorrect_index = random.choice(incorrect_indices)
        incorrect_choice = choice_images[incorrect_index]
        
        # 查找对应的插入后图像
        inserted_images = item.get("inserted_images", [])
        
        if len(inserted_images) <= max(correct_index, incorrect_index):
            continue
            
        correct_inserted = inserted_images[correct_index]
        incorrect_inserted = inserted_images[incorrect_index]
        
        # 创建判断题1：正确的选项 - 应该回答"Yes" (直接比较)
        yes_item1 = {
            "id": f"{item['id']}_direct_yes",
            "original_id": item['id'],
            "split": item['split'],
            "base_image": base_image,
            "choice_image": correct_choice,
            "question_text": "Look at the first image (img_1) with one part missing, and the second image (img_2). Is the second image the missing part of the first image? Carefully observe and compare the edges and content.\n\nYour final answer should be formatted as \\boxed{Yes} or \\boxed{No}.",
            "correct_answer": "Yes",
            "judgment_type": "direct_comparison"
        }
        
        # 创建判断题2：错误的选项 - 应该回答"No" (直接比较)
        no_item1 = {
            "id": f"{item['id']}_direct_no",
            "original_id": item['id'],
            "split": item['split'],
            "base_image": base_image,
            "choice_image": incorrect_choice,
            "question_text": "Look at the first image (img_1) with one part missing, and the second image (img_2). Is the second image the missing part of the first image? Carefully observe and compare the edges and content.\n\nYour final answer should be formatted as \\boxed{Yes} or \\boxed{No}.",
            "correct_answer": "No",
            "judgment_type": "direct_comparison"
        }
        
        # 添加到直接比较数据集
        direct_comparison_dataset.extend([yes_item1, no_item1])
        
        # 创建判断题3：插入正确选项后 - 应该回答"Yes" (带插入比较)
        yes_item2 = {
            "id": f"{item['id']}_inserted_yes",
            "original_id": item['id'],
            "split": item['split'],
            "base_image": base_image,
            "choice_image": correct_choice,
            "inserted_image": correct_inserted,
            "question_text": "Look at the first image (img_1) with one part missing, the second image (img_2), and the third image (img_3) where the second image has been placed in the missing area of the first image. Does the second image correctly fill the missing part? Carefully observe if it fits well and completes the picture properly.\n\nYour final answer should be formatted as \\boxed{Yes} or \\boxed{No}.",
            "correct_answer": "Yes",
            "judgment_type": "with_insertion"
        }
        
        # 创建判断题4：插入错误选项后 - 应该回答"No" (带插入比较)
        no_item2 = {
            "id": f"{item['id']}_inserted_no",
            "original_id": item['id'],
            "split": item['split'],
            "base_image": base_image,
            "choice_image": incorrect_choice,
            "inserted_image": incorrect_inserted,
            "question_text": "Look at the first image (img_1) with one part missing, the second image (img_2), and the third image (img_3) where the second image has been placed in the missing area of the first image. Does the second image correctly fill the missing part? Carefully observe if it fits well and completes the picture properly.\n\nYour final answer should be formatted as \\boxed{Yes} or \\boxed{No}.",
            "correct_answer": "No",
            "judgment_type": "with_insertion"
        }
        
        # 添加到带插入比较数据集
        with_insertion_dataset.extend([yes_item2, no_item2])
        
        
    
    print(f"直接比较判断题数据集包含 {len(direct_comparison_dataset)} 个条目")
    print(f"带插入比较判断题数据集包含 {len(with_insertion_dataset)} 个条目")
    random.shuffle(direct_comparison_dataset)
    random.shuffle(with_insertion_dataset)
    
    if len(direct_comparison_dataset)>max_num:
        direct_comparison_dataset = direct_comparison_dataset[:max_num]
    
    if len(with_insertion_dataset)>max_num:
        with_insertion_dataset = with_insertion_dataset[:max_num]
    print(f"截断后，直接比较判断题数据集包含 {len(direct_comparison_dataset)} 个条目")
    print(f"截断后，带插入比较判断题数据集包含 {len(with_insertion_dataset)} 个条目")
    
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存直接比较数据集
    direct_output_file = os.path.join(output_dir, f"{base_filename}_direct.json")
    with open(direct_output_file, 'w', encoding='utf-8') as f:
        json.dump(direct_comparison_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"直接比较判断题数据集已保存到 {direct_output_file}")
    
    # 保存直接比较数据集 (JSONL格式)
    direct_jsonl_output = direct_output_file.replace('.json', '.jsonl')
    with open(direct_jsonl_output, 'w', encoding='utf-8') as f:
        for item in direct_comparison_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"直接比较判断题数据集(JSONL格式)已保存到 {direct_jsonl_output}")
    
    # 保存带插入比较数据集
    insertion_output_file = os.path.join(output_dir, f"{base_filename}_insertion.json")
    with open(insertion_output_file, 'w', encoding='utf-8') as f:
        json.dump(with_insertion_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"带插入比较判断题数据集已保存到 {insertion_output_file}")
    
    # 保存带插入比较数据集 (JSONL格式)
    insertion_jsonl_output = insertion_output_file.replace('.json', '.jsonl')
    with open(insertion_jsonl_output, 'w', encoding='utf-8') as f:
        for item in with_insertion_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"带插入比较判断题数据集(JSONL格式)已保存到 {insertion_jsonl_output}")
    
    # 为每种类型保存一个小型示例数据集
    for dataset, prefix in [(direct_comparison_dataset, "direct"), (with_insertion_dataset, "insertion")]:
        sample_size = min(10, len(dataset))
        if sample_size > 0:
            sample_dataset = random.sample(dataset, sample_size)
            sample_output = os.path.join(output_dir, f"{base_filename}_{prefix}_sample.json")
            with open(sample_output, 'w', encoding='utf-8') as f:
                json.dump(sample_dataset, f, indent=2, ensure_ascii=False)
            print(f"{prefix}示例数据集已保存到 {sample_output}")
    
    return {
        "direct_comparison": direct_comparison_dataset,
        "with_insertion": with_insertion_dataset
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="将选择题数据集转换为判断题数据集")
    parser.add_argument("--input", type=str, default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/jigsaw/jigsaw_metadata_v1/splits/test/dataset_test.json", help="输入JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/jigsaw/jigsaw_metadata_v1/splits/test/", help="输出目录路径")
    parser.add_argument("--base_filename", type=str, default="subtask", help="输出文件的基本名称")
    parser.add_argument("--max_num",type=int, default=500,)
    args = parser.parse_args()
    
    convert_dataset(args.input, args.output_dir, args.base_filename, args.max_num)

if __name__ == "__main__":
    main()