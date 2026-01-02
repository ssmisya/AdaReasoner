#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any

# --- 辅助函数 ---

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}")
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def write_json(data: List[Dict[str, Any]], file_path: str):
    """将数据写入JSON文件"""
    Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"成功将 {len(data)} 条数据写入 {file_path}")

# --- 对话构建函数 ---

# 从你的参考文件中提取的提示模板
PATH_VERIFY_TASK_INSTRUCTION_SHORT =  """
You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success. 

Now please determine if the action sequence is safe for the given maze. Your final answer should be formatted as \\boxed{Yes} or \\boxed{No}.

The action sequence is:

<ACTION-SEQ>
"""

PATH_FINDING_TASK_INSTRUCTION_SHORT = """
You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success. Your final answer should be formatted as \\boxed{L,R,U,D}.

Please generate action plan for the input maze image.
"""
def create_navigation_conversation(metadata: Dict[str, Any], correct_answer: Dict[str, Any], image_dir: str) -> Dict[str, Any]:
    """为导航任务创建ShareGPT对话"""
    user_message = {
        "from": "human",
        "value": PATH_FINDING_TASK_INSTRUCTION_SHORT + "\n<image>"
    }
    
    assistant_message = {
        "from": "gpt",
        "value": correct_answer["answer"] # 使用已知的正确答案
    }
    
    image_path = os.path.join(image_dir, metadata["image_path"])
    if not os.path.exists(image_path):
        print(f"警告: 导航任务图片不存在 {image_path}")

    return {
        "qid": correct_answer["original_id"],
        "conversations": [user_message, assistant_message],
        "images": [image_path]
    }

def create_verify_conversation(metadata: Dict[str, Any], correct_answer: Dict[str, Any], image_dir: str) -> Dict[str, Any]:
    """为验证任务创建ShareGPT对话"""
    # 从元数据中获取要验证的随机路径
    path_to_verify = metadata["path_drawings"]["random"]["path"]
    if not path_to_verify:
        print(f"警告: 在元数据中找不到用于验证的路径，ID: {metadata['id']}")
        return None

    instruction = PATH_VERIFY_TASK_INSTRUCTION_SHORT.replace("<ACTION-SEQ>", path_to_verify)
    
    user_message = {
        "from": "human",
        "value": instruction + "\n<image>"
    }
    
    assistant_message = {
        "from": "gpt",
        "value": correct_answer["answer"] # 使用已知的正确答案
    }
    
    image_path = os.path.join(image_dir, metadata["image_path"])
    if not os.path.exists(image_path):
        print(f"警告: 验证任务图片不存在 {image_path}")

    return {
        "qid": correct_answer["original_id"],
        "conversations": [user_message, assistant_message],
        "images": [image_path]
    }

# --- 主逻辑 ---

def generate_sft_from_correct_answers(
    correct_answers_path: str,
    sft_metadata_path: str,
    output_dir: str,
    image_dir: str,
    nav_samples: int = 100,
    verify_samples: int = 100
):
    """从正确答案生成SFT训练数据"""
    
    print("步骤 1: 加载文件...")
    correct_answers = load_jsonl(correct_answers_path)
    sft_metadata = load_jsonl(sft_metadata_path)
    
    if not correct_answers or not sft_metadata:
        print("错误: 输入文件为空或无法加载，程序终止。")
        return

    # 将SFT元数据转换为字典以便快速查找
    sft_metadata_dict = {item['id']: item for item in sft_metadata}
    print(f"加载了 {len(correct_answers)} 条正确答案和 {len(sft_metadata_dict)} 条元数据。")

    print("\n步骤 2: 按任务类型分类正确答案...")
    navigation_correct = []
    verify_correct = []
    for answer in correct_answers:
        if answer.get("classification") == "navigation" and answer["split"] == "sft":
            navigation_correct.append(answer)
        elif answer.get("classification") == "verify" and answer["split"] == "sft": # 假设验证任务的分类是 'verification'
            verify_correct.append(answer)
            
    print(f"找到 {len(navigation_correct)} 条导航任务的正确答案。")
    print(f"找到 {len(verify_correct)} 条验证任务的正确答案。")

    # --- 处理导航任务 ---
    print(f"\n步骤 3: 生成导航任务SFT数据 (目标: {nav_samples} 条)...")
    navigation_sft_data = []
    for answer in tqdm(navigation_correct, desc="处理导航任务"):
        if len(navigation_sft_data) >= nav_samples:
            break
        original_id = answer.get("original_id")
        if original_id in sft_metadata_dict:
            metadata = sft_metadata_dict[original_id]
            conversation = create_navigation_conversation(metadata, answer, image_dir)
            if conversation:
                navigation_sft_data.append(conversation)
    
    # --- 处理验证任务 ---
    print(f"\n步骤 4: 生成验证任务SFT数据 (目标: {verify_samples} 条)...")
    verify_sft_data = []
    for answer in tqdm(verify_correct, desc="处理验证任务"):
        if len(verify_sft_data) >= verify_samples:
            break
        original_id = answer.get("original_id")
        if original_id in sft_metadata_dict:
            metadata = sft_metadata_dict[original_id]
            conversation = create_verify_conversation(metadata, answer, image_dir)
            if conversation:
                verify_sft_data.append(conversation)

    print("\n步骤 5: 保存输出文件...")
    nav_output_path = os.path.join(output_dir, "navigation_correct_sft.json")
    verify_output_path = os.path.join(output_dir, "verify_correct_sft.json")
    
    write_json(navigation_sft_data, nav_output_path)
    write_json(verify_sft_data, verify_output_path)
    
    print("\n处理完成！")

def main():
    parser = argparse.ArgumentParser(description="从正确答案元数据生成SFT训练数据")
    parser.add_argument('--correct_answers', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/merged_sft_file/correct_answers_zs/correct_answers_metadata.jsonl", help='correct_answers_metadata.jsonl 文件路径')
    parser.add_argument('--sft_metadata', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/dataset.jsonl", help='包含所有元数据的 sft.jsonl 文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出JSON文件的目录')
    parser.add_argument('--image_dir', type=str, required=True, help='图片所在的根目录')
    parser.add_argument('--nav_count', type=int, default=100, help='要生成的导航任务样本数')
    parser.add_argument('--verify_count', type=int, default=100, help='要生成的验证任务样本数')
    
    args = parser.parse_args()
    
    generate_sft_from_correct_answers(
        correct_answers_path=args.correct_answers,
        sft_metadata_path=args.sft_metadata,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        nav_samples=args.nav_count,
        verify_samples=args.verify_count
    )

if __name__ == "__main__":
    # 为了直接运行，可以取消下面的注释并修改路径
    generate_sft_from_correct_answers(
        correct_answers_path="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/merged_sft_file/correct_answers_zs/correct_answers_metadata.jsonl",
        sft_metadata_path="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/dataset.jsonl",
        output_dir='/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/merged_sft_file/correct_answers_zs',
        image_dir="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation"
    )
    # main()