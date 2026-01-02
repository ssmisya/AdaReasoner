#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from tqdm import tqdm
from typing import List, Dict, Any

def load_json(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON文件时出错: {e}")
        return []

def get_last_turn(conversations: List[Dict[str, str]]) -> (str, str):
    """获取最后一轮的人机对话"""
    last_human = "N/A"
    last_gpt = "N/A"
    
    # 从后向前查找，找到第一个gpt和第一个human的对话
    found_gpt = False
    for conv in reversed(conversations):
        if not found_gpt and conv.get("from") == "gpt":
            last_gpt = conv.get("value", "N/A")
            found_gpt = True
        elif found_gpt and conv.get("from") == "human":
            last_human = conv.get("value", "N/A")
            break
            
    return last_human, last_gpt

def visualize_sharegpt_item(item: Dict[str, Any], output_dir: str):
    """可视化单个ShareGPT条目"""
    qid = item.get("qid", "unknown_id")
    conversations = item.get("conversations", [])
    image_paths = item.get("images", [])
    
    if not image_paths:
        print(f"警告: QID {qid} 没有图片，跳过可视化。")
        return

    # 获取最后一轮对话
    last_human, last_gpt = get_last_turn(conversations)
    
    # --- 创建可视化 ---
    # 根据图片数量调整布局
    num_images = len(image_paths)
    fig = plt.figure(figsize=(6 * num_images, 8))
    gs = gridspec.GridSpec(2, num_images, height_ratios=[3, 1])

    # 显示图片
    for i, img_path in enumerate(image_paths):
        ax = fig.add_subplot(gs[0, i])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(f"Image {i+1}", fontsize=10)
        else:
            ax.text(0.5, 0.5, f"Image Not Found\n{os.path.basename(img_path)}", 
                    ha='center', va='center', fontsize=9, color='red')
        ax.axis('off')

    # 显示对话文本
    text_ax = fig.add_subplot(gs[1, :])
    text_ax.axis('off')
    
    # 格式化文本以便更好地显示
    human_text = f"Human:\n" + "\n".join([line.strip() for line in last_human.split('\n')])
    gpt_text = f"GPT:\n" + "\n".join([line.strip() for line in last_gpt.split('\n')])
    
    full_text = f"{human_text}\n\n{gpt_text}"
    
    text_ax.text(0.01, 0.95, full_text, ha='left', va='top', fontsize=9, wrap=True,
                 bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="lightsteelblue", lw=1))

    plt.suptitle(f"QID: {qid}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图像
    output_file = os.path.join(output_dir, f"{qid}.png")
    plt.savefig(output_file, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="可视化ShareGPT数据，展示图片和最后一轮对话")
    parser.add_argument('--sharegpt_data', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/merged_sft_file/correct_answers_zs/navigation_correct_sft.json",
                        help='ShareGPT格式的JSON文件路径')
    parser.add_argument('--output_dir', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/merged_sft_file/correct_answers_zs/visualize",
                        help='保存可视化结果的目录')
    parser.add_argument('--limit', type=int, default=20,
                        help='要处理的最大项目数 (可选)')
    
    args = parser.parse_args()
    
    # 加载数据
    sharegpt_data = load_json(args.sharegpt_data)
    if not sharegpt_data:
        return
        
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 限制处理数量
    if args.limit:
        sharegpt_data = sharegpt_data[:args.limit]
        
    # 遍历并可视化
    for item in tqdm(sharegpt_data, desc="生成可视化图像"):
        visualize_sharegpt_item(item, args.output_dir)
        
    print(f"\n可视化完成！结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    main()