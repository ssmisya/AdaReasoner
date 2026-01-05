# upload_jigsaw.py
import os
import json
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage, Sequence
from huggingface_hub import HfApi, login
from PIL import Image
import argparse
from tqdm import tqdm
import yaml
from collections import defaultdict
from pathlib import Path

def load_config(config_path="config.yaml"):
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        print(f"Warning: Config file {config_path} not found, using default settings")
        return {}

def load_json_file(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jigsaw_data(dataset_path, num_samples=None):
    """
    加载Jigsaw COCO数据集
    
    Args:
        dataset_path: 数据集JSON文件路径
        num_samples: 限制加载的样本数量，None表示加载全部
        
    Returns:
        list: 包含数据项的列表
    """
    print(f"Loading Jigsaw COCO dataset from {dataset_path}...")
    
    # 加载JSON数据
    dataset = load_json_file(dataset_path)
    
    # 如果指定了样本数量，则限制数据集大小
    if num_samples:
        dataset = dataset[:min(num_samples, len(dataset))]
    
    print(f"Loaded {len(dataset)} samples from dataset")
    
    # 处理数据集中的每个项目
    meta_data = []
    skipped_count = 0
    missing_image_count = 0
    
    for idx, item in enumerate(tqdm(dataset, desc="Processing items")):
        # 跳过不完整的数据项
        if not all(key in item for key in ["id", "question_image", "question_text", "correct_answer"]):
            print(f"Warning: 跳过不完整的数据项: {item.get('id', f'item_{idx}')}")
            skipped_count += 1
            continue
        
        item_id = item["id"]
        question_image_path = item["question_image"]
        
        # 验证图像文件存在
        if not os.path.exists(question_image_path):
            print(f"Warning: 图像文件不存在: {question_image_path}, 跳过此项: {item_id}")
            missing_image_count += 1
            continue
        
        # 验证所有选项图像存在
        all_choices_exist = True
        for choice in item["choices"]:
            if not os.path.exists(choice["image"]):
                print(f"Warning: 选项图像不存在: {choice['image']}, 跳过此项: {item_id}")
                all_choices_exist = False
                missing_image_count += 1
                break
        
        if not all_choices_exist:
            continue
        
        # 获取问题文本和正确答案
        question_text = item["question_text"]
        correct_answer = item["correct_answer"]["letter"]
        
        # 构建选项信息
        choices_info = []
        for choice in item["choices"]:
            choices_info.append({
                "image_path": choice["image"]
            })
        
        # 添加到元数据列表
        meta_data.append({
            "idx": item_id,
            "question_image_path": question_image_path,
            "question_text": question_text,
            "correct_answer": correct_answer,
            "choices": choices_info,
            "category": "jigsaw_coco"
        })
    
    # 显示统计信息
    print(f"\n数据集统计:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  成功加载: {len(meta_data)}")
    print(f"  跳过（不完整）: {skipped_count}")
    print(f"  跳过（图像缺失）: {missing_image_count}")
    
    return meta_data

def convert_to_hf_dataset(meta_data):
    """
    将元数据转换为HuggingFace Dataset格式
    
    Args:
        meta_data: 元数据列表
        
    Returns:
        DatasetDict: HuggingFace数据集字典
    """
    print("\nConverting to HuggingFace Dataset format...")
    
    # 定义特征结构
    features = Features({
        'idx': Value('string'),
        'question_image': HFImage(),
        'choice_images': Sequence(HFImage()),
        'question_text': Value('string'),
        'correct_answer': Value('string'),
        'category': Value('string'),
    })
    
    # 准备数据
    processed_items = []
    
    for item in tqdm(meta_data, desc="Loading images"):
        try:
            # 加载问题图像
            question_image = Image.open(item['question_image_path']).convert('RGB')
            
            # 加载所有选项图像
            choice_images = []
            for choice in item['choices']:
                choice_img = Image.open(choice['image_path']).convert('RGB')
                choice_images.append(choice_img)
            
            # 构建数据项
            data_item = {
                'idx': item['idx'],
                'question_image': question_image,
                'choice_images': choice_images,
                'question_text': item['question_text'],
                'correct_answer': item['correct_answer'],
                'category': item['category'],
            }
            
            processed_items.append(data_item)
            
        except Exception as e:
            print(f"Error processing item {item['idx']}: {e}")
            continue
    
    if not processed_items:
        raise ValueError("No items were successfully processed!")
    
    # 创建Dataset
    print(f"Creating dataset with {len(processed_items)} items...")
    dataset = Dataset.from_list(processed_items, features=features)
    
    # 创建DatasetDict (使用test split)
    dataset_dict = DatasetDict({
        "test": dataset
    })
    
    print(f"Dataset created successfully with {len(dataset)} samples")
    
    return dataset_dict

def generate_readme(dataset_dict, repo_id):
    """生成数据集README"""
    
    # 统计信息
    total_samples = sum(len(ds) for ds in dataset_dict.values())
    splits_info = "\n".join([f"- `{name}`: {len(ds)} samples" for name, ds in dataset_dict.items()])
    
    readme = f"""---
license: apache-2.0
task_categories:
- visual-question-answering
- image-classification
language:
- en
tags:
- jigsaw-puzzle
- spatial-reasoning
- visual-reasoning
- coco
size_categories:
- 1K<n<10K
---

# Jigsaw COCO Dataset

## Dataset Description

This dataset contains jigsaw puzzle tasks based on COCO images. The task is to identify which image piece correctly completes a given jigsaw puzzle.

### Dataset Summary

- **Total samples**: {total_samples}
- **Task**: Visual spatial reasoning with jigsaw puzzles
- **Splits**:
{splits_info}

### Task Description

Given a jigsaw puzzle image with one missing piece and three candidate pieces (A, B, C), the model needs to identify which piece correctly completes the puzzle.

Each sample consists of:
1. A question image showing the jigsaw puzzle with a missing piece
2. Three choice images (labeled A, B, C) showing candidate pieces
3. A text question asking which piece fits
4. The correct answer (A, B, or C)

### Data Fields

- `idx`: Unique identifier for each sample
- `question_image`: PIL Image of the jigsaw puzzle with missing piece
- `choice_images`: List of 3 PIL Images (choices A, B, C)
- `question_text`: Text description of the task
- `correct_answer`: The correct choice (A, B, or C)
- `choices_info`: JSON string containing choice metadata
- `category`: Task category (jigsaw_coco)

## Usage

### Load Dataset

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}")

# Access test split
test_data = dataset["test"]

# Get a sample
sample = test_data[0]
print(f"Question: {{sample['question_text']}}")
print(f"Correct Answer: {{sample['correct_answer']}}")

# Display images
sample['question_image'].show()
for i, choice_img in enumerate(sample['choice_images']):
    print(f"Choice {{chr(65+i)}}:")
    choice_img.show()

#  Evaluation

import re

def evaluate_answer(prediction, ground_truth):
    \"\"\"
    Evaluate if the prediction matches the ground truth
    
    Args:
        prediction: Model's predicted answer
        ground_truth: Correct answer (A, B, or C)
    
    Returns:
        bool: True if correct, False otherwise
    \"\"\"
    # Normalize strings
    pred = prediction.strip().upper()
    gold = ground_truth.strip().upper()
    
    # Direct match
    if pred == gold:
        return True
    
    # Extract letter from various formats
    patterns = [
        r'\\\\boxed{{([A-C])}}',
        r'\\b([A-C])\\b',
        r'(?:answer|option|choice)\\s*:?\\s*([A-C])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, pred, re.IGNORECASE)
        if match and match.group(1).upper() == gold:
            return True
    
    return False

# Example usage
correct_count = 0
total_count = 0

for sample in test_data:
    prediction = model.predict(sample)  # Your model prediction
    is_correct = evaluate_answer(prediction, sample['correct_answer'])
    
    if is_correct:
        correct_count += 1
    total_count += 1

accuracy = correct_count / total_count
print(f"Accuracy: {{accuracy:.2%}}")
Dataset Statistics
Distribution by Answer
The dataset is designed to have a balanced distribution across the three choices (A, B, C).

Image Characteristics
Images are sourced from COCO dataset
Jigsaw puzzles have varying complexity
Missing piece locations vary across samples
Dataset Creation
Source Data
Base images: COCO dataset
Jigsaw generation: Programmatic puzzle generation with controlled piece sizes and shapes
Annotations
Each sample is annotated with:

Question image with missing piece
Three candidate pieces (one correct, two distractors)
Correct answer label
Quality Control
All images are verified to exist before inclusion
Incomplete samples are filtered out
Image quality is validated
Limitations and Bias
Dataset is based on COCO images, which may have inherent biases
Jigsaw puzzle difficulty may vary across samples
Primarily focused on spatial reasoning rather than semantic understanding
Citation
If you use this dataset, please cite:

@dataset{{jigsaw_coco,
  title={{Jigsaw COCO Dataset}},
  author={{Your Name}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
License
Apache 2.0

Contact
For questions or issues, please open an issue on the dataset repository. """

    return readme

def upload_to_hub(dataset_dict, repo_id, token=None, private=False):
    """上传数据集到HuggingFace Hub"""
    if token:
        login(token=token)

    print(f"\nUploading dataset to {repo_id}...")

    try:
        # 上传数据集
        dataset_dict.push_to_hub(
            repo_id,
            private=private,
            commit_message="Upload Jigsaw COCO dataset"
        )
        print(f"Successfully uploaded dataset to {repo_id}")
        
        # 创建并上传README
        print("Generating and uploading README...")
        readme_content = generate_readme(dataset_dict, repo_id)
        api = HfApi()
        
        # 保存README到临时文件
        with open("README_temp.md", "w", encoding='utf-8') as f:
            f.write(readme_content)
        
        api.upload_file(
            path_or_fileobj="README_temp.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Add dataset card"
        )
        print("Successfully uploaded README")
        
        # 删除临时文件
        os.remove("README_temp.md")
        
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        raise

def main(): 
    parser = argparse.ArgumentParser(description='Upload Jigsaw COCO dataset to HuggingFace Hub') 
    parser.add_argument( '--repo_id', type=str, default="hitsmy/AdaEval-Jigsaw-COCO", help='HuggingFace repository ID (username/dataset-name)' ) 
    parser.add_argument( '--token', type=str, default=None, help='HuggingFace API token (or set HF_TOKEN env variable)' ) 
    parser.add_argument( '--private', action='store_true', help='Make the dataset private' ) 
    parser.add_argument( '--config', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/tasks/jigsaw_coco/config.yaml", help='Path to config file' ) 
    parser.add_argument( '--dataset_path', type=str, default=None, help='Override dataset path from config' ) 
    parser.add_argument( '--num_samples', type=int, default=None, help='Limit number of samples to upload (for testing)' )
    args = parser.parse_args()

    # 获取token
    token = args.token or os.environ.get('HF_TOKEN')
    if not token:
        print("Warning: No HuggingFace token provided. You may need to login manually.")

    # 加载配置
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)

    # 获取数据集路径
    dataset_path = args.dataset_path or config.get("dataset_path")
    if not dataset_path:
        print("Error: dataset_path not specified in config or command line")
        return

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        return

    # 获取样本数量限制
    num_samples = args.num_samples or config.get("num_sample")

    # 加载数据
    print("\nLoading Jigsaw COCO data...")
    meta_data = load_jigsaw_data(dataset_path, num_samples)

    if not meta_data:
        print("Error: No data loaded")
        return

    # 转换为HF Dataset
    print("\nConverting to HuggingFace Dataset format...")
    dataset_dict = convert_to_hf_dataset(meta_data)

    # 显示数据集信息
    print("\n" + "="*60)
    print("Dataset structure:")
    print("="*60)
    for name, ds in dataset_dict.items():
        print(f"\n{name}:")
        print(f"  Samples: {len(ds)}")
        print(f"  Features: {list(ds.features.keys())}")
        
        # 显示第一个样本的部分信息
        if len(ds) > 0:
            sample = ds[0]
            print(f"  Sample info:")
            print(f"    - idx: {sample['idx']}")
            print(f"    - question_text: {sample['question_text'][:100]}...")
            print(f"    - correct_answer: {sample['correct_answer']}")
            print(f"    - number of choices: {len(sample['choice_images'])}")

    print("\n" + "="*60)

    # 询问用户是否继续上传
    response = input("\nDo you want to proceed with upload? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Upload cancelled.")
        return

    # 上传到Hub
    print("\nUploading to HuggingFace Hub...")
    upload_to_hub(dataset_dict, args.repo_id, token, args.private)

    print(f"\n{'='*60}")
    print(f"✅ Dataset successfully uploaded!")
    print(f"{'='*60}")
    print(f"View your dataset at: https://huggingface.co/datasets/{args.repo_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()