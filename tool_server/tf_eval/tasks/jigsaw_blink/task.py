from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os
import sys
import re
import json
from PIL import Image
from pathlib import Path
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

QUESTION_PROMPT = "Given the first image (img_1) with one part missing, can you tell which one of the second image (img_2) or the third image (img_3) is the missing part? Imagine which image would be more appropriate to place in the missing spot. You can also carefully observe and compare the edges of the images.\n\nSelect from the following choices.\n\n(A) The second image (img_2)\n(B) The third image (img_3)\nYour final answer should be formatted as \\boxed{Your Choice}, for example, \\boxed{A} or \\boxed{B} or \\boxed{C}."

def load_data_function():
    """
    加载Jigsaw BLINK数据集
    
    Returns:
        list: 包含数据项的列表
    """
    dataset_path = task_config.get("dataset_path")
    num_samples = task_config.get("num_sample")
    splits = task_config.get("split", ["test"])
    
    if not isinstance(splits, list):
        splits = [splits]
    
    # 加载BLINK数据集
    blink_dataset = load_dataset(dataset_path, "Jigsaw")
    
    # 处理所有指定的数据分割
    meta_data = []
    
    for split in splits:
        if split not in blink_dataset:
            logger.warning(f"数据分割 {split} 不存在于BLINK数据集中")
            continue
            
        dataset_split = blink_dataset[split]
        
        # 如果指定了样本数量，则限制数据集大小
        if num_samples:
            dataset_split = dataset_split[:min(num_samples, len(dataset_split))]
        
        # 处理数据集中的每个项目
        for idx, item in enumerate(dataset_split):
            # 跳过没有图像的数据项
            if item["image_1"] is None or (item["image_2"] is None and item["image_3"] is None):
                logger.warning(f"跳过缺少图像的数据项: {item.get('idx', f'{split}_item_{idx}')}")
                continue
            
            # 主图像
            question_image = item["image_1"]
            
            # 选项图像
            choices_images = []
            if item["image_2"] is not None:
                choices_images.append(item["image_2"])
            if item["image_3"] is not None:
                choices_images.append(item["image_3"])
            if item["image_4"] is not None:
                choices_images.append(item["image_4"])
            
            # 组合所有图像
            images = [question_image] + choices_images
            
            # 获取问题和选项
            question_text = QUESTION_PROMPT
            choices = item["choices"]
            
            # 提取答案（对于测试集，答案可能是隐藏的）
            answer = item["answer"]
            answer = f"{answer[1]}"

            # 将选项转换为字母答案格式（A、B）
            letter_mapping = {}
            for i, choice in enumerate(choices):
                letter = chr(65 + i)  # A, B, C, ...
                letter_mapping[choice] = letter
            
            # 如果答案不是隐藏的，转换为字母格式
            letter_answer = "hidden"
            if answer != "hidden" and answer in letter_mapping:
                letter_answer = letter_mapping[answer]
            
            # 添加到元数据列表
            meta_data.append({
                "idx": item.get("idx", f"{split}_Jigsaw_{idx+1}"),
                "images": images,  # 所有图像，第一个是问题图像
                "text": question_text,  # 问题文本
                "answer": answer,  
                "category": "jigsaw",  # 任务类别
                "split": split  # 数据分割
            })
    
    # 显示统计信息
    logger.info(f"总数据数量: {len(meta_data)}")
    
    return meta_data

def rule_based_verify(gold: str, pred: str, choices: list) -> float:
    """
    验证预测的选项是否与正确答案相同
    
    Args:
        gold (str): 正确答案（A、B等或原始文本）
        pred (str): 预测的答案
        choices (list): 选项文本列表
        
    Returns:
        float: 如果预测正确返回1.0，否则返回0.0
    """
    if gold == "hidden":
        logger.warning("无法验证隐藏答案")
        return 0.0
    
    # 去除双引号，两端的空白并转换成小写
    if gold:
        gold = gold.replace("\"", "").strip().lower()
    else:
        return 0.0
    
    # 处理预测答案
    if pred:
        pred = pred.replace("\"", "").strip().lower()
    else:
        return 0.0
    
    # 首先检查直接匹配
    if gold == pred:
        return 1.0
    
    # 转换选项文本到小写以便匹配
    choices_lower = [choice.lower() for choice in choices]
    
    # 检查预测是否匹配原始选项文本
    if pred in choices_lower:
        choice_idx = choices_lower.index(pred)
        letter_pred = chr(65 + choice_idx)
        if letter_pred.lower() == gold.lower():
            return 1.0
    
    # 提取pred中的选项字母
    option_patterns = [
        r'\\boxed{([a-z])}',  # 匹配\boxed{A}格式
        r'\b([a-z])[.、)）]?\b',  # 匹配单独的A、B，可能带有.、)等后缀
        r'(?:option|answer):*\s*([a-z])[.、)）]?\b',  # 匹配"选项是A"、"答案为B"等形式
        r'(?:the answer is|i choose|i pick):*\s*([a-z])[.、)）]?\b',  # 匹配"The answer is A"等形式
        r'\(([a-z])\)'  # 匹配(A)格式
    ]
    
    for pattern in option_patterns:
        match = re.search(pattern, pred)
        if match:
            extracted_option = match.group(1).lower()
            if extracted_option == gold.lower():
                return 1.0
    
    # 检查预测是否提到了正确的选项文本
    for i, choice in enumerate(choices):
        choice_letter = chr(65 + i).lower()
        if choice_letter == gold.lower() and choice.lower() in pred.lower():
            return 1.0
    
    return 0.0

def evaluate_function(results, meta_data):
    """
    评估模型预测结果
    
    Args:
        results: 模型生成的预测结果
        meta_data: 数据集元数据
        
    Returns:
        dict: 评估结果
    """
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    
    compare_logs = []
    total_correct = 0
    total_samples = 0
    
    # 按数据分割跟踪结果
    split_results = {}
    
    # 统计每个数据项的评估结果
    for idx, meta in meta_dict.items():
        if meta["answer"] == "hidden":
            continue
            
        if idx in results_dict:
            prediction = results_dict[idx]["results"]["final_answer"]
        else:
            prediction = "None"
        
        ground_truth = meta["answer"]
        choices = meta.get("choices", [])
        score = rule_based_verify(ground_truth, prediction, choices)
        
        total_correct += score
        total_samples += 1
        
        # 更新分割统计
        split = meta.get("split", "unknown")
        if split not in split_results:
            split_results[split] = {"correct": 0, "total": 0}
        
        split_results[split]["correct"] += score
        split_results[split]["total"] += 1
        
        # 记录详细比较日志
        compare_logs.append({
            "idx": idx,
            "category": meta.get("category", "jigsaw"),
            "gold": ground_truth,
            "pred": prediction,
            "score": score,
            "question": meta["text"],
            "split": meta.get("split", "unknown")
        })
    
    # 计算总体准确率
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # 计算每个分割的准确率
    split_accuracy = {}
    for split, counts in split_results.items():
        split_accuracy[split] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
    
    # 组织评估结果
    result_dict = {
        "accuracy": accuracy,
        "total_correct": total_correct,
        "total_samples": total_samples,
        "split_accuracy": split_accuracy,
        "compare_logs": compare_logs,
        "meta_data": meta_dict,
        "results": results
    }
    
    # 打印评估结果摘要
    logger.info(f"总体准确率: {accuracy:.4f} ({total_correct}/{total_samples})")
    for split, acc in split_accuracy.items():
        split_counts = split_results[split]
        logger.info(f"{split} 集合准确率: {acc:.4f} ({split_counts['correct']}/{split_counts['total']})")
    
    return result_dict