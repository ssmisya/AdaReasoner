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
from tool_server.utils.utils import load_image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

def load_data_function():
    """
    加载Jigsaw COCO数据集
    
    Returns:
        list: 包含数据项的列表
    """
    dataset_path = task_config.get("dataset_path")
    num_samples = task_config.get("num_sample")
    
    dataset = load_json_file(dataset_path)

    
    # 如果指定了样本数量，则限制数据集大小
    if num_samples:
        dataset = dataset[:min(num_samples, len(dataset))]
    
    # 处理数据集中的每个项目
    meta_data = []
    for idx, item in enumerate(dataset):
        # 跳过不完整的数据项
        if not all(key in item for key in ["id", "question_image", "question_text", "correct_answer"]):
            logger.warning(f"跳过不完整的数据项: {item.get('id', f'item_{idx}')}")
            continue
        
        item_id = item["id"]
        question_image_path = item["question_image"]
        
        # 验证图像文件存在
        if not os.path.exists(question_image_path):
            logger.warning(f"图像文件不存在: {question_image_path}, 跳过此项: {item_id}")
            continue
        

        question_image = load_image(question_image_path)
        images = [question_image]
        
        # 加载选项图像
        choices = []
        for choice in item["choices"]:
            choices.append(load_image(choice["image"]))
        images.extend(choices)

        # 获取问题文本和正确答案
        question_text = item["question_text"]
        correct_answer = item["correct_answer"]["letter"]
        
        # 添加到元数据列表
        meta_data.append({
            "idx": item_id,
            "images": images,  # 主要问题图像
            "text": question_text,    # 问题文本
            "answer": correct_answer, # 正确答案 (A, B, C)
            "category": "jigsaw"      # 任务类别
        })
    
    # 显示统计信息
    logger.info(f"总数据数量: {len(meta_data)}")
    
    return meta_data

def rule_based_verify(gold: str, pred: str) -> float:
    """
    验证预测的选项是否与正确答案相同
    
    Args:
        gold (str): 正确答案（A、B、C等）
        pred (str): 预测的答案
        
    Returns:
        float: 如果预测正确返回1.0，否则返回0.0
    """
    # 去除双引号，两端的空白并转换成小写
    gold = gold.replace("\"", "").strip().lower()
    
    # 处理预测答案
    if pred:
        pred = pred.replace("\"", "").strip().lower()
    else:
        pred = ""
    
    # 如果pred字符串为空，则返回0
    if pred == "" or pred is None or pred == "None":
        return 0.0
    
    if gold == pred:
        return 1.0
    
    # 提取pred中的选项字母
    option_patterns = [
        r'\\boxed{([a-c])}',  # 匹配\boxed{A}格式
        r'\b([a-c])[.、)）]?\b',  # 匹配单独的A、B、C，可能带有.、)等后缀
        r'(?:option|answer):*([a-c])[.、)）]?\b',  # 匹配"选项是A"、"答案为B"等形式
        r'(?:the answer is|i choose|i pick):*([a-c])[.、)）]?\b'  # 匹配"The answer is A"等形式
    ]
    
    for pattern in option_patterns:
        match = re.search(pattern, pred)
        if match:
            extracted_option = match.group(1).lower()
            if extracted_option == gold:
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
    
    # 统计每个数据项的评估结果
    for idx, meta in meta_dict.items():
        if idx in results_dict:
            prediction = results_dict[idx]["results"]["final_answer"]
        else:
            prediction = "None"
        
        ground_truth = meta["answer"]
        score = rule_based_verify(ground_truth, prediction)
        
        total_correct += score
        total_samples += 1
        
        # 记录详细比较日志
        compare_logs.append({
            "idx": idx,
            "category": meta.get("category", "jigsaw"),
            "gold": ground_truth,
            "pred": prediction,
            "score": score,
            "question": meta["text"]
        })
    
    # 计算总体准确率
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # 组织评估结果
    result_dict = {
        "accuracy": accuracy,
        "total_correct": total_correct,
        "total_samples": total_samples,
        "compare_logs": compare_logs,
        "meta_data": meta_dict,
        "results": results
    }
    
    # 打印评估结果摘要
    logger.info(f"总体准确率: {accuracy:.4f} ({total_correct}/{total_samples})")
    
    return result_dict