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
from io import BytesIO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

def load_data_function():
    """
    加载Jigsaw COCO数据集
    根据配置从Hugging Face Hub或本地加载
    
    Returns:
        list: 包含数据项的列表
    """
    dataset_type = task_config.get("dataset_type", "unique")
    
    if dataset_type == "huggingface":
        return load_data_from_huggingface()
    else:
        # 保留原有的本地加载逻辑作为备份
        return load_data_from_local()

def load_data_from_huggingface():
    """从Hugging Face Hub加载数据集"""
    dataset_repo = task_config.get("dataset_repo")
    splits = task_config.get("splits", ["test"])
    num_samples = task_config.get("num_sample")
    use_auth_token = task_config.get("use_auth_token", False)
    hf_token = task_config.get("hf_token") or os.environ.get("HF_TOKEN")
    
    if not dataset_repo:
        raise ValueError("dataset_repo must be specified in config when using huggingface dataset_type")
    
    logger.info(f"Loading dataset from Hugging Face Hub: {dataset_repo}")
    
    meta_data = []
    
    # 加载每个split
    for split_name in splits:
        try:
            logger.info(f"Loading split: {split_name}")
            
            # 从Hugging Face加载数据集
            dataset = load_dataset(
                dataset_repo,
                split=split_name,
                token=hf_token if use_auth_token else None
            )
            
            logger.info(f"Loaded {len(dataset)} samples from split {split_name}")
            
            # 限制样本数量（如果指定）
            if num_samples and num_samples < len(dataset):
                dataset = dataset.select(range(num_samples))
                logger.info(f"Limited to {num_samples} samples")
            
            # 转换为meta_data格式
            for item in dataset:
                # 获取问题图像（PIL Image对象）
                question_image = item['question_image']
                
                # 获取选项图像（List of PIL Image对象）
                choice_images = item['choice_images']
                
                # 构建images列表：[问题图像, 选项A, 选项B, 选项C]
                images = [question_image] + choice_images
                
                # 构建数据项
                data_item = {
                    'idx': item['idx'],
                    'images': images,  # [question_image, choice_A, choice_B, choice_C]
                    'text': item['question_text'],
                    'answer': item['correct_answer'],
                    'category': item.get('category', 'jigsaw_coco')
                }
                
                meta_data.append(data_item)
                
        except Exception as e:
            logger.error(f"Error loading split {split_name}: {e}")
            continue
    
    # 数据集统计信息
    logger.info(f"Total data loaded from Hugging Face: {len(meta_data)}")
    
    # 统计答案分布
    answer_counts = {}
    for item in meta_data:
        answer = item.get("answer", "unknown")
        answer_counts[answer] = answer_counts.get(answer, 0) + 1
    
    logger.info("Answer distribution:")
    for answer, count in sorted(answer_counts.items()):
        logger.info(f"  {answer}: {count}")
    
    return meta_data

def load_data_from_local():
    """从本地加载数据集（原有逻辑，作为备份）"""
    dataset_path = task_config.get("dataset_path")
    num_samples = task_config.get("num_sample")
    
    if not dataset_path:
        raise ValueError("dataset_path must be specified in config when using local dataset")
    
    logger.info(f"Loading dataset from local path: {dataset_path}")
    
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
        
        # 加载问题图像
        question_image = load_image(question_image_path)
        images = [question_image]
        
        # 加载选项图像
        choices = []
        for choice in item["choices"]:
            choice_img = load_image(choice["image"])
            choices.append(choice_img)
        images.extend(choices)
        
        # 获取问题文本和正确答案
        question_text = item["question_text"]
        correct_answer = item["correct_answer"]["letter"]
        
        # 添加到元数据列表
        meta_data.append({
            "idx": item_id,
            "images": images,  # [question_image, choice_A, choice_B, choice_C]
            "text": question_text,
            "answer": correct_answer,
            "category": "jigsaw"
        })
    
    # 显示统计信息
    logger.info(f"Total data loaded from local: {len(meta_data)}")
    
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
        r'(?:option|answer):*\s*([a-c])[.、)）]?\b',  # 匹配"选项是A"、"答案为B"等形式
        r'(?:the answer is|i choose|i pick):*\s*([a-c])[.、)）]?\b'  # 匹配"The answer is A"等形式
    ]
    
    for pattern in option_patterns:
        match = re.search(pattern, pred, re.IGNORECASE)
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
    
    # 统计每个答案选项的表现
    answer_stats = {'a': {'correct': 0, 'total': 0}, 
                    'b': {'correct': 0, 'total': 0}, 
                    'c': {'correct': 0, 'total': 0}}
    
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
        
        # 统计每个答案选项的表现
        gt_lower = ground_truth.lower()
        if gt_lower in answer_stats:
            answer_stats[gt_lower]['total'] += 1
            answer_stats[gt_lower]['correct'] += score
        
        # 记录详细比较日志
        compare_logs.append({
            "idx": idx,
            "category": meta.get("category", "jigsaw_coco"),
            "gold": ground_truth,
            "pred": prediction,
            "score": score,
            "question": meta["text"]
        })
    
    # 计算总体准确率
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # 计算每个答案选项的准确率
    answer_accuracy = {}
    for answer, stats in answer_stats.items():
        if stats['total'] > 0:
            answer_accuracy[answer] = stats['correct'] / stats['total']
        else:
            answer_accuracy[answer] = 0.0
    
    # 组织评估结果
    result_dict = {
        "accuracy": accuracy,
        "total_correct": total_correct,
        "total_samples": total_samples,
        "answer_stats": answer_stats,
        "answer_accuracy": answer_accuracy,
        "compare_logs": compare_logs,
        "meta_data": meta_dict,
        "results": results
    }
    
    # 打印评估结果摘要
    logger.info(f"总体准确率: {accuracy:.4f} ({total_correct}/{total_samples})")
    logger.info("各选项准确率:")
    for answer in sorted(answer_accuracy.keys()):
        stats = answer_stats[answer]
        acc = answer_accuracy[answer]
        logger.info(f"  {answer.upper()}: {acc:.4f} ({stats['correct']}/{stats['total']})")
    
    return result_dict