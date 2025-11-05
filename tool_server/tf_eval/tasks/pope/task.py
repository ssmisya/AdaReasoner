from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os, sys
from datasets import load_dataset
import logging
import warnings

# 禁用相关库的日志
logging.getLogger('datasets').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)


def load_data_function():
    dataset_path = task_config["dataset_path"]
    num_sample = task_config.get("num_sample", None)
    
    # 加载POPE数据集
    dataset = load_dataset(dataset_path, split="test", token=True)
    
    # 只处理前num_sample个样本
    if num_sample:
        dataset = dataset.select(range(min(num_sample, len(dataset))))
    
    meta_data = []
    
    for idx, item in enumerate(dataset):
        item_id = f"pope_{idx}"
        image = item["image"].convert("RGB")
        question = item["question"].strip()
        text = f"{question}\nAnswer the question using a single word or phrase."
        answer = item["answer"].lower().strip()
        question_id = item.get("question_id", f"q_{idx}")
        
        meta_data.append({
            "idx": item_id,
            "image": image,
            "text": text,
            "answer": answer,
            "question_id": question_id,
            "original_question": question
        })
    
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


def evaluate_function(results, meta_data):
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    
    compare_logs = []
    accuracy_scores = []
    precision_data = []
    recall_data = []
    f1_data = []
    yes_ratio_data = []
    
    for idx, meta in meta_dict.items():
        if idx in results_dict:
            prediction = results_dict[idx]["results"]["final_answer"]
        else:
            prediction = "None"
        
        if prediction is None:
            prediction = "None"
        
        pred = prediction.lower().strip()
        gt_ans = meta["answer"]
        
        # 确保答案是yes或no
        assert gt_ans in ["yes", "no"], f"Invalid ground truth answer: {gt_ans}"
        
        # 计算分数
        if "yes" in pred and "no" in pred:
            score = 0.0
        else:
            score = 1.0 if gt_ans in pred else 0.0
        
        accuracy_scores.append(score)
        
        # 为precision, recall, f1收集数据
        result_data = {
            "question_id": meta["question_id"],
            "score": score,
            "prediction": pred,
            "ground_truth": gt_ans
        }
        precision_data.append(result_data)
        recall_data.append(result_data)
        f1_data.append(result_data)
        yes_ratio_data.append(result_data)
        
        compare_logs.append({
            "idx": idx,
            "question_id": meta["question_id"],
            "question": meta["original_question"],
            "gold": gt_ans,
            "pred": pred,
            "score": score
        })
    
    # 计算各种指标
    accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
    precision = calculate_precision(precision_data)
    recall = calculate_recall(recall_data)
    f1_score = calculate_f1_score(precision_data)
    yes_ratio = calculate_yes_ratio(yes_ratio_data)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "yes_ratio": yes_ratio,
        "compare_logs": compare_logs,
        "results": results,
        "meta_data": meta_data
    }


def calculate_precision(results):
    true_positives = 0
    false_positives = 0
    
    for result in results:
        pred = result["prediction"].lower()
        gt = result["ground_truth"]
        
        if "yes" in pred and "no" in pred:
            continue
            
        if gt == "yes" and "yes" in pred:
            true_positives += 1
        elif gt == "no" and "yes" in pred:
            false_positives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision


def calculate_recall(results):
    true_positives = 0
    false_negatives = 0
    
    for result in results:
        pred = result["prediction"].lower()
        gt = result["ground_truth"]
        
        if "yes" in pred and "no" in pred:
            continue
            
        if gt == "yes" and "yes" in pred:
            true_positives += 1
        elif gt == "yes" and "no" in pred:
            false_negatives += 1
    
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall


def calculate_f1_score(results):
    precision = calculate_precision(results)
    recall = calculate_recall(results)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def calculate_yes_ratio(results):
    yes_count = 0
    no_count = 0
    
    for result in results:
        gt = result["ground_truth"]
        if gt == "yes":
            yes_count += 1
        elif gt == "no":
            no_count += 1
    
    yes_ratio = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    return yes_ratio