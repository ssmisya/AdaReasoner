from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os, sys
from datasets import load_dataset
import logging
import warnings
from collections import defaultdict

# 禁用相关库的日志
logging.getLogger('datasets').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

# MME评估类型字典
eval_type_dict = {
    "Perception": [
        "existence",
        "count", 
        "position",
        "color",
        "posters",
        "celebrity",
        "scene",
        "landmark",
        "artwork",
        "OCR",
    ],
    "Cognition": [
        "commonsense_reasoning",
        "numerical_calculation", 
        "text_translation",
        "code_reasoning",
    ],
}


def load_data_function():
    dataset_path = task_config["dataset_path"]
    num_sample = task_config.get("num_sample", None)
    
    # 加载MME数据集
    dataset = load_dataset(dataset_path, split="test", token=True)
    
    # 只处理前num_sample个样本
    if num_sample:
        dataset = dataset.select(range(min(num_sample, len(dataset))))
    
    meta_data = []
    
    for idx, item in enumerate(dataset):
        item_id = f"mme_{idx}"
        image = item["image"].convert("RGB")
        question = item["question"].strip()
        text = f"{question}\nAnswer the question using a single word or phrase."
        answer = item["answer"].lower().strip()
        question_id = item.get("question_id", f"q_{idx}")
        category = item.get("category", "unknown")
        
        meta_data.append({
            "idx": item_id,
            "image": image,
            "text": text,
            "answer": answer,
            "question_id": question_id,
            "category": category,
            "original_question": question
        })
    
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


def parse_pred_ans(pred_ans):
    """解析预测答案，改编自Otter Eval"""
    pred_ans = pred_ans.lower().strip().replace(".", "")
    pred_label = None
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    elif len(pred_ans) == 1:
        if pred_ans == "y":
            pred_label = "yes"
        elif pred_ans == "n":
            pred_label = "no"
        else:
            pred_label = "other"
    else:
        prefix_pred_ans = pred_ans[:4]
        if "yes" in prefix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"
    return pred_label


def mme_aggregate_results(results, eval_type):
    """
    MME评分聚合函数
    Args:
        results: 包含question_id, category, score的结果列表
        eval_type: "Perception" 或 "Cognition"
    Returns:
        总分数
    """
    category2score = defaultdict(dict)
    target_categories = eval_type_dict[eval_type]
    
    for result in results:
        question_id = result["question_id"]
        score = result["score"]
        category = result["category"]
        
        # 只处理属于当前评估类型的类别
        if category not in target_categories:
            continue
            
        if question_id not in category2score[category]:
            category2score[category][question_id] = []
        category2score[category][question_id].append(score)
    
    category2avg_score = {}
    for category, question2scores in category2score.items():
        total_score = 0
        for question_id, scores in question2scores.items():
            if len(scores) != 2:
                # 如果不是成对评估，使用平均分
                acc = sum(scores) / len(scores) * 100.0
                acc_plus = (sum(scores) == len(scores)) * 100.0
            else:
                # MME标准成对评估
                acc = sum(scores) / len(scores) * 100.0
                acc_plus = (sum(scores) == 2) * 100.0
            score = acc_plus + acc
            total_score += score
        
        if len(question2scores) > 0:
            avg_score = total_score / len(question2scores)
            category2avg_score[category] = avg_score
            logger.info(f"{category}: {avg_score:.2f}")
    
    total_score = sum(category2avg_score.values())
    return total_score, category2avg_score


def evaluate_function(results, meta_data):
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    
    compare_logs = []
    perception_results = []
    cognition_results = []
    
    for idx, meta in meta_dict.items():
        if idx in results_dict:
            prediction = results_dict[idx]["results"]["final_answer"]
        else:
            prediction = "None"
        
        if prediction is None:
            prediction = "None"
        
        # 解析预测答案
        pred_ans = parse_pred_ans(prediction)
        gt_ans = meta["answer"]
        category = meta["category"]
        question_id = meta["question_id"]
        
        # 确保答案是yes或no
        assert gt_ans in ["yes", "no"], f"Invalid ground truth answer: {gt_ans}"
        assert pred_ans in ["yes", "no", "other"], f"Invalid prediction: {pred_ans}"
        
        # 计算分数
        score = 1.0 if pred_ans == gt_ans else 0.0
        
        # 根据类别分类结果
        result_data = {
            "question_id": question_id,
            "category": category,
            "score": score
        }
        
        if category in eval_type_dict["Perception"]:
            perception_results.append(result_data)
        elif category in eval_type_dict["Cognition"]:
            cognition_results.append(result_data)
        
        compare_logs.append({
            "idx": idx,
            "question_id": question_id,
            "category": category,
            "question": meta["original_question"],
            "gold": gt_ans,
            "pred": pred_ans,
            "score": score
        })
    
    # 计算感知和认知分数
    perception_score, perception_details = mme_aggregate_results(perception_results, "Perception")
    cognition_score, cognition_details = mme_aggregate_results(cognition_results, "Cognition")
    total_score = perception_score + cognition_score
    
    logger.info(f"MME Perception Score: {perception_score:.2f}")
    logger.info(f"MME Cognition Score: {cognition_score:.2f}")
    logger.info(f"MME Total Score: {total_score:.2f}")
    
    return {
        "mme_perception_score": perception_score,
        "mme_cognition_score": cognition_score,
        "mme_total_score": total_score,
        "perception_details": perception_details,
        "cognition_details": cognition_details,
        "compare_logs": compare_logs,
        "results": results,
        "meta_data": meta_data
    }