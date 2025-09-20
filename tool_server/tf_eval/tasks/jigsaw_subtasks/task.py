# task.py - jigsaw_subtasks
from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os
import sys
import re
import json
import math
from PIL import Image
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

# 直接判断任务说明 - 判断子图是否是缺失部分
DIRECT_COMPARISON_INSTRUCTION = """
Look at the first image (img_1) with one part missing, and the second image (img_2). 
Is the second image the missing part of the first image? Carefully observe and compare the edges and content.

Your final answer should be formatted as \\boxed{Yes} or \\boxed{No}.
"""

# 插入判断任务说明 - 判断插入后的图像是否正确
INSERTION_COMPARISON_INSTRUCTION = """
Examine the three images provided: the first image (img_1) with a missing region, the second image (img_2) as a candidate sub-image, and the third image (img_3) where img_2 has been inserted into the missing region of img_1.

Determine whether img_2 is indeed the correct piece that fills the missing part of img_1. Check img_3 carefully to see if the inserted part aligns seamlessly with the surrounding areas in terms of edges, colors, and overall content.

Your final answer should be formatted as \\boxed{Yes} or \\boxed{No}.
"""

# INSERTION_COMPARISON_INSTRUCTION = """
# Please check whether this image shows any signs of manipulation, such as cropping, splicing, or insertion. In other words, determine whether it is a natural photo or one that has been artificially stitched or altered.

# Your final answer should be formatted as \\boxed{Yes} or \\boxed{No}.
# """

def load_data_function():
    """加载Jigsaw子任务数据集"""
    tasks = task_config.get("tasks", ["direct_comparison", "with_insertion"])
    
    # 从配置文件获取数据路径字典
    dataset_path_dict = task_config.get("dataset_path_dict", {})
    if not dataset_path_dict:
        logger.error("No dataset_path_dict found in config")
        return []
    
    img_dir = task_config.get("image_dir", "")
    
    meta_data = []
    
    # 加载不同任务的数据
    for task in tasks:
        if task not in dataset_path_dict:
            logger.warning(f"No path specified for task {task} in dataset_path_dict")
            continue
        
        task_file_path = dataset_path_dict[task]
        
        # 根据文件扩展名选择加载方法
        if task_file_path.endswith(".jsonl"):
            raw_data = process_jsonl(task_file_path)
        elif task_file_path.endswith(".json"):
            raw_data = load_json_file(task_file_path)
        else:
            logger.warning(f"Unsupported file format: {task_file_path}")
            continue
        
        logger.info(f"Loaded {len(raw_data)} raw records for task {task} from {task_file_path}")
        
        # 根据任务类型处理数据
        if task == "direct_comparison":
            task_data = process_direct_comparison_data(raw_data, task, img_dir)
        elif task == "with_insertion":
            task_data = process_insertion_comparison_data(raw_data, task, img_dir)
        else:
            logger.warning(f"Unknown task type: {task}")
            continue
        
        meta_data.extend(task_data)
    
    # 数据集统计信息
    logger.info(f"Total processed data: {len(meta_data)}")
    
    # 统计各任务的数据量
    task_counts = defaultdict(int)
    answer_counts = defaultdict(lambda: defaultdict(int))
    
    for item in meta_data:
        task_type = item.get("task_type", "unknown")
        answer = item.get("answer", "unknown")
        
        task_counts[task_type] += 1
        answer_counts[task_type][answer] += 1
    
    for task_type, count in task_counts.items():
        logger.info(f"Task type: {task_type}, count: {count}")
        for answer, ans_count in answer_counts[task_type].items():
            logger.info(f"  Answer '{answer}': {ans_count} ({ans_count/count*100:.1f}%)")
    
    return meta_data

def process_direct_comparison_data(raw_data, task_type, img_dir):
    """处理直接比较任务数据"""
    processed_data = []
    
    for item in raw_data:
        # 检查item是否包含必要的字段
        if not all(key in item for key in ["id", "base_image", "choice_image", "correct_answer"]):
            logger.warning(f"Missing required keys in item: {item.get('id', 'unknown')}")
            continue
        
        # 构建提示
        text_prompt = DIRECT_COMPARISON_INSTRUCTION
        
        # 处理图像路径
        base_image_path = os.path.join(img_dir, item["base_image"]) if img_dir and not item["base_image"].startswith("/") else item["base_image"]
        choice_image_path = os.path.join(img_dir, item["choice_image"]) if img_dir and not item["choice_image"].startswith("/") else item["choice_image"]
        
        base_image = Image.open(base_image_path)
        choice_image = Image.open(choice_image_path)
        
        # 验证图像文件是否存在
        if not os.path.exists(base_image_path):
            logger.warning(f"Base image file does not exist: {base_image_path}")
            continue
        
        if not os.path.exists(choice_image_path):
            logger.warning(f"Choice image file does not exist: {choice_image_path}")
            continue
        
        processed_data.append({
            "idx": item["id"],
            "original_id": item.get("original_id", item["id"]),
            "images": [base_image, choice_image],  # 基础图像和选择图像
            "text": text_prompt,
            "answer": item["correct_answer"],  # "Yes" 或 "No"
            "task_type": task_type,
            "judgment_type": item.get("judgment_type", "direct_comparison")
        })
    
    logger.info(f"Processed {len(processed_data)} records for {task_type}")
    return processed_data

def process_insertion_comparison_data(raw_data, task_type, img_dir):
    """处理插入比较任务数据"""
    processed_data = []
    
    for item in raw_data:
        # 检查item是否包含必要的字段
        if not all(key in item for key in ["id", "base_image", "choice_image", "inserted_image", "correct_answer"]):
            logger.warning(f"Missing required keys in item: {item.get('id', 'unknown')}")
            continue
        
        # 构建提示
        text_prompt = INSERTION_COMPARISON_INSTRUCTION
        
        # 处理图像路径
        base_image_path = os.path.join(img_dir, item["base_image"]) if img_dir and not item["base_image"].startswith("/") else item["base_image"]
        choice_image_path = os.path.join(img_dir, item["choice_image"]) if img_dir and not item["choice_image"].startswith("/") else item["choice_image"]
        inserted_image_path = os.path.join(img_dir, item["inserted_image"]) if img_dir and not item["inserted_image"].startswith("/") else item["inserted_image"]
        
        base_image = Image.open(base_image_path)
        choice_image = Image.open(choice_image_path)
        inserted_image = Image.open(inserted_image_path)
        
        # 验证图像文件是否存在
        if not os.path.exists(base_image_path):
            logger.warning(f"Base image file does not exist: {base_image_path}")
            continue
        
        if not os.path.exists(choice_image_path):
            logger.warning(f"Choice image file does not exist: {choice_image_path}")
            continue
            
        if not os.path.exists(inserted_image_path):
            logger.warning(f"Inserted image file does not exist: {inserted_image_path}")
            continue
        
        processed_data.append({
            "idx": item["id"],
            "original_id": item.get("original_id", item["id"]),
            "images": [base_image, choice_image, inserted_image],  # 基础图像、选择图像和插入后图像
            "text": text_prompt,
            "answer": item["correct_answer"],  # "Yes" 或 "No"
            "task_type": task_type,
            "judgment_type": item.get("judgment_type", "with_insertion")
        })
    
    logger.info(f"Processed {len(processed_data)} records for {task_type}")
    return processed_data

def evaluate_function(results, meta_data):
    """评估函数，根据任务类型对结果进行评估"""
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    
    # 按任务类型统计结果
    task_results = {}
    
    # 按任务类型和正确答案类型统计结果
    answer_type_results = {}
    
    # 添加合规输出统计
    compliance_stats = {
        "overall": {"compliant": 0, "total": 0, "correct_if_compliant": 0},
        "direct_comparison": {"compliant": 0, "total": 0, "correct_if_compliant": 0},
        "with_insertion": {"compliant": 0, "total": 0, "correct_if_compliant": 0}
    }
    
    overall_correct = 0
    overall_total = 0
    
    compare_logs = []
    
    for idx, meta in meta_dict.items():
        task_type = meta["task_type"]
        
        # 初始化任务类型统计
        if task_type not in task_results:
            task_results[task_type] = {"correct": 0, "total": 0}
        
        # 初始化答案类型统计
        answer_type = meta["answer"]  # "Yes" 或 "No"
        task_answer_key = f"{task_type}_{answer_type}"
        if task_answer_key not in answer_type_results:
            answer_type_results[task_answer_key] = {
                "correct": 0, "total": 0, "task_type": task_type, "answer_type": answer_type
            }
        
        if idx in results_dict:
            prediction = results_dict[idx]["results"]["final_answer"]
            meta["prediction"] = prediction
        else:
            prediction = None
            meta["prediction"] = None
        
        # 评估结果
        score, message, is_compliant = evaluate_yes_no_answer(prediction, meta)
        
        # 记录答案类型结果
        answer_type_results[task_answer_key]["correct"] += score
        answer_type_results[task_answer_key]["total"] += 1
        
        # 记录任务类型结果
        task_results[task_type]["correct"] += score
        task_results[task_type]["total"] += 1
        
        # 记录总体结果
        overall_correct += score
        overall_total += 1
        
        # 更新合规性统计
        compliance_stats[task_type]["total"] += 1
        compliance_stats["overall"]["total"] += 1
        if is_compliant:
            compliance_stats[task_type]["compliant"] += 1
            compliance_stats["overall"]["compliant"] += 1
            if score == 1:  # 如果答案正确
                compliance_stats[task_type]["correct_if_compliant"] += 1
                compliance_stats["overall"]["correct_if_compliant"] += 1
        
        # 日志记录
        log_entry = {
            "idx": idx,
            "original_id": meta.get("original_id", idx),
            "task_type": task_type,
            "judgment_type": meta.get("judgment_type", task_type),
            "gold": meta["answer"],
            "pred": prediction,
            "score": score,
            "message": message,
            "is_compliant": is_compliant
        }
        
        compare_logs.append(log_entry)
    
    # 计算每个任务类型的准确率
    for task_type in task_results:
        if task_results[task_type]["total"] > 0:
            task_results[task_type]["accuracy"] = task_results[task_type]["correct"] / task_results[task_type]["total"]
        else:
            task_results[task_type]["accuracy"] = 0
    
    # 计算每个答案类型的准确率
    for key in answer_type_results:
        if answer_type_results[key]["total"] > 0:
            answer_type_results[key]["accuracy"] = answer_type_results[key]["correct"] / answer_type_results[key]["total"]
        else:
            answer_type_results[key]["accuracy"] = 0
    
    # 计算合规率和合规答案的准确率
    compliance_rates = {}
    for task_type, stats in compliance_stats.items():
        if stats["total"] > 0:
            compliance_rate = stats["compliant"] / stats["total"]
        else:
            compliance_rate = 0
            
        if stats["compliant"] > 0:
            compliant_accuracy = stats["correct_if_compliant"] / stats["compliant"]
        else:
            compliant_accuracy = 0
            
        compliance_rates[task_type] = {
            "compliance_rate": compliance_rate,
            "compliant_accuracy": compliant_accuracy,
            "compliant_count": stats["compliant"],
            "total_count": stats["total"],
            "correct_if_compliant": stats["correct_if_compliant"]
        }
    
    # 将结果转换为列表，便于排序
    answer_type_results_list = list(answer_type_results.values())
    answer_type_results_list.sort(key=lambda x: (x["task_type"], x["answer_type"]))
    
    # 计算总体准确率
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    
    result = {
        "overall_accuracy": overall_accuracy,
        "task_results": task_results,
        "answer_type_results": answer_type_results_list,
        "compliance_stats": compliance_rates,
        "compare_logs": compare_logs
    }
    
    # 打印结果摘要
    logger.info(f"Overall accuracy: {overall_accuracy:.4f} ({overall_correct}/{overall_total})")
    
    for task_type in task_results:
        logger.info(f"Task type {task_type}: {task_results[task_type]['accuracy']:.4f} "
                   f"({task_results[task_type]['correct']}/{task_results[task_type]['total']})")
    
    # 打印按答案类型的结果
    logger.info("Answer type breakdown:")
    for item in answer_type_results_list:
        logger.info(f"  {item['task_type']} - {item['answer_type']}: {item['accuracy']:.4f} "
                   f"({item['correct']}/{item['total']})")
    
    # 打印合规性统计
    logger.info("\nCompliance statistics:")
    for task_type, stats in compliance_rates.items():
        logger.info(f"{task_type}:")
        logger.info(f"  Compliance rate: {stats['compliance_rate']:.4f} ({stats['compliant_count']}/{stats['total_count']})")
        logger.info(f"  Accuracy if compliant: {stats['compliant_accuracy']:.4f} ({stats['correct_if_compliant']}/{stats['compliant_count']})")
    
    return result

def evaluate_yes_no_answer(prediction, meta):
    """评估是/否回答"""
    if prediction is None:
        return 0, "No prediction", False
    
    # 预处理预测结果
    yes_pattern = r'\\boxed\s*{\s*yes\s*}|boxed\s*{\s*yes\s*}|boxed{yes}|boxed\(\s*yes\s*\)'
    no_pattern = r'\\boxed\s*{\s*no\s*}|boxed\s*{\s*no\s*}|boxed{no}|boxed\(\s*no\s*\)'
    
    prediction = prediction.lower().strip()
    
    # 判断是否合规
    is_compliant = False
    pred_answer = None
    
    # 检查是否只包含yes或no答案
    if prediction in ["yes", "no", "yes.", "no."]:
        is_compliant = True
        pred_answer = "yes" if prediction.startswith("yes") else "no"
    elif re.search(yes_pattern, prediction, re.IGNORECASE) and not re.search(no_pattern, prediction, re.IGNORECASE):
        is_compliant = True
        pred_answer = "yes"
    elif re.search(no_pattern, prediction, re.IGNORECASE) and not re.search(yes_pattern, prediction, re.IGNORECASE):
        is_compliant = True
        pred_answer = "no"
    else:
        # 尝试从最后一段文本中提取答案
        last_paragraph = prediction.split('\n')[-1].strip()
        if "yes" in last_paragraph.lower() and "no" not in last_paragraph.lower():
            pred_answer = "yes"
        elif "no" in last_paragraph.lower() and "yes" not in last_paragraph.lower():
            pred_answer = "no"
        else:
            return 0, "Invalid prediction format: cannot determine yes/no", False
    
    # 获取正确答案
    gold_answer = meta["answer"].lower().strip()
    
    # 比较答案
    if pred_answer == gold_answer.lower():
        return 1, "Correct", is_compliant
    else:
        return 0, f"Incorrect. Expected: {gold_answer}, Got: {pred_answer}", is_compliant