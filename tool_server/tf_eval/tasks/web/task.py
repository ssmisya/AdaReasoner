from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os, sys
from datasets import load_dataset
import math
import nltk
from thefuzz import fuzz
import numpy as np
nltk.download("punkt", quiet=True)

try:
    from math_verify import parse, verify
except ImportError:
    print("math_verify package not found. Please install it to use math verification features.")


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)


def load_data_function():
    
    dataset_path = task_config["dataset_path"]
    num_samples = task_config.get("num_sample", None)

    dataset = load_dataset(dataset_path,split="validation")
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    meta_data = []
    for idx,item in enumerate(dataset):
        item_id = f"web_{idx}"
        image = item["image"]
        # text = item["question"]
        text = item["question_answer_with_box"]
        answer = item["answer"]
        meta_data.append({"idx":item_id, "image":image, "text":text, "answer":answer})

    ## Show statistics
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


def evaluate_function(results,meta_data):
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    res_list = []
    compare_logs = []
    
    for idx, meta in meta_dict.items():
        if idx in results_dict:
            meta["prediction"] = results_dict[idx]["results"]["final_answer"]
        else:
            meta["prediction"] = "None"
        meta["prediction"] = "None" if not meta["prediction"] else meta["prediction"]
        prediction = meta["prediction"]
        ground_truth = meta["answer"]
        score = rule_based_verify(ground_truth, prediction)
        res_list.append(score)
        compare_logs.append({"idx":idx, "gold":ground_truth, "pred":prediction, "score":score,"question":meta["text"]})
    return {"Acc":sum(res_list) / len(res_list), "compare_logs":compare_logs, "results":results,"meta_data":meta_data}


def rule_based_verify(
    gold: str,
    pred: str,
    anls_threshold: float = 0.5
) -> bool:
    """
    A rule-based verification function to check if the prediction matches the gold standard.
    
    Args:
        gold (str): The ground truth answer.
        pred (str): The predicted answer.
        
    Returns:
        bool: True if the prediction is correct, False otherwise.
    """
    # 如果pred字符串为空，则返回0
    if pred == "" or pred is None or pred == "None":
        return 0.0
    # 去除双引号，两端的空白并转换成小写
    gold = gold.replace("\"","").strip().lower()
    pred = pred.replace("\"","").strip().lower()
    
    dist = levenshtein_distance(pred, gold)
    length = max(len(pred), len(gold))
    normalized_dist = 0.0 if length == 0 else float(dist) / float(length)
    score = 1 - normalized_dist
    if score < anls_threshold:
        score = 0
    return score

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]