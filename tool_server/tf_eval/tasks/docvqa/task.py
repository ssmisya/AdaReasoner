from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os, sys
from datasets import load_dataset
import math
import nltk
from thefuzz import fuzz
import numpy as np
import collections
from transformers.data.metrics.squad_metrics import squad_evaluate
nltk.download("punkt", quiet=True)

try:
    from math_verify import parse, verify
except ImportError:
    print("math_verify package not found. Please install it to use math verification features.")

'''
所有可能的question_types:
Image/Photo
Yes/No
figure/diagram
form
free_text
handwritten
layout
others
table/list

每种question_type的数量:
layout: 1981
table/list: 1780
form: 1021
free_text: 765
handwritten: 319
figure/diagram: 265
others: 236
Image/Photo: 98
Yes/No: 28

所有type数量加起来超过了testset，是因为，question_types中，可能同时包含多个类别，所以不计算类别的准确度
'''


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)


def load_data_function():
    
    dataset_path = task_config["dataset_path"]
    num_samples = task_config["num_sample"]

    dataset = load_dataset(dataset_path, name="DocVQA", split="validation")
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    meta_data = []
    for idx,item in enumerate(dataset):
        item_id = f"docvqa_{idx}"
        image = item["image"]
        text = item["question"]
        answer = item["answers"]
        meta_data.append({"idx":item_id, "image":image, "text":text, "answer":answer})

    ## Show statistics
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


def evaluate_function(results,meta_data):
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    # res_list = []
    exact_score_list = []
    f1_score_list = []
    compare_logs = []
    
    for idx, meta in meta_dict.items():
        if idx in results_dict:
            meta["prediction"] = results_dict[idx]["results"]["final_answer"]
        else:
            meta["prediction"] = "None"
        prediction = meta["prediction"]
        ground_truth = meta["answer"]
        # ground_truth是一个列表，遍历每个答案进行分数计算，取最高值作为score
        max_exact_score = 0.0
        max_f1_score = 0.0
        for gt in ground_truth:
            max_exact_score, max_f1_score = rule_based_verify(gt, prediction)
        exact_score_list.append(max_exact_score)
        f1_score_list.append(max_f1_score)
        compare_logs.append({"idx":idx, "gold":ground_truth, "pred":prediction, "exact_score":max_exact_score, "f1_score":max_f1_score})

    return {"exact_score":sum(exact_score_list) / len(exact_score_list), "f1_score":sum(f1_score_list) / len(f1_score_list), "compare_logs":compare_logs, "results":results,"meta_data":meta_data}


def rule_based_verify(
    gold: str,
    pred: str
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
        return 0.0, 0.0

    # 去除双引号，两端的空白并转换成小写
    gold = gold.replace("\"","").strip().lower()
    pred = pred.replace("\"","").strip().lower()


    SquadExample = collections.namedtuple("SquadExample", ["qas_id", "answers"])
    # 将gold和pred转换为正确格式
    examples = [SquadExample(qas_id="", answers=[{"text": gold}])]
    preds = {"": pred}  # 这里需要是字典，键为qas_id，值为预测文本
    
    # 使用squad_evaluate计算exact和f1
    result = squad_evaluate(examples, preds)
    # 除以100
    return result['exact'] / 100, result['f1'] / 100