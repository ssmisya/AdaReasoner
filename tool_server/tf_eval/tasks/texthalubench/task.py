from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os, sys
from datasets import load_dataset
import math
import nltk
from thefuzz import fuzz
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download("punkt", quiet=True)
import re

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

    dataset = load_dataset(dataset_path, split="validation")
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    meta_data = []
    for idx,item in enumerate(dataset):
        item_id = f"texthalubench_{idx}"
        image = item["image"]
        text = item["question"]
        # answer是一个列表，对于Understanding任务，answer[0]就是答案，对于Spotting任务，是整个列表
        answer = item["ground_truth"]
        category = item["category"]
        meta_data.append({"idx":item_id, "image":image, "text":text, "answer":answer, "category":category})

    ## Show statistics
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


# 有两种qa_type，一种是recognizing，一种是reasoning，需要分别计算准确率
def evaluate_function(results,meta_data):
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    compare_logs = []
    category_dict = {item : [] for item in ["Spotting", "Understanding"]}
    
    for idx, meta in meta_dict.items():
        if idx in results_dict:
            meta["prediction"] = results_dict[idx]["results"]["final_answer"]
        else:
            meta["prediction"] = "None"
        item_category = meta["category"]
        prediction = meta["prediction"]
        ground_truth = meta["answer"]
        question = meta["text"]
        score = rule_based_verify(ground_truth, prediction,item_category,question)
        category_dict[item_category].append(score)
        compare_logs.append({"idx":idx, "category":item_category, "gold":ground_truth, "pred":prediction, "score":score,"question":meta["text"]})
    for k,v in category_dict.items():
        if len(v) > 0:
            category_dict[k] = sum(v) / len(v)
        else:
            category_dict[k] = 0.0
    res_dict = dict(
        category_res = category_dict,
        compare_logs = compare_logs,
        meta_data = meta_dict,
        results = results,
    )
    return res_dict


def rule_based_verify(
    gold: str,
    pred: str,
    category: str,
    question: str
) -> float:

    if category == "Spotting":
        tp, fp, fn = 0, 0, 0
        gt_words = [x.upper() for x in gold if x and x != "###"]
        pred_words = re.findall(r'\b\w+\b', pred.upper())
        pred_words = list(set(pred_words) - {"TEXT"})
        unmatched = gt_words.copy()
        for word in pred_words:
            if word in unmatched:
                tp+=1
                unmatched.remove(word)
            else:
                fp+=1
        fn += len(unmatched)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        return round(f1, 4)
    elif category == "Understanding":
        # 需要question
        tp, fp, fn = 0, 0, 0
        answer_gt = gold[0].strip().split()
        formatted_q = add_linebreaks(question)
        options = parse_options(formatted_q)
        predicted_letters = extract_answers(pred, options)
        unmatched = answer_gt.copy()
        for p in predicted_letters:
            if p in unmatched:
                tp += 1
                unmatched.remove(p)
            else:
                fp += 1
        fn += len(unmatched)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        return round(f1, 4)
    else:
        raise ValueError("the question type not in Spotting or Understanding!")


def add_linebreaks(question):
    return re.sub(r'(?<!\n)(?=[ABCD]\.)', '\n', question)

def parse_options(text):
    pattern = r"([A-D])\.\s*(.+?)(?=(?:\n[A-D]\.|$))"
    return {m[0]: " ".join(m[1].strip().split()) for m in re.findall(pattern, text, re.DOTALL)}

def extract_answers(response, options):
    predicted = []
    for letter, opt in options.items():
        if opt in response or f"{letter}. {opt}" in response or letter in response:
            predicted.append(letter)
    return predicted