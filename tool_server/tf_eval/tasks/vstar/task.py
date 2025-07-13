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

    dataset = load_dataset(dataset_path, split="test")
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    meta_data = []
    for idx,item in enumerate(dataset):
        item_id = f"vstar_{idx}"
        # 他这里的image是路径
        image_path = os.path.join(dataset_path, item["image"])
        image = Image.open(image_path)
        text = item["text"]
        answer = item["label"]
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
    category_dict = {item : [] for item in ["direct_attributes", "relative_position"]}
    
    for idx, meta in meta_dict.items():
        if idx in results_dict:
            meta["prediction"] = results_dict[idx]["results"]["final_answer"]
        else:
            meta["prediction"] = "None"
        item_category = meta["category"]
        prediction = meta["prediction"]
        ground_truth = meta["answer"]
        score = rule_based_verify(ground_truth, prediction)
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

def is_convertible_to_float(s: str) -> bool:
    """辅助函数，检查一个字符串是否可以被转换为浮点数。"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def rule_based_verify(
    gold: str,
    pred: str
) -> bool:
    """
    验证预测的选项是否与正确答案相同。
    
    Args:
        gold (str): 正确答案（A、B、C、D中的一个）。
        pred (str): 预测的答案。
        comparator: LLMAnswerComparator实例（此参数在当前函数中不使用）
        
    Returns:
        float: 如果预测正确返回1.0，否则返回0.0。
    """
    
    # 去除双引号，两端的空白并转换成小写
    gold = gold.replace("\"","").strip().lower()
    # 当不为none时，去除双引号，两端的空白并转换成小写
    if pred:
        pred = pred.replace("\"","").strip().lower()
    else:
        pred = ""

    # 如果pred字符串为空，则返回0
    if pred == "" or pred is None or pred == "None":
        return 0.0
        
    # 提取pred中的选项字母
    import re
    option_patterns = [
        r'\b([a-d])[.、)）]?\b',  # 匹配单独的A、B、C、D，可能带有.、)等后缀
        r'(?:option|answer):*([a-d])[.、)）]?\b',  # 匹配"选项是A"、"答案为B"等形式
        r'(?:the answer is|i choose|i pick):*([a-d])[.、)）]?\b'  # 匹配"The answer is A"等形式
    ]
    
    for pattern in option_patterns:
        match = re.search(pattern, pred)
        if match:
            extracted_option = match.group(1).lower()
            if extracted_option == gold:
                return 1.0
    
    # 直接比较选项是否相同
    if gold == pred:
        return 1.0
    else:
        return 0.0

