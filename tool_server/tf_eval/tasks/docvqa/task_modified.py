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


# 有两种qa_type，一种是recognizing，一种是reasoning，需要分别计算准确率
def evaluate_function(results, meta_data):
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    res_list = []
    compare_logs = []
    
    # 使用squad_evaluate方式评估
    SquadExample = collections.namedtuple("SquadExample", ["qas_id", "answers"])
    squad_examples = []
    squad_preds = {}
    
    for idx, meta in meta_dict.items():
        if idx in results_dict:
            meta["prediction"] = results_dict[idx]["results"]["final_answer"]
        else:
            meta["prediction"] = "None"
        prediction = meta["prediction"]
        ground_truth = meta["answer"]
        
        # 为squad_evaluate准备数据
        squad_answers = [{"text": gt} for gt in ground_truth]
        squad_examples.append(SquadExample(qas_id=idx, answers=squad_answers))
        squad_preds[idx] = prediction
        
        # 继续计算原来的分数，作为对比和备份
        max_score = 0.0
        for gt in ground_truth:
            score = rule_based_verify(gt, prediction)
            max_score = max(max_score, score)
        res_list.append(max_score)
        compare_logs.append(
            f"Ground Truth: {ground_truth}, Prediction: {prediction}, Score: {score}"
        )

    # 使用squad_evaluate计算分数
    squad_results = squad_evaluate(squad_examples, squad_preds)
    
    return {
        "Acc": sum(res_list) / len(res_list),
        "exact": squad_results["exact"],
        "f1": squad_results["f1"],
        "compare_logs": compare_logs,
        "results": results,
        "meta_data": meta_data
    }


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
    A rule-based verification function to check if the prediction matches the gold standard.
    
    Args:
        gold (str): The ground truth answer.
        pred (str): The predicted answer.
        
    Returns:
        bool: True if the prediction is correct, False otherwise.
    """
    
    # 去除双引号，两端的空白并转换成小写
    gold = gold.replace("\"","").strip().lower()
    pred = pred.replace("\"","").strip().lower()

    # 如果pred字符串为空，则返回0
    if pred == "" or pred == "none":
        return 0.0

    # 直接字符串比较
    if gold == pred:
        return 1.0
    
    # 处理时间格式 (如 -72:00, 24:00)，直接进行字符串比较
    if ':' in gold or ':' in pred:
        return 1.0 if gold == pred else 0.0
    # 不是时间格式，则判断是否为数字
    else:
        is_gold_numeric = is_convertible_to_float(gold)
        is_pred_numeric = is_convertible_to_float(pred)

        if is_gold_numeric and is_pred_numeric:
            gold_value = float(gold)
            pred_value = float(pred)
            if math.isclose(gold_value, pred_value, rel_tol=1e-9, abs_tol=1e-9):
                return 1.0
            else:
                # 两个数字不相等，判定为错误，不再继续
                return 0.0
    
    # 添加希腊字母和拉丁字母的映射
    greek_to_latin = {
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 
        'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
        'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu',
        'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron', 'π': 'pi',
        'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
        'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega'
    }
    
    # 规范化希腊字母
    normalized_gold = gold
    normalized_pred = pred
    
    for greek, latin in greek_to_latin.items():
        normalized_gold = normalized_gold.replace(greek, latin)
        normalized_pred = normalized_pred.replace(greek, latin)
    
    # 提取等式两边的值（如果存在）
    gold_eq_parts = normalized_gold.split('=')
    pred_eq_parts = normalized_pred.split('=')
    
    # 如果两边都是等式，检查是否完全匹配
    if len(gold_eq_parts) == 2 and len(pred_eq_parts) == 2:
        gold_var = gold_eq_parts[0].strip()
        gold_val = gold_eq_parts[1].strip()
        pred_var = pred_eq_parts[0].strip()
        pred_val = pred_eq_parts[1].strip()
        
        # 严格比较：变量名和值都必须匹配
        # 如果变量名或值不同，直接返回0
        if gold_var != pred_var or gold_val != pred_val:
            return 0.0
    
    # 移除空格和下划线，以便更好地比较变量名
    normalized_gold = normalized_gold.replace(" ", "").replace("_", "")
    normalized_pred = normalized_pred.replace(" ", "").replace("_", "")
    
    # 使用规范化后的字符串进行比较
    if normalized_gold == normalized_pred:
        return 1.0
    
    # 安全地使用 LaTeX 解析比较
    # 由于我们已经处理了等式的情况，这里可以跳过可能导致错误的情况
    try:
        gold_latex = parse('${0}$'.format(gold))
        pred_latex = parse('${0}$'.format(pred))
        if verify(gold_latex, pred_latex):
            return 1.0
    except (TypeError, AttributeError):
        # 特定错误类型，跳过 LaTeX 比较
        pass
    except Exception:
        pass
    
    # 安全地尝试直接使用 verify 函数
    try:
        if verify(gold, pred):
            return 1.0
    except (TypeError, AttributeError):
        # 特定错误类型，跳过直接验证
        pass
    except Exception:
        pass
        
    # 处理百分比
    if '%' in pred:
        pred_no_percent = pred.replace('%','')
        if '.' in pred_no_percent:
            pred_no_percent = pred_no_percent.split('.')[0]
        try:
            if pred_no_percent == gold:
                return 1.0
        except Exception:
            pass

    # 处理小数点
    elif '.' in pred:
        pred_no_decimal = pred.split('.')[0]
        try:
            if pred_no_decimal == gold:
                return 1.0
        except Exception:
            pass

    
    return 0.0 