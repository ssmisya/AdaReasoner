from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os, sys
from datasets import load_dataset
import math
import nltk
from thefuzz import fuzz
import numpy as np
import logging
import warnings
from tqdm import tqdm

# 禁用tqdm进度条
tqdm.pandas = lambda *args, **kwargs: None

# 禁用SentenceTransformer和相关库的日志
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('datasets').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

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
    num_sample = task_config.get("num_sample", None)
    
    # 这里的测试应该是validation
    testset = load_dataset(dataset_path,split="validation")
    # 只处理testset的前num_sample个样本
    if num_sample:
        testset = testset.select(range(min(num_sample, len(testset))))
    meta_data = []

    for idx, item in enumerate(testset):
        item_id = f"toorleval_{idx}"
        image = item["images"][0]
        text = item["problem"].replace("<image>","")
        answer = item["answer"]
        meta_data.append({"idx":item_id, "image":image, "text":text, "answer":answer})

    return meta_data


def evaluate_function(results,meta_data):
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    res_list = []
    compare_logs = []
    # breakpoint()
    # comparator_path = task_config.get("answer_comparator_path", None)
    # comparator = LLMAnswerComparator(threshold=0.8, method="bert", model_path=comparator_path)
    for idx, meta in meta_dict.items():
        if idx in results_dict:
            meta["prediction"] = results_dict[idx]["results"]["final_answer"]
        else:
            meta["prediction"] = "None"
        
        gold = meta["answer"].replace("\"","").strip().lower()
        # pred = meta["prediction"].replace("\"","").strip().lower()
        pred = meta["prediction"]
        if pred is None:
            score = 0.0
        else:
            pred = pred.replace("\"","").strip().lower()
        
        score = rule_based_verify(gold, pred)
        res_list.append(score)
        compare_logs.append({"idx":idx,"gold":gold,"pred":pred,"score":score,"question":meta["text"]})

    accuracy = sum(res_list) / len(res_list) if len(res_list) > 0 else 0
    
    return {"Acc":accuracy, "compare_logs":compare_logs,  "results":results, "meta_data":meta_data} #

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
        comparator: The comparator to use.
    Returns:
        bool: True if the prediction is correct, False otherwise.
    """
    
    if pred == "None" or pred == "" or pred is None:
        return 0.0

    # 去除双引号，两端的空白并转换成小写
    gold = gold.replace("\"","").strip().lower()
    pred = pred.replace("\"","").strip().lower()
    
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
    
    return 0