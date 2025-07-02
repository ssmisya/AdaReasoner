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
        item_id = f"reachqa_{idx}"
        image = item["image"]
        text = item["question"]
        answer = item["answer"]
        chart_type = item["chart_type"]
        qa_type = item["qa_type"]
        meta_data.append({"idx":item_id, "image":image, "text":text, "answer":answer, "chart_type":chart_type, "qa_type":qa_type})

    ## Show statistics
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


# 有两种qa_type，一种是recognizing，一种是reasoning，需要分别计算准确率
def evaluate_function(results,meta_data):
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    res_list = []
    compare_logs = []
    chart_type_dict = {item : [] for item in ['Scatter Charts', 'Specific Charts', 'Area Charts', 'Box Charts', 'Node Charts', 'Pie Charts', 'Line Charts', 'Bar Charts', '3D Bar Charts', 'Radar Charts']}
    qa_type_dict = {item : [] for item in ["Recognizing", "Reasoning"]}
    ########################
    comparator_path = task_config.get("answer_comparator_path", None)
    comparator = LLMAnswerComparator(threshold=0.8, method="bert", model_path=comparator_path)
    
    for idx, meta in meta_dict.items():
        if idx in results_dict:
            meta["prediction"] = results_dict[idx]["results"]["final_answer"]
        else:
            meta["prediction"] = "None"
            
        meta["prediction"] = "None" if not meta["prediction"] else meta["prediction"]
        item_chart_type = meta["chart_type"]
        item_qa_type = meta["qa_type"]
        prediction = meta["prediction"]
        ground_truth = meta["answer"]
        ########################
        score = rule_based_verify(ground_truth, prediction, comparator)
        # score = rule_based_verify(ground_truth, prediction)
        chart_type_dict[item_chart_type].append(score)
        qa_type_dict[item_qa_type].append(score)
        compare_logs.append(
            f"QID: {idx}, Chart Type: {item_chart_type}, QA Type: {item_qa_type}, "
            f"Ground Truth: {ground_truth}, Prediction: {prediction}, Score: {score}"
        )
    for k,v in chart_type_dict.items():
        if len(v) > 0:
            chart_type_dict[k] = sum(v) / len(v)
        else:
            chart_type_dict[k] = 0.0
    for k,v in qa_type_dict.items():
        if len(v) > 0:
            qa_type_dict[k] = sum(v) / len(v)
        else:
            qa_type_dict[k] = 0.0
    res_dict = dict(
        chart_type_res = chart_type_dict,
        qa_type_res = qa_type_dict,
        compare_logs = compare_logs,
        meta_data = meta_dict,
    )
    return res_dict


# 使用sentence_transformer
def is_convertible_to_float(s: str) -> bool:
    """辅助函数，检查一个字符串是否可以被转换为浮点数。"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False   
    

class LLMAnswerComparator:
    def __init__(self, threshold=0.8, method="ensemble", model_path="paraphrase-MiniLM-L6-v2"):
        self.threshold = threshold
        # self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char_wb',  # 使用字符级分析
            ngram_range=(1, 2),  # 使用1-2个字符的n-gram
            min_df=1,  # 不过滤低频词
            token_pattern=r'(?u)\b\w+\b|[(),]'  # 包括括号和逗号
        )
        self.method = method
        self.bert_model = SentenceTransformer(model_path)

    def tfidf_similarity(self, text1, text2):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    def bert_similarity(self, text1, text2):
        embeddings = self.bert_model.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def compare(self, response, oracle_response, method=None):
        """
        Compare an LLM response with an oracle (reference) response using the specified method.

        Args:
        response (str): The LLM-generated response to evaluate.
        oracle_response (str): The reference (correct) response.
        method (str): The comparison method to use ('tfidf', 'bert', or 'ensemble').

        Returns:
        tuple: (similarity score, boolean indicating if the response is considered correct)
        """
        if method is None:
            method = self.method

        if method == "tfidf":
            similarity = self.tfidf_similarity(response, oracle_response)
        elif method == "bert":
            similarity = self.bert_similarity(response, oracle_response)
        elif method == "ensemble":
            tfidf_sim = self.tfidf_similarity(response, oracle_response)
            bert_sim = self.bert_similarity(response, oracle_response)
            similarity = np.mean([tfidf_sim, bert_sim])
        else:
            raise ValueError("Invalid method. Choose 'tfidf', 'bert', or 'ensemble'.")

        is_correct = similarity >= self.threshold
        return similarity, is_correct


def rule_based_verify(
    gold: str,
    pred: str,
    #####################
    comparator: LLMAnswerComparator
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
    
    # 使用LLMAnswerComparator进行比较
   # print("pred: ", pred, "\n", "gold: ", gold)
    similarity, is_correct = comparator.compare(pred, gold)
    
    return 1.0 if is_correct else 0.0
    # return float(similarity)

