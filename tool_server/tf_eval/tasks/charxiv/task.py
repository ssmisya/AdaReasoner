
from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os, sys
from datasets import load_dataset
try:
    from math_verify import parse, verify
except ImportError:
    print("math_verify package not found. Please install it to use math verification features.")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from charxiv_constants import DESCRIPTIVE_RESP_INST, DESCRIPTIVE_CLASSIFICATION_MAP, REASONING_CLASSIFICATION

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)


def load_data_function():
    
    # raw_data = load_dir_of_jsonl_data_function_default(task_config)
    dataset_path = task_config["dataset_path"]
    dataset = load_dataset(dataset_path)
    valset = dataset["validation"]
    meta_data = []
    
    for idx,item in enumerate(valset):
        image = item["image"]
        category = item["category"]
        
        # Descriptive evaluation
        for qid_idx in [1,2,3,4]:
            qid = item[f"descriptive_q{qid_idx}"]
            answer = item[f"descriptive_a{qid_idx}"]
            if answer == "Not Applicable":
                # Skip questions that are not applicable
                continue
            
            descriptive_classification = DESCRIPTIVE_CLASSIFICATION_MAP[qid]
            if qid in [18, 19]:
                # skip subplot location when asking about the layout of the subplots
                desc_question = DESCRIPTIVE_RESP_INST[qid] 
            else:
                subplot_row = item[f"subplot_row"]
                subplot_col = item[f"subplot_col"]
                subplot_loc = item[f"subplot_loc"]
                if subplot_row:
                    if subplot_row == 0:
                        # when there is only one subplot
                        prefix = "For the current plot, "
                    else:
                        # when there are multiple subplots
                        prefix = f"For the subplot at row {subplot_row} and column {subplot_col}, "
                elif subplot_loc:
                    prefix = f"For {subplot_loc}, "
                else:
                    raise ValueError(f"Either 'subplot_row' or 'subplot_loc' must be provided for question {qid}.")
                # return the question with the subplot location
                desc_question = DESCRIPTIVE_RESP_INST[qid].format(prefix)
            item_id = f"charxiv_{idx}_descriptive_{qid_idx}"
            res = dict(
                image = image,
                text = desc_question,
                idx = item_id,
                category = category,
                type = "descriptive",
                classification = descriptive_classification,
                answer = answer
            )
            meta_data.append(res)
        
        # Reasoning Questions
        reasoning_question = item["reasoning_q"]
        reasoning_a = item["reasoning_a"]
        reasoning_classification = REASONING_CLASSIFICATION[item["reasoning_a_type"]]
        res = dict(
            image = image,
            text = reasoning_question,
            idx = f"charxiv_{idx}_reasoning",
            category = category,
            type = "reasoning",
            classification = reasoning_classification,
            answer = reasoning_a
        )
        meta_data.append(res)

    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data



def evaluate_function(results, meta_data):
    """
    Evaluate the model predictions based on the provided results and metadata.

    Args:
        results (list): A list of model predictions, each corresponding to a question.
        meta_data (list): A list of metadata dictionaries, each containing:
                          - 'question' (str): The question asked.
                          - 'ground_truth' (str): The correct answer.
                          - 'type' (str): Type of question ('descriptive' or 'reasoning').

    Returns:
        dict: A dictionary containing evaluation metrics like accuracy (ACC).
    """
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    res_list = []
    compare_logs = []
    classification_dict = {item : [] for item in set(DESCRIPTIVE_CLASSIFICATION_MAP.values()) | set(REASONING_CLASSIFICATION.values())}
    type_dict = {item : [] for item in ["descriptive", "reasoning"]}
    
    for idx, meta in meta_dict.items():
        if idx in results_dict:
            meta["prediction"] = results_dict[idx]["results"]["final_answer"]
        else:
            meta["prediction"] = "None"
        classification = meta["classification"]
        item_type = meta["type"]
        prediction = meta["prediction"]
        ground_truth = meta["answer"]
        score = rule_based_verify(ground_truth, prediction)
        meta["score"] = score
        classification_dict[classification].append(score)
        type_dict[item_type].append(score)
        compare_logs.append(
            f"QID: {idx}, Type: {item_type}, Classification: {classification}, "
            f"Ground Truth: {ground_truth}, Prediction: {prediction}, Score: {score}"
        )
    for k,v in classification_dict.items():
        if len(v) > 0:
            classification_dict[k] = sum(v) / len(v)
        else:
            classification_dict[k] = 0.0
    
    for k,v in type_dict.items():
        if len(v) > 0:
            type_dict[k] = sum(v) / len(v)
        else:
            type_dict[k] = 0.0
    
    res_dict = dict(
        type_res = type_dict,
        classification_res = classification_dict,
        compare_logs = compare_logs,
        meta_data = meta_dict,
    )
    return res_dict
    
        
        
        
        

def rule_based_verify(
    gold: str,
    pred: str,
) -> bool:
    """
    A rule-based verification function to check if the prediction matches the gold standard.
    
    Args:
        gold (str): The ground truth answer.
        pred (str): The predicted answer.
        
    Returns:
        bool: True if the prediction is correct, False otherwise.
    """
    
    gold = gold.replace("\"","").strip().lower()
    pred = pred.replace("\"","").strip().lower()
    gold_latex = parse('${0}$'.format(gold))
    pred_latex = parse('${0}$'.format(pred))
    score = 0.0
    if verify(gold_latex, pred_latex) or gold == pred or verify(gold, pred):
        score = 1.0
        
    elif '%' in pred:
        pred = pred.replace('%','')
        if '.' in pred:
            pred = pred.split('.')[0]
        if pred == gold or verify(gold, pred):
            score = 1.0
        else:
            score = 0.0

    elif '.' in pred:
        pred = pred.split('.')[0]
        if pred == gold or verify(gold, pred):
            score = 1.0
        else:
            score = 0.0
    else:
        score = 0.0
    return score