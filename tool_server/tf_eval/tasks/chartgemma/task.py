
from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os
from datasets import Dataset
from thefuzz import fuzz
try:
    from math_verify import parse, verify
except ImportError:
    print("math_verify package not found. Please install it to use math verification features.")


logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

# def load_dataset(file_path, already_processed_path, num_samples=None):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         dataset = json.load(f)
#     process_data = set()
#     with open(already_processed_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line)
#             process_data.add(data['results']['results']['meta_data']['text'])
#     selected_dataset = []
#     for data in dataset:
#         if data['question'] not in process_data:
#             selected_dataset.append(data)

#     if num_samples is None:
#         return Dataset.from_dict({"data": selected_dataset})
#     return Dataset.from_dict({"data": selected_dataset[:num_samples]})

def load_dataset(file_path, num_samples=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if num_samples is None:
        return Dataset.from_dict({"data": dataset})
    return Dataset.from_dict({"data": dataset[:num_samples]})

def load_data_function():
    
    # raw_data = load_dir_of_jsonl_data_function_default(task_config)
    dataset_path = task_config["dataset_path"]
    image_dir_path = task_config["image_dir_path"]
    num_samples = task_config['num_sample']

    # dataset = load_dataset(dataset_path, already_processed_path, num_samples)
    dataset = load_dataset(dataset_path, num_samples)

    meta_data = []
    for idx,item in enumerate(dataset):
        item = item["data"]
        item_id = f"chartgemma_{idx}"
        image_file = item.get("image_path")

        image_path = image_file
        text = item["question"]
        label = item.pop("label").replace("<answer> ", "").replace(" </answer>", "").strip()
        data_item = dict(idx=item_id, text=text, label=label, **item)
        data_item["image_path"] = image_path
        meta_data.append(data_item)

    ## Show statistics
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


def evaluate_function(results,meta_data):
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    res_list = []
    compare_logs = []
    # breakpoint()
    for idx, meta in meta_dict.items():
        if idx in results_dict:
            meta["prediction"] = results_dict[idx]["results"]["final_answer"]
        else:
            meta["prediction"] = "None"
        
        gold = meta["label"].split("<answer>")[-1].split("</answer>")[0].strip().lower()
        pred = meta["prediction"].lower()
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
            score = fuzz.ratio(pred, gold) / 100
            
        res_list.append(score)
        compare_logs.append({"idx":idx,"gold":gold,"pred":pred,"score":score})

    accuracy = sum(res_list) / len(res_list) if len(res_list) > 0 else 0
    
    return {"Acc":accuracy, "compare_logs":compare_logs, "results":results,"meta_data":meta_data}
