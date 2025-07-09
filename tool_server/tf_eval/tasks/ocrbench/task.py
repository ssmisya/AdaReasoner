
from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os, sys
from datasets import Dataset, load_dataset
from thefuzz import fuzz
import math
try:
    from math_verify import parse, verify
except ImportError:
    print("math_verify package not found. Please install it to use math verification features.")


'''
10 question_type
    "Regular Text Recognition"
    "Irregular Text Recognition"
    "Artistic Text Recognition"
    "Handwriting Recognition"
    "Digit String Recognition"
    "Non-Semantic Text Recognition"
    "Scene Text-centric VQA"
    "Doc-oriented VQA"
    "Key Information Extraction"
    "Handwritten Mathematical Expression Recognition"

dataset type:
    "HME100k"
    others not special

'''

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

def pad_small_image(image, min_size=28):
    """
    检查图像尺寸，如果宽度或高度小于min_size，则用白色像素填充到min_size
    
    参数:
        image: PIL.Image对象
        min_size: 最小尺寸要求，默认为28像素
        
    返回:
        处理后的PIL.Image对象
    """
    width, height = image.size
    
    if width < min_size or height < min_size:
        # 计算新尺寸
        new_width = max(width, min_size)
        new_height = max(height, min_size)
        
        # 创建新的白色背景图像
        new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))
        
        # 将原图粘贴到新图像上，居中放置
        paste_x = (new_width - width) // 2
        paste_y = (new_height - height) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        return new_image
    
    return image

def load_data_function():

    dataset_path = task_config["dataset_path"]
    num_sample = task_config.get("num_sample", None)
    
    testset = load_dataset(dataset_path,split="test")
    # 只处理testset的前num_sample个样本
    if num_sample:
        testset = testset.select(range(min(num_sample, len(testset))))
    meta_data = []

    for idx, item in enumerate(testset):
        item_id = f"ocrbench_{idx}"
        image = item["image"]
        width, height = image.size
        if width < 28 or height < 28:
            image = pad_small_image(image)
        text = item["question"]
        answer = item["answer"][0]
        dataset_type = item["dataset"]
        question_type = item["question_type"]
        meta_data.append({"idx":item_id, "image":image, "text":text, "answer":answer, "dataset_type":dataset_type, "question_type":question_type})

    return meta_data


def evaluate_function(results,meta_data):
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    res_list = []
    compare_logs = []
    question_type_dict = {item : [] for item in ["Regular Text Recognition", "Irregular Text Recognition", "Artistic Text Recognition", "Handwriting Recognition", "Digit String Recognition", "Non-Semantic Text Recognition", "Scene Text-centric VQA", "Doc-oriented VQA", "Key Information Extraction", "Handwritten Mathematical Expression Recognition"]}
    for idx, meta in meta_dict.items():
        if idx in results_dict:
            meta["prediction"] = results_dict[idx]["results"]["final_answer"]
        else:
            meta["prediction"] = "None"

        meta["prediction"] = "None" if not meta["prediction"] else meta["prediction"]
        pred = meta["prediction"]
        gold = meta["answer"]
        question_type = meta["question_type"]
        score = 0.0
        
        if pred == "" or pred is None or pred == "None":
            score = 0.0
        else:
            if meta["dataset_type"] == "HME100k":
                if type(gold) == list:
                    for j in range(len(gold)):
                        gold = gold[j].strip().replace("\n", " ").replace(" ", "")
                        pred = pred[j].strip().replace("\n", " ").replace(" ", "")
                        if gold in pred:
                            score = 1.0
                else:
                    gold = gold.strip().replace("\n", " ").replace(" ", "")
                    pred = pred.strip().replace("\n", " ").replace(" ", "")
                    if gold in pred:
                        score = 1.0
            else:
                if type(gold) == list:
                    for j in range(len(gold)):
                        gold = gold[j].lower().strip().replace("\n", " ")
                        pred = pred[j].lower().strip().replace("\n", " ")
                        if gold in pred:
                            score = 1.0
                else:
                    gold = gold.lower().strip().replace("\n", " ")
                    pred = pred.lower().strip().replace("\n", " ")
                    if gold in pred:
                        score = 1.0
                
        res_list.append(score)
        question_type_dict[question_type].append(score)
        compare_logs.append({"idx":idx,"gold":gold,"pred":pred,"score":score,"question_type":question_type})

    accuracy = sum(res_list) / len(res_list) if len(res_list) > 0 else 0
    for k,v in question_type_dict.items():
        if len(v) > 0:
            question_type_dict[k] = sum(v) / len(v)
        else:
            question_type_dict[k] = 0.0
    
    return {"Acc":accuracy, "Score":sum(res_list), "question_type_res":question_type_dict, "compare_logs":compare_logs,"meta_data":meta_data, "results":results}
