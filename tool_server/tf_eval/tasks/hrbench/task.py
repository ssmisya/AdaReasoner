from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os, sys
import base64
import io
import json
import string
import time
import copy as cp
from collections import defaultdict
from datasets import load_dataset
import logging
import warnings
import requests
import pandas as pd
from PIL import Image

# 禁用相关库的日志
logging.getLogger('datasets').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)


class HRBenchEval:
    def __init__(self, api_key, gpt_model="gpt-4o-2024-11-20", max_workers=1):
        self.api_key = api_key
        self.gpt_model = gpt_model
        self.max_workers = max_workers
        self.API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

    def _post_request(self, payload):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def can_infer_option(self, answer, choices):
        if "Failed to obtain answer via API" in answer:
            return False

        reject_to_answer = ["Sorry, I can't help with images of people yet.", "I can't process this file.", "I'm sorry, but without the image provided", "Cannot determine the answer"]
        for err in reject_to_answer:
            if err in answer:
                return "Z"

        def count_choice(splits, choices, prefix="", suffix=""):
            cnt = 0
            for c in choices:
                if prefix + c + suffix in splits:
                    cnt += 1
            return cnt

        answer_mod = cp.copy(answer)
        chars = ".()[],:;!*#{}"
        for c in chars:
            answer_mod = answer_mod.replace(c, " ")

        splits = [x.strip() for x in answer_mod.split()]
        count = count_choice(splits, choices)

        if count == 1:
            for ch in choices:
                if ch in splits:
                    return ch
        elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
            return "Z"
        return False

    def can_infer_text(self, answer, choices):
        answer = answer.lower()
        assert isinstance(choices, dict)
        for k in choices:
            assert k in string.ascii_uppercase
            choices[k] = str(choices[k]).lower()
        cands = []
        for k in choices:
            if choices[k] in answer:
                cands.append(k)
        if len(cands) == 1:
            return cands[0]
        return False

    def can_infer(self, answer, choices):
        answer = str(answer)
        copt = self.can_infer_option(answer, choices)
        return copt if copt else self.can_infer_text(answer, choices)

    def build_prompt(self, question, options, prediction):
        options_prompt = ""
        for key, item in options.items():
            options_prompt += f"{key}. {item}\n"
        tmpl = (
            "You are an AI assistant who will help me to match "
            "an answer with several options of a single-choice question. "
            "You are provided with a question, several options, and an answer, "
            "and you need to find which option is most similar to the answer. "
            "If the meaning of all options are significantly different from the answer, output Z. "
            "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n"
            "Example 1: \n"
            "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
            "Answer: a cute teddy bear\nYour output: A\n"
            "Example 2: \n"
            "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
            "Answer: Spider\nYour output: Z\n"
            "Example 3: \n"
            "Question: {}\nOptions: {}\nAnswer: {}\nYour output: "
        )
        return tmpl.format(question, options_prompt, prediction)

    def get_chat_response(self, data, temperature=0, max_tokens=256, patience=10, sleep_time=0):
        question = data["question"]
        options = data["options"]
        prediction = data["prediction"]
        ret = self.can_infer(prediction, options)
        if ret:
            data["gpt_prediction"] = ret
            return data

        prompt = self.build_prompt(question, options, prediction)
        messages = [
            {"role": "user", "content": prompt},
        ]
        payload = {"model": self.gpt_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "n": 1}

        while patience > 0:
            patience -= 1
            try:
                response = self._post_request(payload)
                prediction = response["choices"][0]["message"]["content"].strip()
                if prediction and prediction != "" and "Failed to obtain answer via API" not in prediction:
                    ret = self.can_infer(prediction, options)
                    data["gpt_prediction"] = ret
                    return data

            except Exception as e:
                logger.error(f"GPT API error: {e}")
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        data["gpt_prediction"] = "Z"  # 默认返回Z
        return data


def decode_base64_to_image(base64_string, target_size=-1):
    """解码base64图像"""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        if target_size > 0:
            image.thumbnail((target_size, target_size))
        return image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None


def extract_options(doc):
    """提取选项"""
    options = {}
    for cand in string.ascii_uppercase:
        if cand in doc and not pd.isna(doc[cand]) and doc[cand]:
            options[cand] = doc[cand]
    return options


def load_data_function():
    dataset_path = task_config["dataset_path"]
    dataset_name = task_config.get("dataset_name", "hrbench_version_split")
    test_split = task_config.get("test_split", "hrbench_4k")
    num_sample = task_config.get("num_sample", None)
    
    # 加载HRBENCH数据集
    dataset = load_dataset(dataset_path, dataset_name, split=test_split, token=True)
    
    # 只处理前num_sample个样本
    if num_sample:
        dataset = dataset.select(range(min(num_sample, len(dataset))))
    
    meta_data = []
    
    for idx, item in enumerate(dataset):
        item_id = f"hrbench_{idx}"
        
        # 解码base64图像
        image = decode_base64_to_image(item["image"])
        if image is None:
            continue
            
        question = item["question"].strip()
        options = extract_options(item)
        
        # 构建问题文本
        options_prompt = ""
        for key, value in options.items():
            options_prompt += f"{key}. {value}\n"
        text = f"{question}\n{options_prompt}Answer the option letter directly."
        
        answer = item["answer"]
        category = item.get("category", "unknown")
        cycle_category = item.get("cycle_category", "unknown")
        index = item.get("index", idx)
        
        meta_data.append({
            "idx": item_id,
            "image": image,
            "text": text,
            "answer": answer,
            "question": question,
            "options": options,
            "category": category,
            "cycle_category": cycle_category,
            "index": index
        })
    
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


def evaluate_function(results, meta_data):
    # 初始化GPT评估器
    api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    gpt_model = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")
    evaluator = HRBenchEval(api_key=api_key, gpt_model=gpt_model, max_workers=1)
    
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    
    compare_logs = []
    category_results = defaultdict(list)
    cycle_category_results = defaultdict(list)
    
    for idx, meta in meta_dict.items():
        if idx in results_dict:
            prediction = results_dict[idx]["results"]["final_answer"]
        else:
            prediction = "None"
        
        if prediction is None:
            prediction = "None"
        
        pred = prediction.strip()
        gt = meta["answer"]
        question = meta["question"]
        options = meta["options"]
        category = meta["category"]
        cycle_category = meta["cycle_category"]
        
        # 使用GPT评估
        resp_dic = evaluator.get_chat_response({
            "question": question,
            "options": options,
            "prediction": pred
        })
        gpt_prediction = resp_dic["gpt_prediction"]
        
        # 计算分数
        score = 1.0 if gt.lower() == gpt_prediction.lower() else 0.0
        
        category_results[category].append(score)
        cycle_category_results[cycle_category].append(score)
        
        compare_logs.append({
            "idx": idx,
            "category": category,
            "cycle_category": cycle_category,
            "question": question,
            "options": options,
            "gold": gt,
            "pred": pred,
            "gpt_prediction": gpt_prediction,
            "score": score
        })
    
    # 计算各类别平均分
    category_avg_scores = {}
    for category, scores in category_results.items():
        avg_score = sum(scores) / len(scores) if scores else 0
        category_avg_scores[category] = avg_score
        logger.info(f"Category {category}: {avg_score:.4f}")
    
    cycle_category_avg_scores = {}
    for cycle_category, scores in cycle_category_results.items():
        avg_score = sum(scores) / len(scores) if scores else 0
        cycle_category_avg_scores[cycle_category] = avg_score
        logger.info(f"Cycle Category {cycle_category}: {avg_score:.4f}")
    
    # 计算总体平均分
    overall_avg = sum(category_avg_scores.values()) / len(category_avg_scores) if category_avg_scores else 0
    cycle_overall_avg = sum(cycle_category_avg_scores.values()) / len(cycle_category_avg_scores) if cycle_category_avg_scores else 0
    
    logger.info(f"Overall Average (by category): {overall_avg:.4f}")
    logger.info(f"Overall Average (by cycle_category): {cycle_overall_avg:.4f}")
    
    return {
        "overall_average": overall_avg,
        "cycle_overall_average": cycle_overall_avg,
        "category_scores": category_avg_scores,
        "cycle_category_scores": cycle_category_avg_scores,
        "compare_logs": compare_logs,
        "results": results,
        "meta_data": meta_data
    }