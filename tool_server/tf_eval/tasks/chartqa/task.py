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
import re
import string
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
    num_sample = task_config["num_sample"]
    
    testset = load_dataset(dataset_path,split="test")
    # 只处理testset的前num_sample个样本
    testset = testset.select(range(min(num_sample, len(testset))))
    meta_data = []

    for idx, item in enumerate(testset):
        item_id = f"chartqa_{idx}"
        image = item["image"]
        text = item["query"]
        answer = item["label"][0] # 因为答案都是放在一个列表中的
        category = item["human_or_machine"]
        # 0为Human，1为Machine
        category = "human" if category == 0 else "machine"
        meta_data.append({"idx":item_id, "image":image, "text":text, "answer":answer, "category":category})

    return meta_data


def evaluate_function(results,meta_data):
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    exact_match_scores = []
    relaxed_accuracy_scores = []
    anywhere_accuracy_scores = []
    compare_logs = []
    
    for idx, meta in meta_dict.items():
        if idx in results_dict:
            meta["prediction"] = results_dict[idx]["results"]["final_answer"]
        else:
            meta["prediction"] = "None"
        
        gold = meta["answer"]
        pred = meta["prediction"]
        
        # 检查pred是否为null或None
        if pred is None:
            em_score = 0.0
            ra_score = 0.0
            aa_score = 0.0
        else:
            # 计算三个指标，确保gold和pred是列表形式
            em_score = exact_match([gold], [pred])["exact_match"]
            ra_score = relaxed_accuracy([gold], [pred])["relaxed_accuracy"]
            aa_score = anywhere_accuracy([gold], [pred])["anywhere_accuracy"]
        
        exact_match_scores.append(em_score)
        relaxed_accuracy_scores.append(ra_score)
        anywhere_accuracy_scores.append(aa_score)
        
        compare_logs.append({
            "idx": idx,
            "gold": gold,
            "pred": pred,
            "exact_match": em_score,
            "relaxed_accuracy": ra_score,
            "anywhere_accuracy": aa_score,
            "question": meta["text"]
        })

    # 计算平均分数
    em_accuracy = sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0
    ra_accuracy = sum(relaxed_accuracy_scores) / len(relaxed_accuracy_scores) if relaxed_accuracy_scores else 0
    aa_accuracy = sum(anywhere_accuracy_scores) / len(anywhere_accuracy_scores) if anywhere_accuracy_scores else 0
    
    return {
        "Exact_Match_Acc": em_accuracy,
        "Relaxed_Acc": ra_accuracy,
        "Anywhere_Acc": aa_accuracy,
        "compare_logs": compare_logs,
        "results": results,
        "meta_data": meta_data
    }

# 从utils.py移植的代码
def _normalize_string(s):
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]
    return s


def _remove_end_punctuation(unnormalized_string: str) -> str:
    while (
        unnormalized_string
        and (
            unnormalized_string[-1] in string.punctuation
            or unnormalized_string[-1].isspace()
        )
        and unnormalized_string[-1] != "%"
    ):
        unnormalized_string = unnormalized_string[:-1]
    return unnormalized_string


class RelaxedCorrectness:
    """Relaxed correctness metrics.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    "Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct."
    """

    def _relaxed_correctness(
        self, prediction: str, targets: list[str], max_relative_change: float = 0.05
    ) -> float:
        def _to_float(text: str) -> tuple[float | None, bool]:
            text = text.strip()
            is_percent = text.endswith("%")
            try:
                value = float(text.rstrip("%"))
                return value, is_percent
            except ValueError:
                return None, False

        def _is_letter(text: str) -> bool:
            return text.isalpha() and len(text) == 1

        def _preprocess_text(text: str) -> str:
            if not any(char.isdigit() for char in text):
                return _normalize_string(text)
            else:
                return _remove_end_punctuation(text).replace(",", "").replace("$", "")

        def calculate_relative_change(prediction: float, target: float) -> float:
            return abs(prediction - target) / max(abs(target), 1e-10)

        def _compare_numeric_values(
            prediction: float, target: float, max_relative_change: float
        ) -> float:
            relative_change = calculate_relative_change(prediction, target)
            return 1.0 if relative_change <= max_relative_change else 0.0

        def _compare_text_values(prediction: str, target: str) -> float:
            while prediction and prediction[-1] in string.punctuation:
                prediction = prediction[:-1]
            return 1.0 if prediction.lower() == target.lower() else 0.0

        def _to_decimal(value: float, is_percent: bool) -> float:
            return value / 100 if is_percent else value

        def _compare_numeric_with_percent(
            prediction: float,
            prediction_is_percent: bool,
            target: float,
            target_is_percent: bool,
            max_relative_change: float,
        ) -> float:
            # Compare as-is
            value = _compare_numeric_values(prediction, target, max_relative_change)

            # If not equal and one is percent, try other comparisons
            if value != 1.0 and (prediction_is_percent or target_is_percent):
                value = max(
                    value,
                    _compare_numeric_values(
                        _to_decimal(prediction, prediction_is_percent),
                        target,
                        max_relative_change,
                    ),
                    _compare_numeric_values(
                        prediction,
                        _to_decimal(target, target_is_percent),
                        max_relative_change,
                    ),
                )
            return value

        prediction = _preprocess_text(prediction)
        prediction_float, prediction_is_percent = _to_float(prediction)

        value_list = []
        for target in targets:
            target = _preprocess_text(target)
            target_float, target_is_percent = _to_float(target)

            if prediction_float is not None and target_float is not None:
                # Compare as numeric values
                value = _compare_numeric_with_percent(
                    prediction_float,
                    prediction_is_percent,
                    target_float,
                    target_is_percent,
                    max_relative_change,
                )
            elif _is_letter(target) and len(prediction) > 0:
                # Compare as multiple choice options: take first letter from prediction
                value = 1.0 if prediction[0].lower() == target.lower() else 0.0
            else:
                # Compare as text values
                value = _compare_text_values(prediction, target)

            value_list.append(value)

        return max(value_list)

    def score(self, model_answer: str, reference_answer: str | list[str]) -> float:
        reference_answer = (
            reference_answer
            if isinstance(reference_answer, list)
            else [reference_answer]
        )
        return self._relaxed_correctness(model_answer, reference_answer)


class ExplicitPromptRelaxedCorrectness(RelaxedCorrectness):
    """Relaxed correctness for explicit prompt."""

    @property
    def name(self) -> str:
        return "explicit_prompt_relaxed_correctness"

    def _get_final_answer(self, generation: str) -> str:
        def _find_last_occurrence(pattern: str, string: str):
            return string.rfind(pattern)

        # Strip extraneous markdown around the answer:
        generation = re.sub(r"([aA]nswer)\**:\**", "\\1:", generation)

        final_answer_index = _find_last_occurrence("answer:", generation.lower())

        if final_answer_index != -1:
            # Find the start of the answer (after "final answer:")
            start_index = final_answer_index + len("answer:")

            # Split the remaining text into lines
            lines = generation[start_index:].split("\n")

            # Find the first non-empty line
            final_answer = next((line.strip() for line in lines if line.strip()), "")

            # Remove any markdown formatting
            final_answer = re.sub(r"[*_\[\]\(\)]", "", final_answer)

            return final_answer
        else:
            return ""

    def score(self, model_answer: str, reference_answer: str | list[str]) -> float:
        parsed_model_answer = self._get_final_answer(model_answer)
        if not parsed_model_answer:
            # Parsing failed.
            return 0.0
        return super().score(parsed_model_answer, reference_answer)


class AnywhereInAnswerRelaxedCorrectness(ExplicitPromptRelaxedCorrectness):
    """Falls back to handle cases where reference answer appears anywhere in generation.

    NOTE: This is an overly generous metric and is likely to falsely inflate scores.
    """

    @property
    def name(self) -> str:
        return "anywhere_in_answer_relaxed_correctness"

    def score(self, model_answer: str, reference_answer: str | list[str]) -> float:
        reference_answer = (
            reference_answer
            if isinstance(reference_answer, list)
            else [reference_answer]
        )
        parsed_model_answer = self._get_final_answer(model_answer)
        if parsed_model_answer:
            return self._relaxed_correctness(parsed_model_answer, reference_answer)

        # Fallback: check if reference answer appears anywhere in the model answer.
        for ref in reference_answer:
            try:
                # Try to parse as a float
                number = float(ref)

                # Revert to int if it is actually an int.
                if int(number) == number:
                    number = int(number)
                # Check if the number is in the model answer with commas (e.g. 1,000)
                if format(number, ",") in model_answer:
                    return 1.0
                # Check if the number is in the model answer without commas (e.g. 1000)
                elif str(number) in model_answer:
                    return 1.0
                elif str(number) + "%" in model_answer:
                    return 1.0
            except ValueError:
                # Reference answer was a text string. We search for typical patterns
                # in the model answer. Note that directly searching for the reference
                # is not a good idea for letter-option choice questions, hence we look
                # for common patterns. This is still heuristic, and might have false
                # positives as well as false negatives.
                candidates = []
                for ref in reference_answer:
                    candidates.extend(
                        [
                            f"is {ref}",
                            f"was {ref}",
                            f" {ref}.",
                            f"are {ref}",
                            f"\n\n{ref}",
                        ]
                    )
                if any([c.lower() in model_answer.lower() for c in candidates]):
                    return 1.0

        return 0


def exact_match(references, predictions):
    pred = predictions[0]
    ref = references[0]

    match = re.search(r"(?:Final Answer|FINAL ANSWER): (.+)$", pred, re.IGNORECASE)
    if match:
        extracted_pred = match.group(1).strip()
        if extracted_pred.lower().removesuffix(".") == ref.strip().lower():
            return {"exact_match": 1.0}
        else:
            return {"exact_match": 0.0}
    else:
        return {"exact_match": 0.0}


def relaxed_accuracy(references, predictions):
    pred = predictions[0]
    ref = references[0]
    score = ExplicitPromptRelaxedCorrectness().score(pred, ref)
    if score:
        if score == 1.0:
            return {"relaxed_accuracy": 1.0}
    return {"relaxed_accuracy": 0.0}


def anywhere_accuracy(references, predictions):
    pred = predictions[0]
    ref = references[0]
    score = AnywhereInAnswerRelaxedCorrectness().score(pred, ref)
    if score:
        if score == 1.0:
            return {"anywhere_accuracy": 1.0}
    return {"anywhere_accuracy": 0.0}