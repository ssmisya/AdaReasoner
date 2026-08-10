#!/usr/bin/env python3
import argparse
import gc
import importlib
import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

from tool_server.tf_eval.evaluator import TFEvaluator
import tool_server.tf_eval.evaluator as evaluator_module
from tool_server.tf_eval.tasks import get_task_functions as original_get_task_functions
from tool_server.tf_eval.utils.arguments import (
    ModelArguments,
    ScriptArguments,
    TaskArguments,
    parse_str_into_dict,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-matrix", required=True)
    parser.add_argument("--timing-output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_mapping(target, values):
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            update_mapping(target[key], value)
        else:
            target[key] = value


def make_guichat_loader(arrow_path, num_sample):
    def load_data_function():
        from datasets import Dataset

        dataset = Dataset.from_file(arrow_path)
        if num_sample:
            dataset = dataset.select(range(min(int(num_sample), len(dataset))))
        rows = []
        for idx, item in enumerate(dataset):
            rows.append(
                {
                    "idx": f"web_guichat_{idx}",
                    "image": item["image"],
                    "text": item["question"],
                    "answer": item["answer"],
                }
            )
        return rows

    return load_data_function


def make_vsp_loader(dataset_path, splits, num_sample, include_level):
    def load_data_function():
        from datasets import load_dataset

        rows = []
        for split_name in splits:
            dataset = load_dataset(dataset_path, split=split_name)
            if num_sample:
                max_per_task = int(num_sample) // len(splits)
                dataset = dataset.select(range(min(len(dataset), max_per_task)))
            for item in dataset:
                gym_map = (
                    json.loads(item["gym_map"])
                    if isinstance(item["gym_map"], str)
                    else item["gym_map"]
                )
                row = {
                    "idx": item["idx"],
                    "original_id": item["original_id"],
                    "image": item["image"],
                    "text": item["text"],
                    "answer": item["answer"],
                    "task_type": item["task_type"],
                    "split": item["split"],
                    "size": item["size"],
                    "gym_map": gym_map,
                    "map_text_list": gym_map,
                }
                if include_level:
                    row["level"] = item.get("level", "unknown")
                if item["task_type"] == "verify":
                    row["path_length"] = item.get("path_length")
                    row["path"] = item.get("path")
                else:
                    for key in ("start_coords", "goal_coords", "obstacle_coords"):
                        value = item.get(key)
                        row[key] = json.loads(value) if isinstance(value, str) else value
                    row["astar_path"] = item.get("astar_path")
                rows.append(row)
        return rows

    return load_data_function


def make_vstar_loader(dataset_path, num_sample):
    def load_data_function():
        from PIL import Image

        root = Path(dataset_path).resolve()
        questions_path = root / "test_questions.jsonl"
        if not questions_path.is_file():
            raise FileNotFoundError(f"V* question file not found: {questions_path}")

        rows = []
        with questions_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                item = json.loads(line)
                relative_path = Path(item["image"])
                if relative_path.is_absolute():
                    image_path = relative_path.resolve()
                else:
                    image_path = (root / relative_path).resolve()
                if image_path != root and root not in image_path.parents:
                    raise RuntimeError(
                        f"V* image path escapes dataset root: {item['image']}"
                    )
                if not image_path.is_file():
                    raise FileNotFoundError(f"V* image not found: {image_path}")
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                    image.load()
                rows.append(
                    {
                        "idx": f"vstar_{idx}",
                        "image": image,
                        "text": item["text"],
                        "answer": item["label"],
                        "category": item["category"],
                    }
                )
                if num_sample and len(rows) >= int(num_sample):
                    break
        return rows

    return load_data_function


def make_jigsaw_coco_loader(arrow_path, num_sample):
    def load_data_function():
        from datasets import Dataset

        dataset = Dataset.from_file(arrow_path)
        if num_sample:
            dataset = dataset.select(range(min(int(num_sample), len(dataset))))
        return [
            {
                "idx": item["idx"],
                "images": [item["question_image"]] + item["choice_images"],
                "text": item["question_text"],
                "answer": item["correct_answer"],
                "category": item.get("category", "jigsaw_coco"),
            }
            for item in dataset
        ]

    return load_data_function


def make_hrbench_loader(arrow_path, num_sample):
    def load_data_function():
        import pandas as pd
        import string
        from datasets import Dataset
        from tool_server.tf_eval.tasks.hrbench.task import decode_base64_to_image

        dataset = Dataset.from_file(arrow_path)
        if num_sample:
            dataset = dataset.select(range(min(int(num_sample), len(dataset))))
        rows = []
        for idx, item in enumerate(dataset):
            image = decode_base64_to_image(item["image"])
            if image is None:
                raise RuntimeError(f"Failed to decode HRBench image at row {idx}")
            options = {
                key: item[key]
                for key in string.ascii_uppercase
                if key in item and not pd.isna(item[key]) and item[key]
            }
            options_prompt = "".join(f"{key}. {value}\n" for key, value in options.items())
            question = item["question"].strip()
            rows.append(
                {
                    "idx": f"hrbench_{idx}",
                    "image": image,
                    "text": f"{question}\n{options_prompt}Answer the option letter directly.",
                    "answer": item["answer"],
                    "question": question,
                    "options": options,
                    "category": item.get("category", "unknown"),
                    "cycle_category": item.get("cycle_category", "unknown"),
                    "index": item.get("index", idx),
                }
            )
        return rows

    return load_data_function


def make_webmmu_loader(dataset_path, category, num_sample):
    def load_data_function():
        from datasets import load_dataset

        dataset = load_dataset(dataset_path, name="web_qa", split="english")
        indices = [
            idx for idx, item in enumerate(dataset) if item["question_type"] == category
        ]
        dataset = dataset.select(indices)
        if num_sample:
            dataset = dataset.select(range(min(int(num_sample), len(dataset))))
        prompt = (
            "Analyze the website screenshot and provide a detailed answer to the "
            "question. If the question involves locating or interacting with specific "
            "elements on the screen, include the bounding box coordinates "
            "[x_min, y_min, x_max, y_max] in your response."
        )
        return [
            {
                "idx": f"webmmu_{idx}",
                "image": item["image"],
                "text": prompt + "\n" + item["question"],
                "answer": item["ground_truth"],
                "category": item["question_type"],
            }
            for idx, item in enumerate(dataset)
        ]

    return load_data_function


def install_generation_timer(timing):
    from tool_server.tf_eval.models.vllm_models import VllmModels

    original_generate = VllmModels.generate

    def timed_generate(self, batch):
        started = time.perf_counter()
        try:
            return original_generate(self, batch)
        finally:
            timing["generation_s"] += time.perf_counter() - started
            timing["generation_calls"] += 1

    VllmModels.generate = timed_generate


def install_result_metadata(task_matrix, timing):
    original_append_jsonl = evaluator_module.append_jsonl
    original_dataset = evaluator_module.BaseEvalDataset
    current = {"task": None, "samples": None}

    class ValidatedDataset(original_dataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            task_name = current["task"]
            total_count = len(self.full_data)
            remaining_count = len(self)
            expected = task_matrix[task_name].get("expected_samples")
            strict = task_matrix[task_name].get("strict_expected_samples", True)
            if total_count <= 0:
                raise RuntimeError(f"Task {task_name} loaded zero samples")
            if strict and expected is not None and total_count != int(expected):
                raise RuntimeError(
                    f"Task {task_name} loaded {total_count} samples; expected {expected}"
                )
            current["samples"] = total_count
            timing.setdefault("input_samples", {})[task_name] = total_count
            timing.setdefault("remaining_samples_at_start", {})[task_name] = remaining_count

    def append_jsonl_with_metadata(payload, path):
        task_name = current["task"]
        payload["task_name"] = task_name
        payload["input_samples"] = current["samples"]
        payload["primary_metric"] = task_matrix[task_name]["primary_metric"]
        payload["metric_label"] = task_matrix[task_name].get(
            "metric_label", task_matrix[task_name]["primary_metric"]
        )
        original_append_jsonl(payload, path)

    def evaluate_with_metadata(self):
        original_tasks = list(self.tasks)
        try:
            for task_name in original_tasks:
                current["task"] = task_name
                current["samples"] = None
                self.tasks = [task_name]
                TFEvaluator._original_evaluate_for_metadata(self)
        finally:
            self.tasks = original_tasks

    if not hasattr(TFEvaluator, "_original_evaluate_for_metadata"):
        TFEvaluator._original_evaluate_for_metadata = TFEvaluator.evaluate
    evaluator_module.BaseEvalDataset = ValidatedDataset
    evaluator_module.append_jsonl = append_jsonl_with_metadata
    TFEvaluator.evaluate = evaluate_with_metadata


def install_task_overrides(task_matrix):
    def get_task_functions(task_name):
        original_get_task_functions(task_name)
        spec = task_matrix.get(task_name, {})
        module = importlib.import_module(f"tool_server.tf_eval.tasks.{task_name}.task")
        override = spec.get("task_config", {})
        update_mapping(module.task_config, override)

        load_data_function = module.load_data_function
        local_arrow = spec.get("local_arrow")
        if task_name == "vsp":
            load_data_function = make_vsp_loader(
                module.task_config["dataset_path"],
                list(module.task_config["tasks"]),
                module.task_config.get("num_sample"),
                include_level=True,
            )
        if task_name == "vspo":
            splits = [name.replace("-", "_") for name in module.task_config["tasks"]]
            load_data_function = make_vsp_loader(
                module.task_config["dataset_repo"],
                splits,
                module.task_config.get("num_sample"),
                include_level=False,
            )
        if local_arrow and not os.path.isfile(local_arrow):
            raise FileNotFoundError(f"Local Arrow file not found: {local_arrow}")
        if task_name == "web_guichat" and local_arrow:
            load_data_function = make_guichat_loader(
                local_arrow, module.task_config.get("num_sample")
            )
        if task_name == "jigsaw_coco" and local_arrow:
            load_data_function = make_jigsaw_coco_loader(
                local_arrow, module.task_config.get("num_sample")
            )
        if task_name == "hrbench" and local_arrow:
            load_data_function = make_hrbench_loader(
                local_arrow, module.task_config.get("num_sample")
            )
        if task_name == "vstar":
            load_data_function = make_vstar_loader(
                module.task_config["dataset_path"],
                module.task_config.get("num_sample"),
            )
        if task_name == "webmmu" and spec.get("filter_category"):
            load_data_function = make_webmmu_loader(
                module.task_config["dataset_path"],
                spec["filter_category"],
                module.task_config.get("num_sample"),
            )

        return {
            "load_data_function": load_data_function,
            "evaluate_function": module.evaluate_function,
            "task_config": module.task_config,
        }

    evaluator_module.get_task_functions = get_task_functions


def preflight_task_data(task_name, task_functions, task_matrix, timing):
    started = time.perf_counter()
    data = task_functions["load_data_function"]()
    try:
        if not isinstance(data, list):
            raise RuntimeError(
                f"Task {task_name} loader returned {type(data).__name__}, expected list"
            )
        count = len(data)
        spec = task_matrix[task_name]
        expected = spec.get("expected_samples")
        strict = spec.get("strict_expected_samples", True)
        if count <= 0:
            raise RuntimeError(f"Task {task_name} loaded zero samples during preflight")
        if strict and expected is not None and count != int(expected):
            raise RuntimeError(
                f"Task {task_name} preflight loaded {count} samples; expected {expected}"
            )
        identifiers = [item.get("idx") for item in data]
        if None in identifiers or len(set(identifiers)) != count:
            raise RuntimeError(f"Task {task_name} has missing or duplicate sample IDs")
        required = {"idx", "text", "answer"}
        for position, item in enumerate(data):
            missing = sorted(required - set(item))
            if missing:
                raise RuntimeError(
                    f"Task {task_name} row {position} is missing fields: {missing}"
                )
            images = item.get("images")
            if images is None:
                images = [item.get("image")]
            if not isinstance(images, list) or not images or any(
                image is None for image in images
            ):
                raise RuntimeError(
                    f"Task {task_name} row {position} has no usable image input"
                )
        timing.setdefault("preflight_samples", {})[task_name] = count
        timing.setdefault("preflight_s", {})[task_name] = time.perf_counter() - started
    finally:
        del data
        gc.collect()


def build_arguments(config):
    model_args = ModelArguments(**config["model_args"])
    task_args = TaskArguments(**config["task_args"])
    script_args = ScriptArguments(**config["script_args"])

    task_args.task_name = task_args.task_name.split(",")
    if isinstance(model_args.model_args, str):
        model_args.model_args = parse_str_into_dict(model_args.model_args)
    if isinstance(script_args.wandb_args, str):
        script_args.wandb_args = parse_str_into_dict(script_args.wandb_args)
    return model_args, task_args, script_args


def write_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(output)


def main():
    args = parse_args()
    process_start = time.perf_counter()
    timing = {
        "seed": args.seed,
        "status": "running",
        "model_load_s": None,
        "evaluation_s": None,
        "process_total_s": None,
        "generation_s": 0.0,
        "generation_calls": 0,
    }
    write_json(args.timing_output, timing)

    try:
        set_seed(args.seed)
        with open(args.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        with open(args.task_matrix, "r", encoding="utf-8") as handle:
            task_matrix = json.load(handle)

        install_generation_timer(timing)
        install_task_overrides(task_matrix)
        install_result_metadata(task_matrix, timing)
        model_args, task_args, script_args = build_arguments(config)

        # Load and validate every selected dataset before allocating GPU memory.
        for task_name in task_args.task_name:
            task_functions = evaluator_module.get_task_functions(task_name)
            preflight_task_data(task_name, task_functions, task_matrix, timing)
        write_json(args.timing_output, timing)

        model_start = time.perf_counter()
        evaluator = TFEvaluator(model_args, task_args, script_args)
        timing["model_load_s"] = time.perf_counter() - model_start
        write_json(args.timing_output, timing)

        evaluation_start = time.perf_counter()
        evaluator.evaluate()
        timing["evaluation_s"] = time.perf_counter() - evaluation_start
        timing["status"] = "success"
        write_json(args.timing_output, timing)
    except Exception as exc:
        timing["status"] = "failed"
        timing["error"] = f"{type(exc).__name__}: {exc}"
        timing["traceback"] = traceback.format_exc()
        raise
    finally:
        timing["process_total_s"] = time.perf_counter() - process_start
        write_json(args.timing_output, timing)


if __name__ == "__main__":
    main()
