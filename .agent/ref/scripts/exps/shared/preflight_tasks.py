#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPT_DIR))

import tool_server.tf_eval.evaluator as evaluator_module
from eval_entry import install_task_overrides, preflight_task_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-matrix", required=True)
    parser.add_argument("--tasks", required=True)
    args = parser.parse_args()

    with open(args.task_matrix, "r", encoding="utf-8") as handle:
        task_matrix = json.load(handle)
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    unknown = sorted(set(tasks) - set(task_matrix))
    if unknown:
        raise RuntimeError("Unknown tasks: " + ", ".join(unknown))

    timing = {}
    install_task_overrides(task_matrix)
    for task_name in tasks:
        functions = evaluator_module.get_task_functions(task_name)
        preflight_task_data(task_name, functions, task_matrix, timing)
        print(
            f"PREFLIGHT_OK task={task_name} "
            f"samples={timing['preflight_samples'][task_name]} "
            f"seconds={timing['preflight_s'][task_name]:.3f}"
        )


if __name__ == "__main__":
    main()
