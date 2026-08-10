#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


TASK_ORDER = [
    "vsp",
    "vspo",
    "jigsaw_coco",
    "jigsaw_blink",
    "vstar",
    "web_guichat",
    "webmmu",
    "hrbench",
]


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_last_jsonl(path):
    last = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    if last is None:
        raise ValueError(f"No JSON record in {path}")
    return last


def mean_scores(items):
    values = [float(item["score"]) for item in items if "score" in item]
    return statistics.fmean(values) if values else None


def result_records(result):
    records = result.get("compare_logs")
    if records is not None:
        return records
    records = result.get("meta_data")
    if isinstance(records, dict):
        return list(records.values())
    if isinstance(records, list):
        return records
    return []


def extract_metric(task, result):
    if task in {"vsp", "vspo"}:
        return float(result["overall_accuracy"])
    if task in {"jigsaw_coco", "jigsaw_blink"}:
        return float(result["accuracy"])
    if task == "vstar":
        score = mean_scores(result.get("compare_logs", []))
        if score is not None:
            return score
        return statistics.fmean(float(v) for v in result["category_res"].values())
    if task == "web_guichat":
        return float(result["Acc"])
    if task == "webmmu":
        return float(result["category_res"]["Functional"])
    if task == "hrbench":
        return float(result["overall_average"])
    raise KeyError(f"Unsupported task: {task}")


def sample_count(task, result):
    if "total_samples" in result:
        return int(result["total_samples"])
    records = result_records(result)
    if task == "webmmu":
        return sum(1 for item in records if item.get("category") == "Functional")
    if records:
        return len(records)
    if task in {"vsp", "vspo"}:
        return sum(int(v.get("total", 0)) for v in result.get("task_results", {}).values())
    return None


def describe(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "variance": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "variance": statistics.variance(values) if len(values) > 1 else None,
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def load_run(run_dir, task, seed):
    result_path = run_dir / "result.jsonl"
    timing_path = run_dir / "timing.json"
    done_path = run_dir / "DONE.json"
    if not result_path.is_file() or not timing_path.is_file() or not done_path.is_file():
        return None

    result = read_last_jsonl(result_path)
    timing = read_json(timing_path)
    done = read_json(done_path)
    if timing.get("status") != "success" or not done.get("validated"):
        return None
    if result.get("task_name") != task or int(done.get("seed", -1)) != seed:
        return None

    metric = extract_metric(task, result)
    samples = sample_count(task, result)
    run = {
        "task": task,
        "seed": seed,
        "primary_metric": done.get("primary_metric"),
        "metric_label": done.get("metric_label"),
        "stochastic_decoding": done.get("stochastic_decoding"),
        "metric": metric,
        "metric_percent": metric * 100.0,
        "samples": samples,
        "model_load_s": timing.get("model_load_s"),
        "evaluation_s": timing.get("evaluation_s"),
        "process_total_s": timing.get("process_total_s"),
        "generation_s": timing.get("generation_s"),
        "generation_calls": timing.get("generation_calls"),
        "instance_latency_file": done.get("instance_latency_file"),
        "instance_latency_mean_s": done.get("instance_latency_mean_s"),
        "round_latency_mean_s": done.get("round_latency_mean_s"),
        "round_latency_count": done.get("round_latency_count"),
    }
    if samples and timing.get("evaluation_s") is not None:
        run["evaluation_ms_per_sample"] = timing["evaluation_s"] * 1000.0 / samples

    stage_path = run_dir / "stage_latency.json"
    if stage_path.is_file():
        stage = read_json(stage_path)
        for key in (
            "tool_exec_s",
            "orchestration_s",
            "wall_total_s",
            "gen_calls",
            "tool_batches",
        ):
            if key in stage:
                run[key] = stage[key]

    latency_path = run_dir / "latency_summary.json"
    if latency_path.is_file():
        latency = read_json(latency_path)
        run["latency_coverage"] = latency.get("coverage")
        run["instance_e2e_s"] = latency.get("instance_e2e_s")
        run["round_e2e_s"] = latency.get("round_e2e_s")
        run["round_e2e_s_by_round"] = latency.get("round_e2e_s_by_round")
        run["backend_request_e2e_s"] = latency.get("backend_request_e2e_s")
        run["backend_ttft_s"] = latency.get("backend_ttft_s")
        run["backend_queue_s"] = latency.get("backend_queue_s")
        run["tool_call_s"] = latency.get("tool_call_s")
    return run


def summarize(experiment_dir, expected_seeds):
    tasks = []
    all_runs = []
    for task in TASK_ORDER:
        task_dir = experiment_dir / task
        runs = []
        for seed in expected_seeds:
            run = load_run(task_dir / f"seed_{seed}", task, seed)
            if run:
                runs.append(run)
                all_runs.append(run)

        metric_stats = describe([item["metric_percent"] for item in runs])
        latency_stats = describe(
            [float(item["evaluation_s"]) for item in runs if item.get("evaluation_s") is not None]
        )
        per_sample_stats = describe(
            [
                float(item["evaluation_ms_per_sample"])
                for item in runs
                if item.get("evaluation_ms_per_sample") is not None
            ]
        )
        generation_stats = describe(
            [float(item["generation_s"]) for item in runs if item.get("generation_s") is not None]
        )
        tool_stats = describe(
            [float(item["tool_exec_s"]) for item in runs if item.get("tool_exec_s") is not None]
        )
        instance_latency_stats = describe(
            [
                float(item["instance_e2e_s"]["mean"])
                for item in runs
                if item.get("instance_e2e_s", {}).get("mean") is not None
            ]
        )
        round_latency_stats = describe(
            [
                float(item["round_e2e_s"]["mean"])
                for item in runs
                if item.get("round_e2e_s", {}).get("mean") is not None
            ]
        )
        tool_call_latency_stats = describe(
            [
                float(item["tool_call_s"]["mean"])
                for item in runs
                if item.get("tool_call_s", {}).get("mean") is not None
            ]
        )
        tasks.append(
            {
                "task": task,
                "primary_metric": runs[0].get("primary_metric") if runs else None,
                "metric_label": runs[0].get("metric_label") if runs else None,
                "stochastic_decoding": runs[0].get("stochastic_decoding") if runs else None,
                "complete": len(runs) == len(expected_seeds),
                "expected_seeds": expected_seeds,
                "completed_seeds": [item["seed"] for item in runs],
                "metric_percent": metric_stats,
                "evaluation_s": latency_stats,
                "evaluation_ms_per_sample": per_sample_stats,
                "generation_s": generation_stats,
                "tool_exec_s": tool_stats,
                "instance_e2e_mean_s_across_seeds": instance_latency_stats,
                "round_e2e_mean_s_across_seeds": round_latency_stats,
                "tool_call_mean_s_across_seeds": tool_call_latency_stats,
                "runs": runs,
            }
        )

    completed_task_means = [
        item["metric_percent"]["mean"]
        for item in tasks
        if item["complete"] and item["metric_percent"]["mean"] is not None
    ]
    all_tasks_complete = len(completed_task_means) == len(TASK_ORDER)
    return {
        "experiment_dir": str(experiment_dir),
        "variance_definition": (
            f"sample variance across {len(expected_seeds)} controlled inference seed(s) "
            "(ddof=1); undefined when only one seed is present"
        ),
        "task_count": len(tasks),
        "complete_task_count": sum(1 for item in tasks if item["complete"]),
        "all_tasks_complete": all_tasks_complete,
        "macro_average_percent": (
            statistics.fmean(completed_task_means) if all_tasks_complete else None
        ),
        "tasks": tasks,
        "runs": all_runs,
    }


def write_csv(path, summary):
    fields = [
        "task",
        "primary_metric",
        "metric_label",
        "stochastic_decoding",
        "complete",
        "completed_seeds",
        "metric_mean_percent",
        "metric_variance",
        "metric_std",
        "evaluation_mean_s",
        "evaluation_std_s",
        "evaluation_mean_ms_per_sample",
        "generation_mean_s",
        "generation_std_s",
        "tool_exec_mean_s",
        "tool_exec_std_s",
        "instance_e2e_mean_s",
        "round_e2e_mean_s",
        "tool_call_mean_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summary["tasks"]:
            writer.writerow(
                {
                    "task": item["task"],
                    "primary_metric": item["primary_metric"],
                    "metric_label": item["metric_label"],
                    "stochastic_decoding": item["stochastic_decoding"],
                    "complete": item["complete"],
                    "completed_seeds": ",".join(map(str, item["completed_seeds"])),
                    "metric_mean_percent": item["metric_percent"]["mean"],
                    "metric_variance": item["metric_percent"]["variance"],
                    "metric_std": item["metric_percent"]["std"],
                    "evaluation_mean_s": item["evaluation_s"]["mean"],
                    "evaluation_std_s": item["evaluation_s"]["std"],
                    "evaluation_mean_ms_per_sample": item["evaluation_ms_per_sample"]["mean"],
                    "generation_mean_s": item["generation_s"]["mean"],
                    "generation_std_s": item["generation_s"]["std"],
                    "tool_exec_mean_s": item["tool_exec_s"]["mean"],
                    "tool_exec_std_s": item["tool_exec_s"]["std"],
                    "instance_e2e_mean_s": item["instance_e2e_mean_s_across_seeds"]["mean"],
                    "round_e2e_mean_s": item["round_e2e_mean_s_across_seeds"]["mean"],
                    "tool_call_mean_s": item["tool_call_mean_s_across_seeds"]["mean"],
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--seeds", default="42,1234,2026")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    summary = summarize(experiment_dir, seeds)

    output_json = experiment_dir / "summary.json"
    output_csv = experiment_dir / "summary.csv"
    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(output_csv, summary)

    print(f"summary: {output_json}")
    print(f"csv:     {output_csv}")
    print(f"complete tasks: {summary['complete_task_count']}/{summary['task_count']}")
    if summary["macro_average_percent"] is not None:
        print(f"macro average: {summary['macro_average_percent']:.4f}")


if __name__ == "__main__":
    main()
