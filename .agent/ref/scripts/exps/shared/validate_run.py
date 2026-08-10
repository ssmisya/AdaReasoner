#!/usr/bin/env python3
import argparse
import json
import math
import os
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import extract_metric, read_last_jsonl, result_records, sample_count


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def require_latency(value, label, allow_zero=True):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Missing or invalid latency {label}: {value!r}") from exc
    lower_bound_ok = number >= 0.0 if allow_zero else number > 0.0
    if not math.isfinite(number) or not lower_bound_ok:
        raise RuntimeError(f"Missing or invalid latency {label}: {value!r}")
    return number


def validate_instance_latencies(raw_results):
    rows = []
    for result_item in raw_results:
        idx = result_item.get("idx")
        payload = result_item.get("results")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Missing instance result payload for {idx}")
        latency = payload.get("latency")
        if not isinstance(latency, dict):
            raise RuntimeError(f"Missing instance latency for {idx}")
        instance_e2e_s = require_latency(
            latency.get("instance_e2e_s"), f"{idx}.instance_e2e_s", allow_zero=False
        )
        rounds = latency.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            raise RuntimeError(f"Missing round latency records for {idx}")
        if int(latency.get("round_count", -1)) != len(rounds):
            raise RuntimeError(f"Round count mismatch for {idx}")
        normalized_rounds = []
        for position, round_record in enumerate(rounds, start=1):
            if not isinstance(round_record, dict):
                raise RuntimeError(f"Invalid round record {idx}/{position}")
            normalized = dict(round_record)
            normalized["generation_batch_wall_s"] = require_latency(
                round_record.get("generation_batch_wall_s"),
                f"{idx}.round{position}.generation_batch_wall_s",
                allow_zero=False,
            )
            normalized["tool_latency_s"] = require_latency(
                round_record.get("tool_latency_s", 0.0),
                f"{idx}.round{position}.tool_latency_s",
            )
            normalized["round_e2e_s"] = require_latency(
                round_record.get("round_e2e_s"),
                f"{idx}.round{position}.round_e2e_s",
                allow_zero=False,
            )
            normalized["orchestration_and_queue_s"] = require_latency(
                round_record.get("orchestration_and_queue_s", 0.0),
                f"{idx}.round{position}.orchestration_and_queue_s",
            )
            normalized_rounds.append(normalized)
        rows.append(
            {
                "idx": idx,
                "instance_e2e_s": instance_e2e_s,
                "round_count": len(normalized_rounds),
                "model_generation_batch_wall_s": require_latency(
                    latency.get("model_generation_batch_wall_s"),
                    f"{idx}.model_generation_batch_wall_s",
                    allow_zero=False,
                ),
                "tool_latency_s": require_latency(
                    latency.get("tool_latency_s", 0.0),
                    f"{idx}.tool_latency_s",
                ),
                "definitions": {
                    key: latency.get(key)
                    for key in (
                        "definition",
                        "instance_definition",
                        "round_definition",
                        "generation_attribution",
                    )
                },
                "rounds": normalized_rounds,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-matrix", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    matrix = read_json(args.task_matrix)
    spec = matrix[args.task]

    required = [
        run_dir / "result.jsonl",
        run_dir / "timing.json",
        run_dir / "run_metadata.json",
        run_dir / "exit_code.txt",
        run_dir / "config.yaml",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing run artifacts: " + ", ".join(missing))

    if (run_dir / "exit_code.txt").read_text(encoding="utf-8").strip() != "0":
        raise RuntimeError("Non-zero exit code")

    timing = read_json(run_dir / "timing.json")
    if timing.get("status") != "success":
        raise RuntimeError(f"Timing status is not success: {timing.get('status')}")

    result = read_last_jsonl(run_dir / "result.jsonl")
    if result.get("task_name") != args.task:
        raise RuntimeError(
            f"Result task mismatch: {result.get('task_name')} != {args.task}"
        )

    if result.get("primary_metric") != spec.get("primary_metric"):
        raise RuntimeError(
            f"Primary metric protocol mismatch: {result.get('primary_metric')} != "
            f"{spec.get('primary_metric')}"
        )
    metric = extract_metric(args.task, result)
    if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
        raise RuntimeError(f"Invalid primary metric: {metric}")

    samples = sample_count(args.task, result)
    input_samples = result.get("input_samples")
    expected = spec.get("expected_samples")
    strict = spec.get("strict_expected_samples", True)
    if samples is None or int(samples) <= 0:
        raise RuntimeError(f"Missing or invalid scored sample count: {samples}")
    if input_samples is None or int(input_samples) != int(samples):
        raise RuntimeError(
            f"Input/scored sample mismatch: {input_samples} vs {samples}"
        )
    if strict and expected is not None and int(samples) != int(expected):
        raise RuntimeError(f"Scored {samples} samples; expected {expected}")

    failed_generation_text = "Vllm failed to generate response due to"
    records = result_records(result)
    raw_results = result.get("results", [])
    if not isinstance(raw_results, list) or len(raw_results) != int(samples):
        raise RuntimeError(
            f"Generated/scored sample mismatch: {len(raw_results) if isinstance(raw_results, list) else 'invalid'} vs {samples}"
        )
    generated_ids = [item.get("idx") for item in raw_results]
    if None in generated_ids or len(set(generated_ids)) != int(samples):
        raise RuntimeError("Generated result IDs are missing or duplicated")
    scored_ids = [item.get("idx") for item in records]
    if None in scored_ids or set(generated_ids) != set(scored_ids):
        raise RuntimeError("Generated and scored result IDs do not match")
    latency_rows = validate_instance_latencies(raw_results)

    failed_ids = [
        item.get("idx")
        for item in records
        if failed_generation_text
        in str(item.get("pred", item.get("prediction", "")))
    ]
    if failed_ids:
        raise RuntimeError(
            f"vLLM generation failed for {len(failed_ids)} samples; first IDs: {failed_ids[:10]}"
        )

    if args.task == "hrbench":
        invalid_judges = [
            item.get("idx")
            for item in records
            if item.get("gpt_prediction") not in {"A", "B", "C", "D", "Z"}
            or item.get("judge_source") not in {"rule", "yunwu"}
        ]
        if invalid_judges:
            raise RuntimeError(
                f"HRBench has {len(invalid_judges)} unaudited judge results; "
                f"first IDs: {invalid_judges[:10]}"
            )

    metadata = read_json(run_dir / "run_metadata.json")
    if os.path.realpath(metadata.get("model_path", "")) != os.path.realpath(args.model_path):
        raise RuntimeError("Model path does not match run metadata")
    if int(metadata.get("seed", -1)) != args.seed:
        raise RuntimeError("Seed does not match run metadata")
    if metadata.get("task") != args.task:
        raise RuntimeError("Task does not match run metadata")

    latency_path = run_dir / "latency.jsonl"
    latency_summary_path = run_dir / "latency_summary.json"
    if not latency_path.is_file() or not latency_summary_path.is_file():
        raise RuntimeError("Missing latency.jsonl or latency_summary.json")
    latency_summary = read_json(latency_summary_path)
    if int(latency_summary.get("latency_records", -1)) != int(samples):
        raise RuntimeError(
            f"Latency coverage mismatch: {latency_summary.get('latency_records')} vs {samples}"
        )
    all_rounds = [
        round_record
        for row in latency_rows
        for round_record in row["rounds"]
    ]

    done = {
        "validated": True,
        "task": args.task,
        "seed": args.seed,
        "model_path": os.path.realpath(args.model_path),
        "samples": int(samples),
        "stochastic_decoding": metadata.get("stochastic_decoding"),
        "primary_metric": spec["primary_metric"],
        "metric_label": spec.get("metric_label", spec["primary_metric"]),
        "metric": metric,
        "metric_percent": metric * 100.0,
        "evaluation_s": timing.get("evaluation_s"),
        "instance_latency_file": latency_path.name,
        "instance_latency_mean_s": sum(
            row["instance_e2e_s"] for row in latency_rows
        ) / len(latency_rows),
        "round_latency_mean_s": sum(
            row["round_e2e_s"] for row in all_rounds
        ) / len(all_rounds),
        "round_latency_count": len(all_rounds),
    }
    fd, temp_name = tempfile.mkstemp(prefix="DONE.", suffix=".tmp", dir=run_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(done, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_name, run_dir / "DONE.json")
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


if __name__ == "__main__":
    main()
