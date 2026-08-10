#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "std": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def read_checkpoint(path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def extract_result(record):
    result = record.get("results", {})
    if isinstance(result, dict) and isinstance(result.get("results"), dict):
        return result["results"], result.get("idx")
    return result, result.get("idx") if isinstance(result, dict) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output_jsonl = Path(args.output_jsonl)
    summary_path = Path(args.summary)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total_records = 0
    instance_values = []
    round_values = []
    generation_batch_values = []
    backend_request_values = []
    ttft_values = []
    queue_values = []
    tool_values = []
    by_round = {}

    for record in read_checkpoint(checkpoint):
        total_records += 1
        result, fallback_idx = extract_result(record)
        latency = result.get("latency") if isinstance(result, dict) else None
        if not isinstance(latency, dict) or latency.get("instance_e2e_s") is None:
            continue

        idx = result.get("meta_data", {}).get("idx", fallback_idx)
        row = {
            "task_name": record.get("task_name"),
            "model_name": record.get("model_name"),
            "idx": idx,
            "latency": latency,
        }
        rows.append(row)
        instance_values.append(latency.get("instance_e2e_s"))

        for round_record in latency.get("rounds", []):
            round_index = int(round_record.get("round_index", 0))
            round_values.append(round_record.get("round_e2e_s"))
            generation_batch_values.append(
                round_record.get("generation_batch_wall_s")
            )
            backend = round_record.get("backend_request_metrics", {})
            backend_request_values.append(backend.get("request_e2e_s"))
            ttft_values.append(backend.get("ttft_s"))
            queue_values.append(backend.get("queue_s"))
            by_round.setdefault(round_index, []).append(
                round_record.get("round_e2e_s")
            )
            for tool_call in round_record.get("tool_calls", []):
                tool_values.append(tool_call.get("latency_s"))

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": 1,
        "definitions": {
            "instance_e2e_s": "wall time from instance admission before conversation construction until the final round is recognized as finished; excludes final result serialization and checkpoint writes",
            "round_e2e_s": "wall time from the start of a model round through generation and, when used, tool result conversion into the next-round input",
            "generation_batch_wall_s": "synchronous wall time of the whole model batch; shared by all requests in that batch and not summed for throughput",
            "backend_request_metrics": "per-request vLLM metrics when exposed by the installed vLLM version",
            "tool_latency_s": "wall time of one ToolManager.call_tool RPC",
        },
        "checkpoint": str(checkpoint),
        "total_checkpoint_records": total_records,
        "latency_records": len(rows),
        "coverage": (len(rows) / total_records) if total_records else None,
        "instance_e2e_s": describe(instance_values),
        "round_e2e_s": describe(round_values),
        "round_e2e_s_by_round": {
            str(index): describe(values) for index, values in sorted(by_round.items())
        },
        "generation_batch_wall_s": describe(generation_batch_values),
        "backend_request_e2e_s": describe(backend_request_values),
        "backend_ttft_s": describe(ttft_values),
        "backend_queue_s": describe(queue_values),
        "tool_call_s": describe(tool_values),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"latency coverage: {len(rows)}/{total_records}; "
        f"summary={summary_path}; instances={output_jsonl}"
    )
    if args.require_complete and len(rows) != total_records:
        raise SystemExit(
            f"latency coverage is incomplete: {len(rows)}/{total_records} records"
        )


if __name__ == "__main__":
    main()
