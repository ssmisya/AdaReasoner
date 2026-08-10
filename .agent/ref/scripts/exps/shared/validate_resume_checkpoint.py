#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

FAILURE_MARKER = "vllm failed to generate response due to"


def iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)


def finite_number(value, *, positive=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and (number > 0 if positive else number >= 0)


def extract_result(record):
    wrapper = record.get("results")
    if not isinstance(wrapper, dict):
        raise ValueError("missing results wrapper")
    result = wrapper.get("results")
    if not isinstance(result, dict):
        raise ValueError("missing instance result payload")
    idx = result.get("meta_data", {}).get("idx", wrapper.get("idx"))
    if idx is None:
        raise ValueError("missing sample ID")
    return str(idx), result


def validate_latency(idx, result):
    latency = result.get("latency")
    if not isinstance(latency, dict):
        raise ValueError(f"{idx}: missing latency")
    if not finite_number(latency.get("instance_e2e_s"), positive=True):
        raise ValueError(f"{idx}: invalid instance_e2e_s")
    if not finite_number(latency.get("model_generation_batch_wall_s"), positive=True):
        raise ValueError(f"{idx}: invalid model_generation_batch_wall_s")
    rounds = latency.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError(f"{idx}: missing round latency records")
    if int(latency.get("round_count", -1)) != len(rounds):
        raise ValueError(f"{idx}: round_count mismatch")
    for position, round_record in enumerate(rounds, 1):
        if not isinstance(round_record, dict):
            raise ValueError(f"{idx}: invalid round {position}")
        if not finite_number(round_record.get("generation_batch_wall_s"), positive=True):
            raise ValueError(f"{idx}: invalid round {position} generation latency")
        if not finite_number(round_record.get("round_e2e_s"), positive=True):
            raise ValueError(f"{idx}: invalid round {position} end-to-end latency")
        if not finite_number(round_record.get("tool_latency_s", 0.0)):
            raise ValueError(f"{idx}: invalid round {position} tool latency")
        if not finite_number(round_record.get("orchestration_and_queue_s", 0.0)):
            raise ValueError(f"{idx}: invalid round {position} orchestration latency")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise SystemExit("checkpoint is missing or empty")

    ids = set()
    records = 0
    with checkpoint.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"invalid JSON at line {line_number}: {exc}"
                ) from exc
            if any(FAILURE_MARKER in text.lower() for text in iter_strings(record)):
                raise SystemExit(
                    f"vLLM failure placeholder found at line {line_number}"
                )
            try:
                idx, result = extract_result(record)
                if idx in ids:
                    raise ValueError(f"duplicate sample ID: {idx}")
                validate_latency(idx, result)
            except ValueError as exc:
                raise SystemExit(
                    f"checkpoint is not safe to resume at line {line_number}: {exc}"
                ) from exc
            ids.add(idx)
            records += 1

    if records == 0:
        raise SystemExit("checkpoint has zero records")
    print(f"resume checkpoint validated: records={records}, latency_records={records}")


if __name__ == "__main__":
    main()
