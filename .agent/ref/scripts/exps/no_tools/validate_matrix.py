#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

MODEL_SIZES = ("3b", "7b", "32b", "72b")
MODEL_PATHS = {
    size: f"/data/songmingyang/models/baselines/Qwen2.5-VL-{size.upper()}-Instruct"
    for size in MODEL_SIZES
}


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def marker_is_valid(path, task, seed, model_path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("validated") is True
        and payload.get("task") == task
        and int(payload.get("seed", -1)) == int(seed)
        and Path(payload.get("model_path", "")).resolve() == Path(model_path).resolve()
    )


def jobs(args):
    tasks = parse_csv(args.tasks)
    seeds = parse_csv(args.seeds)
    for size in MODEL_SIZES:
        for task in tasks:
            for seed in seeds:
                run_dir = args.result_root / "no_tools" / f"qwen25vl_{size}" / task / f"seed_{seed}"
                yield size, task, seed, run_dir


def count_markers(args):
    counts = {size: 0 for size in MODEL_SIZES}
    invalid = []
    for size, task, seed, run_dir in jobs(args):
        marker = run_dir / "DONE.json"
        if marker_is_valid(marker, task, seed, MODEL_PATHS[size]):
            counts[size] += 1
        elif marker.exists():
            invalid.append(str(marker))
    payload = {
        "counts": counts,
        "complete": sum(counts.values()),
        "expected": len(parse_csv(args.tasks)) * len(parse_csv(args.seeds)) * len(MODEL_SIZES),
        "invalid_markers": invalid,
    }
    return payload


def validate_all(args):
    failures = []
    for size, task, seed, run_dir in jobs(args):
        marker = run_dir / "DONE.json"
        if not marker.exists():
            failures.append({"size": size, "task": task, "seed": seed, "reason": "missing DONE.json"})
            continue
        command = [
            sys.executable,
            str(args.validate_run),
            "--run-dir",
            str(run_dir),
            "--task",
            task,
            "--task-matrix",
            str(args.task_matrix),
            "--model-path",
            MODEL_PATHS[size],
            "--seed",
            seed,
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            reason = (completed.stderr or completed.stdout).strip().splitlines()
            failures.append(
                {
                    "size": size,
                    "task": task,
                    "seed": seed,
                    "reason": reason[-1] if reason else f"validator rc={completed.returncode}",
                }
            )
    report = count_markers(args)
    report["fully_validated"] = not failures and report["complete"] == report["expected"]
    report["failures"] = failures
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temp = args.report.with_suffix(args.report.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(args.report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["fully_validated"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--task-matrix", required=True, type=Path)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--validate-run", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    if args.count_only:
        print(json.dumps(count_markers(args), ensure_ascii=False))
        return
    if args.report is None:
        parser.error("--report is required unless --count-only is used")
    raise SystemExit(validate_all(args))


if __name__ == "__main__":
    main()
