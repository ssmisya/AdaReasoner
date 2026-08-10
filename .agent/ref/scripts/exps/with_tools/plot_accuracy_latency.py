#!/usr/bin/env python3
"""Build dependency-free accuracy/latency comparison artifacts from model summaries."""

import argparse
import csv
import json
import statistics
from html import escape
from pathlib import Path


DEFAULT_TASKS = [
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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def describe(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
    }


def latency_from_run(run):
    latency = run.get("instance_e2e_s")
    if isinstance(latency, dict) and latency.get("mean") is not None:
        return float(latency["mean"])
    value = run.get("instance_latency_mean_s")
    return float(value) if value is not None else None


def task_latency_stats(task):
    aggregate = task.get("instance_e2e_mean_s_across_seeds")
    if isinstance(aggregate, dict) and aggregate.get("mean") is not None:
        return {
            "count": int(aggregate.get("count", 0)),
            "mean": float(aggregate["mean"]),
            "std": (
                float(aggregate["std"])
                if aggregate.get("std") is not None
                else None
            ),
        }
    return describe(latency_from_run(run) for run in task.get("runs", []))


def task_rows(summary, model_slug, model_label, requested_tasks):
    by_name = {task["task"]: task for task in summary.get("tasks", [])}
    rows = []
    for task_name in requested_tasks:
        task = by_name.get(task_name, {})
        metric = task.get("metric_percent", {})
        latency = task_latency_stats(task)
        complete = bool(task.get("complete")) and metric.get("mean") is not None and latency["mean"] is not None
        rows.append(
            {
                "scope": "task",
                "task": task_name,
                "model_slug": model_slug,
                "model_label": model_label,
                "seed_count": int(metric.get("count", 0)),
                "accuracy_mean_percent": metric.get("mean"),
                "accuracy_seed_std_percent": metric.get("std"),
                "instance_e2e_mean_s": latency["mean"],
                "instance_e2e_seed_std_s": latency["std"],
                "complete": complete,
            }
        )
    return rows


def macro_row(summary, model_slug, model_label, requested_tasks, rows):
    complete_rows = [row for row in rows if row["complete"]]
    if len(complete_rows) != len(requested_tasks):
        return {
            "scope": "macro",
            "task": "macro",
            "model_slug": model_slug,
            "model_label": model_label,
            "seed_count": 0,
            "accuracy_mean_percent": None,
            "accuracy_seed_std_percent": None,
            "instance_e2e_mean_s": None,
            "instance_e2e_seed_std_s": None,
            "complete": False,
        }

    by_name = {task["task"]: task for task in summary.get("tasks", [])}
    per_seed_accuracy = {}
    per_seed_latency = {}
    for task_name in requested_tasks:
        for run in by_name[task_name].get("runs", []):
            seed = int(run["seed"])
            per_seed_accuracy.setdefault(seed, []).append(float(run["metric_percent"]))
            latency = latency_from_run(run)
            if latency is not None:
                per_seed_latency.setdefault(seed, []).append(latency)

    seed_accuracy = [
        statistics.fmean(values)
        for values in per_seed_accuracy.values()
        if len(values) == len(requested_tasks)
    ]
    seed_latency = [
        statistics.fmean(values)
        for values in per_seed_latency.values()
        if len(values) == len(requested_tasks)
    ]
    accuracy_stats = describe(seed_accuracy)
    latency_stats = describe(seed_latency)

    return {
        "scope": "macro",
        "task": "macro",
        "model_slug": model_slug,
        "model_label": model_label,
        "seed_count": accuracy_stats["count"],
        "accuracy_mean_percent": statistics.fmean(
            float(row["accuracy_mean_percent"]) for row in complete_rows
        ),
        "accuracy_seed_std_percent": accuracy_stats["std"],
        "instance_e2e_mean_s": statistics.fmean(
            float(row["instance_e2e_mean_s"]) for row in complete_rows
        ),
        "instance_e2e_seed_std_s": latency_stats["std"],
        "complete": True,
    }


def write_csv(path, rows):
    fields = [
        "scope",
        "task",
        "model_slug",
        "model_label",
        "seed_count",
        "accuracy_mean_percent",
        "accuracy_seed_std_percent",
        "instance_e2e_mean_s",
        "instance_e2e_seed_std_s",
        "complete",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def padded_range(values):
    low, high = min(values), max(values)
    if low == high:
        pad = max(abs(low) * 0.1, 1.0)
    else:
        pad = (high - low) * 0.12
    return low - pad, high + pad


def render_svg(path, rows, panels, model_order):
    width, height = 1500, 1120
    cols, rows_count = 3, 3
    outer_x, outer_y = 55, 105
    gap_x, gap_y = 28, 42
    panel_w = (width - 2 * outer_x - (cols - 1) * gap_x) / cols
    panel_h = (height - outer_y - 55 - (rows_count - 1) * gap_y) / rows_count
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"]
    color_by_model = {
        slug: colors[index % len(colors)] for index, slug in enumerate(model_order)
    }

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:25px;font-weight:700}.subtitle{font-size:13px;fill:#526075}.panel{font-size:15px;font-weight:700}.tick{font-size:10px;fill:#667085}.label{font-size:10px;font-weight:600}.axis{stroke:#98a2b3;stroke-width:1}.grid{stroke:#e4e7ec;stroke-width:1}.curve{fill:none;stroke:#98a2b3;stroke-width:1.5;stroke-dasharray:4 3}</style>',
        '<text x="55" y="38" class="title">With-tools Accuracy–Latency Comparison</text>',
        '<text x="55" y="62" class="subtitle">X: client-observed mean instance E2E latency (s) · Y: official task accuracy (%) · points connect model configurations</text>',
    ]

    legend_x = 55
    for slug in model_order:
        label = next(row["model_label"] for row in rows if row["model_slug"] == slug)
        color = color_by_model[slug]
        svg.append(f'<circle cx="{legend_x + 6}" cy="84" r="5" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 17}" y="88" class="subtitle">{escape(label)}</text>')
        legend_x += 17 + max(145, len(label) * 7)

    grouped = {}
    for row in rows:
        if row["complete"]:
            grouped.setdefault(row["task"], []).append(row)

    for panel_index, panel_name in enumerate(panels):
        col = panel_index % cols
        row_index = panel_index // cols
        x0 = outer_x + col * (panel_w + gap_x)
        y0 = outer_y + row_index * (panel_h + gap_y)
        plot_left, plot_right = x0 + 55, x0 + panel_w - 18
        plot_top, plot_bottom = y0 + 30, y0 + panel_h - 45
        points = grouped.get(panel_name, [])
        title = "Macro average" if panel_name == "macro" else panel_name
        svg.append(f'<text x="{x0}" y="{y0 + 15}" class="panel">{escape(title)}</text>')
        svg.append(f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" class="axis"/>')
        svg.append(f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" class="axis"/>')
        if not points:
            svg.append(f'<text x="{plot_left + 15}" y="{plot_top + 30}" class="subtitle">No complete data</text>')
            continue

        x_min, x_max = padded_range([float(point["instance_e2e_mean_s"]) for point in points])
        y_min, y_max = padded_range([float(point["accuracy_mean_percent"]) for point in points])

        def x_coord(value):
            return plot_left + (float(value) - x_min) / (x_max - x_min) * (plot_right - plot_left)

        def y_coord(value):
            return plot_bottom - (float(value) - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

        for tick_index in range(3):
            fraction = tick_index / 2
            x_value = x_min + fraction * (x_max - x_min)
            y_value = y_min + fraction * (y_max - y_min)
            x_tick = plot_left + fraction * (plot_right - plot_left)
            y_tick = plot_bottom - fraction * (plot_bottom - plot_top)
            svg.append(f'<line x1="{x_tick:.1f}" y1="{plot_top}" x2="{x_tick:.1f}" y2="{plot_bottom}" class="grid"/>')
            svg.append(f'<line x1="{plot_left}" y1="{y_tick:.1f}" x2="{plot_right}" y2="{y_tick:.1f}" class="grid"/>')
            svg.append(f'<text x="{x_tick:.1f}" y="{plot_bottom + 17}" text-anchor="middle" class="tick">{x_value:.2f}</text>')
            svg.append(f'<text x="{plot_left - 7}" y="{y_tick + 3:.1f}" text-anchor="end" class="tick">{y_value:.1f}</text>')

        ordered = sorted(points, key=lambda point: float(point["instance_e2e_mean_s"]))
        polyline = " ".join(
            f'{x_coord(point["instance_e2e_mean_s"]):.1f},{y_coord(point["accuracy_mean_percent"]):.1f}'
            for point in ordered
        )
        svg.append(f'<polyline points="{polyline}" class="curve"/>')
        for point in ordered:
            px = x_coord(point["instance_e2e_mean_s"])
            py = y_coord(point["accuracy_mean_percent"])
            color = color_by_model[point["model_slug"]]
            short_label = point["model_slug"].replace("adareasoner_randomized_", "Ada-").replace("qwen25vl_", "Qwen-")
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
            svg.append(f'<text x="{px + 7:.1f}" y="{py - 7:.1f}" class="label">{escape(short_label)}</text>')

        svg.append(f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{plot_bottom + 35}" text-anchor="middle" class="tick">Instance E2E latency (s)</text>')
        svg.append(f'<text x="{x0 + 10}" y="{(plot_top + plot_bottom) / 2:.1f}" text-anchor="middle" class="tick" transform="rotate(-90 {x0 + 10} {(plot_top + plot_bottom) / 2:.1f})">Accuracy (%)</text>')

    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def parse_model(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must use SLUG=DISPLAY_NAME")
    slug, label = value.split("=", 1)
    if not slug or not label:
        raise argparse.ArgumentTypeError("model must use non-empty SLUG=DISPLAY_NAME")
    return slug, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    result_root = Path(args.result_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else result_root / "with_tools" / "accuracy_latency"
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_tasks = [task for task in args.tasks.split(",") if task]

    all_rows = []
    missing = []
    for model_slug, model_label in args.model:
        summary_path = result_root / "with_tools" / model_slug / "summary.json"
        if not summary_path.is_file():
            missing.append(f"missing summary: {summary_path}")
            continue
        summary = read_json(summary_path)
        rows = task_rows(summary, model_slug, model_label, requested_tasks)
        all_rows.extend(rows)
        all_rows.append(macro_row(summary, model_slug, model_label, requested_tasks, rows))
        for row in rows:
            if not row["complete"]:
                missing.append(f"incomplete accuracy/latency: {model_slug}/{row['task']}")

    if args.require_complete and missing:
        raise SystemExit("Cannot build formal curve:\n  " + "\n  ".join(missing))
    if not all_rows:
        raise SystemExit("No model summaries were available")

    csv_path = output_dir / "accuracy_latency.csv"
    json_path = output_dir / "accuracy_latency.json"
    svg_path = output_dir / "accuracy_latency.svg"
    write_csv(csv_path, all_rows)
    available_model_order = [
        slug
        for slug, _ in args.model
        if any(row["model_slug"] == slug for row in all_rows)
    ]
    payload = {
        "schema_version": 1,
        "definition": {
            "accuracy": "official per-task metric in percent; macro is the unweighted mean across requested tasks",
            "latency": "client-observed instance_e2e_s; macro is the unweighted mean of per-task seed means",
            "uncertainty": "standard deviation across controlled inference seeds; undefined for a one-seed point estimate",
            "curve_scope": "matched-protocol model comparison, not a max-round/tool-call budget sweep",
        },
        "requested_tasks": requested_tasks,
        "models": available_model_order,
        "complete": not missing,
        "warnings": missing,
        "rows": all_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    panels = ["macro", *requested_tasks]
    render_svg(svg_path, all_rows, panels, available_model_order)

    print(f"accuracy/latency csv:  {csv_path}")
    print(f"accuracy/latency json: {json_path}")
    print(f"accuracy/latency svg:  {svg_path}")


if __name__ == "__main__":
    main()
