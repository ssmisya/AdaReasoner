#!/usr/bin/env bash
set -euo pipefail

REPO="/data/songmingyang/code/reasoning/AdaReasoner-rebuttal"
PYTHON_BIN="${PYTHON_BIN:-/data/songmingyang/miniforge3/envs/vllm-latest/bin/python}"
MODEL="${GPT5_MODEL:-gpt-5-2025-08-07}"
TASKS="${GPT5_TASKS:-vsp,jigsaw_blink,web_guichat,webmmu,hrbench,vstar}"
TOOLS_OVERRIDE="${GPT5_TOOLS:-}"
BATCH_SIZE="${GPT5_BATCH_SIZE:-4}"
REQUEST_CONCURRENCY="${GPT5_REQUEST_CONCURRENCY:-4}"
MAX_ROUNDS="${GPT5_MAX_ROUNDS:-6}"
CONTROLLER="${TOOL_CONTROLLER_URL:-http://127.0.0.1:21112}"
OUTPUT_DIR=""
DRY_RUN=0
SKIP_PREFLIGHT=0
AUTO_PROXY=1

usage() {
    cat <<'EOF'
Usage: rebuttal_exps/run_gpt5_tools_eval.sh [options]

Options:
  --tasks TASK1,TASK2      Evaluation tasks (default: rebuttal task suite)
  --model MODEL            GPT-5 model ID (default: gpt-5-2025-08-07)
  --tools TOOL1,TOOL2      Override task-specific tool selections
  --batch-size N           Active evaluation batch size (default: 4)
  --concurrency N          Concurrent GPT-5 requests (default: 4)
  --max-rounds N           Maximum model/tool rounds (default: 6)
  --controller URL         Local tool controller URL
  --output-dir DIR         Output directory (default: timestamped directory)
  --dry-run                Generate config and run preflight checks only
  --skip-preflight         Skip API/model/tool availability checks
  --no-auto-proxy          Do not reuse the machine's configured proxy
  -h, --help               Show this help

Examples:
  bash rebuttal_exps/run_gpt5_tools_eval.sh --tasks vsp --dry-run
  bash rebuttal_exps/run_gpt5_tools_eval.sh --tasks webmmu,hrbench --batch-size 2
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Missing value for $1" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks) require_value "$@"; TASKS="$2"; shift 2 ;;
        --model) require_value "$@"; MODEL="$2"; shift 2 ;;
        --tools) require_value "$@"; TOOLS_OVERRIDE="$2"; shift 2 ;;
        --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
        --concurrency) require_value "$@"; REQUEST_CONCURRENCY="$2"; shift 2 ;;
        --max-rounds) require_value "$@"; MAX_ROUNDS="$2"; shift 2 ;;
        --controller) require_value "$@"; CONTROLLER="${2%/}"; shift 2 ;;
        --output-dir) require_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        --no-auto-proxy) AUTO_PROXY=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value in "$BATCH_SIZE" "$REQUEST_CONCURRENCY" "$MAX_ROUNDS"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Batch size, concurrency, and max rounds must be positive integers." >&2
        exit 2
    fi
done
if [[ ! "$MODEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid model ID: $MODEL" >&2
    exit 2
fi

IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"
if [[ ${#TASK_ARRAY[@]} -eq 0 ]]; then
    echo "At least one task is required." >&2
    exit 2
fi
for i in "${!TASK_ARRAY[@]}"; do
    task="${TASK_ARRAY[$i]//[[:space:]]/}"
    if [[ -z "$task" || ! "$task" =~ ^[A-Za-z0-9_]+$ ]]; then
        echo "Invalid task name: ${TASK_ARRAY[$i]}" >&2
        exit 2
    fi
    TASK_ARRAY[$i]="$task"
done
TASKS="$(IFS=','; echo "${TASK_ARRAY[*]}")"

# Remote model generation and benchmark judges must use this OpenAI-compatible
# yunwu endpoint. Do not inherit or fall back to the OpenAI official endpoint.
export OPENAI_API_URL="https://yunwu.ai/v1"
export OPENAI_API_KEY="sk-8e3WFT4PoKHtgXvgQq1fFkxNgsG1yw2mWEgPRpNI8mpkFHVu"

export YUNWU_PROXY_URL="${YUNWU_PROXY_URL:-http://songmingyang:SDNiquomCZL6QL8PpKk5Tf2CTzWNQIAvdp6XYE5GBIHFvGvsSt96xOKoZ9W6@10.1.20.50:23128/}"
if [[ "$AUTO_PROXY" -eq 1 ]]; then
    # Only remote HTTPS traffic needs the proxy. Do not set HTTP_PROXY: the
    # tool controller and local services use HTTP and must remain direct.
    export HTTPS_PROXY="$YUNWU_PROXY_URL"
    export https_proxy="$YUNWU_PROXY_URL"
fi
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,::1,.svc,.cluster.local,10.0.0.0/8"
export no_proxy="$NO_PROXY"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO/rebuttal_exps/gpt5_tools/$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
CONFIG_PATH="$OUTPUT_DIR/config.yaml"
RESULT_PATH="$OUTPUT_DIR/results.jsonl"
LOG_PATH="$OUTPUT_DIR/run.log"

export ADA_EVAL_TASKS="$TASKS"
export ADA_EVAL_TOOLS_OVERRIDE="$TOOLS_OVERRIDE"
export ADA_EVAL_MODEL="$MODEL"
export ADA_EVAL_BATCH_SIZE="$BATCH_SIZE"
export ADA_EVAL_REQUEST_CONCURRENCY="$REQUEST_CONCURRENCY"
export ADA_EVAL_MAX_ROUNDS="$MAX_ROUNDS"
export ADA_EVAL_CONTROLLER="$CONTROLLER"
export ADA_EVAL_OUTPUT_DIR="$OUTPUT_DIR"
export ADA_EVAL_RESULT_PATH="$RESULT_PATH"
export ADA_EVAL_CONFIG_PATH="$CONFIG_PATH"

"$PYTHON_BIN" <<'PY'
import os
from pathlib import Path

import yaml

TASK_TOOLS = {
    "vsp": "AStarWithPixelCoordinate,Draw2DPath,Point",
    "jigsaw_blink": "DetectBlackArea,InsertImage",
    "web_guichat": "OCR,Point,Crop",
    "webmmu": "OCR,Crop",
    "hrbench": "Point,OCR,Crop,AStarWithPixelCoordinate",
    "vstar": "Point,OCR,Crop,AStarWithPixelCoordinate",
}
DEFAULT_TOOLS = "Point,OCR,Crop,AStarWithPixelCoordinate,Draw2DPath,DetectBlackArea,InsertImage"

tasks = os.environ["ADA_EVAL_TASKS"].split(",")
output_dir = Path(os.environ["ADA_EVAL_OUTPUT_DIR"])
tools_override = os.environ.get("ADA_EVAL_TOOLS_OVERRIDE", "").strip()
checkpoint_paths = {
    task: str(output_dir / f"{task}_ckpt.jsonl") for task in tasks
}
task_args = {
    "task_name": ",".join(tasks),
    "save_to_ckpt": checkpoint_paths,
    "middle_images_save_dir": {
        task: str(output_dir / "middle_images" / task) for task in tasks
    },
}
resume_paths = {
    task: path for task, path in checkpoint_paths.items() if Path(path).is_file()
}
if resume_paths:
    task_args["resume_from_ckpt"] = resume_paths
if tools_override:
    task_args["tool_selection"] = tools_override
else:
    task_args["tool_selection_dict"] = {
        task: TASK_TOOLS.get(task, DEFAULT_TOOLS) for task in tasks
    }

config = {
    "model_args": {
        "model": "openai",
        "model_args": (
            f"pretrained={os.environ['ADA_EVAL_MODEL']},"
            "limit_mm_per_prompt=10,max_retry=5,request_timeout=600,"
            "retry_base_seconds=2,"
            f"request_concurrency={os.environ['ADA_EVAL_REQUEST_CONCURRENCY']}"
        ),
        "batch_size": int(os.environ["ADA_EVAL_BATCH_SIZE"]),
        "max_rounds": int(os.environ["ADA_EVAL_MAX_ROUNDS"]),
        "model_mode": "general",
    },
    "task_args": task_args,
    "script_args": {
        "verbosity": "INFO",
        "output_path": os.environ["ADA_EVAL_RESULT_PATH"],
        "controller_addr": os.environ["ADA_EVAL_CONTROLLER"],
        "if_use_tool": True,
        "if_randomize_tool": False,
    },
}

config_path = Path(os.environ["ADA_EVAL_CONFIG_PATH"])
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
for path in task_args["middle_images_save_dir"].values():
    Path(path).mkdir(parents=True, exist_ok=True)
PY

if [[ "$SKIP_PREFLIGHT" -eq 0 ]]; then
    PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import contextlib
import io
import os
import sys

import requests
import yaml
from openai import OpenAI

config_path = sys.argv[1]
with open(config_path, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

model_arg = config["model_args"]["model_args"].split(",")[0]
model = model_arg.split("=", 1)[1]
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_API_URL"].rstrip("/"),
    timeout=60,
    max_retries=0,
)
model_ids = {item.id for item in client.models.list().data}
if model not in model_ids:
    raise SystemExit(f"GPT model is not available at the configured endpoint: {model}")

args = config["task_args"]
if args.get("tool_selection"):
    selected = set(args["tool_selection"].split(","))
else:
    selected = {
        tool
        for tools in args.get("tool_selection_dict", {}).values()
        for tool in tools.split(",")
    }

controller = config["script_args"]["controller_addr"].rstrip("/")
session = requests.Session()
session.trust_env = False
response = session.post(f"{controller}/list_models", timeout=10)
response.raise_for_status()
online = set(response.json().get("models", []))

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from tool_server.tool_workers.offline_workers import get_available_tools
    offline = set(get_available_tools())

missing = selected - online - offline
if missing:
    raise SystemExit(f"Configured tools are unavailable: {', '.join(sorted(missing))}")
print(
    f"Preflight OK: model={model}; "
    f"online_tools={','.join(sorted(selected & online)) or 'none'}; "
    f"offline_tools={','.join(sorted(selected & offline)) or 'none'}"
)
PY
fi

cat <<EOF
GPT-5 + Tools evaluation
  model:      $MODEL
  tasks:      $TASKS
  config:     $CONFIG_PATH
  results:    $RESULT_PATH
  log:        $LOG_PATH
  controller: $CONTROLLER
EOF

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run complete; evaluation was not started."
    exit 0
fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
unset VLLM_BASE_URL VLLM_API_KEY BASE_URL
cd "$REPO"
"$PYTHON_BIN" -m tool_server.tf_eval --config "$CONFIG_PATH" 2>&1 | tee "$LOG_PATH"
