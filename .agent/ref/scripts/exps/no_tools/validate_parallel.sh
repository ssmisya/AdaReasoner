#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$EXPS_ROOT/shared/common.sh"

SMOKE_LOAD=0
if [[ "${1:-}" == "--smoke-load" ]]; then
    SMOKE_LOAD=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--smoke-load]" >&2
    exit 2
fi

ok() { printf 'OK   %s\n' "$*"; }
fail() { printf 'FAIL %s\n' "$*" >&2; exit 1; }

for command in bash python3 tmux flock setsid nvidia-smi timeout; do
    command -v "$command" >/dev/null || fail "missing command: $command"
done
ok "required commands"

bash "$EXPS_ROOT/validate.sh" >/dev/null
ok "datasets, models, conda and evaluation prerequisites"

bash -n \
    "$SCRIPT_DIR/run_all_parallel.sh" \
    "$SCRIPT_DIR/parallel_worker.sh" \
    "$SCRIPT_DIR/parallel_dashboard.sh" \
    "$SCRIPT_DIR/validate_parallel.sh" \
    "$EXPS_ROOT/shared/run_one.sh" \
    "$EXPS_ROOT/shared/common.sh"
python3 -m py_compile \
    "$SCRIPT_DIR/smoke_load.py" \
    "$SCRIPT_DIR/validate_matrix.py" \
    "$EXPS_ROOT/shared/eval_entry.py" \
    "$EXPS_ROOT/shared/preflight_tasks.py" \
    "$REPO/tool_server/tf_eval/tasks/hrbench/task.py"
ok "shell and Python syntax"

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits)
[[ ${#GPU_ROWS[@]} -eq 4 ]] || fail "exactly 4 GPUs required; found ${#GPU_ROWS[@]}"
for row in "${GPU_ROWS[@]}"; do
    IFS=',' read -r index total free <<< "$row"
    index="${index// /}"
    total="${total// /}"
    free="${free// /}"
    (( total >= 80000 )) || fail "GPU $index has only ${total} MiB; A100 80GB-class GPU required"
    if [[ "${ALLOW_BUSY:-0}" != "1" ]]; then
        (( free >= 76000 )) || fail "GPU $index has only ${free} MiB free; stop existing evaluations first"
    fi
done
ok "4 x 80GB GPUs and free-memory layout"

python3 - "$TASK_MATRIX" "$TASKS" "$SEEDS" <<'PY'
import json, sys
matrix = json.load(open(sys.argv[1], encoding="utf-8"))
tasks = [x for x in sys.argv[2].split(",") if x]
seeds = [x for x in sys.argv[3].split(",") if x]
unknown = sorted(set(tasks) - set(matrix))
if unknown:
    raise SystemExit("unknown tasks: " + ", ".join(unknown))
if not tasks or not seeds or any(not seed.isdigit() for seed in seeds):
    raise SystemExit("TASKS/SEEDS must be non-empty and seeds must be integers")
print(f"matrix: {len(tasks)} tasks x {len(seeds)} seeds x 4 models = {len(tasks)*len(seeds)*4} jobs")
PY
ok "task/seed matrix"

source /data/songmingyang/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python3 "$EXPS_ROOT/shared/preflight_tasks.py" \
    --task-matrix "$TASK_MATRIX" \
    --tasks "$TASKS"
ok "all task datasets, image paths, fields and sample counts"

if [[ ",$TASKS," == *,hrbench,* ]]; then
    load_yunwu_env
    python3 - <<'PY'
import os
import time
import requests

base = os.environ.get("OPENAI_API_URL", "https://yunwu.ai/v1").rstrip("/")
if base != "https://yunwu.ai/v1":
    raise SystemExit(f"unexpected YunwuAI base URL: {base}")
proxy = os.environ["YUNWU_PROXY_URL"]
session = requests.Session()
session.trust_env = False
last_error = None
for attempt in range(1, 6):
    try:
        response = session.post(
            base + "/chat/completions",
            headers={
                "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                "Content-Type": "application/json",
            },
            json={
                "model": os.environ.get("MODEL_VERSION", "gpt-4o-2024-11-20"),
                "messages": [{"role": "user", "content": "Reply with only A."}],
                "temperature": 0,
                "max_tokens": 2,
            },
            proxies={"http": proxy, "https": proxy},
            timeout=(30, 120),
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or not body.get("choices"):
            raise RuntimeError("YunwuAI preflight returned no choices")
        answer = body["choices"][0]["message"]["content"].strip().upper().rstrip(".")
        if answer != "A":
            raise RuntimeError(f"YunwuAI preflight returned invalid answer: {answer!r}")
        print(f"YUNWU_PREFLIGHT_OK attempt={attempt}")
        break
    except Exception as exc:
        last_error = exc
        if attempt < 5:
            time.sleep(2 ** attempt)
else:
    raise SystemExit(f"YunwuAI preflight failed after 5 attempts: {last_error}")
PY
    ok "YunwuAI proxy, authentication and chat completion"
fi

python3 - <<'PY'
from pathlib import Path
models = {
    "32B": (Path("/data/songmingyang/models/baselines/Qwen2.5-VL-32B-Instruct"), 1, 80 * 1024**3 * 0.92),
    "72B": (Path("/data/songmingyang/models/baselines/Qwen2.5-VL-72B-Instruct"), 2, 2 * 80 * 1024**3 * 0.95),
}
for size, (path, tp, budget) in models.items():
    weights = sum(item.stat().st_size for item in path.glob("*.safetensors"))
    if weights >= budget:
        raise SystemExit(f"{size} weights {weights/1024**3:.1f} GiB exceed layout budget {budget/1024**3:.1f} GiB")
    print(f"{size}: weights={weights/1024**3:.1f} GiB, TP={tp}, reserved-budget={budget/1024**3:.1f} GiB")
PY
ok "model weight/layout feasibility"

if [[ "${ALLOW_ACTIVE:-0}" != "1" ]]; then
    if pgrep -af "$EXPS_ROOT/shared/(run_one.sh|eval_entry.py)" >/dev/null 2>&1 || \
       pgrep -af "$SCRIPT_DIR/parallel_worker.sh" >/dev/null 2>&1; then
        fail "an evaluation process is already active"
    fi
fi

if (( SMOKE_LOAD )); then
    LOG_DIR="${RESULT_ROOT}/.parallel_smoke_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    source /data/songmingyang/miniforge3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
    export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

    echo "Running concurrent real model-load test on the production layout..."
    timeout 1800 env CUDA_VISIBLE_DEVICES=0,1 python3 "$SCRIPT_DIR/smoke_load.py" \
        --model "$(model_path 72B)" --tp 2 --gpu-memory 0.95 >"$LOG_DIR/72b.log" 2>&1 &
    pid72=$!
    timeout 1800 env CUDA_VISIBLE_DEVICES=2 python3 "$SCRIPT_DIR/smoke_load.py" \
        --model "$(model_path 32B)" --tp 1 --gpu-memory 0.92 >"$LOG_DIR/32b.log" 2>&1 &
    pid32=$!
    timeout 1800 env CUDA_VISIBLE_DEVICES=3 python3 "$SCRIPT_DIR/smoke_load.py" \
        --model "$(model_path 7B)" --tp 1 --gpu-memory 0.85 >"$LOG_DIR/7b.log" 2>&1 &
    pid7=$!

    rc=0
    wait "$pid72" || rc=1
    wait "$pid32" || rc=1
    wait "$pid7" || rc=1
    if (( rc )); then
        tail -40 "$LOG_DIR"/*.log >&2 || true
        fail "concurrent model-load test failed; logs: $LOG_DIR"
    fi
    grep -H 'SMOKE_OK' "$LOG_DIR"/*.log
    ok "concurrent 72B(TP2) + 32B(TP1) + 7B(TP1) model load and generation"
fi

echo "Parallel no-tools validation passed."
