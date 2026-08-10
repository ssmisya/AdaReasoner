#!/usr/bin/env bash
# Formal with-tools matrix. Models run serially because each one occupies both
# non-tool GPUs with TP=2; Point/OCR remain resident on GPU 1,2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(cd "$EXPS_ROOT/../../../.." && pwd)"
# shellcheck disable=SC1091
source "$EXPS_ROOT/shared/common.sh"

SESSION="${TMUX_SESSION:-ada_eval_with_tools}"
MODEL_GPUS="${MODEL_GPUS:-0,3}"
MODEL_ORDER="${MODEL_ORDER:-adareasoner_randomized_7b,qwen25vl_7b,qwen25vl_72b}"
TASKS="${TASKS:-vsp,vspo,jigsaw_coco,jigsaw_blink,vstar,web_guichat,webmmu,hrbench}"
SEEDS="${SEEDS:-42,1234,2026}"
ADAREASONER_SEEDS="${ADAREASONER_SEEDS:-42}"
ADAREASONER_MODEL_PATH="${ADAREASONER_MODEL_PATH:-/data/songmingyang/model/adareasoner/AdaReasoner-7B-Randomized}"
ADAREASONER_MODEL_NAME="${ADAREASONER_MODEL_NAME:-AdaReasoner-7B-Randomized-Final}"
QWEN7B_MODEL_PATH="${QWEN7B_MODEL_PATH:-/data/songmingyang/models/baselines/Qwen2.5-VL-7B-Instruct}"
QWEN72B_MODEL_PATH="${QWEN72B_MODEL_PATH:-/data/songmingyang/models/baselines/Qwen2.5-VL-72B-Instruct}"
BATCH_ALL="${BATCH_ALL:-64}"
GPU_MEMORY_7B="${GPU_MEMORY_7B:-0.85}"
GPU_MEMORY_72B="${GPU_MEMORY_72B:-0.95}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
DRY_RUN="${DRY_RUN:-0}"
AUTO_START_TOOLS="${AUTO_START_TOOLS:-0}"
GENERATE_CURVE="${GENERATE_CURVE:-1}"
EVAL_BACKEND="${EVAL_BACKEND:-native}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-32}"
TOOL_CONCURRENCY="${TOOL_CONCURRENCY:-16}"
TF_EVAL_PIPELINE="${TF_EVAL_PIPELINE:-1}"
TF_EVAL_PIPELINE_MAX_ACTIVE="${TF_EVAL_PIPELINE_MAX_ACTIVE:-$((BATCH_ALL * 3))}"
START_TOOLS="$REPO/.agent/ref/scripts/start_tools.sh"
WORKER="$SCRIPT_DIR/tp2_worker.sh"
DASHBOARD="$SCRIPT_DIR/dashboard.sh"
CHECK_TOOLS="$SCRIPT_DIR/check_tools.sh"

attach_session() {
    if [[ -n "${TMUX:-}" ]]; then
        tmux switch-client -t "$SESSION"
    else
        tmux attach-session -t "$SESSION"
    fi
}

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Tool Eval dry-run (backend=$EVAL_BACKEND; models are serial; each model uses TP=2):"
    echo "  1. AdaReasoner-7B-Randomized Final: seeds=$ADAREASONER_SEEDS (default: one run)"
    echo "  2. Qwen2.5-VL-7B + Tools: seeds=$SEEDS"
    echo "  3. Qwen2.5-VL-72B + Tools: seeds=$SEEDS"
    echo "  tasks=$TASKS model_gpus=$MODEL_GPUS tool_gpus=1,2 batch=$BATCH_ALL"
    echo "  backend=$EVAL_BACKEND vllm_port=$VLLM_PORT lm_concurrency=$VLLM_CONCURRENCY tool_concurrency=$TOOL_CONCURRENCY"
    echo "  pipeline=$TF_EVAL_PIPELINE max_active=$TF_EVAL_PIPELINE_MAX_ACTIVE"
    echo "  latency=required per instance and per round; accuracy/latency curve=$GENERATE_CURVE"
    exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    CURRENT_STATE="$RESULT_ROOT/.with_tools_state/current"
    WORKER_PANE_DEAD=0
    if [[ -s "$CURRENT_STATE/worker.pane" ]]; then
        WORKER_PANE="$(cat "$CURRENT_STATE/worker.pane")"
        WORKER_PANE_DEAD="$(tmux display-message -p -t "$WORKER_PANE" '#{pane_dead}' 2>/dev/null || echo 1)"
    fi
    if [[ -f "$CURRENT_STATE/ALL_DONE" || -f "$CURRENT_STATE/FINISHED_WITH_FAILURES" || "$WORKER_PANE_DEAD" == "1" ]]; then
        echo "发现已结束的 tmux session '$SESSION'，清理旧界面后创建新的可恢复运行。"
        tmux kill-session -t "$SESSION"
    else
        echo "tmux session '$SESSION' 正在运行，不创建重复 worker。"
        if [[ "${NO_ATTACH:-0}" != "1" ]]; then
            attach_session
        else
            echo "查看命令: tmux attach -t $SESSION"
        fi
        exit 0
    fi
fi

for command in bash python3 tmux curl ss flock setsid nvidia-smi; do
    command -v "$command" >/dev/null || { echo "ERROR: missing command: $command" >&2; exit 2; }
done
case "$EVAL_BACKEND" in
    native|legacy|embedded) ;;
    server|openai|async)
        if ! curl -fsS --max-time 3 "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
            echo "ERROR: EVAL_BACKEND=$EVAL_BACKEND requires a healthy vLLM server on port $VLLM_PORT." >&2
            echo "Start it first with: source $SCRIPT_DIR/vllm_serve.sh && vllm_start <model-path> $MODEL_GPUS 2 $VLLM_PORT" >&2
            exit 1
        fi
        ;;
    *) echo "ERROR: invalid EVAL_BACKEND=$EVAL_BACKEND" >&2; exit 2 ;;
esac

python3 - "$TASK_MATRIX" "$MODEL_ORDER" "$TASKS" "$SEEDS" "$ADAREASONER_SEEDS" "$BATCH_ALL" "$MAX_ATTEMPTS" <<'PY'
import json
import sys

matrix = json.load(open(sys.argv[1], encoding="utf-8"))
models = [value for value in sys.argv[2].split(",") if value]
tasks = [value for value in sys.argv[3].split(",") if value]
allowed_models = {"adareasoner_randomized_7b", "qwen25vl_7b", "qwen25vl_72b"}
if not models or len(models) != len(set(models)) or set(models) - allowed_models:
    raise SystemExit(f"ERROR: invalid or duplicate MODEL_ORDER: {models}")
if not tasks or len(tasks) != len(set(tasks)) or set(tasks) - set(matrix):
    raise SystemExit(f"ERROR: invalid or duplicate TASKS: {tasks}")
for label, value in (("SEEDS", sys.argv[4]), ("ADAREASONER_SEEDS", sys.argv[5])):
    seeds = [seed for seed in value.split(",") if seed]
    if not seeds or len(seeds) != len(set(seeds)) or any(not seed.isdigit() for seed in seeds):
        raise SystemExit(f"ERROR: invalid or duplicate {label}: {value}")
if int(sys.argv[6]) <= 0 or int(sys.argv[7]) <= 0:
    raise SystemExit("ERROR: BATCH_ALL and MAX_ATTEMPTS must be positive integers")
PY

# Static preflight only: this does not load a model or start an evaluation.
SKIP_MODEL_CHECK=1 bash "$EXPS_ROOT/validate.sh" >/dev/null
for model_dir in "$ADAREASONER_MODEL_PATH" "$QWEN7B_MODEL_PATH" "$QWEN72B_MODEL_PATH"; do
    [[ -f "$model_dir/config.json" ]] || { echo "ERROR: invalid model: $model_dir" >&2; exit 2; }
    compgen -G "$model_dir/*.safetensors" >/dev/null || {
        echo "ERROR: model has no safetensors weights: $model_dir" >&2
        exit 2
    }
done
[[ "$MODEL_GPUS" == "0,3" ]] || {
    echo "ERROR: MODEL_GPUS must be 0,3 because GPU1/2 are reserved for Point/OCR." >&2
    exit 2
}
GPU_INDICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')"
for gpu in 0 1 2 3; do
    grep -qx "$gpu" <<< "$GPU_INDICES" || {
        echo "ERROR: this layout requires physical GPU $gpu." >&2
        exit 2
    }
done

if [[ "${ALLOW_BUSY_MODEL_GPUS:-0}" != "1" ]]; then
    python3 - "$MODEL_GPUS" "$GPU_MEMORY_72B" <<'PY'
import subprocess
import sys

gpu_ids = [int(value) for value in sys.argv[1].split(",")]
utilization = float(sys.argv[2])
output = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"],
    text=True,
)
info = {}
for line in output.splitlines():
    index, total, free = (part.strip() for part in line.split(","))
    info[int(index)] = (float(total), float(free))
for gpu_id in gpu_ids:
    total, free = info[gpu_id]
    required = total * utilization
    if free < required:
        raise SystemExit(
            f"ERROR: GPU{gpu_id} has {free:.0f} MiB free, but the 72B TP=2 run "
            f"reserves {required:.0f} MiB ({utilization:.2f} of {total:.0f} MiB). "
            "Stop other evaluations first; ALLOW_BUSY_MODEL_GPUS=1 only bypasses this early check."
        )
PY
fi

if ! bash "$CHECK_TOOLS" --quiet; then
    if [[ "$AUTO_START_TOOLS" == "1" ]]; then
        echo "工具服务未就绪，正在启动 GPU1/2 上的 Point/OCR 和 CPU Crop..."
        bash "$START_TOOLS"
    else
        echo "ERROR: tool services are not ready. Run: bash $START_TOOLS" >&2
        exit 1
    fi
fi
bash "$CHECK_TOOLS"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
START_EPOCH="$(date +%s)"
STATE_ROOT="$RESULT_ROOT/.with_tools_state"
STATE_DIR="$STATE_ROOT/$RUN_ID"
mkdir -p "$STATE_DIR/failures"
ln -sfn "$STATE_DIR" "$STATE_ROOT/current"

{
    printf 'REPO=%q\n' "$REPO"
    printf 'RESULT_ROOT=%q\n' "$RESULT_ROOT"
    printf 'CONDA_ENV=%q\n' "$CONDA_ENV"
    printf 'TASKS=%q\n' "$TASKS"
    printf 'ADAREASONER_SEEDS=%q\n' "$ADAREASONER_SEEDS"
    printf 'SEEDS=%q\n' "$SEEDS"
    printf 'MODEL_ORDER=%q\n' "$MODEL_ORDER"
    printf 'MODEL_GPUS=%q\n' "$MODEL_GPUS"
    printf 'ADAREASONER_MODEL_PATH=%q\n' "$ADAREASONER_MODEL_PATH"
    printf 'ADAREASONER_MODEL_NAME=%q\n' "$ADAREASONER_MODEL_NAME"
    printf 'QWEN7B_MODEL_PATH=%q\n' "$QWEN7B_MODEL_PATH"
    printf 'QWEN72B_MODEL_PATH=%q\n' "$QWEN72B_MODEL_PATH"
    printf 'BATCH_ALL=%q\n' "$BATCH_ALL"
    printf 'MAX_ATTEMPTS=%q\n' "$MAX_ATTEMPTS"
    printf 'MAX_MODEL_LEN=%q\n' "$MAX_MODEL_LEN"
    printf 'GPU_MEMORY_7B=%q\n' "$GPU_MEMORY_7B"
    printf 'GPU_MEMORY_72B=%q\n' "$GPU_MEMORY_72B"
    printf 'GENERATE_CURVE=%q\n' "$GENERATE_CURVE"
    printf 'EVAL_BACKEND=%q\n' "$EVAL_BACKEND"
    printf 'VLLM_PORT=%q\n' "$VLLM_PORT"
    printf 'VLLM_CONCURRENCY=%q\n' "$VLLM_CONCURRENCY"
    printf 'TOOL_CONCURRENCY=%q\n' "$TOOL_CONCURRENCY"
    printf 'TF_EVAL_PIPELINE=%q\n' "$TF_EVAL_PIPELINE"
    printf 'TF_EVAL_PIPELINE_MAX_ACTIVE=%q\n' "$TF_EVAL_PIPELINE_MAX_ACTIVE"
    printf 'SESSION=%q\n' "$SESSION"
    printf 'START_EPOCH=%q\n' "$START_EPOCH"
} > "$STATE_DIR/config.env"

PANE0="$(tmux new-session -d -P -F '#{pane_id}' -s "$SESSION" -n with_tools)"
printf '%s\n' "$PANE0" > "$STATE_DIR/worker.pane"
PANE1="$(tmux split-window -d -h -P -F '#{pane_id}' -t "$PANE0")"
PANE2="$(tmux split-window -d -v -P -F '#{pane_id}' -t "$PANE0")"
PANE3="$(tmux split-window -d -v -P -F '#{pane_id}' -t "$PANE1")"

tmux set-option -t "$SESSION" remain-on-exit on
tmux set-option -t "$SESSION" history-limit 200000
tmux set-option -t "$SESSION" pane-border-status top
tmux set-option -t "$SESSION" pane-border-format ' #{pane_title} '
tmux set-option -t "$SESSION" status on
tmux set-option -t "$SESSION" status-left ' Ada with-tools '
tmux set-option -t "$SESSION" status-right ' Ctrl-b d: detach | %F %T '

tmux select-pane -t "$PANE0" -T "三模型 TP2 实时日志 | GPU0,3"
tmux select-pane -t "$PANE1" -T "实时总览 + latency"
tmux select-pane -t "$PANE2" -T "工具健康 | GPU1,2"
tmux select-pane -t "$PANE3" -T "工具服务日志"

CMD0="clear; exec bash '$WORKER' '$STATE_DIR'"
CMD1="clear; exec bash '$DASHBOARD' '$STATE_DIR'"
CMD2="while true; do clear; date '+%F %T'; echo; bash '$CHECK_TOOLS' || true; echo; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader; sleep 5; done"
LOGDIR="$REPO/rebuttal_exps/toolserver_logs"
CMD3="mkdir -p '$LOGDIR'; touch '$LOGDIR'/controller.log '$LOGDIR'/point_50002.log '$LOGDIR'/point_50003.log '$LOGDIR'/ocr_50010.log '$LOGDIR'/ocr_50011.log '$LOGDIR'/crop_50012.log; tail -n 20 -F '$LOGDIR'/controller.log '$LOGDIR'/point_50002.log '$LOGDIR'/point_50003.log '$LOGDIR'/ocr_50010.log '$LOGDIR'/ocr_50011.log '$LOGDIR'/crop_50012.log"

tmux send-keys -t "$PANE0" "$CMD0" C-m
tmux send-keys -t "$PANE1" "$CMD1" C-m
tmux send-keys -t "$PANE2" "$CMD2" C-m
tmux send-keys -t "$PANE3" "$CMD3" C-m
tmux select-layout -t "$SESSION:0" tiled >/dev/null

echo "已启动 with-tools 三模型评测: $SESSION"
echo "模型顺序: $MODEL_ORDER"
echo "模型 GPU: $MODEL_GPUS (TP=2)；工具 GPU: 1,2"
echo "评测后端: $EVAL_BACKEND；vLLM port: $VLLM_PORT；LM/工具并发: $VLLM_CONCURRENCY/$TOOL_CONCURRENCY"
echo "流水线: $TF_EVAL_PIPELINE；最大活跃样本: $TF_EVAL_PIPELINE_MAX_ACTIVE"
echo "AdaReasoner seeds: $ADAREASONER_SEEDS；Qwen seeds: $SEEDS"
echo "accuracy/latency 产物: $RESULT_ROOT/with_tools/accuracy_latency/"
echo "状态目录: $STATE_DIR"
echo "重新进入: tmux attach -t $SESSION"
echo "后台分离: Ctrl+B D"

if [[ "${NO_ATTACH:-0}" != "1" ]]; then
    sleep 1
    attach_session
fi
