#!/usr/bin/env bash
set -uo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 <state-dir>" >&2; exit 2; }
STATE_DIR="$1"
[[ -f "$STATE_DIR/config.env" ]] || { echo "ERROR: missing $STATE_DIR/config.env" >&2; exit 2; }

# shellcheck disable=SC1090
source "$STATE_DIR/config.env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$EXPS_ROOT/shared/common.sh"

RUN_ONE="$SCRIPT_DIR/run_one.sh"
CHECK_TOOLS="$SCRIPT_DIR/check_tools.sh"
STATUS_FILE="$STATE_DIR/worker.status"
CHILD_PID=""
CURRENT_STARTED="$(date +%s)"
KEEPALIVE_PID=""
KEEPALIVE_PIDFILE="$STATE_DIR/keepalive.pid"
KEEPALIVE_LOG="$STATE_DIR/keepalive.log"
KEEPALIVE_SIGNAL="$STATE_DIR/keepalive.signal"
mkdir -p "$STATE_DIR/failures"

# ---- GPU keep-alive lifecycle ----
start_keepalive() {
    local keepalive_script="$EXPS_ROOT/shared/gpu_keepalive.py"
    if [[ ! -f "$keepalive_script" ]]; then
        echo "[WARN] gpu_keepalive.py not found at $keepalive_script, skip keep-alive" >&2
        return 0
    fi
    echo "[$(date '+%F %T')] 启动 GPU 保活守护进程 (GPUs=$MODEL_GPUS) ..."
    local conda_python="/data/songmingyang/miniforge3/envs/${CONDA_ENV:-vllm-latest}/bin/python3"
    if [[ ! -x "$conda_python" ]]; then
        echo "[ERROR] conda python not found at $conda_python, skip keep-alive" >&2
        return 0
    fi
    setsid "$conda_python" "$keepalive_script" \
        --gpus "$MODEL_GPUS" \
        --idle-threshold 5.0 \
        --idle-seconds 30 \
        --warm-memory-mb 50 \
        --check-interval 5 \
        --pid-file "$KEEPALIVE_PIDFILE" \
        --signal-file "$KEEPALIVE_SIGNAL" \
        >> "$KEEPALIVE_LOG" 2>&1 &
    KEEPALIVE_PID=$!
    sleep 1
    if ! kill -0 "$KEEPALIVE_PID" 2>/dev/null; then
        echo "[ERROR] GPU keep-alive failed to start, check $KEEPALIVE_LOG" >&2
        KEEPALIVE_PID=""
    else
        echo "[$(date '+%F %T')] GPU 保活已启动 PID=$KEEPALIVE_PID"
    fi
}

stop_keepalive() {
    if [[ -n "$KEEPALIVE_PID" ]] && kill -0 "$KEEPALIVE_PID" 2>/dev/null; then
        echo "[$(date '+%F %T')] 停止 GPU 保活 PID=$KEEPALIVE_PID"
        kill "$KEEPALIVE_PID" 2>/dev/null || true
        wait "$KEEPALIVE_PID" 2>/dev/null || true
    fi
    if [[ -f "$KEEPALIVE_PIDFILE" ]]; then
        local pid
        pid="$(cat "$KEEPALIVE_PIDFILE" 2>/dev/null || true)"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$KEEPALIVE_PIDFILE"
    fi
    rm -f "$KEEPALIVE_SIGNAL"
    KEEPALIVE_PID=""
}

# Ensure keep-alive is always cleaned up on exit
cleanup_keepalive_on_exit() {
    stop_keepalive
}
trap cleanup_keepalive_on_exit EXIT

split_csv "$TASKS"
TASK_LIST=("${CSV_ITEMS[@]}")
split_csv "$SEEDS"
QWEN_SEED_LIST=("${CSV_ITEMS[@]}")
split_csv "$MODEL_ORDER"
MODEL_LIST=("${CSV_ITEMS[@]}")

model_spec() {
    case "$1" in
        adareasoner_randomized_7b)
            MODEL_SIZE_CURRENT=7B
            MODEL_SLUG_CURRENT=adareasoner_randomized_7b
            MODEL_NAME_CURRENT="$ADAREASONER_MODEL_NAME"
            MODEL_PATH_CURRENT="$ADAREASONER_MODEL_PATH"
            ;;
        qwen25vl_7b)
            MODEL_SIZE_CURRENT=7B
            MODEL_SLUG_CURRENT=qwen25vl_7b
            MODEL_NAME_CURRENT=Qwen2.5-VL-7B-Instruct
            MODEL_PATH_CURRENT="$QWEN7B_MODEL_PATH"
            ;;
        qwen25vl_72b)
            MODEL_SIZE_CURRENT=72B
            MODEL_SLUG_CURRENT=qwen25vl_72b
            MODEL_NAME_CURRENT=Qwen2.5-VL-72B-Instruct
            MODEL_PATH_CURRENT="$QWEN72B_MODEL_PATH"
            ;;
        *)
            echo "ERROR: unknown model id: $1" >&2
            return 2
            ;;
    esac
}

write_status() {
    local phase="$1" model_id="${2:--}" model_name="${3:--}" task="${4:--}" seed="${5:--}" message="${6:-}"
    local tmp="$STATUS_FILE.tmp.$$"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$phase" "$model_id" "$model_name" "$task" "$seed" "$MODEL_GPUS" "$CURRENT_STARTED" "$message" > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

stop_child() {
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -TERM -- "-$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    CHILD_PID=""
}

on_signal() {
    write_status interrupted - - - - "received signal; checkpoint preserved"
    stop_child
    exit 130
}
trap on_signal INT TERM HUP

wait_for_tools() {
    while ! bash "$CHECK_TOOLS" --quiet; do
        CURRENT_STARTED="$(date +%s)"
        write_status waiting-tools - - - - "Point/OCR/Crop not ready; retry in 30s"
        echo "[$(date '+%F %T')] 工具服务未完全就绪，30 秒后重试。"
        sleep 30
    done
}

failure_file() {
    printf '%s/failures/%s_%s_seed_%s.count\n' "$STATE_DIR" "$1" "$2" "$3"
}

run_job() {
    local model_id="$1" task="$2" seed="$3"
    local failure count attempt rc
    model_spec "$model_id" || return $?
    failure="$(failure_file "$MODEL_SLUG_CURRENT" "$task" "$seed")"
    count=0
    [[ -s "$failure" ]] && count="$(cat "$failure")"

    while (( count < MAX_ATTEMPTS )); do
        wait_for_tools
        attempt=$((count + 1))
        CURRENT_STARTED="$(date +%s)"
        write_status running "$model_id" "$MODEL_NAME_CURRENT" "$task" "$seed" "attempt $attempt/$MAX_ATTEMPTS"

        echo
        echo "================================================================================"
        echo "model=$MODEL_NAME_CURRENT task=$task seed=$seed"
        echo "mode=with_tools GPU=$MODEL_GPUS TP=2 batch=$BATCH_ALL attempt=$attempt/$MAX_ATTEMPTS"
        echo "start=$(date '+%F %T')"
        echo "================================================================================"

        # Signal keep-alive: eval is starting → do NOT warm GPU
        touch "$KEEPALIVE_SIGNAL" 2>/dev/null || true

        setsid env \
            MODEL_PATH_OVERRIDE="$MODEL_PATH_CURRENT" \
            MODEL_SLUG_OVERRIDE="$MODEL_SLUG_CURRENT" \
            MODEL_DISPLAY_NAME_OVERRIDE="$MODEL_NAME_CURRENT" \
            GPU_7B="$MODEL_GPUS" TP_7B=2 GPU_MEMORY_7B="$GPU_MEMORY_7B" \
            GPU_72B="$MODEL_GPUS" TP_72B=2 GPU_MEMORY_72B="$GPU_MEMORY_72B" \
            BATCH_ALL="$BATCH_ALL" RESUME=1 CONTINUE_ON_ERROR=1 AUTO_START_TOOLS=0 REQUIRE_LATENCY=1 \
            EVAL_BACKEND="${EVAL_BACKEND:-native}" VLLM_PORT="${VLLM_PORT:-8000}" \
            VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-32}" TOOL_CONCURRENCY="${TOOL_CONCURRENCY:-16}" \
            TF_EVAL_TOOL_CONCURRENCY="${TOOL_CONCURRENCY:-16}" \
            TF_EVAL_PIPELINE="${TF_EVAL_PIPELINE:-1}" \
            TF_EVAL_PIPELINE_MAX_ACTIVE="${TF_EVAL_PIPELINE_MAX_ACTIVE:-$((BATCH_ALL * 3))}" \
            bash "$RUN_ONE" "$MODEL_SIZE_CURRENT" with_tools "$task" "$seed" &
        CHILD_PID=$!
        wait "$CHILD_PID"
        rc=$?
        CHILD_PID=""

        # Signal keep-alive: eval finished → keep-alive may resume if GPU idle
        rm -f "$KEEPALIVE_SIGNAL"

        if [[ $rc -eq 0 || $rc -eq 100 ]]; then
            write_status completed "$model_id" "$MODEL_NAME_CURRENT" "$task" "$seed" "rc=$rc"
            echo "完成: $MODEL_NAME_CURRENT/$task/seed_$seed rc=$rc"
            return 0
        fi

        count=$((count + 1))
        echo "$count" > "$failure"
        write_status failed "$model_id" "$MODEL_NAME_CURRENT" "$task" "$seed" "rc=$rc; attempts=$count/$MAX_ATTEMPTS"
        echo "失败: $MODEL_NAME_CURRENT/$task/seed_$seed rc=$rc ($count/$MAX_ATTEMPTS)" >&2
        (( count < MAX_ATTEMPTS )) && sleep 10
    done
    return 1
}

split_csv "$ADAREASONER_SEEDS"
ADAREASONER_SEED_LIST=("${CSV_ITEMS[@]}")
TOTAL_JOBS=0
for model_id in "${MODEL_LIST[@]}"; do
    if [[ "$model_id" == "adareasoner_randomized_7b" ]]; then
        seed_count=${#ADAREASONER_SEED_LIST[@]}
    else
        seed_count=${#QWEN_SEED_LIST[@]}
    fi
    TOTAL_JOBS=$((TOTAL_JOBS + ${#TASK_LIST[@]} * seed_count))
done
JOB_INDEX=0
FAILED_JOBS=0
start_keepalive
write_status starting - - - - "total_jobs=$TOTAL_JOBS"

for model_id in "${MODEL_LIST[@]}"; do
    model_spec "$model_id" || exit $?
    [[ -f "$MODEL_PATH_CURRENT/config.json" ]] || { echo "ERROR: invalid model path: $MODEL_PATH_CURRENT" >&2; exit 2; }
    if [[ "$model_id" == "adareasoner_randomized_7b" ]]; then
        MODEL_SEEDS_CSV="$ADAREASONER_SEEDS"
    else
        MODEL_SEEDS_CSV="$SEEDS"
    fi
    split_csv "$MODEL_SEEDS_CSV"
    MODEL_SEED_LIST=("${CSV_ITEMS[@]}")
    for task in "${TASK_LIST[@]}"; do
        for seed in "${MODEL_SEED_LIST[@]}"; do
            JOB_INDEX=$((JOB_INDEX + 1))
            echo "调度 [$JOB_INDEX/$TOTAL_JOBS]: $MODEL_NAME_CURRENT/$task/seed_$seed"
            run_job "$model_id" "$task" "$seed" || FAILED_JOBS=$((FAILED_JOBS + 1))
        done
    done

    python3 "$EXPS_ROOT/shared/summarize.py" \
        "$RESULT_ROOT/with_tools/$MODEL_SLUG_CURRENT" --seeds "$MODEL_SEEDS_CSV" || true
done

CURVE_RC=0
if (( FAILED_JOBS == 0 )) && [[ "${GENERATE_CURVE:-1}" == "1" ]]; then
    CURVE_ARGS=()
    for model_id in "${MODEL_LIST[@]}"; do
        model_spec "$model_id" || exit $?
        CURVE_ARGS+=(--model "$MODEL_SLUG_CURRENT=$MODEL_NAME_CURRENT")
    done
    echo "正在生成 accuracy/latency CSV、JSON 和 SVG 曲线..."
    python3 "$SCRIPT_DIR/plot_accuracy_latency.py" \
        --result-root "$RESULT_ROOT" \
        --tasks "$TASKS" \
        --require-complete \
        "${CURVE_ARGS[@]}" || CURVE_RC=$?
    if (( CURVE_RC != 0 )); then
        FAILED_JOBS=$((FAILED_JOBS + 1))
        echo "ERROR: accuracy/latency curve generation failed (rc=$CURVE_RC)" >&2
    fi
fi

CURRENT_STARTED="$(date +%s)"
if (( FAILED_JOBS )); then
    write_status done-with-failures - - - - "failed_jobs_or_artifacts=$FAILED_JOBS; total_jobs=$TOTAL_JOBS"
    touch "$STATE_DIR/FINISHED_WITH_FAILURES"
    echo "调度结束但存在失败: total=$TOTAL_JOBS failed_or_artifacts=$FAILED_JOBS time=$(date '+%F %T')" >&2
    exit 1
fi

write_status done - - - - "all $TOTAL_JOBS jobs and artifacts completed"
touch "$STATE_DIR/ALL_DONE"
echo "全部调度和曲线生成完成: total=$TOTAL_JOBS time=$(date '+%F %T')"
exit 0
