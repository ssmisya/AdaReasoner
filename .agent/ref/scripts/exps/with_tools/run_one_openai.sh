#!/usr/bin/env bash
# ============================================================================
# run_one_openai.sh — 单任务评测入口 (OpenaiModels → 本地 vLLM API Server)
#
# 与 run_one.sh 的区别:
#   1. 使用 model: openai (OpenaiModels) 而非 vllm_models (嵌入式 VllmModels)
#   2. 不分配 GPU (CUDA_VISIBLE_DEVICES="")，GPU 由 vLLM API server 独占
#   3. 通过 VLLM_BASE_URL / VLLM_API_KEY 连接共享 vLLM server
#   4. 使用 eval_entry_openai.py (计时针对 OpenaiModels)
#   5. 复用相同的 DONE.json / validate / resume 逻辑
#
# 环境变量:
#   VLLM_PORT           vLLM server 端口 (默认 8000)
#   RESULT_ROOT         结果根目录
#   CONTROLLER_ADDR     工具 controller 地址
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_DIR="$(cd "$SCRIPT_DIR/../shared" && pwd)"
source "$SHARED_DIR/common.sh"

usage() {
    echo "Usage: $0 <3B|7B|32B|72B> <no_tools|with_tools> <task> <seed>" >&2
    exit 2
}

[[ $# -eq 4 ]] || usage
MODEL_SIZE="${1^^}"
MODE="$2"
TASK="$3"
SEED="$4"

case "$MODEL_SIZE" in 3B|7B|32B|72B) ;; *) usage ;; esac
case "$MODE" in no_tools|with_tools) ;; *) usage ;; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "ERROR: seed must be an integer" >&2; exit 2; }

# ---- 验证 task 存在于 task_matrix ----
python3 - "$TASK_MATRIX" "$TASK" <<'PY'
import json,sys
matrix=json.load(open(sys.argv[1], encoding="utf-8"))
if sys.argv[2] not in matrix:
    raise SystemExit(f"Unknown task: {sys.argv[2]}")
PY

MODEL_PATH="${MODEL_PATH_OVERRIDE:-$(model_path "$MODEL_SIZE")}"
MODEL_SLUG="${MODEL_SLUG_OVERRIDE:-qwen25vl_${MODEL_SIZE,,}}"
MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME_OVERRIDE:-Qwen2.5-VL-${MODEL_SIZE}}"
SERVED_MODEL_NAME="$(basename "$MODEL_PATH")"
TOOLS="$(tool_selection "$TASK")"

IF_USE_TOOL=false
[[ "$MODE" == "with_tools" ]] && IF_USE_TOOL=true
IF_RANDOMIZE_TOOL="${IF_RANDOMIZE_TOOL:-false}"

USE_STOCHASTIC=False
if python3 - "$TASK_MATRIX" "$TASK" <<'PY'
import json,sys
matrix=json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if float(matrix[sys.argv[2]]["task_config"]["generation_config"].get("temperature", 0)) > 0 else 1)
PY
then
    USE_STOCHASTIC=True
fi

TEMPERATURE="$(python3 - "$TASK_MATRIX" "$TASK" <<'PY'
import json,sys
matrix=json.load(open(sys.argv[1], encoding="utf-8"))
tc=matrix[sys.argv[2]]["task_config"]["generation_config"]
print(float(tc.get("temperature",0)))
PY
)"

require_dir "$MODEL_PATH"
require_file "$MODEL_PATH/config.json"
require_file "$SCRIPT_DIR/eval_entry_openai.py"
require_file "$TASK_MATRIX"

# ---- 路径 & 锁 ----
# Keep server checkpoints separate from native checkpoints: their model_name
# and request protocols are intentionally different.
SERVER_RUN_SLUG="${SERVER_RUN_SLUG_OVERRIDE:-${MODEL_SLUG}_openai}"
if [[ "$IF_RANDOMIZE_TOOL" == "true" ]]; then
    SERVER_RUN_SLUG="${SERVER_RUN_SLUG}_randomized"
fi
RUN_DIR="$RESULT_ROOT/$MODE/$SERVER_RUN_SLUG/$TASK/seed_$SEED"
LOCK_DIR="$RESULT_ROOT/.job_locks/$MODE/$SERVER_RUN_SLUG"
mkdir -p "$LOCK_DIR"
if [[ "${JOB_LOCK_HELD:-0}" != "1" ]]; then
    exec 9>"$LOCK_DIR/${TASK}_seed_${SEED}.lock"
    if ! flock -n 9; then
        echo "BUSY: another process owns $MODEL_SIZE/$TASK/seed_$SEED" >&2
        exit 101
    fi
fi

# ---- 跳过已完成 / Resume ----
if [[ -f "$RUN_DIR/DONE.json" && "${FORCE:-0}" != "1" ]]; then
    if python3 "$SHARED_DIR/validate_run.py" \
        --run-dir "$RUN_DIR" \
        --task "$TASK" \
        --task-matrix "$TASK_MATRIX" \
        --model-path "$MODEL_PATH" \
        --seed "$SEED"; then
        echo "SKIP: validated completed run exists: $RUN_DIR/DONE.json"
        exit 100
    fi
    echo "ERROR: existing DONE validation failed; use FORCE=1 to archive and rerun: $RUN_DIR" >&2
    exit 1
fi

RESUMED_SAMPLES=0
if [[ -d "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    if [[ "${FORCE:-0}" == "1" ]]; then
        BACKUP="${RUN_DIR}.bak_$(date +%Y%m%d_%H%M%S)"
        mv "$RUN_DIR" "$BACKUP"
        echo "  -> Archived previous run to $BACKUP"
    elif [[ "${RESUME:-1}" == "1" ]]; then
        if [[ -s "$RUN_DIR/ckpt.jsonl" ]]; then
            RESUMED_SAMPLES="$(grep -cve '^[[:space:]]*$' "$RUN_DIR/ckpt.jsonl" || true)"
            echo "  -> [RESUME] preserving checkpoint with $RESUMED_SAMPLES samples"
        else
            echo "  -> [RESUME] no usable checkpoint; restarting this incomplete job"
            rm -f "$RUN_DIR/result.jsonl" "$RUN_DIR/timing.json" "$RUN_DIR/exit_code.txt"
        fi
        rm -f "$RUN_DIR/DONE.json" "$RUN_DIR"/DONE.*.tmp
    else
        echo "ERROR: incomplete run directory exists: $RUN_DIR" >&2
        echo "Set FORCE=1 to archive it, or RESUME=1 to continue from ckpt.jsonl." >&2
        exit 1
    fi
fi
mkdir -p "$RUN_DIR/middle_images"

# ---- 验证工具服务可用 ----
if [[ "$MODE" == "with_tools" ]]; then
    if ! curl -fsS -m 3 -X POST "$CONTROLLER_ADDR/list_models" >/dev/null 2>&1; then
        echo "ERROR: tool controller not reachable at $CONTROLLER_ADDR" >&2
        echo "  Please start tools first: bash .agent/ref/scripts/start_tools.sh" >&2
        exit 1
    fi
fi

# ---- 生成 OpenaiModels 配置 ----
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-32}"
VLLM_REQUEST_TIMEOUT="${VLLM_REQUEST_TIMEOUT:-300}"
TOOL_CONCURRENCY="${TOOL_CONCURRENCY:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$VLLM_CONCURRENCY}"
REQUEST_CONCURRENCY="${TF_EVAL_REQUEST_CONCURRENCY:-$VLLM_CONCURRENCY}"
PIPELINE_ENABLED="${TF_EVAL_PIPELINE:-1}"
PIPELINE_MAX_ACTIVE="${TF_EVAL_PIPELINE_MAX_ACTIVE:-$((EVAL_BATCH_SIZE * 3))}"
CONFIG="$RUN_DIR/config.yaml"
cat > "$CONFIG" <<YAML
model_args:
  model: openai
  model_args: pretrained=$SERVED_MODEL_NAME,temperature=$TEMPERATURE,limit_mm_per_prompt=10,max_retry=5,tensor_parallel=1,seed=$SEED,request_concurrency=$REQUEST_CONCURRENCY,request_timeout=$VLLM_REQUEST_TIMEOUT
  batch_size: $EVAL_BATCH_SIZE
  max_rounds: 6
  model_mode: general
task_args:
  task_name: $TASK
  tool_selection: $TOOLS
  resume_from_ckpt:
    $TASK: $RUN_DIR/ckpt.jsonl
  save_to_ckpt:
    $TASK: $RUN_DIR/ckpt.jsonl
  middle_images_save_dir:
    $TASK: $RUN_DIR/middle_images
script_args:
  verbosity: INFO
  output_path: $RUN_DIR/result.jsonl
  controller_addr: $CONTROLLER_ADDR
  if_use_tool: $IF_USE_TOOL
  if_randomize_tool: $IF_RANDOMIZE_TOOL
YAML

# ---- 元数据 ----
python3 - "$RUN_DIR/run_metadata.json" <<PY
import json,platform,time
payload={
  "model_size":"$MODEL_SIZE",
  "model_path":"$MODEL_PATH",
  "served_model_name":"$SERVED_MODEL_NAME",
  "mode":"$MODE",
  "task":"$TASK",
  "seed":$SEED,
  "vllm_port":$VLLM_PORT,
  "backend":"openai_server",
  "eval_batch_size":$EVAL_BATCH_SIZE,
  "lm_request_concurrency":$REQUEST_CONCURRENCY,
  "tool_concurrency":$TOOL_CONCURRENCY,
  "pipeline_enabled":"$PIPELINE_ENABLED",
  "pipeline_max_active":$PIPELINE_MAX_ACTIVE,
  "resumed_samples":$RESUMED_SAMPLES,
  "tool_selection":"$TOOLS",
  "if_randomize_tool":bool("$IF_RANDOMIZE_TOOL".lower()=="true"),
  "stochastic_decoding":$USE_STOCHASTIC,
  "temperature":$TEMPERATURE,
  "controller_addr":"$CONTROLLER_ADDR",
  "created_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "hostname":platform.node(),
}
json.dump(payload,open("$RUN_DIR/run_metadata.json","w"),indent=2)
PY

# ---- 环境 ----
source /data/songmingyang/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_CACHE
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONSAFEPATH=1
export PYTHONHASHSEED="$SEED"
if [[ "$TASK" == "hrbench" ]]; then
    load_yunwu_env
fi
# HRBench explicitly proxies only its YunwuAI request. Keep local vLLM and
# tool-controller traffic outside every process-wide proxy.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,::1,.svc,.cluster.local,10.0.0.0/8"
export no_proxy="$NO_PROXY"

# ---- OpenAI API 配置 (连接本地 vLLM) ----
export CUDA_VISIBLE_DEVICES=""          # eval 进程不占 GPU
export VLLM_API_KEY="${VLLM_API_KEY:-vllm}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
export TF_EVAL_REQUEST_CONCURRENCY="$REQUEST_CONCURRENCY"
export TF_EVAL_TOOL_CONCURRENCY="$TOOL_CONCURRENCY"
export TF_EVAL_TOOL_TIMEOUT="${TOOL_TIMEOUT:-300}"
# While some samples wait for tools, ready samples keep feeding vLLM. The
# active window is bounded because each sample retains conversation images.
export TF_EVAL_PIPELINE="$PIPELINE_ENABLED"
export TF_EVAL_PIPELINE_MAX_ACTIVE="$PIPELINE_MAX_ACTIVE"

cd "$REPO"

# ---- 时延日志 ----
unset E3_LATENCY_LOG || true
if [[ "$MODE" == "with_tools" ]]; then
    export E3_LATENCY_LOG="$RUN_DIR/stage_latency.json"
fi

echo "============================================================"
echo "model=$MODEL_DISPLAY_NAME mode=$MODE task=$TASK seed=$SEED (shared vLLM server backend)"
echo "vllm=$VLLM_BASE_URL model=$SERVED_MODEL_NAME lm_concurrency=$TF_EVAL_REQUEST_CONCURRENCY tool_concurrency=$TOOL_CONCURRENCY"
echo "pipeline=$TF_EVAL_PIPELINE max_active=$TF_EVAL_PIPELINE_MAX_ACTIVE"
echo "result=$RUN_DIR"
echo "start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

set +e
python "$SCRIPT_DIR/eval_entry_openai.py" \
    --config "$CONFIG" \
    --task-matrix "$TASK_MATRIX" \
    --timing-output "$RUN_DIR/timing.json" \
    --seed "$SEED" \
    2>&1 | tee "$RUN_DIR/run.log"
STATUS=${PIPESTATUS[0]}
set -e

echo "$STATUS" > "$RUN_DIR/exit_code.txt"
if [[ "$STATUS" -ne 0 ]]; then
    echo "FAILED: model=$MODEL_SIZE mode=$MODE task=$TASK seed=$SEED" >&2
    exit "$STATUS"
fi

# ---- 时延汇总与验证 ----
LATENCY_ARGS=()
if [[ "${REQUIRE_LATENCY:-1}" == "1" ]]; then
    LATENCY_ARGS+=(--require-complete)
fi
python3 "$SHARED_DIR/summarize_latency.py" \
    --checkpoint "$RUN_DIR/ckpt.jsonl" \
    --output-jsonl "$RUN_DIR/latency.jsonl" \
    --summary "$RUN_DIR/latency_summary.json" \
    "${LATENCY_ARGS[@]}"

python3 "$SHARED_DIR/validate_run.py" \
    --run-dir "$RUN_DIR" \
    --task "$TASK" \
    --task-matrix "$TASK_MATRIX" \
    --model-path "$MODEL_PATH" \
    --seed "$SEED"

echo "DONE: $RUN_DIR"
