#!/usr/bin/env bash
# ============================================================================
# vllm_serve.sh — vLLM OpenAI-兼容 API Server 生命周期管理
#
# 用法:
#   source vllm_serve.sh
#   vllm_start   /path/to/model 0,3 2 8000 "--enforce-eager --seed 42"
#   vllm_wait    8000 300
#   vllm_health  8000
#   vllm_stop    8000
# ============================================================================
set -euo pipefail

VLLM_CONDA="${VLLM_CONDA:-vllm-latest}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-/data/songmingyang/code/reasoning/AdaReasoner-rebuttal/rebuttal_exps/vllm_server_logs}"

# ---- 启动 vLLM API Server ----
vllm_start() {
    local model_path="$1"
    local gpu_ids="$2"
    local tp_size="$3"
    local port="$4"
    local extra_args="${5:-}"

    local logfile="$VLLM_LOG_DIR/vllm_serve_${port}.log"
    mkdir -p "$VLLM_LOG_DIR"

    if vllm_health "$port" 2>/dev/null; then
        echo "  -> vLLM server on :$port already running (GPU=$gpu_ids)"
        return 0
    fi

    echo ""
    echo "--- 启动 vLLM API Server ---"
    echo "  Model:     $model_path"
    echo "  GPUs:      $gpu_ids"
    echo "  TP size:   $tp_size"
    echo "  Port:      $port"
    echo "  Extra:     $extra_args"
    echo "  Log:       $logfile"
    echo ""

    source /data/songmingyang/miniforge3/etc/profile.d/conda.sh
    conda activate "$VLLM_CONDA"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

    local served_name
    served_name="$(basename "$model_path")"

    nohup env CUDA_VISIBLE_DEVICES="$gpu_ids" \
        python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --served-model-name "$served_name" \
        --tensor-parallel-size "$tp_size" \
        --gpu-memory-utilization 0.90 \
        --max-model-len 8192 \
        --port "$port" \
        --host 0.0.0.0 \
        $extra_args \
        > "$logfile" 2>&1 &

    local pid=$!
    echo "  -> vLLM server PID=$pid  (port=$port, GPU=$gpu_ids)"
    echo "$pid" > "$VLLM_LOG_DIR/vllm_serve_${port}.pid"
}

# ---- 等待 server 就绪 ----
vllm_wait() {
    local port="$1"
    local timeout="${2:-600}"
    local interval=2

    echo -n "  -> 等待 vLLM server :$port 就绪..."
    local elapsed=0
    while (( elapsed < timeout )); do
        if vllm_health "$port" 2>/dev/null; then
            echo " ✓ (${elapsed}s)"
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
        if (( elapsed % 10 == 0 )); then
            echo -n "."
        fi
    done
    echo " ✗ 超时 (${timeout}s)"
    return 1
}

# ---- 健康检查 ----
vllm_health() {
    local port="$1"
    local health_url="http://localhost:${port}/health"
    curl -fsS -m 3 "$health_url" >/dev/null 2>&1
}

# ---- 检查模型是否已加载 ----
vllm_models_ready() {
    local port="$1"
    local models_url="http://localhost:${port}/v1/models"
    curl -fsS -m 3 "$models_url" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); exit(0 if len(d.get('data',[]))>0 else 1)" 2>/dev/null
}

# ---- 获取 GPU 显存使用 ----
vllm_gpu_info() {
    local gpu_ids="$1"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader --id="$gpu_ids" 2>/dev/null | cat
}

# ---- 停止 server ----
vllm_stop() {
    local port="$1"
    local pidfile="$VLLM_LOG_DIR/vllm_serve_${port}.pid"

    echo -n "  -> 停止 vLLM server :$port..."

    # 优雅关闭
    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile" 2>/dev/null || true)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            # 等待最多 30 秒
            for i in $(seq 1 15); do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 2
            done
            # 强制 kill
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$pidfile"
    fi

    # 备用：通过 fuser 杀端口
    local pids
    pids=$(fuser "$port/tcp" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        kill -TERM $pids 2>/dev/null || true
        sleep 3
        pids=$(fuser "$port/tcp" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            kill -9 $pids 2>/dev/null || true
        fi
    fi

    echo " done"
}

# ---- 显示 server 状态 ----
vllm_status() {
    local port="$1"
    echo "--- vLLM Server :$port ---"
    if vllm_health "$port" 2>/dev/null; then
        echo "  Status: RUNNING"
        echo "  Models:"
        curl -sS -m 3 "http://localhost:${port}/v1/models" 2>/dev/null | \
            python3 -c "import json,sys; [print(f'    - {m[\"id\"]}') for m in json.load(sys.stdin).get('data',[])]" 2>/dev/null || echo "    (query failed)"
    else
        echo "  Status: NOT RUNNING"
    fi
}
