#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/data/songmingyang/code/reasoning/AdaReasoner-rebuttal}"
EXPS_ROOT="$REPO/.agent/ref/scripts/exps"
SHARED_DIR="$EXPS_ROOT/shared"
TASK_MATRIX="$SHARED_DIR/task_matrix.json"
RESULT_ROOT="${RESULT_ROOT:-$REPO/rebuttal_exps/qwen25vl_eval}"
CONDA_ENV="${CONDA_ENV:-vllm-latest}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$REPO/configs/accelerate.yaml}"
CONTROLLER_ADDR="${CONTROLLER_ADDR:-http://127.0.0.1:21112}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/songmingyang/data/hf_datasets}"
SEEDS="${SEEDS:-42,1234,2026}"
TASKS="${TASKS:-vsp,vspo,jigsaw_coco,jigsaw_blink,vstar,web_guichat,webmmu,hrbench}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
YUNWU_ENV_FILE="${YUNWU_ENV_FILE:-$EXPS_ROOT/.env}"

load_yunwu_env() {
    if [[ -f "$YUNWU_ENV_FILE" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "$YUNWU_ENV_FILE"
        set +a
    fi
    : "${OPENAI_API_KEY:?ERROR: OPENAI_API_KEY is required in $YUNWU_ENV_FILE or the environment}"
    : "${YUNWU_PROXY_URL:?ERROR: YUNWU_PROXY_URL is required in $YUNWU_ENV_FILE or the environment}"
    export OPENAI_API_URL="${OPENAI_API_URL:-https://yunwu.ai/v1}"
    export MODEL_VERSION="${MODEL_VERSION:-gpt-4o-2024-11-20}"
}

model_path() {
    local size="${1^^}"
    printf '/data/songmingyang/models/baselines/Qwen2.5-VL-%s-Instruct\n' "$size"
}

tensor_parallel() {
    local size="${1^^}"
    local override="TP_${size}"
    if [[ -n "${!override:-}" ]]; then
        printf '%s\n' "${!override}"
        return
    fi
    case "$size" in
        7B|72B) printf '2\n' ;;
        *) printf '1\n' ;;
    esac
}

gpu_devices() {
    local size="${1^^}"
    local override="GPU_${size}"
    if [[ -n "${!override:-}" ]]; then
        printf '%s\n' "${!override}"
        return
    fi
    case "$size" in
        7B) printf '%s\n' "${GPU_7B:-0,3}" ;;
        72B) printf '%s\n' "${GPU_72B:-0,3}" ;;
        *) printf '%s\n' "${GPU_SINGLE:-0}" ;;
    esac
}

batch_size() {
    local size="${1^^}"
    local override="BATCH_${size}"
    case "$size" in 3B|7B|32B|72B) ;; *) return 1 ;; esac
    printf '%s\n' "${!override:-${BATCH_ALL:-64}}"
}

gpu_memory_utilization() {
    local size="${1^^}"
    local override="GPU_MEMORY_${size}"
    if [[ -n "${!override:-}" ]]; then
        printf '%s\n' "${!override}"
        return
    fi
    case "$size" in
        72B) printf '0.95\n' ;;
        32B) printf '0.92\n' ;;
        *) printf '%s\n' "${GPU_MEMORY_SMALL:-0.85}" ;;
    esac
}

tool_selection() {
    local task="$1"
    python3 - "$TASK_MATRIX" "$task" <<'PY'
import json,sys
matrix=json.load(open(sys.argv[1], encoding="utf-8"))
print(matrix[sys.argv[2]]["tool_selection"])
PY
}

split_csv() {
    local value="$1"
    local old_ifs="$IFS"
    IFS=',' read -ra CSV_ITEMS <<< "$value"
    IFS="$old_ifs"
}

require_file() {
    [[ -f "$1" ]] || { echo "ERROR: file not found: $1" >&2; return 1; }
}

require_dir() {
    [[ -d "$1" ]] || { echo "ERROR: directory not found: $1" >&2; return 1; }
}
