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

split_csv "$TASKS"
TASK_LIST=("${CSV_ITEMS[@]}")
split_csv "$SEEDS"
SEED_LIST=("${CSV_ITEMS[@]}")
TOTAL_PER_MODEL=$(( ${#TASK_LIST[@]} * ${#SEED_LIST[@]} ))
split_csv "$ADAREASONER_SEEDS"
ADAREASONER_SEED_LIST=("${CSV_ITEMS[@]}")
TOTAL_ADAREASONER=$(( ${#TASK_LIST[@]} * ${#ADAREASONER_SEED_LIST[@]} ))
split_csv "$MODEL_ORDER"
MODEL_LIST=("${CSV_ITEMS[@]}")
START_EPOCH="${START_EPOCH:-$(date +%s)}"
SESSION="${SESSION:-ada_eval_with_tools}"

model_slug() {
    case "$1" in
        adareasoner_randomized_7b) echo adareasoner_randomized_7b ;;
        qwen25vl_7b) echo qwen25vl_7b ;;
        qwen25vl_72b) echo qwen25vl_72b ;;
        *) echo "$1" ;;
    esac
}

format_duration() {
    local seconds="$1"
    printf '%02d:%02d:%02d' $((seconds / 3600)) $(((seconds % 3600) / 60)) $((seconds % 60))
}

latest_latency() {
    local checkpoint="$1"
    [[ -s "$checkpoint" ]] || { echo '-'; return; }
    tail -n 1 "$checkpoint" | python3 -c '
import json, sys
try:
    latency = json.load(sys.stdin)["results"]["results"]["latency"]
    print("inst={:.2f}s rounds={}".format(
        latency.get("instance_e2e_s", 0), latency.get("round_count", 0)
    ))
except Exception:
    print("legacy/no-latency")
' 2>/dev/null || echo '-'
}

while true; do
    now="$(date +%s)"
    clear
    echo "With-tools 正式矩阵：每个模型双卡 TP=2，模型间串行"
    echo "时间: $(date '+%F %T')    已运行: $(format_duration $((now - START_EPOCH)))"
    echo "模型卡: GPU $MODEL_GPUS (TP=2)    工具卡: GPU 1,2 (Point+OCR)    batch=$BATCH_ALL"
    echo "================================================================================================"
    printf '%-4s %-23s %-13s %-9s %-8s\n' GPU 型号 显存 利用率 角色
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '
        $1==0 || $1==3 {role="Model-TP2"}
        $1==1 || $1==2 {role="Tools"}
        {printf "%-4s %-23s %5s/%-5sMiB %6s%%   %-8s\n",$1,$2,$3,$4,$5,role}'

    echo "================================================================================================"
    if bash "$SCRIPT_DIR/check_tools.sh" --quiet 2>/dev/null; then
        echo "工具状态: READY — Controller + Point×2 + OCR×2 + Crop"
    else
        echo "工具状态: NOT READY — worker 会暂停并每 30 秒重试，不会在工具缺失时产生正式结果"
    fi

    echo "================================================================================================"
    echo "完整 job 进度（完整 task×seed 通过校验后 +1）"
    printf '%-30s %-10s %-34s\n' 模型 完成job 进度
    for model_id in "${MODEL_LIST[@]}"; do
        slug="$(model_slug "$model_id")"
        expected_jobs="$TOTAL_PER_MODEL"
        [[ "$model_id" == "adareasoner_randomized_7b" ]] && expected_jobs="$TOTAL_ADAREASONER"
        done_count=0
        for task_name in "${TASK_LIST[@]}"; do
            if [[ "$model_id" == "adareasoner_randomized_7b" ]]; then
                model_seeds=("${ADAREASONER_SEED_LIST[@]}")
            else
                model_seeds=("${SEED_LIST[@]}")
            fi
            for model_seed in "${model_seeds[@]}"; do
                [[ -f "$RESULT_ROOT/with_tools/$slug/$task_name/seed_$model_seed/DONE.json" ]] \
                    && done_count=$((done_count + 1))
            done
        done
        width=28
        filled=$((done_count * width / expected_jobs))
        (( filled > width )) && filled=$width
        bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' $((width-filled)) '' | tr ' ' '-')"
        printf '%-30s %3d/%-6d [%s]\n' "$slug" "$done_count" "$expected_jobs" "$bar"
    done

    echo "================================================================================================"
    if [[ -s "$STATE_DIR/worker.status" ]]; then
        IFS=$'\t' read -r phase model_id model_name task seed gpus started message < "$STATE_DIR/worker.status" || true
        echo "当前: status=$phase model=$model_name task=$task seed=$seed GPU=$gpus"
        if [[ "$phase" == "running" ]]; then
            slug="$(model_slug "$model_id")"
            run_dir="$RESULT_ROOT/with_tools/$slug/$task/seed_$seed"
            samples=0
            [[ -s "$run_dir/ckpt.jsonl" ]] && samples="$(grep -cve '^[[:space:]]*$' "$run_dir/ckpt.jsonl" || true)"
            expected="$(python3 - "$TASK_MATRIX" "$task" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]].get("expected_samples", "?"))
PY
)"
            echo "样本级进度: $samples/$expected    最新 instance latency: $(latest_latency "$run_dir/ckpt.jsonl")"
            echo "输出: $run_dir/{ckpt.jsonl,latency.jsonl,latency_summary.json,result.jsonl}"
        else
            echo "详情: $message"
        fi
    else
        echo "worker 尚未启动"
    fi

    echo "================================================================================================"
    echo "操作: Ctrl+B + 方向键切 pane；Ctrl+B D 分离；tmux attach -t $SESSION 重连"
    sleep 3
done
