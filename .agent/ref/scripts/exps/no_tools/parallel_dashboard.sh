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
MATRIX_VALIDATOR="$SCRIPT_DIR/validate_matrix.py"
VALIDATE_RUN="$EXPS_ROOT/shared/validate_run.py"

split_csv "$TASKS"
TASK_LIST=("${CSV_ITEMS[@]}")
split_csv "$SEEDS"
SEED_LIST=("${CSV_ITEMS[@]}")
TOTAL_PER_MODEL=$(( ${#TASK_LIST[@]} * ${#SEED_LIST[@]} ))
TOTAL_ALL=$(( TOTAL_PER_MODEL * 4 ))
START_EPOCH="${START_EPOCH:-$(date +%s)}"

all_lanes_finished() {
    [[ -f "$STATE_DIR/lanes/lane0.finished" ]] &&
    [[ -f "$STATE_DIR/lanes/lane1.finished" ]] &&
    [[ -f "$STATE_DIR/lanes/lane2.finished" ]]
}

format_duration() {
    local seconds="$1"
    printf '%02d:%02d:%02d' $((seconds / 3600)) $(((seconds % 3600) / 60)) $((seconds % 60))
}

read_counts() {
    python3 "$MATRIX_VALIDATOR" \
        --result-root "$RESULT_ROOT" \
        --task-matrix "$TASK_MATRIX" \
        --tasks "$TASKS" \
        --seeds "$SEEDS" \
        --validate-run "$VALIDATE_RUN" \
        --count-only | python3 -c '
import json,sys
x=json.load(sys.stdin)
print(x["counts"]["3b"], x["counts"]["7b"], x["counts"]["32b"], x["counts"]["72b"], x["complete"], len(x["invalid_markers"]))
'
}

failure_attempts() {
    python3 - "$STATE_DIR/failures" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
total=0
for path in root.glob("*.count"):
    try:
        total += int(path.read_text().strip())
    except (OSError, ValueError):
        pass
print(total)
PY
}

print_dashboard() {
    local now elapsed count3 count7 count32 count72 total_done invalid_markers total_failed
    now="$(date +%s)"
    elapsed=$((now - START_EPOCH))
    read -r count3 count7 count32 count72 total_done invalid_markers <<< "$(read_counts)"
    total_failed="$(failure_attempts)"

    clear
    echo "AdaReasoner no-tools 四卡动态评测"
    echo "时间: $(date '+%F %T')    已运行: $(format_duration "$elapsed")    tmux: ada_eval_no_tools"
    echo "调度: GPU0,1=72B TP2；GPU2/GPU3=32B/7B/3B；空闲 GPU 动态接力"
    echo "================================================================================================"
    echo "GPU 实时状态"
    printf '%-4s %-24s %-12s %-12s %-8s %-8s\n' "GPU" "型号" "显存使用" "显存总量" "利用率" "温度"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' '{printf "%-4s %-24s %8s MiB %8s MiB %6s%% %6s C\n",$1,$2,$3,$4,$5,$6}'

    echo "================================================================================================"
    echo "完整 job 进度（只统计当前精确路径且 validated=true 的 DONE.json）"
    printf '%-7s %-12s %-42s\n' "模型" "完整job" "进度"
    for item in "3B:$count3" "7B:$count7" "32B:$count32" "72B:$count72"; do
        size="${item%%:*}"
        done_count="${item##*:}"
        width=30
        filled=$(( done_count * width / TOTAL_PER_MODEL ))
        empty=$((width - filled))
        bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' "$empty" '' | tr ' ' '-')"
        printf '%-7s %3d/%-8d [%s]\n' "$size" "$done_count" "$TOTAL_PER_MODEL" "$bar"
    done
    echo "完整总进度: $total_done/$TOTAL_ALL jobs"
    echo "失败尝试次数: $total_failed；无效 DONE 标志: $invalid_markers"

    echo "================================================================================================"
    echo "正在运行的样本级进度（每 3 秒刷新）"
    printf '%-8s %-24s %-13s %-7s %-12s\n' "GPU" "当前 job" "样本" "百分比" "状态"
    shopt -s nullglob
    status_files=("$STATE_DIR"/workers/*.status)
    if ((${#status_files[@]} == 0)); then
        echo "worker 尚未启动"
    else
        for file in "${status_files[@]}"; do
            IFS=$'\t' read -r phase model task seed gpus pid started message < "$file" || true
            sample_progress="-"
            percent="-"
            job="-"
            if [[ "$phase" == "running" && "$model" != "-" ]]; then
                run_dir="$RESULT_ROOT/no_tools/qwen25vl_${model,,}/$task/seed_$seed"
                completed_samples=0
                [[ -s "$run_dir/ckpt.jsonl" ]] && completed_samples="$(grep -cve '^[[:space:]]*$' "$run_dir/ckpt.jsonl" || true)"
                expected="$(python3 - "$TASK_MATRIX" "$task" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]].get("expected_samples", 0))
PY
)"
                sample_progress="$completed_samples/$expected"
                if [[ "$expected" =~ ^[0-9]+$ ]] && (( expected > 0 )); then
                    percent="$(awk -v n="$completed_samples" -v d="$expected" 'BEGIN {printf "%.1f%%", 100*n/d}')"
                fi
                job="$model/$task/s$seed"
            fi
            printf '%-8s %-24s %-13s %-7s %-12s\n' "$gpus" "$job" "$sample_progress" "$percent" "$phase"
        done
    fi
    shopt -u nullglob

    echo "================================================================================================"
    echo "操作: Ctrl+B 后按方向键切换 pane；Ctrl+B D 后台分离；tmux attach -t ada_eval_no_tools 重连"
    echo "结果目录: $RESULT_ROOT/no_tools"
}

while true; do
    print_dashboard
    if all_lanes_finished; then
        echo
        echo "所有 GPU lane 已退出，正在逐项执行最终 96-job 完整验证..."
        rm -f "$STATE_DIR/ALL_DONE" "$STATE_DIR/INCOMPLETE"
        if python3 "$MATRIX_VALIDATOR" \
            --result-root "$RESULT_ROOT" \
            --task-matrix "$TASK_MATRIX" \
            --tasks "$TASKS" \
            --seeds "$SEEDS" \
            --validate-run "$VALIDATE_RUN" \
            --report "$STATE_DIR/final_validation.json" >/dev/null; then
            bash "$EXPS_ROOT/summarize_all.sh" >/dev/null 2>&1 || true
            touch "$STATE_DIR/ALL_DONE"
            print_dashboard
            echo
            echo "调度成功：全部 $TOTAL_ALL 个 job 已通过最终验证。"
            exit 0
        fi
        touch "$STATE_DIR/INCOMPLETE"
        print_dashboard
        echo
        echo "调度未完成：存在缺失或无效 job；详见 $STATE_DIR/final_validation.json" >&2
        exit 1
    fi
    sleep 3
done
