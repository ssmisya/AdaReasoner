#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$EXPS_ROOT/shared/common.sh"

SESSION="${TMUX_SESSION:-ada_eval_no_tools}"
WORKER="$SCRIPT_DIR/parallel_worker.sh"
DASHBOARD="$SCRIPT_DIR/parallel_dashboard.sh"
PREFLIGHT="$SCRIPT_DIR/validate_parallel.sh"

attach_session() {
    if [[ -n "${TMUX:-}" ]]; then
        tmux switch-client -t "$SESSION"
    else
        tmux attach-session -t "$SESSION"
    fi
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
    EXISTING_STATE="$(readlink -f "$RESULT_ROOT/.parallel_state/current" 2>/dev/null || true)"
    ACTIVE_PATTERN="parallel_(worker|dashboard)\.sh.*${EXISTING_STATE//\//\\/}"
    if [[ -n "$EXISTING_STATE" ]] && pgrep -af "$ACTIVE_PATTERN" >/dev/null 2>&1; then
        echo "tmux session '$SESSION' 正在执行评测，直接进入。"
        attach_session
        exit 0
    fi
    echo "检测到无活动 worker/dashboard 的旧 tmux session，正在清理后恢复评测。"
    tmux kill-session -t "$SESSION"
fi

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    bash "$PREFLIGHT"
fi

START_EPOCH="$(date +%s)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
STATE_ROOT="$RESULT_ROOT/.parallel_state"
STATE_DIR="$STATE_ROOT/$RUN_ID"
mkdir -p "$STATE_DIR/workers" "$STATE_DIR/failures" "$STATE_DIR/lanes"
ln -sfn "$STATE_DIR" "$STATE_ROOT/current"

{
    printf 'REPO=%q\n' "$REPO"
    printf 'RESULT_ROOT=%q\n' "$RESULT_ROOT"
    printf 'CONDA_ENV=%q\n' "$CONDA_ENV"
    printf 'TASKS=%q\n' "$TASKS"
    printf 'SEEDS=%q\n' "$SEEDS"
    printf 'MAX_MODEL_LEN=%q\n' "$MAX_MODEL_LEN"
    printf 'BATCH_ALL=%q\n' "${BATCH_ALL:-64}"
    printf 'MAX_ATTEMPTS=%q\n' "${MAX_ATTEMPTS:-2}"
    printf 'START_EPOCH=%q\n' "$START_EPOCH"
} > "$STATE_DIR/config.env"

PANE0="$(tmux new-session -d -P -F '#{pane_id}' -s "$SESSION" -n no_tools)"
PANE1="$(tmux split-window -d -h -P -F '#{pane_id}' -t "$PANE0")"
PANE2="$(tmux split-window -d -v -P -F '#{pane_id}' -t "$PANE0")"
PANE3="$(tmux split-window -d -v -P -F '#{pane_id}' -t "$PANE1")"

tmux set-option -t "$SESSION" remain-on-exit on
tmux set-option -t "$SESSION" history-limit 200000
tmux set-option -t "$SESSION" pane-border-status top
tmux set-option -t "$SESSION" pane-border-format ' #{pane_title} '
tmux set-option -t "$SESSION" status on
tmux set-option -t "$SESSION" status-left ' Ada no-tools '
tmux set-option -t "$SESSION" status-right ' Ctrl-b d: detach | %F %T '

tmux select-pane -t "$PANE0" -T "72B primary | GPU 0,1"
tmux select-pane -t "$PANE1" -T "dynamic lane | GPU 2 -> GPU 2,3"
tmux select-pane -t "$PANE2" -T "dynamic lane | GPU 3"
tmux select-pane -t "$PANE3" -T "实时总览"

CMD0="clear; bash '$WORKER' 72-primary 72b 0,1 '$STATE_DIR' || true; if [[ ! -f '$STATE_DIR/workers/single-gpu2.done' || ! -f '$STATE_DIR/workers/single-gpu3.done' ]]; then echo '72B worker 已退出，GPU0/1 转入单卡队列抢剩余任务'; bash '$WORKER' steal-gpu0 single 0 '$STATE_DIR' & p0=\$!; bash '$WORKER' steal-gpu1 single 1 '$STATE_DIR' & p1=\$!; wait \$p0 || true; wait \$p1 || true; fi; touch '$STATE_DIR/lanes/lane0.finished'; echo 'GPU0/1 lane finished'; exec bash"
CMD1="clear; bash '$WORKER' single-gpu2 single 2 '$STATE_DIR' || true; while [[ ! -f '$STATE_DIR/workers/single-gpu3.finished' ]]; do sleep 5; done; if [[ ! -f '$STATE_DIR/workers/72-primary.done' ]]; then echo '单卡初始 worker 已退出，GPU2/3 组成第二个 72B TP2 worker'; bash '$WORKER' 72-secondary 72b 2,3 '$STATE_DIR' || true; else echo '72B 队列已完整验证，无需第二 worker'; touch '$STATE_DIR/workers/72-secondary.done' '$STATE_DIR/workers/72-secondary.finished'; fi; touch '$STATE_DIR/lanes/lane1.finished'; echo 'GPU2 lane finished'; exec bash"
CMD2="clear; bash '$WORKER' single-gpu3 single 3 '$STATE_DIR' || true; echo 'GPU3 单卡 worker 已退出，等待可能的 72B 接力'; while [[ ! -f '$STATE_DIR/workers/72-secondary.finished' ]]; do sleep 5; done; touch '$STATE_DIR/lanes/lane2.finished'; echo 'GPU3 lane finished'; exec bash"
CMD3="clear; exec bash '$DASHBOARD' '$STATE_DIR'"

tmux send-keys -t "$PANE0" "$CMD0" C-m
tmux send-keys -t "$PANE1" "$CMD1" C-m
tmux send-keys -t "$PANE2" "$CMD2" C-m
tmux send-keys -t "$PANE3" "$CMD3" C-m
tmux select-layout -t "$SESSION:0" tiled >/dev/null

echo "已启动四卡动态评测: $SESSION"
echo "状态目录: $STATE_DIR"
echo "重新进入: tmux attach -t $SESSION"
echo "后台分离: Ctrl+B D"

if [[ "${NO_ATTACH:-0}" != "1" ]]; then
    sleep 1
    attach_session
fi
