#!/bin/bash
# Point worker supervisor (rebuttal 鲁棒性)
# 为每个 (GPU, port) 维持一个 molmo_point_worker; 进程退出(尤其CUDA损坏exit 3)就自动重启并重新注册controller。
# 用法: bash point_supervisor.sh   (前台常驻; 建议用 run_in_background 起)
# 布局: GPU3:[50002,50003]

set -u
PY=/data/songmingyang/miniforge3/envs/vllm2/bin/python
REPO=/data/songmingyang/code/reasoning/AdaReasoner-rebuttal
export PYTHONPATH=$REPO
CTRL=http://127.0.0.1:21112
MODEL=/data/songmingyang/models/Molmo-7B-D-0924
LOGDIR=$REPO/rebuttal_exps/point_worker_logs
mkdir -p "$LOGDIR"

# (gpu port) 映射
SPECS=("3 50002" "3 50003")

declare -A PIDS
start_one() {
  local gpu=$1 port=$2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -m tool_server.tool_workers.online_workers.molmo_point_worker \
    --host 127.0.0.1 --port $port --controller_addr $CTRL \
    --model_path $MODEL --model_name Point \
    > "$LOGDIR/worker_${port}.log" 2>&1 &
  PIDS[$port]=$!
  echo "$(date '+%H:%M:%S') started worker port=$port gpu=$gpu pid=${PIDS[$port]}"
}

# 初始全部拉起
for spec in "${SPECS[@]}"; do
  set -- $spec; start_one "$1" "$2"
  sleep 8   # 错开加载, 避免同卡两个worker同时抢显存峰值
done

echo "$(date '+%H:%M:%S') all workers launched, entering supervise loop"
# 守护循环: 谁死了就按其gpu/port重启
while true; do
  for spec in "${SPECS[@]}"; do
    set -- $spec; gpu=$1; port=$2
    pid=${PIDS[$port]:-}
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      echo "$(date '+%H:%M:%S') worker port=$port (pid=$pid) DOWN, restarting..."
      start_one "$gpu" "$port"
      sleep 10   # 等它重新加载+注册
    fi
  done
  sleep 15
done
