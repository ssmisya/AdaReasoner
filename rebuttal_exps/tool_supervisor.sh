#!/bin/bash
# 工具栈 supervisor (rebuttal 鲁棒性, 新布局: Point 1卡2worker + OCR + Crop)
# 布局:
#   GPU3: Point worker ×2 (50002, 50003)  ← Molmo, 1卡2worker(用户指定)
#   GPU2: OCR worker (50010, PaddleOCR PP-OCRv5) + Crop worker (50011, 无模型)
# 进程退出(尤其CUDA损坏exit 3)自动按原gpu/port/model重启+重注册。
# OCR worker 需要 opengl bundle 才能 import GUI cv2 → 启动前注入 LD_LIBRARY_PATH。

set -u
PY=/data/songmingyang/miniforge3/envs/vllm2/bin/python
OCRPY=/data/songmingyang/miniforge3/envs/ocr-server/bin/python   # OCR用独立env(paddle-gpu, 避免与torch cu124争CUDA)
REPO=/data/songmingyang/code/reasoning/AdaReasoner-rebuttal
export PYTHONPATH=$REPO
CTRL=http://127.0.0.1:21112
MOLMO=/data/songmingyang/models/Molmo-7B-D-0924
OPENGL=/apdcephfs_cq11/share_1567347/share_info/myangsong/opengl_libs
# 全组件现在都用GUI opencv → 都需要opengl bundle才能import cv2/transformers
export LD_LIBRARY_PATH="$OPENGL:${LD_LIBRARY_PATH:-}"
LOGDIR=$REPO/rebuttal_exps/tool_worker_logs
mkdir -p "$LOGDIR"

declare -A PIDS   # port -> pid
declare -A KIND   # port -> point|ocr|crop
declare -A GPU    # port -> gpu id

start_point() {
  local gpu=$1 port=$2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -m tool_server.tool_workers.online_workers.molmo_point_worker \
    --host 127.0.0.1 --port $port --controller_addr $CTRL \
    --model_path $MOLMO --model_name Point \
    > "$LOGDIR/point_${port}.log" 2>&1 &
  PIDS[$port]=$!; KIND[$port]=point; GPU[$port]=$gpu
  echo "$(date '+%H:%M:%S') started Point port=$port gpu=$gpu pid=${PIDS[$port]}"
}
start_ocr() {
  local gpu=$1 port=$2
  CUDA_VISIBLE_DEVICES=$gpu LD_LIBRARY_PATH="$OPENGL:${LD_LIBRARY_PATH:-}" \
    nohup $OCRPY -m tool_server.tool_workers.online_workers.ocr_worker \
    --host 127.0.0.1 --port $port --controller_addr $CTRL \
    --model_name OCR --gpu_ids "0" \
    > "$LOGDIR/ocr_${port}.log" 2>&1 &
  PIDS[$port]=$!; KIND[$port]=ocr; GPU[$port]=$gpu
  echo "$(date '+%H:%M:%S') started OCR port=$port gpu=$gpu pid=${PIDS[$port]}"
}
start_crop() {
  local port=$1
  CUDA_VISIBLE_DEVICES="" nohup $PY -m tool_server.tool_workers.online_workers.crop_worker_prompt \
    --host 127.0.0.1 --port $port --controller_addr $CTRL --model_name Crop \
    > "$LOGDIR/crop_${port}.log" 2>&1 &
  PIDS[$port]=$!; KIND[$port]=crop; GPU[$port]="-"
  echo "$(date '+%H:%M:%S') started Crop port=$port pid=${PIDS[$port]}"
}

restart_one() {
  local port=$1
  case "${KIND[$port]}" in
    point) start_point "${GPU[$port]}" "$port" ;;
    ocr)   start_ocr   "${GPU[$port]}" "$port" ;;
    crop)  start_crop  "$port" ;;
  esac
}

# 初始拉起 (布局: Point×2@GPU3, OCR@GPU0, Crop@CPU; GPU1/2常被外部容器占)
start_point 3 50002; sleep 8
start_point 3 50003; sleep 8
start_ocr   0 50010; sleep 8
start_crop  50011;   sleep 5
echo "$(date '+%H:%M:%S') all workers launched, entering supervise loop"

while true; do
  for port in "${!PIDS[@]}"; do
    pid=${PIDS[$port]:-}
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      echo "$(date '+%H:%M:%S') worker port=$port kind=${KIND[$port]} (pid=$pid) DOWN, restarting..."
      restart_one "$port"
      sleep 10
    fi
  done
  sleep 15
done
