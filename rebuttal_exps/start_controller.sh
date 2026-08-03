#!/bin/bash
# 启动 tool server: controller (21112) + Point worker (Molmo, GPU3)
cd /home/myangsong/AdaReasoner
export PYTHONPATH=/home/myangsong/AdaReasoner
PY=/home/myangsong/.conda/envs/tool-server/bin/python
LOGDIR=/home/myangsong/AdaReasoner/rebuttal_exps/toolserver_logs
mkdir -p "$LOGDIR"

# 1. controller (无GPU)
CUDA_VISIBLE_DEVICES="" nohup $PY tool_server/tool_workers/online_workers/controller.py \
  --host 0.0.0.0 --port 21112 > "$LOGDIR/controller.log" 2>&1 &
echo "controller PID=$!"
