#!/bin/bash
# Start tool-server controller on port 21112 (no GPU)
set -euo pipefail
REPO=/data/songmingyang/code/reasoning/AdaReasoner-rebuttal
source /data/songmingyang/miniforge3/etc/profile.d/conda.sh
conda activate vllm-latest
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO"
LOGDIR="$REPO/rebuttal_exps/toolserver_logs"
mkdir -p "$LOGDIR"
cd "$REPO"

if ss -tlnp 2>/dev/null | grep -q ':21112 '; then
  echo "controller already listening on 21112"
  exit 0
fi

CUDA_VISIBLE_DEVICES="" nohup python tool_server/tool_workers/online_workers/controller.py \
  --host 0.0.0.0 --port 21112 \
  > "$LOGDIR/controller.log" 2>&1 &
echo "controller PID=$! log=$LOGDIR/controller.log"
sleep 2
ss -tlnp 2>/dev/null | grep 21112 || true
