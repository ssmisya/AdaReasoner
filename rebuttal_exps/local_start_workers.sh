#!/bin/bash
# Start online tool workers adapted for this machine:
#   Point (Molmo) on GPU1 @ 50002
#   Crop (CPU) @ 50011
# OCR is optional (paddle CUDA conflicts) — set START_OCR=1 to try.
set -euo pipefail
REPO=/data/songmingyang/code/reasoning/AdaReasoner-rebuttal
source /data/songmingyang/miniforge3/etc/profile.d/conda.sh
conda activate vllm-latest
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO"
CTRL=http://127.0.0.1:21112
MOLMO=/data/songmingyang/shared_models/adareasoner/tools/Molmo-7B-D-0924
LOGDIR="$REPO/rebuttal_exps/toolserver_logs"
mkdir -p "$LOGDIR"
cd "$REPO"
START_OCR="${START_OCR:-0}"

# wait for controller
for i in $(seq 1 30); do
  if curl -s -X POST "$CTRL/list_models" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# Point on GPU1
if ! ss -tlnp 2>/dev/null | grep -q ':50002 '; then
  CUDA_VISIBLE_DEVICES=1 nohup python -m tool_server.tool_workers.online_workers.molmo_point_worker \
    --host 127.0.0.1 --port 50002 --controller_addr "$CTRL" \
    --model_path "$MOLMO" --model_name Point \
    > "$LOGDIR/point_50002.log" 2>&1 &
  echo "Point PID=$! gpu=1 port=50002 log=$LOGDIR/point_50002.log"
else
  echo "Point already on 50002"
fi

# Crop on CPU
if ! ss -tlnp 2>/dev/null | grep -q ':50011 '; then
  CUDA_VISIBLE_DEVICES="" nohup python -m tool_server.tool_workers.online_workers.crop_worker_prompt \
    --host 127.0.0.1 --port 50011 --controller_addr "$CTRL" \
    --model_name Crop \
    > "$LOGDIR/crop_50011.log" 2>&1 &
  echo "Crop PID=$! port=50011 log=$LOGDIR/crop_50011.log"
else
  echo "Crop already on 50011"
fi

# Optional OCR (separate env if present)
if [ "$START_OCR" = "1" ]; then
  OCRPY=/data/songmingyang/miniforge3/envs/ocr-server/bin/python
  if [ -x "$OCRPY" ] && ! ss -tlnp 2>/dev/null | grep -q ':50010 '; then
    CUDA_VISIBLE_DEVICES=2 nohup "$OCRPY" -m tool_server.tool_workers.online_workers.ocr_worker \
      --host 127.0.0.1 --port 50010 --controller_addr "$CTRL" \
      --model_name OCR --gpu_ids "0" \
      > "$LOGDIR/ocr_50010.log" 2>&1 &
    echo "OCR PID=$! gpu=2 port=50010 log=$LOGDIR/ocr_50010.log"
  else
    echo "OCR skip (env missing or port busy)"
  fi
else
  echo "OCR skipped (set START_OCR=1 to try)"
fi

echo "Workers launched. Check: curl -s -X POST $CTRL/list_models"
