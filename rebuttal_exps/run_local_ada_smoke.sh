#!/bin/bash
set -euo pipefail
REPO=/data/songmingyang/code/reasoning/AdaReasoner-rebuttal
OUTDIR=$REPO/rebuttal_exps/local_ada_smoke
LOCK=$OUTDIR/run.lock
LOG=$OUTDIR/run2.log
mkdir -p "$OUTDIR/middle_images"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Another ada smoke is already running (lock $LOCK)"
  exit 1
fi

source /data/songmingyang/miniforge3/etc/profile.d/conda.sh
conda activate vllm-latest
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0
cd "$REPO"

# ensure no leftover eval on this host
pgrep -af 'local_ada_smoke_vsp' || true

echo "==== $(date -u) START ada smoke (single) ====" | tee "$LOG"
echo "models=$(curl -s -m 3 -X POST http://127.0.0.1:21112/list_models)" | tee -a "$LOG"
echo "gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';')" | tee -a "$LOG"
echo "$$" > "$OUTDIR/eval.pid"

accelerate launch --config_file configs/accelerate.yaml -m tool_server.tf_eval \
  --config rebuttal_exps/local_ada_smoke_vsp.yaml \
  >> "$LOG" 2>&1
EC=$?
echo "==== $(date -u) END exit=$EC ====" | tee -a "$LOG"
exit $EC
