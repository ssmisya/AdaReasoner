#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

usage() {
    echo "Usage: $0 <3B|7B|32B|72B> <no_tools|with_tools> <task> <seed>" >&2
}

[[ $# -eq 4 ]] || { usage; exit 2; }
if [[ "${EVAL_BACKEND:-native}" == "server" ]]; then
    exec bash "$SCRIPT_DIR/../with_tools/run_one_openai.sh" "$@"
fi
MODEL_SIZE="${1^^}"
MODE="$2"
TASK="$3"
SEED="$4"

case "$MODEL_SIZE" in 3B|7B|32B|72B) ;; *) usage; exit 2 ;; esac
case "$MODE" in no_tools|with_tools) ;; *) usage; exit 2 ;; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "ERROR: seed must be an integer" >&2; exit 2; }

python3 - "$TASK_MATRIX" "$TASK" <<'PY'
import json,sys
matrix=json.load(open(sys.argv[1], encoding="utf-8"))
if sys.argv[2] not in matrix:
    raise SystemExit(f"Unknown task: {sys.argv[2]}")
PY

MODEL_PATH="${MODEL_PATH_OVERRIDE:-$(model_path "$MODEL_SIZE")}"
MODEL_SLUG="${MODEL_SLUG_OVERRIDE:-qwen25vl_${MODEL_SIZE,,}}"
MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME_OVERRIDE:-Qwen2.5-VL-${MODEL_SIZE}}"
[[ "$MODEL_SLUG" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || { echo "ERROR: invalid MODEL_SLUG: $MODEL_SLUG" >&2; exit 2; }
TP="$(tensor_parallel "$MODEL_SIZE")"
GPUS="$(gpu_devices "$MODEL_SIZE")"
BATCH="$(batch_size "$MODEL_SIZE")"
GPU_MEMORY="$(gpu_memory_utilization "$MODEL_SIZE")"
PIPELINE_ENABLED="${TF_EVAL_PIPELINE:-0}"
PIPELINE_MAX_ACTIVE="${TF_EVAL_PIPELINE_MAX_ACTIVE:-$((BATCH * 3))}"
TOOL_CONCURRENCY_EFFECTIVE="${TF_EVAL_TOOL_CONCURRENCY:-${TOOL_CONCURRENCY:-$BATCH}}"
export TF_EVAL_PIPELINE="$PIPELINE_ENABLED"
export TF_EVAL_PIPELINE_MAX_ACTIVE="$PIPELINE_MAX_ACTIVE"
export TF_EVAL_TOOL_CONCURRENCY="$TOOL_CONCURRENCY_EFFECTIVE"
TASK_MIN_MODEL_LEN="$(python3 - "$TASK_MATRIX" "$TASK" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]].get("min_model_len", 0))
PY
)"
EFFECTIVE_MAX_MODEL_LEN="${EFFECTIVE_MAX_MODEL_LEN:-$MAX_MODEL_LEN}"
if (( EFFECTIVE_MAX_MODEL_LEN < TASK_MIN_MODEL_LEN )); then
    EFFECTIVE_MAX_MODEL_LEN="$TASK_MIN_MODEL_LEN"
fi
TOOLS="$(tool_selection "$TASK")"
IFS=',' read -ra GPU_LIST <<< "$GPUS"
if [[ "${#GPU_LIST[@]}" -ne "$TP" ]]; then
    echo "ERROR: GPU count (${#GPU_LIST[@]}: $GPUS) must equal tensor_parallel ($TP)" >&2
    exit 2
fi
IF_USE_TOOL=false
[[ "$MODE" == "with_tools" ]] && IF_USE_TOOL=true
IF_RANDOMIZE_TOOL="${IF_RANDOMIZE_TOOL:-false}"
MAX_ROUNDS=6
[[ "$MODE" == "no_tools" ]] && MAX_ROUNDS=1
USE_STOCHASTIC=False
if python3 - "$TASK_MATRIX" "$TASK" <<'PY'
import json,sys
matrix=json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if float(matrix[sys.argv[2]]["task_config"]["generation_config"].get("temperature", 0)) > 0 else 1)
PY
then
    USE_STOCHASTIC=True
fi

require_dir "$MODEL_PATH"
require_file "$MODEL_PATH/config.json"
require_file "$ACCELERATE_CONFIG"
require_file "$SCRIPT_DIR/eval_entry.py"
require_file "$SCRIPT_DIR/validate_resume_checkpoint.py"
require_file "$TASK_MATRIX"

RUN_DIR="$RESULT_ROOT/$MODE/$MODEL_SLUG/$TASK/seed_$SEED"
LOCK_DIR="$RESULT_ROOT/.job_locks/$MODE/$MODEL_SLUG"
mkdir -p "$LOCK_DIR"
if [[ "${JOB_LOCK_HELD:-0}" != "1" ]]; then
    exec 9>"$LOCK_DIR/${TASK}_seed_${SEED}.lock"
    if ! flock -n 9; then
        echo "BUSY: another process owns $MODEL_SIZE/$TASK/seed_$SEED" >&2
        exit 101
    fi
fi

# Never resume checkpoints produced through a different model backend. The old
# OpenAI-compatible runner used the same result directory as native AdaEval.
if [[ -f "$RUN_DIR/config.yaml" ]]; then
    EXISTING_BACKEND="$(awk '$1 == "model:" {print $2; exit}' "$RUN_DIR/config.yaml")"
    if [[ -n "$EXISTING_BACKEND" && "$EXISTING_BACKEND" != "vllm_models" ]]; then
        if [[ "${ARCHIVE_INCOMPATIBLE_BACKEND:-1}" == "1" ]]; then
            BACKUP="${RUN_DIR}.bak_${EXISTING_BACKEND}_$(date +%Y%m%d_%H%M%S)"
            mv "$RUN_DIR" "$BACKUP"
            echo "  -> Archived incompatible $EXISTING_BACKEND run to $BACKUP"
        else
            echo "ERROR: incompatible checkpoint backend '$EXISTING_BACKEND': $RUN_DIR" >&2
            echo "Set ARCHIVE_INCOMPATIBLE_BACKEND=1 to archive it and start native AdaEval." >&2
            exit 1
        fi
    fi
fi

CURRENT_PROTOCOL="$(python3 - "$MODEL_PATH" "$MODE" "$MODEL_SLUG" "$TOOLS" "$IF_RANDOMIZE_TOOL" "$TP" "$BATCH" "$EFFECTIVE_MAX_MODEL_LEN" <<'PY'
import hashlib, json, os, sys
payload = {
    "model_path": os.path.realpath(sys.argv[1]),
    "mode": sys.argv[2],
    "model_slug": sys.argv[3],
    "tool_selection": sys.argv[4],
    "if_randomize_tool": sys.argv[5].lower() == "true",
    "tensor_parallel": int(sys.argv[6]),
    "batch_size": int(sys.argv[7]),
    "max_model_len": int(sys.argv[8]),
}
text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
print(hashlib.sha256(text.encode()).hexdigest())
PY
)"
if [[ -f "$RUN_DIR/run_metadata.json" ]]; then
    readarray -t EXISTING_FIELDS < <(python3 - "$RUN_DIR/run_metadata.json" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    data = {}
print(data.get("model_path", ""))
print(data.get("protocol_fingerprint", ""))
PY
)
    EXISTING_MODEL_PATH="${EXISTING_FIELDS[0]:-}"
    EXISTING_PROTOCOL="${EXISTING_FIELDS[1]:-}"
    MISMATCH=""
    if [[ -n "$EXISTING_MODEL_PATH" && "$(realpath -m "$EXISTING_MODEL_PATH")" != "$(realpath -m "$MODEL_PATH")" ]]; then
        MISMATCH="model"
    elif [[ -s "$RUN_DIR/ckpt.jsonl" && -z "$EXISTING_PROTOCOL" ]]; then
        MISMATCH="legacy_protocol"
    elif [[ -n "$EXISTING_PROTOCOL" && "$EXISTING_PROTOCOL" != "$CURRENT_PROTOCOL" ]]; then
        MISMATCH="protocol"
    fi
    if [[ -n "$MISMATCH" ]]; then
        BACKUP="${RUN_DIR}.bak_${MISMATCH}_mismatch_$(date +%Y%m%d_%H%M%S)"
        mv "$RUN_DIR" "$BACKUP"
        echo "  -> Archived checkpoint with incompatible $MISMATCH to $BACKUP"
    fi
fi

if [[ -f "$RUN_DIR/DONE.json" && "${FORCE:-0}" != "1" ]]; then
    if python3 "$SCRIPT_DIR/validate_run.py" \
        --run-dir "$RUN_DIR" \
        --task "$TASK" \
        --task-matrix "$TASK_MATRIX" \
        --model-path "$MODEL_PATH" \
        --seed "$SEED"; then
        echo "SKIP: validated completed run exists: $RUN_DIR/DONE.json"
        exit 100
    fi
    echo "ERROR: existing DONE validation failed; use FORCE=1 to archive and rerun: $RUN_DIR" >&2
    exit 1
fi
RESUMED_SAMPLES=0
if [[ -d "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    if [[ "${FORCE:-0}" == "1" ]]; then
        BACKUP="${RUN_DIR}.bak_$(date +%Y%m%d_%H%M%S)"
        mv "$RUN_DIR" "$BACKUP"
        echo "  -> Archived previous run to $BACKUP"
    elif [[ "${RESUME:-1}" == "1" ]]; then
        if [[ -s "$RUN_DIR/ckpt.jsonl" ]]; then
            CHECKPOINT_AUDIT=""
            if CHECKPOINT_AUDIT="$(python3 "$SCRIPT_DIR/validate_resume_checkpoint.py" \
                --checkpoint "$RUN_DIR/ckpt.jsonl" 2>&1)"; then
                RESUMED_SAMPLES="$(grep -cve '^[[:space:]]*$' "$RUN_DIR/ckpt.jsonl" || true)"
                echo "  -> [RESUME] $CHECKPOINT_AUDIT"
                rm -f \
                    "$RUN_DIR/result.jsonl" \
                    "$RUN_DIR/timing.json" \
                    "$RUN_DIR/exit_code.txt" \
                    "$RUN_DIR/latency.jsonl" \
                    "$RUN_DIR/latency_summary.json"
            else
                printf '%s\n' "$CHECKPOINT_AUDIT" > "$RUN_DIR/RESUME_REJECTED.txt"
                BACKUP="${RUN_DIR}.bak_resume_rejected_$(date +%Y%m%d_%H%M%S)_$$"
                mv "$RUN_DIR" "$BACKUP"
                echo "  -> [RESUME] rejected unsafe checkpoint: $CHECKPOINT_AUDIT"
                echo "  -> Archived incomplete run to $BACKUP; restarting from zero"
                RESUMED_SAMPLES=0
            fi
        else
            echo "  -> [RESUME] no usable checkpoint; restarting this incomplete job"
            rm -f \
                "$RUN_DIR/result.jsonl" \
                "$RUN_DIR/timing.json" \
                "$RUN_DIR/exit_code.txt" \
                "$RUN_DIR/latency.jsonl" \
                "$RUN_DIR/latency_summary.json"
        fi
        rm -f "$RUN_DIR/DONE.json" "$RUN_DIR"/DONE.*.tmp
    else
        echo "ERROR: incomplete run directory exists: $RUN_DIR" >&2
        echo "Set FORCE=1 to archive it, or RESUME=1 to continue from ckpt.jsonl." >&2
        exit 1
    fi
fi
mkdir -p "$RUN_DIR/middle_images"

if [[ "$MODE" == "with_tools" && "${AUTO_START_TOOLS:-1}" == "1" ]]; then
    if ! curl -fsS -m 3 -X POST "$CONTROLLER_ADDR/list_models" >/dev/null 2>&1; then
        bash "$REPO/.agent/ref/scripts/start_tools.sh"
    fi
fi

if [[ "$MODE" == "with_tools" ]]; then
    ONLINE_REQUIRED="$(python3 - "$TASK_MATRIX" "$TASK" <<'PY'
import json,sys
names=set(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]["tool_selection"].split(","))
print(",".join(sorted(names & {"Point","OCR","Crop"})))
PY
)"
    if [[ -n "$ONLINE_REQUIRED" ]]; then
        REGISTERED="$(curl -fsS -m 5 -X POST "$CONTROLLER_ADDR/list_models" | python3 -c 'import json,sys; print(",".join(json.load(sys.stdin).get("models", [])))')"
        python3 - "$ONLINE_REQUIRED" "$REGISTERED" <<'PY'
import sys
required=set(filter(None,sys.argv[1].split(",")))
registered=set(filter(None,sys.argv[2].split(",")))
missing=sorted(required-registered)
if missing:
    raise SystemExit("Missing online tools: " + ", ".join(missing))
PY
    fi
fi

# Fail early with a clear message instead of letting a vLLM worker die after startup.
python3 - "$GPUS" "$TP" "$GPU_MEMORY" "$MODEL_SIZE" <<'PY'
import subprocess
import sys

gpu_ids = [int(item.strip()) for item in sys.argv[1].split(",") if item.strip()]
tp = int(sys.argv[2])
utilization = float(sys.argv[3])
model_size = sys.argv[4]
if len(gpu_ids) != tp:
    raise SystemExit(
        f"ERROR: selected {len(gpu_ids)} GPUs ({gpu_ids}) but tensor_parallel={tp}"
    )
if len(set(gpu_ids)) != len(gpu_ids):
    raise SystemExit(f"ERROR: duplicate GPU IDs in CUDA_VISIBLE_DEVICES: {gpu_ids}")
if not 0 < utilization <= 1:
    raise SystemExit(f"ERROR: invalid gpu_memory_utilization={utilization}")

output = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
gpus = {}
for line in output.splitlines():
    index, total, free = (part.strip() for part in line.split(","))
    gpus[int(index)] = (float(total), float(free))

for gpu_id in gpu_ids:
    if gpu_id not in gpus:
        raise SystemExit(f"ERROR: GPU {gpu_id} does not exist")
    total, free = gpus[gpu_id]
    required = total * utilization
    used = total - free
    print(
        f"GPU preflight: GPU{gpu_id} used={used:.0f} MiB free={free:.0f} MiB "
        f"vLLM_required={required:.0f} MiB"
    )
    if free < required:
        raise SystemExit(
            f"ERROR: GPU{gpu_id} has only {free:.0f} MiB free, but vLLM requests "
            f"{required:.0f} MiB ({utilization:.2f} of {total:.0f} MiB). "
            "Do not place vLLM on GPU1/2 while Point/OCR tools are running; "
            f"for TP=2 use GPU_{model_size}=0,3 TP_{model_size}=2."
        )
PY

CONFIG="$RUN_DIR/config.yaml"
cat > "$CONFIG" <<YAML
model_args:
  model: vllm_models
  model_args: pretrained=$MODEL_PATH,tensor_parallel=$TP,limit_mm_per_prompt=10,gpu_memory_utilization=$GPU_MEMORY,max_model_len=$EFFECTIVE_MAX_MODEL_LEN,enforce_eager=True,seed=$SEED
  batch_size: $BATCH
  max_rounds: $MAX_ROUNDS
  model_mode: general
task_args:
  task_name: $TASK
  tool_selection: $TOOLS
  resume_from_ckpt:
    $TASK: $RUN_DIR/ckpt.jsonl
  save_to_ckpt:
    $TASK: $RUN_DIR/ckpt.jsonl
  middle_images_save_dir:
    $TASK: $RUN_DIR/middle_images
script_args:
  verbosity: INFO
  output_path: $RUN_DIR/result.jsonl
  controller_addr: $CONTROLLER_ADDR
  if_use_tool: $IF_USE_TOOL
  if_randomize_tool: $IF_RANDOMIZE_TOOL
YAML

python3 - "$RUN_DIR/run_metadata.json" <<PY
import json,platform,time
payload={
  "model_size":"$MODEL_SIZE",
  "model_slug":"$MODEL_SLUG",
  "model_display_name":"$MODEL_DISPLAY_NAME",
  "model_path":"$MODEL_PATH",
  "backend":"vllm_models",
  "mode":"$MODE",
  "task":"$TASK",
  "seed":$SEED,
  "cuda_visible_devices":"$GPUS",
  "tensor_parallel":$TP,
  "batch_size":$BATCH,
  "pipeline_enabled":"$PIPELINE_ENABLED",
  "pipeline_max_active":$PIPELINE_MAX_ACTIVE,
  "tool_concurrency":$TOOL_CONCURRENCY_EFFECTIVE,
  "max_model_len":$EFFECTIVE_MAX_MODEL_LEN,
  "max_rounds":$MAX_ROUNDS,
  "resumed_samples":$RESUMED_SAMPLES,
  "tool_selection":"$TOOLS",
  "if_randomize_tool":bool("$IF_RANDOMIZE_TOOL".lower()=="true"),
  "stochastic_decoding":$USE_STOCHASTIC,
  "controller_addr":"$CONTROLLER_ADDR",
  "latency_schema_version":1,
  "latency_required":bool(int("${REQUIRE_LATENCY:-1}")),
  "latency_definition":{
    "instance_e2e_s":"from admission before conversation construction until the final round is recognized as finished; excludes final result serialization and checkpoint writes",
    "round_model_generate_s":"wall time around model.generate()",
    "round_tool_call_s":"wall time around the tool RPC"
  },
  "protocol_fingerprint":"$CURRENT_PROTOCOL",
  "created_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "hostname":platform.node(),
}
json.dump(payload,open("$RUN_DIR/run_metadata.json","w"),indent=2)
PY

source /data/songmingyang/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_CACHE
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONSAFEPATH=1
export PYTHONHASHSEED="$SEED"
if [[ "$TASK" == "hrbench" ]]; then
    load_yunwu_env
fi
# HRBench passes its proxy directly to requests; never proxy the complete evaluator process.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,::1,.svc,.cluster.local,10.0.0.0/8"
export no_proxy="$NO_PROXY"
export CUDA_VISIBLE_DEVICES="$GPUS"
cd "$REPO"
unset E3_LATENCY_LOG || true
if [[ "$MODE" == "with_tools" ]]; then
    export E3_LATENCY_LOG="$RUN_DIR/stage_latency.json"
fi

echo "============================================================"
echo "model=$MODEL_DISPLAY_NAME slug=$MODEL_SLUG mode=$MODE task=$TASK seed=$SEED"
echo "gpus=$GPUS tp=$TP batch=$BATCH max_model_len=$EFFECTIVE_MAX_MODEL_LEN result=$RUN_DIR"
echo "pipeline=$PIPELINE_ENABLED max_active=$PIPELINE_MAX_ACTIVE tool_concurrency=$TOOL_CONCURRENCY_EFFECTIVE"
echo "start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

set +e
accelerate launch --config_file "$ACCELERATE_CONFIG" \
    "$SCRIPT_DIR/eval_entry.py" \
    --config "$CONFIG" \
    --task-matrix "$TASK_MATRIX" \
    --timing-output "$RUN_DIR/timing.json" \
    --seed "$SEED" \
    2>&1 | tee "$RUN_DIR/run.log"
STATUS=${PIPESTATUS[0]}
set -e

echo "$STATUS" > "$RUN_DIR/exit_code.txt"
if [[ "$STATUS" -ne 0 ]]; then
    echo "FAILED: model=$MODEL_SIZE mode=$MODE task=$TASK seed=$SEED" >&2
    exit "$STATUS"
fi

LATENCY_ARGS=()
if [[ "${REQUIRE_LATENCY:-1}" == "1" ]]; then
    LATENCY_ARGS+=(--require-complete)
fi
python3 "$SCRIPT_DIR/summarize_latency.py" \
    --checkpoint "$RUN_DIR/ckpt.jsonl" \
    --output-jsonl "$RUN_DIR/latency.jsonl" \
    --summary "$RUN_DIR/latency_summary.json" \
    "${LATENCY_ARGS[@]}"

python3 "$SCRIPT_DIR/validate_run.py" \
    --run-dir "$RUN_DIR" \
    --task "$TASK" \
    --task-matrix "$TASK_MATRIX" \
    --model-path "$MODEL_PATH" \
    --seed "$SEED"

echo "DONE: $RUN_DIR"
