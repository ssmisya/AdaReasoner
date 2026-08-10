#!/usr/bin/env bash
# Fix 14 jobs that completed evaluation but are missing DONE.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="/data/songmingyang/code/reasoning/AdaReasoner-rebuttal"
SHARED_DIR="$REPO/.agent/ref/scripts/exps/shared"
TASK_MATRIX="$SHARED_DIR/task_matrix.json"
RESULT_ROOT="$REPO/rebuttal_exps/qwen25vl_eval"

source "$SHARED_DIR/common.sh"
source /data/songmingyang/miniforge3/etc/profile.d/conda.sh
conda activate vllm-latest

export HF_DATASETS_CACHE="/data/songmingyang/data/hf_datasets"

# List of jobs that have complete eval but no DONE.json
JOBS=(
    "qwen25vl_7b:vsp:seed_42"
    "qwen25vl_7b:web_guichat:seed_42"
    "qwen25vl_7b:web_guichat:seed_1234"
    "qwen25vl_7b:web_guichat:seed_2026"
    "qwen25vl_32b:jigsaw_coco:seed_42"
    "qwen25vl_32b:vsp:seed_1234"
    "qwen25vl_32b:vspo:seed_2026"
    "qwen25vl_32b:web_guichat:seed_42"
    "qwen25vl_32b:web_guichat:seed_1234"
    "qwen25vl_32b:web_guichat:seed_2026"
    "qwen25vl_72b:vspo:seed_42"
    "qwen25vl_72b:web_guichat:seed_42"
    "qwen25vl_72b:web_guichat:seed_1234"
    "qwen25vl_72b:web_guichat:seed_2026"
)

PASS=0
FAIL=0

for job in "${JOBS[@]}"; do
    IFS=':' read -r model_slug task seed_dir <<< "$job"
    RUN_DIR="$RESULT_ROOT/no_tools/$model_slug/$task/$seed_dir"
    DONE_FILE="$RUN_DIR/DONE.json"
    
    echo "================================================"
    echo "Processing: $model_slug / $task / $seed_dir"
    
    # Safety checks
    if [[ -f "$DONE_FILE" ]]; then
        echo "  SKIP: DONE.json already exists"
        ((PASS++)) || true
        continue
    fi
    if [[ ! -f "$RUN_DIR/exit_code.txt" ]] || [[ "$(cat "$RUN_DIR/exit_code.txt")" != "0" ]]; then
        echo "  SKIP: exit_code is not 0 or missing"
        ((FAIL++)) || true
        continue
    fi
    if [[ ! -f "$RUN_DIR/ckpt.jsonl" ]]; then
        echo "  SKIP: ckpt.jsonl missing"
        ((FAIL++)) || true
        continue
    fi
    
    # Get model size (uppercase)
    model_size="${model_slug##*_}"
    MODEL_PATH="${MODEL_PATH_OVERRIDE:-$(model_path "$model_size")}"
    
    # Step 1: summarize latency
    echo "  -> summarize_latency.py ..."
    if python3 "$SHARED_DIR/summarize_latency.py" \
        --checkpoint "$RUN_DIR/ckpt.jsonl" \
        --output-jsonl "$RUN_DIR/latency.jsonl" \
        --summary "$RUN_DIR/latency_summary.json" \
        "${LATENCY_EXTRA:---require-complete}" 2>&1 | sed 's/^/    /'; then
        echo "  -> latency OK"
    else
        echo "  -> latency FAILED (continuing anyway)"
    fi
    
    # Step 2: validate run (generates DONE.json)
    echo "  -> validate_run.py ..."
    if python3 "$SHARED_DIR/validate_run.py" \
        --run-dir "$RUN_DIR" \
        --task "$task" \
        --task-matrix "$TASK_MATRIX" \
        --model-path "$MODEL_PATH" \
        --seed "${seed_dir#seed_}" 2>&1 | sed 's/^/    /'; then
        echo "  -> DONE.json generated successfully!"
        ((PASS++)) || true
    else
        echo "  -> validate_run FAILED"
        ((FAIL++)) || true
    fi
done

echo ""
echo "================================================"
echo "Summary: $PASS passed, $FAIL failed, total ${#JOBS[@]}"
