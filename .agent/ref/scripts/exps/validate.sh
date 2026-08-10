#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/shared/common.sh"

ERRORS=0
check_dir() {
    if [[ -d "$1" ]]; then echo "OK   $1"; else echo "MISS $1"; ERRORS=$((ERRORS+1)); fi
}
check_file() {
    if [[ -f "$1" ]]; then echo "OK   $1"; else echo "MISS $1"; ERRORS=$((ERRORS+1)); fi
}

if [[ "${SKIP_MODEL_CHECK:-0}" != "1" ]]; then
    for SIZE in 3B 7B 32B 72B; do
        check_dir "$(model_path "$SIZE")"
        check_file "$(model_path "$SIZE")/config.json"
    done
fi
check_file "$TASK_MATRIX"
check_file "$ACCELERATE_CONFIG"
check_file "/data/songmingyang/data/adareasoner/benchmarks/AdaEval-VSP/data/verify_test-00000-of-00001.parquet"
check_file "/data/songmingyang/data/adareasoner/benchmarks/AdaEval-VSP/data/navigation_test-00000-of-00001.parquet"
check_file "/data/songmingyang/data/adareasoner/benchmarks/AdaEval-VSPO/data/verify_test-00000-of-00001.parquet"
check_file "/data/songmingyang/data/adareasoner/benchmarks/AdaEval-VSPO/data/navigation_test-00000-of-00001.parquet"
check_file "/data/songmingyang/data/hf_datasets/hitsmy___ada_eval-jigsaw-coco/default/0.0.0/30272a67390749a9e750dfc09aec14e3cf4c316b/ada_eval-jigsaw-coco-test.arrow"
check_file "/data/songmingyang/data/benchmarks/BLINK/Jigsaw/val-00000-of-00001.parquet"
check_file "/data/songmingyang/data/adareasoner/benchmarks/vstar_bench/test_questions.jsonl"
check_file "/data/songmingyang/data/hf_datasets/web_guichat/default/0.0.0/d30d903eef58ebd5/web_guichat-validation.arrow"
check_file "/data/songmingyang/data/benchmarks/WebMMU/web_qa/english.parquet"
check_file "/data/songmingyang/data/hf_datasets/DreamMr___hr-bench/hrbench_version_split/0.0.0/37c2d2e4cfd7b855187a99b3f6ebf286ca6cf453/hr-bench-hrbench_4k.arrow"

source /data/songmingyang/miniforge3/etc/profile.d/conda.sh
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "OK   conda env: $CONDA_ENV"
else
    echo "MISS conda env: $CONDA_ENV"
    ERRORS=$((ERRORS+1))
fi

if curl -fsS -m 3 -X POST "$CONTROLLER_ADDR/list_models" >/dev/null 2>&1; then
    MODELS="$(curl -fsS -m 3 -X POST "$CONTROLLER_ADDR/list_models" | python3 -c 'import json,sys; print(", ".join(json.load(sys.stdin).get("models", [])))')"
    echo "OK   tool controller: $MODELS"
else
    echo "INFO tool controller is not running; with-tools scripts can start it automatically"
fi

if ((ERRORS)); then
    echo "Validation failed with $ERRORS missing requirements." >&2
    exit 1
fi
echo "Validation passed."
