#!/usr/bin/env bash
# AdaReasoner-7B-Randomized is the final model. Evaluate it once with native Tool Eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_DIR="$(cd "$SCRIPT_DIR/../shared" && pwd)"
source "$SHARED_DIR/common.sh"

export MODEL_PATH_OVERRIDE="${ADAREASONER_MODEL_PATH:-/data/songmingyang/model/adareasoner/AdaReasoner-7B-Randomized}"
export MODEL_SLUG_OVERRIDE="${ADAREASONER_MODEL_SLUG:-adareasoner_randomized_7b}"
export MODEL_DISPLAY_NAME_OVERRIDE="${ADAREASONER_MODEL_NAME:-AdaReasoner-7B-Randomized-Final}"
export SEEDS="${ADAREASONER_SEED:-42}"
export GPU_7B="${GPU_7B:-0,3}"
export TP_7B="${TP_7B:-2}"
export GPU_MEMORY_7B="${GPU_MEMORY_7B:-0.85}"
export REQUIRE_LATENCY="${REQUIRE_LATENCY:-1}"
export TF_EVAL_PIPELINE="${TF_EVAL_PIPELINE:-1}"
export TF_EVAL_TOOL_CONCURRENCY="${TF_EVAL_TOOL_CONCURRENCY:-16}"

bash "$SHARED_DIR/run_matrix.sh" 7B with_tools
