#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH_OVERRIDE="${MODEL_PATH_OVERRIDE:-/data/songmingyang/model/adareasoner/AdaReasoner-7B-Randomized}"
export MODEL_SLUG_OVERRIDE="${MODEL_SLUG_OVERRIDE:-adareasoner_randomized_7b}"
export MODEL_DISPLAY_NAME_OVERRIDE="${MODEL_DISPLAY_NAME_OVERRIDE:-AdaReasoner-7B-Randomized-Final}"
export GPU_7B="${GPU_7B:-0,3}"
export TP_7B="${TP_7B:-2}"
export BATCH_7B="${BATCH_7B:-64}"
export REQUIRE_LATENCY="${REQUIRE_LATENCY:-1}"
export TF_EVAL_PIPELINE="${TF_EVAL_PIPELINE:-1}"
export TF_EVAL_TOOL_CONCURRENCY="${TF_EVAL_TOOL_CONCURRENCY:-16}"

bash "$ROOT/shared/run_matrix.sh" 7B with_tools
