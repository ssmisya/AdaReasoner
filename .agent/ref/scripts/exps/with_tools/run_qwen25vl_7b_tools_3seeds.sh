#!/usr/bin/env bash
# Qwen2.5-VL-7B with tools, 3 seeds
# GPU layout: GPU0,3=model(vLLM,TP=2), GPU1,2=Point+OCR
# Requires 4 GPUs
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../shared" && pwd)"

# -- enable auto-resume: skip completed, auto-clean & restart incomplete --
export RESUME=1
export CONTINUE_ON_ERROR=1

export MODEL_SIZE=7B
export SEEDS=42,1234,2026
export GPU_7B="${GPU_7B:-0,3}"
export TP_7B="${TP_7B:-2}"
export GPU_MEMORY_7B="${GPU_MEMORY_7B:-0.85}"
export BATCH_7B="${BATCH_7B:-64}"
export REQUIRE_LATENCY="${REQUIRE_LATENCY:-1}"
export TF_EVAL_PIPELINE="${TF_EVAL_PIPELINE:-1}"
export TF_EVAL_TOOL_CONCURRENCY="${TF_EVAL_TOOL_CONCURRENCY:-16}"

exec bash "$ROOT/run_matrix.sh" "$MODEL_SIZE" with_tools
