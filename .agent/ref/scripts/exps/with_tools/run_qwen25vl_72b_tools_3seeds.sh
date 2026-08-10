#!/usr/bin/env bash
# Qwen2.5-VL-72B with tools, 3 seeds
# GPU layout: GPU0,3=model(vLLM,TP=2), GPU1=Point+OCR, GPU2=Point+OCR
# Requires >= 4 GPUs
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../shared" && pwd)"

# -- enable auto-resume: skip completed, auto-clean & restart incomplete --
export RESUME=1
export CONTINUE_ON_ERROR=1

export MODEL_SIZE=72B
export SEEDS=42,1234,2026
export GPU_72B="${GPU_72B:-0,3}"
export TP_72B="${TP_72B:-2}"
export GPU_MEMORY_72B="${GPU_MEMORY_72B:-0.95}"
export BATCH_72B="${BATCH_72B:-64}"
export REQUIRE_LATENCY="${REQUIRE_LATENCY:-1}"

exec bash "$ROOT/run_matrix.sh" "$MODEL_SIZE" with_tools
