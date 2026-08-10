#!/usr/bin/env bash
# Select the evaluation backend without changing the established CLI.
# EVAL_BACKEND=native (default): embedded vLLM, legacy evaluation path.
# EVAL_BACKEND=server: shared OpenAI-compatible vLLM server with concurrent requests.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${EVAL_BACKEND:-native}"

case "$BACKEND" in
  native|legacy|embedded)
    exec "$SCRIPT_DIR/../shared/run_one.sh" "$@"
    ;;
  server|openai|async)
    exec "$SCRIPT_DIR/run_one_openai.sh" "$@"
    ;;
  *)
    echo "ERROR: unsupported EVAL_BACKEND=$BACKEND (expected native or server)" >&2
    exit 2
    ;;
esac
