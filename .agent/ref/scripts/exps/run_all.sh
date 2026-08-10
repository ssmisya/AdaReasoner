#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$ROOT/validate.sh"
for SIZE in 3B 7B 32B 72B; do
    bash "$ROOT/no_tools/run_qwen25vl_${SIZE,,}_3seeds.sh"
done
bash "$ROOT/with_tools/run_qwen25vl_7b_tools_3seeds.sh"
bash "$ROOT/with_tools/run_qwen25vl_72b_tools_3seeds.sh"
