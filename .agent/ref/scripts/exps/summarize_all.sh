#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/shared/common.sh"

for MODE_SIZE in \
    "no_tools 3b" \
    "no_tools 7b" \
    "no_tools 32b" \
    "no_tools 72b" \
    "with_tools 7b" \
    "with_tools 72b"; do
    read -r MODE SIZE <<< "$MODE_SIZE"
    DIR="$RESULT_ROOT/$MODE/qwen25vl_$SIZE"
    if [[ -d "$DIR" ]]; then
        python3 "$ROOT/shared/summarize.py" "$DIR" --seeds "$SEEDS"
    else
        echo "SKIP: $DIR does not exist"
    fi
done
