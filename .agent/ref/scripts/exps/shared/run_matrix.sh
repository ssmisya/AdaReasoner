#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

usage() {
    echo "Usage: $0 <3B|7B|32B|72B> <no_tools|with_tools>" >&2
    echo "Optional env: TASKS=vsp,jigsaw_coco SEEDS=42,1234,2026 CONTINUE_ON_ERROR=1 RESUME=1" >&2
}

[[ $# -eq 2 ]] || { usage; exit 2; }
MODEL_SIZE="${1^^}"
MODE="$2"
case "$MODEL_SIZE" in 3B|7B|32B|72B) ;; *) usage; exit 2 ;; esac
case "$MODE" in no_tools|with_tools) ;; *) usage; exit 2 ;; esac

# -- resolve task list & seed list --
split_csv "${TASKS:-}"
TASK_LIST=("${CSV_ITEMS[@]}")
split_csv "${SEEDS:-}"
SEED_LIST=("${CSV_ITEMS[@]}")

FAILURES=()
SKIPPED=()
DONE=()

TOTAL_TASKS=${#TASK_LIST[@]}
TOTAL_SEEDS=${#SEED_LIST[@]}
TOTAL_JOBS=$(( TOTAL_TASKS * TOTAL_SEEDS ))
RUN_IDX=0
OVERALL_START=$(date +%s)

MODEL_SLUG="${MODEL_SLUG_OVERRIDE:-qwen25vl_${MODEL_SIZE,,}}"
MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME_OVERRIDE:-Qwen2.5-VL-${MODEL_SIZE}}"

echo ""
echo "============================ START =============================="
echo "  MODEL     : $MODEL_DISPLAY_NAME ($MODEL_SIZE)"
echo "  MODE      : $MODE"
echo "  TASKS     : ${TASK_LIST[*]}"
echo "  SEEDS     : ${SEED_LIST[*]}"
echo "  TOTAL JOBS: $TOTAL_JOBS ($TOTAL_TASKS tasks x $TOTAL_SEEDS seeds)"
echo "  START     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="
echo ""

for TASK in "${TASK_LIST[@]}"; do
    for SEED in "${SEED_LIST[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))
        RUN_START=$(date +%s)
        # progress bar (simple text)
        printf "[%s/%s] %s seed=%s ... " "$RUN_IDX" "$TOTAL_JOBS" "$TASK" "$SEED"

        set +e
        bash "$SCRIPT_DIR/run_one.sh" "$MODEL_SIZE" "$MODE" "$TASK" "$SEED"
        RC=$?
        set -e

        if [[ $RC -eq 0 ]]; then
            RUN_END=$(date +%s)
            RUN_ELAPSED=$((RUN_END - RUN_START))
            DONE+=("$TASK/seed_$SEED")
            ELAPSED_TOTAL=$((RUN_END - OVERALL_START))
            echo ""
            echo "  -> DONE  [$RUN_IDX/$TOTAL_JOBS] $TASK seed=$SEED  [${RUN_ELAPSED}s | total ${ELAPSED_TOTAL}s]"
        elif [[ $RC -eq 100 ]]; then
            # skip code: already completed
            SKIPPED+=("$TASK/seed_$SEED")
            echo "SKIPPED (already completed)"
        else
            FAILURES+=("$TASK/seed_$SEED")
            echo "FAILED (rc=$RC)"
            if [[ "${CONTINUE_ON_ERROR:-0}" != "1" ]]; then
                echo ""
                echo "FATAL: run failed for $TASK/seed_$SEED (rc=$RC)" >&2
                echo "Set CONTINUE_ON_ERROR=1 to skip failures and continue." >&2
                exit 1
            fi
            echo "  WARNING: continuing to next job (CONTINUE_ON_ERROR=1)" >&2
        fi
    done
done

OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$((OVERALL_END - OVERALL_START))

echo ""
echo "=========================== SUMMARY ============================="
echo "  MODEL     : $MODEL_DISPLAY_NAME ($MODEL_SIZE)"
echo "  MODE      : $MODE"
echo "  TOTAL     : $TOTAL_JOBS jobs"
echo "  DONE      : ${#DONE[@]} (fresh runs)"
echo "  SKIPPED   : ${#SKIPPED[@]} (previously completed)"
echo "  FAILED    : ${#FAILURES[@]}"
echo "  DURATION  : ${OVERALL_ELAPSED}s ($((OVERALL_ELAPSED / 60))m $((OVERALL_ELAPSED % 60))s)"
echo "  FINISH    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

# -- summarize scores --
EXPERIMENT_DIR="$RESULT_ROOT/$MODE/$MODEL_SLUG"
python3 "$SCRIPT_DIR/summarize.py" "$EXPERIMENT_DIR" --seeds "$SEEDS" 2>/dev/null || true

if ((${#FAILURES[@]})); then
    echo ""
    printf 'FAILED RUNS:\n' >&2
    printf '  %s\n' "${FAILURES[@]}" >&2
    exit 1
fi

echo "ALL $TOTAL_JOBS JOBS FINISHED SUCCESSFULLY."
