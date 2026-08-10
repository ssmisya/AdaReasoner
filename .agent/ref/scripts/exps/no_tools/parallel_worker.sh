#!/usr/bin/env bash
set -uo pipefail

usage() {
    echo "Usage: $0 <worker-name> <single|72b> <gpu-list> <state-dir>" >&2
    exit 2
}

[[ $# -eq 4 ]] || usage
WORKER_NAME="$1"
QUEUE="$2"
GPUS="$3"
STATE_DIR="$4"
[[ "$QUEUE" == "single" || "$QUEUE" == "72b" ]] || usage
[[ -f "$STATE_DIR/config.env" ]] || { echo "ERROR: missing $STATE_DIR/config.env" >&2; exit 2; }

# shellcheck disable=SC1090
source "$STATE_DIR/config.env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$EXPS_ROOT/shared/common.sh"

RUN_ONE="$EXPS_ROOT/shared/run_one.sh"
VALIDATE_RUN="$EXPS_ROOT/shared/validate_run.py"
LOCK_ROOT="$RESULT_ROOT/.job_locks/no_tools"
STATUS_FILE="$STATE_DIR/workers/$WORKER_NAME.status"
FINISHED_FILE="$STATE_DIR/workers/$WORKER_NAME.finished"
DONE_FILE="$STATE_DIR/workers/$WORKER_NAME.done"
FAILED_FILE="$STATE_DIR/workers/$WORKER_NAME.failed"
ABORTED_FILE="$STATE_DIR/workers/$WORKER_NAME.aborted"
VALIDATED_CACHE_DIR="$STATE_DIR/validated_jobs"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
CHILD_PID=""
CURRENT_STARTED="$(date +%s)"
mkdir -p "$STATE_DIR/workers" "$STATE_DIR/failures" "$VALIDATED_CACHE_DIR" "$LOCK_ROOT"
rm -f "$FINISHED_FILE" "$DONE_FILE" "$FAILED_FILE" "$ABORTED_FILE"

split_csv "$TASKS"
TASK_LIST=("${CSV_ITEMS[@]}")
split_csv "$SEEDS"
SEED_LIST=("${CSV_ITEMS[@]}")
if [[ "$QUEUE" == "72b" ]]; then
    SIZE_LIST=("72B")
    TP=2
else
    SIZE_LIST=("32B" "7B" "3B")
    TP=1
fi

write_status() {
    local phase="$1" model="${2:--}" task="${3:--}" seed="${4:--}" message="${5:-}"
    local tmp="$STATUS_FILE.tmp.$$"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$phase" "$model" "$task" "$seed" "$GPUS" "$$" "$CURRENT_STARTED" "$message" > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

run_dir() {
    local size="$1" task="$2" seed="$3"
    printf '%s/no_tools/qwen25vl_%s/%s/seed_%s\n' \
        "$RESULT_ROOT" "${size,,}" "$task" "$seed"
}

job_done() {
    local size="$1" task="$2" seed="$3" dir marker cache
    dir="$(run_dir "$size" "$task" "$seed")"
    marker="$dir/DONE.json"
    cache="$VALIDATED_CACHE_DIR/${size,,}_${task}_seed_${seed}.ok"
    [[ -f "$marker" ]] || { rm -f "$cache"; return 1; }
    if [[ -f "$cache" && "$cache" -nt "$marker" ]]; then
        return 0
    fi
    if python3 "$VALIDATE_RUN" \
        --run-dir "$dir" \
        --task "$task" \
        --task-matrix "$TASK_MATRIX" \
        --model-path "$(model_path "$size")" \
        --seed "$seed" >/dev/null 2>&1; then
        touch "$cache"
        return 0
    fi
    rm -f "$cache"
    return 1
}

archive_invalid_done() {
    local size="$1" task="$2" seed="$3" dir marker
    dir="$(run_dir "$size" "$task" "$seed")"
    marker="$dir/DONE.json"
    if [[ -f "$marker" ]] && ! job_done "$size" "$task" "$seed"; then
        mv "$marker" "$dir/DONE.invalid_$(date +%Y%m%d_%H%M%S)_$$.json"
    fi
}

failure_file() {
    local size="$1" task="$2" seed="$3"
    printf '%s/failures/%s_%s_seed_%s.count\n' "$STATE_DIR" "${size,,}" "$task" "$seed"
}

failure_count() {
    local file value
    file="$(failure_file "$1" "$2" "$3")"
    if [[ -s "$file" ]]; then
        value="$(cat "$file")"
        [[ "$value" =~ ^[0-9]+$ ]] && printf '%s\n' "$value" || printf '0\n'
    else
        printf '0\n'
    fi
}

record_failure() {
    local file count
    file="$(failure_file "$1" "$2" "$3")"
    count="$(failure_count "$1" "$2" "$3")"
    echo $((count + 1)) > "$file"
}

stop_child() {
    local signal="${1:-TERM}"
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -"$signal" -- "-$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    CHILD_PID=""
}

on_signal() {
    write_status "interrupted" "-" "-" "-" "received signal"
    stop_child TERM
    touch "$ABORTED_FILE" "$FINISHED_FILE"
    exit 130
}
trap on_signal INT TERM HUP

run_claimed_job() {
    local size="$1" task="$2" seed="$3"
    local gpu_var="GPU_${size}" tp_var="TP_${size}" rc
    CURRENT_STARTED="$(date +%s)"
    write_status "running" "$size" "$task" "$seed" \
        "attempt $(( $(failure_count "$size" "$task" "$seed") + 1 ))/$MAX_ATTEMPTS"

    echo
    echo "=================================================================="
    echo "worker=$WORKER_NAME queue=$QUEUE gpu=$GPUS"
    echo "job=$size/$task/seed_$seed tp=$TP start=$(date '+%F %T')"
    echo "=================================================================="

    archive_invalid_done "$size" "$task" "$seed"
    setsid env \
        "$gpu_var=$GPUS" \
        "$tp_var=$TP" \
        JOB_LOCK_HELD=1 \
        RESUME=1 \
        bash "$RUN_ONE" "$size" no_tools "$task" "$seed" &
    CHILD_PID=$!
    wait "$CHILD_PID"
    rc=$?
    CHILD_PID=""

    if [[ $rc -eq 0 || $rc -eq 100 ]]; then
        if job_done "$size" "$task" "$seed"; then
            echo "worker=$WORKER_NAME validated $size/$task/seed_$seed rc=$rc"
            write_status "completed" "$size" "$task" "$seed" "validated rc=$rc"
            return 0
        fi
        rc=102
        echo "worker=$WORKER_NAME got rc=0/100 without a valid DONE" >&2
    fi

    record_failure "$size" "$task" "$seed"
    echo "worker=$WORKER_NAME failed $size/$task/seed_$seed rc=$rc" >&2
    write_status "failed" "$size" "$task" "$seed" "rc=$rc"
    return "$rc"
}

queue_missing_count() {
    local missing=0 size task seed
    for size in "${SIZE_LIST[@]}"; do
        for task in "${TASK_LIST[@]}"; do
            for seed in "${SEED_LIST[@]}"; do
                job_done "$size" "$task" "$seed" || missing=$((missing + 1))
            done
        done
    done
    printf '%s\n' "$missing"
}

write_status "starting" "-" "-" "-" "queue=$QUEUE"
echo "Worker $WORKER_NAME started: queue=$QUEUE gpu=$GPUS"

while true; do
    acquired=0
    pending=0

    for size in "${SIZE_LIST[@]}"; do
        for task in "${TASK_LIST[@]}"; do
            for seed in "${SEED_LIST[@]}"; do
                if job_done "$size" "$task" "$seed"; then
                    continue
                fi

                attempts="$(failure_count "$size" "$task" "$seed")"
                if (( attempts >= MAX_ATTEMPTS )); then
                    continue
                fi
                pending=1

                lock_dir="$LOCK_ROOT/qwen25vl_${size,,}"
                lock_file="$lock_dir/${task}_seed_${seed}.lock"
                mkdir -p "$lock_dir"
                exec 9>"$lock_file"
                if ! flock -n 9; then
                    exec 9>&-
                    continue
                fi

                acquired=1
                run_claimed_job "$size" "$task" "$seed" || true
                flock -u 9 || true
                exec 9>&-
                break 3
            done
        done
    done

    if (( acquired )); then
        continue
    fi
    if (( pending )); then
        CURRENT_STARTED="$(date +%s)"
        write_status "waiting" "-" "-" "-" "another worker owns remaining jobs"
        sleep 5
        continue
    fi
    break
done

CURRENT_STARTED="$(date +%s)"
missing="$(queue_missing_count)"
touch "$FINISHED_FILE"
if (( missing == 0 )); then
    write_status "done" "-" "-" "-" "queue fully validated"
    touch "$DONE_FILE"
    echo "Worker $WORKER_NAME completed and validated the full $QUEUE queue."
    exit 0
fi

write_status "failed" "-" "-" "-" "$missing queue jobs missing after retry limit"
printf '%s\n' "$missing" > "$FAILED_FILE"
echo "Worker $WORKER_NAME exhausted the queue with $missing jobs still invalid." >&2
exit 1
