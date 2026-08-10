#!/bin/bash
# ============================================================================
# start_tools.sh — 一键启动全套工具服务
#
# GPU 布局 (4×A100 80GB, Point ~16GB + OCR ~0.5GB = 16.5GB 绰绰有余):
#   GPU 0 — 预留 vLLM 实验推理 (TP rank 0)
#   GPU 1 — Point (:50002) + OCR (:50010) — 合占 ~16.3GB
#   GPU 2 — Point (:50003) + OCR (:50011) — 合占 ~16.3GB
#   GPU 3 — 预留 vLLM 实验推理 (TP rank 1)
#
# 端口:
#   :21112 — Controller
#   :50002 — Point (GPU1)    :50003 — Point (GPU2)
#   :50010 — OCR  (GPU1)     :50011 — OCR  (GPU2)
#   :50012 — Crop  (CPU)
#
# 环境: Point/Crop/Controller 用 vllm-latest, OCR 用 ocr-server
#
# 使用方法:
#   启动: bash .agent/ref/scripts/start_tools.sh
#   状态: bash .agent/ref/scripts/status_tools.sh
#   停止: bash .agent/ref/scripts/stop_tools.sh
# ============================================================================
set -euo pipefail

REPO=/data/songmingyang/code/reasoning/AdaReasoner-rebuttal
LOGDIR="$REPO/rebuttal_exps/toolserver_logs"
CTRL=http://127.0.0.1:21112

# ---- 环境 -----
source /data/songmingyang/miniforge3/etc/profile.d/conda.sh

# ---- 路径常量 ----
MOLMO=/data/songmingyang/shared_models/adareasoner/tools/Molmo-7B-D-0924
OCR_PYTHON=/data/songmingyang/miniforge3/envs/ocr-server/bin/python

# ---- 端口常量 ----
PORT_CTRL=21112
PORT_POINT_GPU1=50002
PORT_POINT_GPU2=50003
PORT_OCR_GPU1=50010
PORT_OCR_GPU2=50011
PORT_CROP=50012

mkdir -p "$LOGDIR"
cd "$REPO"

echo "============================================"
echo "  AdaReasoner Tool Server — 一键启动"
echo "  GPU1: Point + OCR    GPU2: Point + OCR"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# ---- util: 检查端口是否已监听 ----
port_alive() { ss -tlnp 2>/dev/null | grep -q ":$1 "; }

# ---- util: 等待 controller 就绪 ----
wait_controller() {
    for i in $(seq 1 30); do
        if curl -s -m 2 -X POST "$CTRL/list_models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: controller 未能在 30s 内就绪"
    return 1
}

# ---- util: 启动一个 GPU worker ----
launch_gpu_worker() {
    local label="$1"         # 显示标签 e.g. "Point (GPU1)"
    local port="$2"          # 绑定端口
    local gpu_idx="$3"       # 物理 GPU 编号 e.g. 1 or 2
    local logfile="$4"       # 日志文件
    local module="$5"        # Python 模块
    local model_name="$6"    # controller 注册名
    local extra_args="${7:-}" # 额外参数 (如 --model_path, --gpu_ids)
    local python_bin="${8:-python}"
    local conda_env="${9:-vllm-latest}"

    if port_alive "$port"; then
        echo "  -> $label 已在运行 (端口 $port)"
        return
    fi

    conda activate "$conda_env"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$REPO"

    CUDA_VISIBLE_DEVICES="$gpu_idx" nohup $python_bin -m "$module" \
        --host 127.0.0.1 --port "$port" \
        --controller_addr "$CTRL" \
        --model_name "$model_name" \
        $extra_args \
        > "$logfile" 2>&1 &
    echo "  -> $label PID=$!  GPU=$gpu_idx  端口=$port"
}

# ---- util: 统计已注册模型数 ----
registered_count() {
    curl -s -m 3 -X POST "$CTRL/list_models" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('models',[])))" 2>/dev/null || echo 0
}
registered_names() {
    curl -s -m 3 -X POST "$CTRL/list_models" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(', '.join(d.get('models',[])))" 2>/dev/null || echo "?"
}

# ===================================================================
# 1. Controller (无 GPU, :21112)
# ===================================================================
echo ""
echo "[1/7] 启动 Controller ..."
conda activate vllm-latest
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO"

if port_alive "$PORT_CTRL"; then
    echo "  -> Controller 已在运行 (端口 $PORT_CTRL)"
else
    CUDA_VISIBLE_DEVICES="" nohup python tool_server/tool_workers/online_workers/controller.py \
        --host 0.0.0.0 --port "$PORT_CTRL" \
        > "$LOGDIR/controller.log" 2>&1 &
    echo "  -> Controller PID=$!  日志: $LOGDIR/controller.log"
    sleep 2
fi
wait_controller
echo "  -> Controller 就绪 ✓"

# ===================================================================
# 2. Point Worker — GPU 1, :50002  [Molmo-7B ~16GB]
# ===================================================================
echo ""
echo "[2/7] 启动 Point Worker on GPU 1 ..."
launch_gpu_worker "Point (GPU1)" "$PORT_POINT_GPU1" 1 \
    "$LOGDIR/point_${PORT_POINT_GPU1}.log" \
    "tool_server.tool_workers.online_workers.molmo_point_worker" \
    "Point" \
    "--model_path $MOLMO"

# ===================================================================
# 3. OCR Worker — GPU 1, :50010  [PaddleOCR ~0.5GB, 独立 env]
# ===================================================================
echo ""
echo "[3/7] 启动 OCR Worker on GPU 1 ..."
launch_gpu_worker "OCR  (GPU1)" "$PORT_OCR_GPU1" 1 \
    "$LOGDIR/ocr_${PORT_OCR_GPU1}.log" \
    "tool_server.tool_workers.online_workers.ocr_worker" \
    "OCR" \
    "--gpu_ids 0" \
    "$OCR_PYTHON" "ocr-server"

# ===================================================================
# 4. Point Worker — GPU 2, :50003  [Molmo-7B ~16GB]
# ===================================================================
echo ""
echo "[4/7] 启动 Point Worker on GPU 2 ..."
launch_gpu_worker "Point (GPU2)" "$PORT_POINT_GPU2" 2 \
    "$LOGDIR/point_${PORT_POINT_GPU2}.log" \
    "tool_server.tool_workers.online_workers.molmo_point_worker" \
    "Point" \
    "--model_path $MOLMO"

# ===================================================================
# 5. OCR Worker — GPU 2, :50011  [PaddleOCR ~0.5GB]
# ===================================================================
echo ""
echo "[5/7] 启动 OCR Worker on GPU 2 ..."
launch_gpu_worker "OCR  (GPU2)" "$PORT_OCR_GPU2" 2 \
    "$LOGDIR/ocr_${PORT_OCR_GPU2}.log" \
    "tool_server.tool_workers.online_workers.ocr_worker" \
    "OCR" \
    "--gpu_ids 0" \
    "$OCR_PYTHON" "ocr-server"

# ===================================================================
# 6. Crop Worker (CPU, :50012)
# ===================================================================
echo ""
echo "[6/7] 启动 Crop Worker (CPU) ..."
if port_alive "$PORT_CROP"; then
    echo "  -> Crop 已在运行 (端口 $PORT_CROP)"
else
    conda activate vllm-latest
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$REPO"
    CUDA_VISIBLE_DEVICES="" nohup python -m tool_server.tool_workers.online_workers.crop_worker_prompt \
        --host 127.0.0.1 --port "$PORT_CROP" \
        --controller_addr "$CTRL" \
        --model_name Crop \
        > "$LOGDIR/crop_${PORT_CROP}.log" 2>&1 &
    echo "  -> Crop PID=$!  端口=$PORT_CROP"
fi

# ===================================================================
# 7. 等待所有 worker 注册 (Pointx2 + OCRx2 + Crop = controller 去重后可能 3 或 5)
#    最长等待 3 分钟 (Molmo 加载约 60s)
# ===================================================================
echo ""
echo "  等待所有 worker 注册到 controller (最长 180s)..."

# 最少去重模型名: Point, OCR, Crop = 3
MIN_MODELS=3
for i in $(seq 1 36); do
    C=$(registered_count)
    M=$(registered_names)
    if [ "$C" -ge "$MIN_MODELS" ]; then
        echo "  -> 已注册 $C 种模型: $M"
        break
    fi
    printf "  -> [%2d/180s] 已注册 %d 种模型: %s\r" "$((i*5))" "$C" "$M"
    sleep 5
done

echo ""
echo ""
echo "--- 注册模型 ---"
echo "  $(registered_names)"

echo ""
echo "--- 端口检测 ---"
for pair in "$PORT_CTRL:Controller" "$PORT_POINT_GPU1:Point (GPU1)" "$PORT_POINT_GPU2:Point (GPU2)" \
            "$PORT_OCR_GPU1:OCR (GPU1)" "$PORT_OCR_GPU2:OCR (GPU2)" "$PORT_CROP:Crop (CPU)"; do
    p="${pair%%:*}"
    d="${pair#*:}"
    if port_alive "$p"; then
        echo "  ✓ :$p  [$d]"
    else
        echo "  ✗ :$p  [$d] 未监听!"
    fi
done

echo ""
echo "--- GPU 显存 ---"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | while read -r line; do
    idx=$(echo "$line" | cut -d',' -f1)
    note=""
    case $idx in
        0) note=" (预留 vLLM TP rank 0)";;
        1) note=" (Point+OCR)";;
        2) note=" (Point+OCR)";;
        3) note=" (预留 vLLM TP rank 1)";;
    esac
    echo "    GPU$line$note"
done

echo ""
echo "============================================"
echo "  启动完成。"
echo ""
echo "  日志: tail -f $LOGDIR/{controller,point_50002,point_50003,ocr_50010,ocr_50011,crop_50012}.log"
echo "  状态: bash .agent/ref/scripts/status_tools.sh"
echo "  停止: bash .agent/ref/scripts/stop_tools.sh"
echo "============================================"
