#!/bin/bash
# ============================================================================
# status_tools.sh — 查看工具服务运行状态
#
# 使用方法: bash .agent/ref/scripts/status_tools.sh
# ============================================================================
set -euo pipefail

CTRL=http://127.0.0.1:21112
LOGDIR=/data/songmingyang/code/reasoning/AdaReasoner-rebuttal/rebuttal_exps/toolserver_logs

echo "============================================"
echo "  AdaReasoner Tool Server 状态"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

echo ""
echo "--- 注册模型 ---"
MODELS=$(curl -s -m 3 -X POST "$CTRL/list_models" 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
ms=d.get('models',[])
print(', '.join(ms) if ms else '(空 — worker 可能还在加载)')
" 2>/dev/null || echo "(不可达)")
echo "  $MODELS"

echo ""
echo "--- 进程 ---"
check_proc() { pgrep -f "$1" >/dev/null 2>&1 && echo "  ✓ $2" || echo "  ✗ $2 (未运行)"; }
check_proc "controller.py.*21112"  "Controller  (:21112)"
check_proc "molmo_point_worker.*50002" "Point GPU1 (:50002)"
check_proc "molmo_point_worker.*50003" "Point GPU2 (:50003)"
check_proc "ocr_worker.*50010"     "OCR GPU1   (:50010)"
check_proc "ocr_worker.*50011"     "OCR GPU2   (:50011)"
check_proc "crop_worker_prompt.*50012" "Crop CPU   (:50012)"

echo ""
echo "--- 端口 ---"
for pair in "21112:Controller" "50002:Point (GPU1)" "50003:Point (GPU2)" \
            "50010:OCR (GPU1)" "50011:OCR (GPU2)" "50012:Crop (CPU)"; do
    p="${pair%%:*}"
    d="${pair#*:}"
    if ss -tlnp 2>/dev/null | grep -q ":$p "; then
        pid=$(ss -tlnp 2>/dev/null | grep ":$p " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
        echo "  ✓ :$p ($d)  PID=$pid"
    else
        echo "  ✗ :$p ($d) 未监听"
    fi
done

echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader | while read -r line; do
    idx=$(echo "$line" | cut -d',' -f1)
    used=$(echo "$line" | cut -d',' -f3 | xargs)
    free=$(echo "$line" | cut -d',' -f4 | xargs)
    util=$(echo "$line" | cut -d',' -f5 | xargs)
    note=""
    case $idx in
        0) note=" (预留 vLLM 实验)";;
        1) note=" (Point+OCR)";;
        2) note=" (Point+OCR)";;
        3) note=" (空闲备用)";;
    esac
    echo "  GPU $idx: $used / $free free, $util util$note"
done

echo ""
echo "--- 最近日志 (controller) ---"
if [ -f "$LOGDIR/controller.log" ]; then
    tail -3 "$LOGDIR/controller.log" | sed 's/^/  /'
else
    echo "  (无日志)"
fi

echo ""
echo "============================================"
