#!/bin/bash
# ============================================================================
# stop_tools.sh — 停止所有工具服务
#
# 使用方法: bash .agent/ref/scripts/stop_tools.sh
# ============================================================================
set -euo pipefail

echo "============================================"
echo "  停止 AdaReasoner Tool Server"
echo "============================================"

kill_by_pattern() {
    local pattern="$1"
    local label="${2:-$pattern}"
    local pids
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  终止 $label (PID: $(echo $pids)) ..."
        kill $pids 2>/dev/null || true
        sleep 1
        pids=$(pgrep -f "$pattern" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "  强杀 $label ..."
            kill -9 $pids 2>/dev/null || true
        fi
    else
        echo "  $label: 未运行"
    fi
}

kill_by_pattern "controller.py.*--port 21112"  "Controller  (:21112)"
kill_by_pattern "molmo_point_worker.*50002"    "Point GPU1 (:50002)"
kill_by_pattern "molmo_point_worker.*50003"    "Point GPU2 (:50003)"
kill_by_pattern "ocr_worker.*50010"            "OCR GPU1   (:50010)"
kill_by_pattern "ocr_worker.*50011"            "OCR GPU2   (:50011)"
kill_by_pattern "crop_worker_prompt.*50012"    "Crop CPU   (:50012)"

# 清理残留
for leftover in $(pgrep -f "molmo_point_worker\|ocr_worker\|crop_worker_prompt\|controller.py" 2>/dev/null || true); do
    echo "  清理残留 PID=$leftover"
    kill -9 $leftover 2>/dev/null || true
done

sleep 1

echo ""
echo "--- 端口残留检查 ---"
for p in 21112 50002 50003 50010 50011 50012; do
    if ss -tlnp 2>/dev/null | grep -q ":$p "; then
        echo "  :$p  仍在监听"
    else
        echo "  :$p  已释放 ✓"
    fi
done

echo ""
echo "============================================"
echo "  已停止。"
echo "============================================"
