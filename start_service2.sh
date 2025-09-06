#!/bin/bash

# 创建一个名为tool-factory的tmux会话，加上时间戳
session_name="tool-factory222-$(date +%Y%m%d_%H%M%S)"
tmux new-session -d -s $session_name

export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-12.4/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/mnt/petrelfs/share/cuda-12.4/bin:$PATH

# 在tmux会话中运行环境设置和服务
tmux send-keys -t $session_name "cd $(pwd)" C-m
tmux send-keys -t $session_name "conda activate tool-server" C-m

# 关键：在启动Python服务前，清除所有的代理环境变量
# 这会确保接下来的python命令在一个无代理的环境中运行
tmux send-keys -t $session_name "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY" C-m

# 运行python服务
tmux send-keys -t $session_name "python -m tool_server.tool_workers.scripts.launch_scripts.start_server_config_shy_指定结点2 --config tool_server/tool_workers/scripts/launch_scripts/config/all_service_example_shy_指定结点2.yaml" C-m

echo "服务已在tmux会话中启动"
echo "你可以通过以下命令查看服务运行状态："
echo "tmux attach -t $session_name"
echo "按Ctrl+B后按D可以从会话中分离，保持服务在后台运行"