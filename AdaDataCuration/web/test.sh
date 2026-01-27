#!/bin/bash

# set -x # 取消注释以打印所有运行的命令

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_0

# 检查是否在tmux会话内运行
# 如果不在，则创建一个新的tmux会话并在其中重新执行此脚本
if [ -z "$TMUX" ]; then
  # 从下面的变量定义中获取项目和实验名，用于命名tmux会话
  SESSION_NAME="test_7b_full_dataset"

  echo "创建新的tmux会话: $SESSION_NAME"
  tmux new-session -d -s "$SESSION_NAME" "bash $0 inside_tmux"
  echo "tmux会话已在后台启动，你可以通过以下命令查看:"
  echo "tmux attach -t $SESSION_NAME"
  exit 0
fi

# 确保脚本是在tmux内部被调用的
if [ "$1" != "inside_tmux" ]; then
  echo "请通过不带参数的方式运行此脚本，以启动新的tmux会话"
  exit 1
fi

# 1. 环境配置 (您的配置)
# ==========================================================
source ~/.bashrc
source ~/miniconda3/bin/activate vllm2

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

export CUDA_HOME=/mnt/petrelfs/share/cuda-12.4
export PATH=/mnt/petrelfs/share/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-12.4/lib64:$LD_LIBRARY_PATH

srun \
  --partition=ai_moe \
  --mpi=pmi2 \
  --job-name=test_eval \
  -c 16 \
  --gres=gpu:0 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  --quotatype=reserved \
  python AdaReasoner/AdaDataCurationweb/test_7b_guichat.py