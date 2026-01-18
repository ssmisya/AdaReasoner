#!/bin/bash

# set -x # 取消注释以打印所有运行的命令

export PROJECT_NAME="tool_rl"
export EXPERIMENT_NAME="frozen_7b_wo_tool"

# 检查是否在tmux会话内运行
# 如果不在，则创建一个新的tmux会话并在其中重新执行此脚本
if [ -z "$TMUX" ]; then
  # 从下面的变量定义中获取项目和实验名，用于命名tmux会话
  SESSION_NAME="${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"

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

export LLM_AS_A_JUDGE_BASE="http://SH-IDC1-10-140-37-71:16113/v1"
export LLM_AS_A_JUDGE_MODEL="/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-72B-Instruct"
export no_proxy="localhost,127.0.0.1,sh-idc1-10-140-37-71,10.140.37.71,sh-idc1-10-140-37-6,10.140.37.6"

export CUDA_HOME=/mnt/petrelfs/share/cuda-12.4
export PATH=/mnt/petrelfs/share/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-12.4/lib64:$LD_LIBRARY_PATH

export LDFLAGS="-ldl"
export CFLAGS="-I/libaio/usr/include $CFLAGS"
export LDFLAGS="-L/libaio/usr/lib $LDFLAGS"
export C_INCLUDE_PATH=/libaio/usr/include
export LD_LIBRARY_PATH="/libaio/usr/lib:$LD_LIBRARY_PATH"

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=ERROR
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset RAY_ADDRESS
export RAY_NODE_IP_ADDRESS=127.0.0.1

# 2. 项目和实验配置 (您的配置)
# ==========================================================
code_base="/DeepEyes"
cd $code_base

if [ -z "$EXPERIMENT_NAME" ]; then
  echo "错误: 环境变量 EXPERIMENT_NAME 未设置或为空。"
  echo "脚本已终止。"
  exit 1
fi

log_dir="${code_base}/logs"
mkdir -p $log_dir
log_file="${log_dir}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"

SAVE_CHECKPOINT_DIR="/DeepEyes/checkpoints"
FROZENLAKE_DATASET_TRAIN="/datasets/vsp_wo_tool_rl/train.parquet"
FROZENLAKE_DATASET_VAL="/datasets/vsp_wo_tool_rl/test.parquet"
REF_MODEL_PATH="/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-VL-7B-Instruct-new"


# 4. 构建并执行训练命令 (已集成srun)
node_list="SH-IDC1-10-140-37-138"
gpus=0 
cpus=2
partition="ai_moe"
quotatype="reserved"
# quotatype="spot"
# export CUDA_VISIBLE_DEVICES=2,3,4,5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 基础的python命令 (您的配置)
python_cmd="python3 -m verl.trainer.main_ppo \
    +debug=False \
    +vs_debug=False \
    data.train_files=[${FROZENLAKE_DATASET_TRAIN}] \
    data.val_files=[${FROZENLAKE_DATASET_VAL}] \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=20480 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=10240 \
    actor_rollout_ref.rollout.agent.max_turns=10 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','rl_logging_board'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100000 \
    trainer.test_freq=10 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.resume_mode=auto \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=10 \
    custom_reward_function.path=/DeepEyes/verl/utils/reward_score/frozenlake_wo_tool.py"

# 使用 srun 包装器构建最终的完整命令
cmd="OMP_NUM_THREADS=8 srun --partition=${partition} -w ${node_list} --job-name=\"${EXPERIMENT_NAME}\" --mpi=pmi2 --export=ALL --no-kill --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
${python_cmd}"


# 执行命令并记录日志
echo "==========================================================" | tee -a ${log_file}
echo "开始训练..." | tee -a ${log_file}
echo "执行命令:" | tee -a ${log_file}
echo "$cmd" | tee -a ${log_file}
echo "==========================================================" | tee -a ${log_file}

# 使用eval执行命令，并将标准输出和标准错误都重定向到tee
eval "$cmd" 2>&1 | tee -a ${log_file}

echo "==========================================================" | tee -a ${log_file}
echo "运行结束。日志已保存到: ${log_file}" | tee -a ${log_file}