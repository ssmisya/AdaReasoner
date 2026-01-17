#!/bin/bash


source ~/.bashrc
source ~/anaconda3/bin/activate toolrl


export CUDA_HOME=/mnt/petrelfs/share/cuda-12.4
export PATH=/mnt/petrelfs/share/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-12.4/lib64:$LD_LIBRARY_PATH



export PYTHONUNBUFFERED=1
export NCCL_DEBUG=ERROR
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset RAY_ADDRESS
export RAY_NODE_IP_ADDRESS=127.0.0.1

cd AdaReasoner/AdaTG


log_dir="./logs"
mkdir -p $log_dir
log_file="${log_dir}/${EXPERIMENT_NAME}.log"

SAVE_CHECKPOINT_DIR=checkpoints/rl/v2

# Prepare Training data
FROZENLAKE_DATASET_TRAIN="/your/path/to/vstar_3tasls/unified_train_4tasks.parquet" 
FROZENLAKE_DATASET_VAL="/your/path/to/vstar_3tasls/unified_test_4tasks.parquet"

# Prepare Reference Modle
REF_MODEL_PATH="/mnt/petrelfs/songmingyang/songmingyang/runs/tool_factory/sft/v1/Qwen2.5-VL-7B-Instruct-unified_jigsaw_vsp_web_v1_randomized/checkpoint-300"

export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL
unset LD_PRELOAD   # 以防之前预加载了别的 OMP



export PROJECT_NAME="tool_rl"
export EXPERIMENT_NAME="unified_all_randomized_sft_normal_rl_4tasks_7b_new"
MAX_PIXELS=1280*28*28*3


# 基础的python命令 (您的配置)
python_cmd="python -m verl.trainer.main_ppo \
    --config-path . \
    --config-name unified \
    +debug=False \
    +vs_debug=False \
    data.train_files=[${FROZENLAKE_DATASET_TRAIN}] \
    data.val_files=[${FROZENLAKE_DATASET_VAL}] \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=20480 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
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
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
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
    actor_rollout_ref.rollout.agent.tool_manager.controller_addr=http://SH-IDC1-10-140-37-24:21112 \
    custom_reward_function.path=./verl/utils/reward_score/unified_tool.py \
    actor_rollout_ref.model.max_pixels=${MAX_PIXELS} "

node_list=""
cluster_addr=10.140.37.57

gpus=0 
cpus=2
partition="ai_moe"
quotatype="reserved"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# 使用 srun 包装器构建最终的完整命令
cmd="OMP_NUM_THREADS=8 srun --partition=${partition} -w ${node_list} --job-name=\"${EXPERIMENT_NAME}\" --mpi=pmi2 --export=ALL --no-kill --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
ray job submit --address="http://${cluster_addr}:10021" --working-dir . \
-- \
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