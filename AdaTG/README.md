
# AdaTG
##  Quick Start

### Environment Setup

```bash
# [Optional] Create a clean Conda environment
conda create -n tool-server python=3.10
conda activate tool-server
# Install PyTorch or prepare a torch-based environment
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124


# Install AdaTG
git clone https://github.com/ssmisya/AdaReasoner.git
cd AdaReasoner/AdaTG
# You can reference our requirements.txt for other dependencies
pip install -r ./requirements.txt 
pip install -e . # We didn't add too many constraints for easier installation
```

### Prepare Data & Reference Models
Please prepare your data in huggingface parquet format. The data format should follow:

```python
prompt = [
    {
        "content": system_prompt,
        "role": "system"
    },
    {
        "content": f"{question_text}",
        "role": "user"
    }
]
item = {
    "data_source": "jigsaw_coco",
    "prompt": prompt,
    "images": [{"bytes": question_image_bytes}] + choice_images,
    "ability": "visual_reasoning",
    "env_name": "jigsaw",
    "reward_model": {
        "ground_truth": correct_letter.lower(),
        "style": "model"
    },
    "extra_info": { # Used for reward calculation
        "extra_info1": "...",
    }
}

```
You can also refer to our [Data & Models](https://github.com/ssmisya/AdaReasoner/tree/main/docs/data_models.md) page for access to the provided datasets and pretrained models.

### Start Ray Cluster (Optional)

You can also start a Ray cluster to enable the dashboard and facilitate easier debugging.

```bash
ray start --head --port=$port --dashboard-host=0.0.0.0 \
--dashboard-port=$dashboard_port \
--ray-client-server-port=${client_server_port}  \
--num-cpus "64" \
--num-gpus "8"  \
--block --temp-dir="$TMPDIR" \
--min-worker-port ${min_worker_port} \
--max-worker-port ${max_worker_port}
```

### Start Training
> ⚠️ **Important:** Online tools require the **tool server** to be running.  
> Please start the tool server before invoking any online tools.

We recommend an 8×A100 GPU setup for training the 7B model.

```bash
python -m verl.trainer.main_ppo \
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
    actor_rollout_ref.rollout.agent.tool_manager.controller_addr=${controller_addr} \
    custom_reward_function.path=./verl/utils/reward_score/unified_tool.py \
    actor_rollout_ref.model.max_pixels=${MAX_PIXELS}
```

You can refer to `AdaReasoner/AdaTG/examples/adareasoner` for additional training and evaluation scripts.
A complete configuration example is provided in `AdaReasoner/AdaTG/examples/adareasoner/unified_randomize.yaml`.

## Advanced Usage

### Custom Reward Functions

You can implement your own reward functions by following the structure in `verl/utils/reward_score/unified_tool.py`.
