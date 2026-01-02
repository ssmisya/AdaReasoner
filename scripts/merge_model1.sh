#!/bin/bash

# --- 环境设置 ---
source ~/.bashrc
source ~/miniconda3/bin/activate vllm2

export CUDA_HOME=/mnt/petrelfs/share/cuda-12.4
export PATH=/mnt/petrelfs/share/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-12.4/lib64:$LD_LIBRARY_PATH

# --- 路径和变量设置 ---
code_base=/mnt/petrelfs/sunhaoyu/visual-code/DeepEyes/scripts
cd $code_base

ref_model_path=/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-VL-7B-Instruct-new
input_ckpt_dir=/mnt/petrelfs/sunhaoyu/visual-code/DeepEyes/checkpoints/tool_rl/web_7b_wo_tool/global_step_150/actor
target_dir=${input_ckpt_dir}/huggingface
hf_model_path=${input_ckpt_dir}/huggingface

# --- srun 执行参数 ---
gpus=0
cpus=12
quotatype="reserved"
OMP_NUM_THREADS=8

# --- 执行 srun 命令 ---
srun --partition=ai_moe \
     --job-name="merge_model_job" \
     --mpi=pmi2 \
     --gres=gpu:${gpus} \
     -n1 \
     --ntasks-per-node=1 \
     -c ${cpus} \
     --kill-on-bad-exit=1 \
     --quotatype=${quotatype} \
     python ./model_merger.py \
       --backend fsdp \
       --tie-word-embedding \
       --local_dir ${input_ckpt_dir} \
       --target_dir ${target_dir} \
       --hf_model_path ${hf_model_path} \
       --test \
       --test_hf_dir ${ref_model_path}