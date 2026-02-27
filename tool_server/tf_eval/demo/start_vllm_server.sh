#!/bin/bash

source ~/.bashrc
source ~/anaconda3/bin/activate vllm2

export CUDA_HOME=/mnt/petrelfs/share/cuda-12.4
export PATH=/mnt/petrelfs/share/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-12.4/lib64:$LD_LIBRARY_PATH

export LDFLAGS="-ldl"


export NCCL_DEBUG=ERROR
export NCCL_DEBUG_SUBSYS=ALL

export VLLM_WORKER_MULTIPROC_METHOD=spawn



export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT


# config_file=$1
export NCCL_DEBUG=ERROR
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_0

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export PYTHONUNBUFFERED=1


quotatype="reserved"

gpus=0
cpus=2

export CUDA_VISIBLE_DEVICES="6,7"
node=SH-IDC1-10-140-37-127

tensor_parallel_size=2
cd /mnt/petrelfs/songmingyang/code/reasoning/opensource/Tool-Factory-Filter/tool_server/tf_eval/demo

# 创建logs目录（如果不存在）
mkdir -p logs

# 获取当前时间作为日志文件名
log_file="logs/vllm_server1_$(date +%Y%m%d_%H%M%S).log"

model_path=/mnt/petrelfs/songmingyang/songmingyang/runs/tool_factory/rl/v2/tool_rl/test_ckpts/unified_all_randomized_sft_randomized_rl_4tasks_7b/global_step_250/actor/huggingface

model_path=/mnt/petrelfs/songmingyang/songmingyang/runs/tool_factory/rl/v2/tool_rl/test_ckpts/unified_all_randomized_sft_normal_rl_4tasks_7b_new/global_step_400/actor/huggingface

export SLURM_JOB_ID=7149367
unset SLURM_JOB_ID
# 构建命令
cmd="OMP_NUM_THREADS=8 srun --partition=ai_moe -w ${node} --job-name=\"vllm-server\" --mpi=pmi2 --export=ALL --no-kill --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
python \
-m vllm.entrypoints.openai.api_server \
--model ${model_path} \
--port 16116 \
--limit-mm-per-prompt image=10 \
--tensor-parallel-size ${tensor_parallel_size} \
--enforce-eager 2>&1 | tee ${log_file}" 

echo "日志将保存到: ${log_file}"

# 执行命令
eval $cmd