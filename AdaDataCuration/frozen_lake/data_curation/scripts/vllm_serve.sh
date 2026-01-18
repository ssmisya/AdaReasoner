source ~/.bashrc
source ~/anaconda3/bin/activate vllm2



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
node_list="SH-IDC1-10-140-37-118"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# unset CUDA_VISIBLE_DEVICES


# 获取当前时间作为日志文件名
log_file="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/scripts/logs/vllm_server.log"

# 构建命令
OMP_NUM_THREADS=8 srun --partition=ai_moe -w ${node_list} --job-name=\"vllm-server\" --mpi=pmi2 --export=ALL --no-kill --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
python \
-m vllm.entrypoints.openai.api_server \
--model /mnt/petrelfs/songmingyang/songmingyang/model/mm/Qwen2.5-VL-7B-Instruct \
--port 7120 \
--tensor-parallel-size 2 \
-dp 4 \
--distributed-executor-backend mp \
--served-model-name "Qwen2.5-VL-7B-Instruct" \
--enforce-eager 2>&1 | tee ${log_file}


# python \
# -m vllm.entrypoints.openai.api_server \
# --model /mnt/petrelfs/songmingyang/songmingyang/model/mm/Qwen2.5-VL-7B-Instruct \
# --port 16112 \
# --tensor-parallel-size 1 \
# --enforce-eager 2>&1 | tee ${log_file}

