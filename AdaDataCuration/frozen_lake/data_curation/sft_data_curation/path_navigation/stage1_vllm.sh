#!/bin/bash
source ~/.bashrc
source ~/anaconda3/bin/activate vllm2

export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT

export NCCL_DEBUG=ERROR
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_0

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

quotatype="reserved"
gpus=0
cpus=16
node_list="SH-IDC1-10-140-37-118"

# VLLM 服务器配置
VLLM_BASE_URL="http://SH-IDC1-10-140-37-118:7120/v1"
VLLM_API_KEY="EMPTY"  # VLLM 通常不需要真实的 API key
MODEL_NAME="Qwen2.5-VL-7B-Instruct"

# 数据生成配置
INPUT_PATH="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/metadata_split/path_navigation/sft.jsonl"
OUTPUT_PATH="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/path_navigation"
IMAGE_DIR="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation"
MAX_SAMPLES=2000
NUM_THREADS=32  # VLLM 支持高并发

# 构建命令
OMP_NUM_THREADS=8 srun --partition=ai_moe -w ${node_list} --job-name="path-nav-gen" --mpi=pmi2 --export=ALL --no-kill --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
python /mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/sft_data_curation/path_navigation/path_navigation_stage1.py \
    --input_path ${INPUT_PATH} \
    --output_path ${OUTPUT_PATH} \
    --model ${MODEL_NAME} \
    --api_provider openai \
    --max_samples ${MAX_SAMPLES} \
    --image_dir ${IMAGE_DIR} \
    --openai_api_key ${VLLM_API_KEY} \
    --openai_base_url ${VLLM_BASE_URL} \
    --num_threads ${NUM_THREADS} \
    --seed 42