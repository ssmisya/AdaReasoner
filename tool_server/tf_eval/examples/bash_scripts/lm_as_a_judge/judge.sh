source ~/.bashrc
source ~/anaconda3/bin/activate vllm2


export CUDA_HOME=/mnt/petrelfs/share/cuda-12.1
export PATH=/mnt/petrelfs/share/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-12.1/lib64:$LD_LIBRARY_PATH



config_file=$1
export NCCL_DEBUG=ERROR
export NCCL_DEBUG_SUBSYS=ALL

export VLLM_WORKER_MULTIPROC_METHOD=spawn


input_path=$1

gpus=0
cpus=16
quotatype="reserved"
OMP_NUM_THREADS=8 srun --partition=ai_moe --job-name="llmjudge" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/llm_eval.py \
--jsonl_paths $input_path

