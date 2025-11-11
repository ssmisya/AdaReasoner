source ~/.bashrc
source ~/anaconda3/bin/activate vllm2

export CUDA_HOME=/mnt/petrelfs/share/cuda-12.4
export PATH=/mnt/petrelfs/share/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-12.4/lib64:$LD_LIBRARY_PATH


gpus=0
cpus=16
quotatype="reserved"
export CUDA_VISIBLE_DEVICES=1
OMP_NUM_THREADS=8 srun --partition=ai_moe --job-name="llm_eval" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/llm_eval.py