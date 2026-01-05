source ~/.bashrc
source ~/anaconda3/bin/activate vllm2

export CUDA_HOME=your-cuda-path
export PATH=your-cuda-path:$PATH
export LD_LIBRARY_PATH=your-cuda-path:$LD_LIBRARY_PATH


export API_TYPE=openai
export OPENAI_API_URL="https://yunwu.ai/v1/chat/completions"
export OPENAI_API_KEY="your-api-key"


config_file=$1
export NCCL_DEBUG=ERROR
export NCCL_DEBUG_SUBSYS=ALL

export VLLM_WORKER_MULTIPROC_METHOD=spawn

export CUDA_VISIBLE_DEVICES="0,1"
node=SH-IDC1-10-140-37-127

gpus=0
cpus=2
quotatype="auto"
OMP_NUM_THREADS=8 srun --partition=ai_moe -w ${node} --job-name="eval" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python \
-m tool_server.tf_eval --config  ${config_file} 

