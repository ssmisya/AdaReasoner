source ~/.bashrc # 这个可以有可以没有吧
source ~/miniconda3/bin/activate visual

proxy_off

# environment variables
# export OMP_NUM_THREADS=4
# AD_NAME=songmingyang
# encrypted_password=iWRsYqbwV4EJgJvU8QjLe00CptZc5jBVH3FMo5i6n9mVdOSoUurpyBTmst1Z
# new_proxy_address=http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/
# export http_proxy=$new_proxy_address
# export https_proxy=$new_proxy_address
# export HTTP_PROXY=$new_proxy_address
# export HTTPS_PROXY=$new_proxy_address
# unset http_proxy
# unset https_proxy
# unset HTTP_PROXY
# unset HTTPS_PROXY

export CUDA_HOME=/mnt/petrelfs/share/cuda-11.8
export PATH=/mnt/petrelfs/share/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.8/lib64:$LD_LIBRARY_PATH

export LDFLAGS="-ldl"

export CFLAGS="-I/mnt/petrelfs/sunhaoyu/visual-code/libaio/usr/include $CFLAGS"
export LDFLAGS="-L/mnt/petrelfs/sunhaoyu/visual-code/libaio/usr/lib $LDFLAGS"
export C_INCLUDE_PATH=/mnt/petrelfs/sunhaoyu/visual-code/libaio/usr/include
export LD_LIBRARY_PATH="/mnt/petrelfs/sunhaoyu/visual-code/libaio/usr/lib:$LD_LIBRARY_PATH"

export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT

code_base=/mnt/petrelfs/sunhaoyu/visual-code/OpenThinkIMG/tool_server
cd $code_base
# job_id=5079273
# export SLURM_JOB_ID=${job_id}
# unset SLURM_JOB_ID


export API_TYPE=openai
export OPENAI_API_URL=https://api.datapipe.app/v1/chat/completions
export OPENAI_API_KEY=sk-B3bRcR0fLubdoSmJ2cE13e57708c439aA14f825eB5Eb25De


# config_file=$1
export NCCL_DEBUG=ERROR
export NCCL_DEBUG_SUBSYS=ALL
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn

gpus=1
cpus=32
quotatype="spot"
OMP_NUM_THREADS=8 srun --partition=ai_moe --job-name="test" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python \
-m tool_server.tf_eval.test

# torchrun --nproc_per_node=${gpus} \
# salloc --partition=MoE --job-name="eval" --gres=gpu:8 -n1 --ntasks-per-node=1 -c 64 --quotatype="reserved"
# salloc --partition=MoE --job-name="interact" --gres=gpu:1 -n1 --ntasks-per-node=1 -c 16 --quotatype="reserved"
