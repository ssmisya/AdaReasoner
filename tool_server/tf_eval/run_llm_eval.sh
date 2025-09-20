source ~/.bashrc
source ~/miniconda3/bin/activate vllm2

export CUDA_HOME=/mnt/petrelfs/share/cuda-12.4
export PATH=/mnt/petrelfs/share/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-12.4/lib64:$LD_LIBRARY_PATH

export LDFLAGS="-ldl"

export CFLAGS="-I/mnt/petrelfs/sunhaoyu/visual-code/libaio/usr/include $CFLAGS"
export LDFLAGS="-L/mnt/petrelfs/sunhaoyu/visual-code/libaio/usr/lib $LDFLAGS"
export C_INCLUDE_PATH=/mnt/petrelfs/sunhaoyu/visual-code/libaio/usr/include
export LD_LIBRARY_PATH="/mnt/petrelfs/sunhaoyu/visual-code/libaio/usr/lib:$LD_LIBRARY_PATH"

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_0

# config_file=$1
export NCCL_DEBUG=ERROR
export NCCL_DEBUG_SUBSYS=ALL
export VLLM_WORKER_MULTIPROC_METHOD=spawn

gpus=0
cpus=8
quotatype="reserved"
export CUDA_VISIBLE_DEVICES=1
OMP_NUM_THREADS=8 srun --partition=ai_moe --job-name="llm_eval" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python \
-m llm_eval
