source ~/.bashrc
source ~/anaconda3/bin/activate toolrl

cd AdaReasoner/AdaTG


export VLLM_WORKER_MULTIPROC_METHOD="spawn"

export TMPDIR="/tmp"

HYDRA_FULL_ERROR=1


export RAY_memory_monitor_refresh_ms=0
gpus=0
cpus=2
quotatype="reserved"

cluster_addr=cluster_addr
cluster_ip=cluster_ip
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

port=6380
dashboard_port=10021
client_server_port=7132

min_worker_port=20000
max_worker_port=20500

echo "Starting ray at IP: $cluster_ip, PORT: $port"
srun --partition=ai_moe \
-w ${cluster_addr} \
--job-name="RAY-CLUSTER" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
ray start --head --port=$port --dashboard-host=0.0.0.0 --dashboard-port=$dashboard_port --ray-client-server-port=${client_server_port}  --num-cpus "64" --num-gpus "8"  --block --temp-dir="$TMPDIR" --min-worker-port ${min_worker_port} --max-worker-port ${max_worker_port}

