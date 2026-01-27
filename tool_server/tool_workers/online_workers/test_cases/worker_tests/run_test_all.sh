source ~/miniconda3/bin/activate visual

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_0

srun \
  --partition=ai_moe \
  --mpi=pmi2 \
  --job-name=test_eval \
  -c 4 \
  --gres=gpu:0 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  --quotatype=spot \
  python /tool_server/tool_workers/online_workers/test_cases/worker_tests/test_all.py --controller_addr http://SH-IDC1-10-140-37-6:21112


