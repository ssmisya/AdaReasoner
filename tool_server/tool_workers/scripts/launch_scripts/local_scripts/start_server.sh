# environment variables
source ~/.bashrc
source ~/anaconda3/bin/activate tool_server

cd /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tool_workers/scripts/launch_scripts

export SLURM_JOB_ID=5079274
# unset SLURM_JOB_ID     

gpus=4
cpus=32
quotatype="auto"

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="dino" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./start_server_local.py \
--config ./config/all_service_example_local.yaml

