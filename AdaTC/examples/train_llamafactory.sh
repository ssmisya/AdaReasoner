#!/usr/bin/bash
source ~/.bashrc
source ~/anaconda3/bin/activate llamafactory
code_base=/your/path/to/LLaMA-Factory
cd $code_base



lf_config=$1

# spot reserved auto
num_nodes=1      # should match with --nodes
gpus=0           # should match with --gres
cpus=2          # should match with --cpus-per-task
quotatype="reserved"


OMP_NUM_THREADS=8 FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 srun  \
--partition=ai_moe --job-name="llamafactory" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
 llamafactory-cli train $lf_config

