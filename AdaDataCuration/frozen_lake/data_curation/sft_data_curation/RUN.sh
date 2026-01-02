# environment variables
source ~/.bashrc
source ~/anaconda3/bin/activate vllm2

cd /mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation

export SLURM_JOB_ID=3273170
unset SLURM_JOB_ID     

gpus=0
cpus=16
quotatype="auto"

candidate_path="/mnt/petrelfs/songmingyang/code/reasoning/data_construction/refocus/data/refocus_chartqa_v_bar_wbb_selfbar_candidates.jsonl"
output_path="/mnt/petrelfs/songmingyang/code/reasoning/data_construction/refocus/data/refocus_chartqa_v_bar_wbb_selfbar_reformatted.jsonl"

srun --partition=ai_moe --mpi=pmi2 --job-name=pruner --gres=gpu:${gpus} -c ${cpus} --nodes=1 --ntasks-per-node=1 --kill-on-bad-exit=1 --quotatype=reserved \
python /mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/gemini_generate/gemini_point_random.py



# srun --partition=MoE --mpi=pmi2 --job-name=pruner --gres=gpu:1 --nodes=1 --ntasks-per-node=1 --kill-on-bad-exit=1 --quotatype=reserved \
#  python ../serve/grounding_dino_worker.py
# -w SH-IDCA1404-10-140-54-67 \