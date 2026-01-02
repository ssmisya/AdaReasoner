# environment variables
source ~/.bashrc
source ~/anaconda3/bin/activate vllm2

cd /mnt/petrelfs/songmingyang/code/reasoning/data_construction/PixelReasoner

export SLURM_JOB_ID=3273170
unset SLURM_JOB_ID     

gpus=0
cpus=16
quotatype="auto"

candidate_path="/mnt/petrelfs/songmingyang/code/reasoning/data_construction/PixelReasoner/data/pixelreasonersft_groundingcrop_candidates.jsonl"
output_path="/mnt/petrelfs/songmingyang/code/reasoning/data_construction/PixelReasoner/data/pixelreasonersft_groundingcrop_formatted.jsonl"

srun --partition=ai_moe --mpi=pmi2 --job-name=pruner --gres=gpu:${gpus} -c ${cpus} --nodes=1 --ntasks-per-node=1 --kill-on-bad-exit=1 --quotatype=reserved \
python ./generate.py \
--input_path $candidate_path \
--output_path $output_path \
--first_prompt "pixelreasonersft_grounding_crop" \
--inference_type pixelreasoner



# srun --partition=MoE --mpi=pmi2 --job-name=pruner --gres=gpu:1 --nodes=1 --ntasks-per-node=1 --kill-on-bad-exit=1 --quotatype=reserved \
#  python ../serve/grounding_dino_worker.py
# -w SH-IDCA1404-10-140-54-67 \