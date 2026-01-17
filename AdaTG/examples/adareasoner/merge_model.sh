source ~/.bashrc
source ~/anaconda3/bin/activate toolrl

cd AdaReasoner/AdaTG/scripts

input_ckpt_dir=$1


ref_model_path=Qwen/Qwen2.5-VL-7B-Instruct

target_dir=${input_ckpt_dir}/huggingface
hf_model_path=${input_ckpt_dir}/huggingface

srun -p ai_moe \
python ./model_merger.py \
--backend fsdp \
--tie-word-embedding \
--local_dir ${input_ckpt_dir} \
--target_dir ${target_dir} \
--hf_model_path ${hf_model_path} \
--test \
--test_hf_dir ${ref_model_path} \
