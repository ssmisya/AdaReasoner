source ~/.bashrc
source ~/miniconda3/bin/activate vllm2


ref_model_path=/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-VL-7B-Instruct-new
input_ckpt_dir=/mnt/petrelfs/sunhaoyu/visual-code/DeepEyes/checkpoints/tool_rl/web_7b_wo_tool/global_step_150/actor
target_dir=${input_ckpt_dir}/huggingface
hf_model_path=${input_ckpt_dir}/huggingface

python ./model_merger.py \
--backend fsdp \
--tie-word-embedding \
--local_dir ${input_ckpt_dir} \
--target_dir ${target_dir} \
--hf_model_path ${hf_model_path} \
--test \
--test_hf_dir ${ref_model_path} \