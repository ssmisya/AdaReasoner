source ~/.bashrc
source ~/miniconda3/bin/activate vllm2

code_base=/mnt/petrelfs/sunhaoyu/visual-code/DeepEyes/scripts
cd $code_base


ref_model_path=/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-VL-7B-Instruct-new
input_ckpt_dir=/mnt/petrelfs/sunhaoyu/visual-code/DeepEyes/checkpoints/tool_rl/web_7b_wo_tool/global_step_150/actor
target_dir=${input_ckpt_dir}/huggingface
hf_model_path=${input_ckpt_dir}/huggingface
# target_dir要包括vocab.json那些文件就可以了，不需要文件夹名称全部对应吧
# target_dir=/mnt/petrelfs/sunhaoyu/visual-code/DeepEyes/checkpoints/tool_rl/totest/actor/huggingface
# hf_model_path=/mnt/petrelfs/sunhaoyu/visual-code/DeepEyes/checkpoints/tool_rl/totest/actor/huggingface


python ./model_merger.py \
--backend fsdp \
--tie-word-embedding \
--local_dir ${input_ckpt_dir} \
--target_dir ${target_dir} \
--hf_model_path ${hf_model_path} \
--test \
--test_hf_dir ${ref_model_path} \