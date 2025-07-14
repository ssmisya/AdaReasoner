log_dir=/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/lm_as_a_judge

bash /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/bash_scripts/lm_as_a_judge/judge.sh /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/qwen25vl3b_sftv1/qwen25_allres.jsonl > $log_dir/3bsftv1_llm.log 2>&1 &

bash /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/bash_scripts/lm_as_a_judge/judge.sh  /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/qwen25vl3b_notool/qwen25_allres.jsonl > $log_dir/3bnotool_llm.log 2>&1 &

bash /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/bash_scripts/lm_as_a_judge/judge.sh /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/qwen25vl3b/qwen25_allres.jsonl > $log_dir/3bzstool_llm.log 2>&1 &

bash /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/bash_scripts/lm_as_a_judge/judge.sh /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/qwen25vl7b_sftv1/qwen25_allres.jsonl > $log_dir/7bsftv1_llm.log 2>&1 &

bash /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/bash_scripts/lm_as_a_judge/judge.sh /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/qwen25vl7b_notool/qwen25_allres.jsonl > $log_dir/7bnotool_llm.log 2>&1 &

bash /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/bash_scripts/lm_as_a_judge/judge.sh /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/qwen25vl7b/qwen25_allres.jsonl > $log_dir/7bzstool_llm.log 2>&1 &
