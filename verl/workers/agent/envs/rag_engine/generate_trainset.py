import os
import datasets
import json
import argparse
import random
random.seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_data_path', default='/cpfs/user/honglingyi/DATA/LLM/RAG-RL-Hotpotqa-with-2wiki/stage_1.jsonl')
    parser.add_argument('--stage2_data_path', default='/cpfs/user/honglingyi/DATA/LLM/RAG-RL-Hotpotqa-with-2wiki/stage_2.jsonl')
    parser.add_argument('--output_dir', default='/cpfs/user/fengyuan/verl_data/r1-searcher/')

    args = parser.parse_args()

    sys_prompt = """The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer. The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think><answer> final answer here </answer>".
During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as "keyword_1 keyword_2...")<|end_of_query|>". **A query must involve only a single triple**. Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>"."""

    prompt_set = set()
    num_filtered = 0

    stage1_data_path = args.stage1_data_path
    stage1_data = []
    with open(stage1_data_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            prompt = line['question']

            if prompt in prompt_set:
                num_filtered += 1
                print(f' [DEBUG] filter duplicated {prompt=}')
                continue
            prompt_set.add(prompt)

            stage1_data.append({
                "data_source": "rag_v2-train",
                "prompt": [
                    {
                        "role": "system",
                        "content": sys_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }],
                "question": line['question'],
                "env_name": "rag_v2",
                "ability": "qa",
                "reward_model": {
                        "style": "rule",
                        "ground_truth": line['answer']
                    },
                "extra_info": {
                    "id": line['idx'],
                    "question": prompt,
                    'answer': line['answer'],
                    # "pred_anses": line['pred_anses']
                }
            })

    stage2_data_path = args.stage2_data_path
    stage2_data = []
    with open(stage2_data_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            prompt = line['question']

            if prompt in prompt_set:
                num_filtered += 1
                print(f' [DEBUG] filter duplicated {prompt=}')
                continue
            prompt_set.add(prompt)

            stage2_data.append({
                "data_source": "rag_v2-train",
                "prompt": [
                    {
                        "role": "system",
                        "content": sys_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }],
                "question": line['question'],
                "ability": "qa",
                "env_name": "rag_v2",
                "reward_model": {
                        "style": "rule",
                        "ground_truth": line['answer']
                    },
                "extra_info": {
                    "id": line['idx'],
                    'answer': line['answer'],
                    "question": prompt
                }
            })

    print(f' [DEBUG] unique_prompt={len(prompt_set)}, {num_filtered=}')

    stage1_dataset = datasets.Dataset.from_list(stage1_data)
    stage2_dataset = datasets.Dataset.from_list(stage2_data)
    total_dataset = datasets.concatenate_datasets([stage1_dataset, stage2_dataset])
    total_dataset = total_dataset.shuffle()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    stage1_dataset.to_parquet(os.path.join(args.output_dir, 'stage1.parquet'))
    stage2_dataset.to_parquet(os.path.join(args.output_dir, 'stage2.parquet'))
    total_dataset.to_parquet(os.path.join(args.output_dir, 'train.parquet'))
