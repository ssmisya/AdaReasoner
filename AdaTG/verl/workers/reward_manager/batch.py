# Copyright 2025 Individual Contributor: Mert Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto


class BatchRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras_raw = data.non_tensor_batch.get("extra_info", [None] * len(data))
        # 确保extras是一个Python列表，而不是numpy数组
        if hasattr(extras_raw, 'tolist'):
            extras = extras_raw.tolist()
        else:
            extras = list(extras_raw) if extras_raw is not None else [None] * len(data)
        tool_rewards = data.non_tensor_batch.get("tool_rewards", [None] * len(data))

        scores = self.compute_score(
            data_sources=data_sources,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
            tool_rewards=tool_rewards,
            **self.reward_kwargs,
        )
        return scores

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        tool_rewards = data.non_tensor_batch.get("tool_rewards", [None] * len(data))

        scores = self.verify(data)  # 现在返回的是详细信息列表
        rewards = []
        reward_details = []  # 新增：收集详细的奖励信息
        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score_info = scores[i]  # 现在是字典格式

            # 提取总分数和详细信息
            if isinstance(score_info, dict):
                reward = score_info["score"]
                # 将详细的奖励信息添加到extra_info中
                for key, value in score_info.items():
                    reward_extra_info[key].append(value)
                # 同时收集详细信息用于wandb
                reward_details.append(score_info)
            else:
                # 兼容旧格式
                reward = score_info
                reward_extra_info["score"].append(score_info)
                # 兼容旧格式的详细信息
                reward_details.append({
                    "score": score_info,
                    "format_reward": 0,
                    "accuracy_reward": 0,
                    "tool_reward": 0,
                    "data_source": data_sources[i] if i < len(data_sources) else "unknown"
                })

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score_details]", score_info)  # 修改：打印详细信息
                print("[tool_reward_raw]", tool_rewards[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {
                "reward_tensor": reward_tensor, 
                "reward_extra_info": reward_extra_info,
                "reward_details": reward_details  # 新增：返回详细的奖励信息
            }
        else:
            return reward_tensor