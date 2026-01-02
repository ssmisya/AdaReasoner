# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from verl import DataProto


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    if 'action_mask' in batch.batch:
        action_mask = batch.batch['action_mask'][:, -batch.batch['responses'].shape[-1]:]
        obs_mask = response_mask * (1 - action_mask)
        obs_length = obs_mask.sum(-1).float()
    else:
        obs_length = torch.zeros_like(response_length)
    response_length -= obs_length

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
        obs_length=obs_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # TODO: add response length

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]
    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    action_or_attn_mask = batch.batch['action_mask'] if 'action_mask' in batch.batch else batch.batch['attention_mask']
    response_mask = action_or_attn_mask[:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]
    obs_length = response_info["obs_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                # 因为使用的是grpo，所以没有values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length

        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),

        # obs length
        'obs_length/mean': torch.mean(obs_length).detach().item(),
        'obs_length/min': torch.min(obs_length).detach().item(),
        'obs_length/max': torch.max(obs_length).detach().item(),

        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def compute_reward_detail_metrics(batch):
    """
    计算奖励详细信息的指标
    
    Args:
        batch: 包含meta_info的batch数据
        
    Returns:
        dict: 奖励详细信息的指标字典
    """
    reward_metrics = {}
    
    # 检查是否有reward_details信息
    if hasattr(batch, 'meta_info') and batch.meta_info and 'reward_details' in batch.meta_info:
        reward_details = batch.meta_info['reward_details']
        
        # 修复：正确检查reward_details是否为空
        if reward_details is not None and len(reward_details) > 0:
            # 整体指标（所有样本）
            scores = [detail['score'] for detail in reward_details]
            format_rewards = [detail['format_reward'] for detail in reward_details]
            accuracy_rewards = [detail['accuracy_reward'] for detail in reward_details]
            tool_rewards = [detail['tool_reward'] for detail in reward_details]
            
            # 转换为numpy数组以便计算
            scores = np.array(scores)
            format_rewards = np.array(format_rewards)
            accuracy_rewards = np.array(accuracy_rewards)
            tool_rewards = np.array(tool_rewards)
            
            # 修复：正确检查数组是否为空
            if len(scores) > 0:
                reward_metrics['rewards/overall/avg_score'] = np.mean(scores)
                reward_metrics['rewards/overall/max_score'] = np.max(scores)
                reward_metrics['rewards/overall/min_score'] = np.min(scores)
            else:
                reward_metrics['rewards/overall/avg_score'] = 0.0
                reward_metrics['rewards/overall/max_score'] = 0.0
                reward_metrics['rewards/overall/min_score'] = 0.0
            
            if len(format_rewards) > 0:
                reward_metrics['rewards/overall/avg_format_reward'] = np.mean(format_rewards)
            else:
                reward_metrics['rewards/overall/avg_format_reward'] = 0.0
            
            if len(accuracy_rewards) > 0:
                reward_metrics['rewards/overall/avg_accuracy_reward'] = np.mean(accuracy_rewards)
            else:
                reward_metrics['rewards/overall/avg_accuracy_reward'] = 0.0
            
            if len(tool_rewards) > 0:
                reward_metrics['rewards/overall/avg_tool_reward'] = np.mean(tool_rewards)
                reward_metrics['rewards/overall/max_tool_reward'] = np.max(tool_rewards)
                reward_metrics['rewards/overall/min_tool_reward'] = np.min(tool_rewards)
            else:
                reward_metrics['rewards/overall/avg_tool_reward'] = 0.0
                reward_metrics['rewards/overall/max_tool_reward'] = 0.0
                reward_metrics['rewards/overall/min_tool_reward'] = 0.0
            
            # 按data_source分组的指标
            source_details = defaultdict(list)
            for detail in reward_details:
                data_source = detail.get('data_source', 'unknown')
                source_details[data_source].append(detail)
            
            # 为每个data_source计算指标
            for data_source, details in source_details.items():
                if not details:
                    continue
                    
                # 提取各种奖励值
                ds_scores = [d["score"] for d in details]
                ds_format_rewards = [d["format_reward"] for d in details]
                ds_accuracy_rewards = [d["accuracy_reward"] for d in details]
                ds_tool_rewards = [d["tool_reward"] for d in details]
                
                prefix = f"rewards/{data_source}"
                
                # 总分指标
                if ds_scores:
                    reward_metrics[f"{prefix}/avg_score"] = np.mean(ds_scores)
                    reward_metrics[f"{prefix}/max_score"] = np.max(ds_scores)
                    reward_metrics[f"{prefix}/min_score"] = np.min(ds_scores)
                
                # 格式奖励指标
                if ds_format_rewards:
                    reward_metrics[f"{prefix}/avg_format_reward"] = np.mean(ds_format_rewards)
                
                # 准确性奖励指标
                if ds_accuracy_rewards:
                    reward_metrics[f"{prefix}/avg_accuracy_reward"] = np.mean(ds_accuracy_rewards)
                
                # 工具奖励指标（包含min和max）
                if ds_tool_rewards:
                    reward_metrics[f"{prefix}/avg_tool_reward"] = np.mean(ds_tool_rewards)
                    reward_metrics[f"{prefix}/max_tool_reward"] = np.max(ds_tool_rewards)
                    reward_metrics[f"{prefix}/min_tool_reward"] = np.min(ds_tool_rewards)
                
                # 样本数量
                reward_metrics[f"{prefix}/sample_count"] = len(details)
                
                print(f"[DEBUG] Training reward metrics for {data_source}: {len(details)} samples, "
                      f"avg_score={np.mean(ds_scores) if ds_scores else 0:.3f}, "
                      f"avg_format={np.mean(ds_format_rewards) if ds_format_rewards else 0:.3f}, "
                      f"avg_accuracy={np.mean(ds_accuracy_rewards) if ds_accuracy_rewards else 0:.3f}, "
                      f"avg_tool={np.mean(ds_tool_rewards) if ds_tool_rewards else 0:.3f}")
    
    return reward_metrics


def _aggregate_multi_gpu_tool_stats(tool_stats_list: list) -> dict:
    """
    聚合多GPU的工具统计信息
    
    参数:
        tool_stats_list: 来自多个GPU的工具统计列表
        
    返回:
        聚合后的工具统计字典
    """
    aggregated = {}
    
    # 遍历所有GPU的统计
    for gpu_stats in tool_stats_list:
        if not gpu_stats:
            continue
            
        for key, value in gpu_stats.items():
            if key not in aggregated:
                aggregated[key] = 0
            aggregated[key] += value
    
    # 重新计算成功率
    if 'tool_total_calls' in aggregated and aggregated['tool_total_calls'] > 0:
        aggregated['tool_success_rate'] = aggregated.get('tool_successful_calls', 0) / aggregated['tool_total_calls']
    
    # 重新计算各工具的成功率
    for key in list(aggregated.keys()):
        if key.endswith('_calls') and not key.endswith('_total_calls') and not key.endswith('_successful_calls'):
            # 对应的success key
            success_key = key.replace('_calls', '_success')
            rate_key = key.replace('_calls', '_success_rate')
            
            if success_key in aggregated:
                calls = aggregated[key]
                success = aggregated[success_key]
                aggregated[rate_key] = success / max(calls, 1)
    
    # 重新计算data_source级别的成功率
    for key in list(aggregated.keys()):
        if key.endswith('_total_calls') and 'tool_' in key and key != 'tool_total_calls':
            # 对应的success key
            success_key = key.replace('_total_calls', '_successful_calls')
            rate_key = key.replace('_total_calls', '_success_rate')
            
            if success_key in aggregated:
                calls = aggregated[key]
                success = aggregated[success_key]
                aggregated[rate_key] = success / max(calls, 1)
    
    return aggregated


def _aggregate_distributed_tool_stats(local_tool_stats: dict) -> dict:
    """
    在分布式环境中聚合所有GPU的工具统计
    
    参数:
        local_tool_stats: 本地GPU的工具统计
        
    返回:
        聚合后的全局工具统计
    """
    try:
        import torch
        import torch.distributed as dist
        
        if not dist.is_initialized():
            return local_tool_stats
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        print(f"[DEBUG] Rank {rank}/{world_size} aggregating tool_stats: {local_tool_stats}")
        
        # 收集所有GPU的工具统计
        all_tool_stats = [None] * world_size
        dist.all_gather_object(all_tool_stats, local_tool_stats)
        
        print(f"[DEBUG] Rank {rank} collected tool_stats from all GPUs: {len(all_tool_stats)} items")
        
        # 聚合所有GPU的统计
        aggregated = {}
        for gpu_stats in all_tool_stats:
            if not gpu_stats:
                continue
                
            for key, value in gpu_stats.items():
                if key not in aggregated:
                    aggregated[key] = 0
                aggregated[key] += value
        
        # 重新计算成功率
        if 'tool_total_calls' in aggregated and aggregated['tool_total_calls'] > 0:
            aggregated['tool_success_rate'] = aggregated.get('tool_successful_calls', 0) / aggregated['tool_total_calls']
        
        # 重新计算各工具的成功率
        for key in list(aggregated.keys()):
            if key.endswith('_calls') and not key.endswith('_total_calls') and not key.endswith('_successful_calls'):
                # 对应的success key
                success_key = key.replace('_calls', '_success')
                rate_key = key.replace('_calls', '_success_rate')
                
                if success_key in aggregated:
                    calls = aggregated[key]
                    success = aggregated[success_key]
                    aggregated[rate_key] = success / max(calls, 1)
        
        # 重新计算data_source级别的成功率
        for key in list(aggregated.keys()):
            if key.endswith('_total_calls') and 'tool_' in key and key != 'tool_total_calls':
                # 对应的success key
                success_key = key.replace('_total_calls', '_successful_calls')
                rate_key = key.replace('_total_calls', '_success_rate')
                
                if success_key in aggregated:
                    calls = aggregated[key]
                    success = aggregated[success_key]
                    aggregated[rate_key] = success / max(calls, 1)
        
        print(f"[DEBUG] Rank {rank} final aggregated tool_stats: {aggregated}")
        return aggregated
        
    except Exception as e:
        print(f"[DEBUG] Error in distributed aggregation: {e}")
        return local_tool_stats


def compute_agent_metrics(batch: DataProto) -> Dict[str, float]:
    """
    计算代理相关的指标，包括工具调用指标
    
    参数:
        batch: 数据批次
        
    返回:
        代理指标字典
    """
    metrics = {}
    batch_size = len(batch.batch) if hasattr(batch.batch, '__len__') else batch.batch.shape[0]
    
    print(f"[DEBUG] Computing agent metrics for batch_size: {batch_size}")
    print(f"[DEBUG] Batch keys: {batch.batch.keys() if hasattr(batch, 'batch') else 'No batch'}")
    print(f"[DEBUG] Non-tensor batch keys: {batch.non_tensor_batch.keys() if hasattr(batch, 'non_tensor_batch') else 'No non_tensor_batch'}")
    
    # 从meta_info中获取已经聚合过的tool_stats
    aggregated_tool_stats = None
    if hasattr(batch, 'meta_info') and batch.meta_info and 'tool_stats' in batch.meta_info:
        aggregated_tool_stats = batch.meta_info['tool_stats']
        print(f"[DEBUG] Got aggregated tool_stats from meta_info: {aggregated_tool_stats}")
    else:
        print(f"[DEBUG] No tool_stats in meta_info. batch.meta_info: {getattr(batch, 'meta_info', 'No meta_info')}")
    
    # 如果没有获得tool_stats，直接返回
    if aggregated_tool_stats is None:
        print(f"[DEBUG] No tool_stats available, skipping agent metrics")
        return metrics
    
    if aggregated_tool_stats:
        # 基础指标
        total_calls = aggregated_tool_stats.get('tool_total_calls', 0)
        successful_calls = aggregated_tool_stats.get('tool_successful_calls', 0)
        success_rate = successful_calls / max(total_calls, 1)
        
        metrics['agent/tool_total_calls'] = total_calls
        metrics['agent/tool_successful_calls'] = successful_calls  
        metrics['agent/tool_success_rate'] = success_rate
        metrics['agent/tool_avg_calls_per_sample'] = total_calls / max(batch_size, 1)
        
        print(f"[DEBUG] Agent metrics: total_calls={total_calls}, successful_calls={successful_calls}, batch_size={batch_size}")
        
        # 从aggregated_tool_stats中动态获取所有可用工具，而不是硬编码
        available_tools = []
        for key in aggregated_tool_stats.keys():
            if key.startswith('tool_') and key.endswith('_calls'):
                # 跳过全局统计和data_source级别的统计
                if (key.startswith('tool_total_') or 
                    key.startswith('tool_successful_') or 
                    '_total_calls' in key or 
                    '_successful_calls' in key):
                    continue
                
                tool_name = key.replace('tool_', '').replace('_calls', '')
                if tool_name != 'unrecognized_tool':  # 单独处理unrecognized_tool
                    available_tools.append(tool_name)
        
        # 处理所有可用工具的指标
        for tool_name in available_tools:
            tool_calls_key = f'tool_{tool_name}_calls'
            tool_success_key = f'tool_{tool_name}_success'
            
            if tool_calls_key in aggregated_tool_stats:
                tool_calls = aggregated_tool_stats[tool_calls_key]
                tool_success = aggregated_tool_stats.get(tool_success_key, 0)
                tool_success_rate = tool_success / max(tool_calls, 1)
                
                metrics[f'agent/tools/{tool_name}/calls'] = tool_calls
                metrics[f'agent/tools/{tool_name}/success'] = tool_success
                metrics[f'agent/tools/{tool_name}/success_rate'] = tool_success_rate
                metrics[f'agent/tools/{tool_name}/calls_per_sample'] = tool_calls / max(batch_size, 1)
                
                print(f"[DEBUG] Tool {tool_name}: calls={tool_calls}, success={tool_success}, rate={tool_success_rate}")
        
        # 特别处理unrecognized_tool的指标 - 只显示总调用次数
        if 'tool_unrecognized_tool_calls' in aggregated_tool_stats:
            unrec_calls = aggregated_tool_stats['tool_unrecognized_tool_calls']
            
            # 只显示总调用次数，不显示成功次数、成功率等（因为都是0，显示多余）
            metrics['agent/tools/unrecognized_tool/calls'] = unrec_calls
            
        
        # 处理按data_source分组的工具统计
        data_source_keys = [key for key in aggregated_tool_stats.keys() if key.startswith('tool_') and key.endswith('_total_calls')]
        for key in data_source_keys:
            # 提取data_source名称，如从'tool_path_nav_total_calls'提取'path_nav'
            data_source = key.replace('tool_', '').replace('_total_calls', '')
            
            # 跳过非data_source的统计（如tool_total_calls）
            if data_source in ['total', 'successful'] or '_' not in key[5:-12]:  # 5是'tool_'的长度，12是'_total_calls'的长度
                continue
            
            ds_total_calls = aggregated_tool_stats.get(f'tool_{data_source}_total_calls', 0)
            ds_successful_calls = aggregated_tool_stats.get(f'tool_{data_source}_successful_calls', 0)
            ds_success_rate = ds_successful_calls / max(ds_total_calls, 1)
            
            metrics[f'agent/data_sources/{data_source}/tool_total_calls'] = ds_total_calls
            metrics[f'agent/data_sources/{data_source}/tool_successful_calls'] = ds_successful_calls
            metrics[f'agent/data_sources/{data_source}/tool_success_rate'] = ds_success_rate
            metrics[f'agent/data_sources/{data_source}/tool_avg_calls_per_sample'] = ds_total_calls / max(batch_size, 1)
            
            print(f"[DEBUG] Data source {data_source}: total_calls={ds_total_calls}, successful_calls={ds_successful_calls}, rate={ds_success_rate}")
            
            # 处理该data_source下各个工具的统计
            for tool_name in available_tools + ['unrecognized_tool']:
                tool_calls_key = f'tool_{data_source}_{tool_name}_calls'
                tool_success_key = f'tool_{data_source}_{tool_name}_success'
                
                if tool_calls_key in aggregated_tool_stats:
                    tool_calls = aggregated_tool_stats[tool_calls_key]
                    tool_success = aggregated_tool_stats.get(tool_success_key, 0)
                    tool_success_rate = tool_success / max(tool_calls, 1)
                    
                    if tool_name == 'unrecognized_tool':
                        # 对于unrecognized_tool，只显示调用次数
                        metrics[f'agent/data_sources/{data_source}/tools/{tool_name}/calls'] = tool_calls
                    else:
                        # 对于正常工具，显示完整统计
                        metrics[f'agent/data_sources/{data_source}/tools/{tool_name}/calls'] = tool_calls
                        metrics[f'agent/data_sources/{data_source}/tools/{tool_name}/success'] = tool_success
                        metrics[f'agent/data_sources/{data_source}/tools/{tool_name}/success_rate'] = tool_success_rate
                        metrics[f'agent/data_sources/{data_source}/tools/{tool_name}/calls_per_sample'] = tool_calls / max(batch_size, 1)
                    
                    if tool_calls > 0:  # 只打印有调用的工具
                        print(f"[DEBUG] Data source {data_source}, Tool {tool_name}: calls={tool_calls}, success={tool_success}, rate={tool_success_rate}")
    else:
        print(f"[DEBUG] No tool_stats found in batch.meta_info")
    
    return metrics


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val

# 不再使用这个
# def process_validation_metrics(
#     data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
# ) -> dict[str, dict[str, dict[str, float]]]:
#     """Process validation metrics into a structured format.

#     Args:
#         data_sources: Array of data source identifiers for each sample
#         sample_inputs: List of input prompts
#         infos_dict: variable name -> list of values for each sample

#     Returns:
#         dict[str, dict[str, dict[str, float]]]: data source -> variable name -> metric value
#     """
#     # Group metrics by data source, prompt and variable
#     data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
#     for sample_idx, data_source in enumerate(data_sources):
#         prompt = sample_inputs[sample_idx]
#         var2vals = data_src2prompt2var2vals[data_source][prompt]
#         for var_name, var_vals in infos_dict.items():
#             var2vals[var_name].append(var_vals[sample_idx])

#     # Calculate metrics for each group
#     data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
#     for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
#         for prompt, var2vals in prompt2var2vals.items():
#             for var_name, var_vals in var2vals.items():
#                 if isinstance(var_vals[0], str):
#                     continue
#                 metric = {}
#                 n_resps = len(var_vals)
#                 metric[f"mean@{n_resps}"] = np.mean(var_vals)
#                 metric[f"std@{n_resps}"] = np.std(var_vals)

#                 ns = []
#                 n = 2
#                 while n < n_resps:
#                     ns.append(n)
#                     n *= 2
#                 ns.append(n_resps)

#                 for n in ns:
#                     # Best/Worst-of-N
#                     [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
#                         data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
#                     )
#                     metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
#                     metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
#                     # Majority voting
#                     if var2vals.get("pred", None) is not None:
#                         vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
#                         [(maj_n_mean, maj_n_std)] = bootstrap_metric(
#                             data=vote_data,
#                             subset_size=n,
#                             reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
#                             seed=seed,
#                         )
#                         metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

#                 data_src2prompt2var2metric[data_source][prompt][var_name] = metric

#     # Aggregate metrics across prompts
#     data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
#     for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
#         for prompt, var2metric in prompt2var2metric.items():
#             for var_name, metric in var2metric.items():
#                 for metric_name, metric_val in metric.items():
#                     data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

#     data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
#     for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
#         for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
#             for metric_name, prompt_vals in metric2prompt_vals.items():
#                 data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

#     return data_src2var2metric2val


# 新增的计算validation相关内容的函数
def compute_simple_validation_metrics(reward_details, tool_stats):
    """
    计算简单直接的验证指标
    
    Args:
        reward_details: 所有样本的详细奖励信息
        tool_stats: 工具统计信息
        
    Returns:
        dict: 简单验证指标
    """
    metrics = {}
    
    if not reward_details:
        return metrics
    
    # 按数据源分组计算指标
    source_details = defaultdict(list)
    for detail in reward_details:
        data_source = detail.get('data_source', 'unknown')
        source_details[data_source].append(detail)
    
    # 为每个数据源计算指标
    for data_source, details in source_details.items():
        if not details:
            continue
            
        # 提取各种奖励值
        scores = [d["score"] for d in details]
        format_rewards = [d["format_reward"] for d in details]
        accuracy_rewards = [d["accuracy_reward"] for d in details]
        tool_rewards = [d["tool_reward"] for d in details]
        
        prefix = f"validation/{data_source}"
        
        # 总分指标
        metrics[f"{prefix}/score/mean"] = np.mean(scores)
        metrics[f"{prefix}/score/max"] = np.max(scores)
        metrics[f"{prefix}/score/min"] = np.min(scores)
        
        # 格式奖励指标
        metrics[f"{prefix}/format_reward/mean"] = np.mean(format_rewards)
        
        # 准确性奖励指标
        metrics[f"{prefix}/accuracy_reward/mean"] = np.mean(accuracy_rewards)
        
        # 工具奖励指标
        metrics[f"{prefix}/tool_reward/mean"] = np.mean(tool_rewards)
        metrics[f"{prefix}/tool_reward/max"] = np.max(tool_rewards)
        metrics[f"{prefix}/tool_reward/min"] = np.min(tool_rewards)
        
        # 样本数量
        metrics[f"{prefix}/sample_count"] = len(details)
    
    # 整体指标（所有数据源合并）
    all_scores = [d["score"] for d in reward_details]
    all_format_rewards = [d["format_reward"] for d in reward_details]
    all_accuracy_rewards = [d["accuracy_reward"] for d in reward_details]
    all_tool_rewards = [d["tool_reward"] for d in reward_details]
    
    metrics["validation/overall/score/mean"] = np.mean(all_scores)
    metrics["validation/overall/score/max"] = np.max(all_scores)
    metrics["validation/overall/score/min"] = np.min(all_scores)
    
    metrics["validation/overall/format_reward/mean"] = np.mean(all_format_rewards)
    
    metrics["validation/overall/accuracy_reward/mean"] = np.mean(all_accuracy_rewards)
    
    metrics["validation/overall/tool_reward/mean"] = np.mean(all_tool_rewards)
    metrics["validation/overall/tool_reward/max"] = np.max(all_tool_rewards)
    metrics["validation/overall/tool_reward/min"] = np.min(all_tool_rewards)
    
    metrics["validation/overall/total_samples"] = len(reward_details)
    
    # 工具调用统计
    if tool_stats:
        # 全局工具统计
        for key, value in tool_stats.items():
            if not key.startswith('tool_') or '_total_calls' in key or '_successful_calls' in key or '_success_rate' in key:
                # 跳过unrecognized_tool的success相关统计
                if 'unrecognized_tool_success' in key and key != 'tool_unrecognized_tool_calls':
                    continue
                metrics[f"validation/tools/{key}"] = value
        
        # 按data_source分组的工具统计
        data_source_keys = [key for key in tool_stats.keys() if key.startswith('tool_') and key.endswith('_total_calls')]
        for key in data_source_keys:
            # 提取data_source名称
            data_source = key.replace('tool_', '').replace('_total_calls', '')
            
            # 跳过非data_source的统计
            if data_source in ['total', 'successful'] or '_' not in key[5:-12]:
                continue
            
            # 添加该data_source的工具统计
            ds_total_calls = tool_stats.get(f'tool_{data_source}_total_calls', 0)
            ds_successful_calls = tool_stats.get(f'tool_{data_source}_successful_calls', 0)
            ds_success_rate = tool_stats.get(f'tool_{data_source}_success_rate', 0)
            
            metrics[f"validation/data_sources/{data_source}/tool_total_calls"] = ds_total_calls
            metrics[f"validation/data_sources/{data_source}/tool_successful_calls"] = ds_successful_calls
            metrics[f"validation/data_sources/{data_source}/tool_success_rate"] = ds_success_rate
            
            # 添加该data_source下各个工具的统计
            for tool_key, tool_value in tool_stats.items():
                if tool_key.startswith(f'tool_{data_source}_') and not tool_key.endswith('_total_calls') and not tool_key.endswith('_successful_calls') and not tool_key.endswith('_success_rate'):
                    # 提取工具名称和统计类型
                    tool_part = tool_key.replace(f'tool_{data_source}_', '')
                    
                    # 对于unrecognized_tool，只显示calls统计，跳过success相关统计
                    if 'unrecognized_tool_success' in tool_part or 'unrecognized_tool_success_rate' in tool_part:
                        continue
                    
                    metrics[f"validation/data_sources/{data_source}/tools/{tool_part}"] = tool_value
    
    return metrics
