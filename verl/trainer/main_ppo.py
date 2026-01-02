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
注意：我们没有将main与ray_trainer合并，因为ray_trainer也被其他main函数使用。
"""

import os

import hydra  # 用于管理配置
import ray    # 分布式计算框架

from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # 导入PPO训练器


def get_custom_reward_fn(config):
    """
    从配置中获取自定义奖励函数
    
    参数:
        config: 包含自定义奖励函数配置的字典
        
    返回:
        wrapped_fn: 包装后的奖励函数，如果没有配置则返回None
    """
    import importlib.util
    import sys

    # 获取自定义奖励函数配置
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    # 动态导入模块
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    # 获取函数名并检查是否存在，就是compute_score
    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    # 获取额外的奖励函数参数
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    # 包装原始函数，注入额外参数
    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """
    主入口函数，由hydra管理配置
    
    参数:
        config: hydra加载的配置对象
    """
    run_ppo(config)


def run_ppo(config) -> None:
    """
    运行PPO训练的主函数
    
    参数:
        config: 训练配置
    """
    # 设置环境变量解决SGLang与ray设备隔离冲突
    AD_NAME="songmingyang"
    encrypted_password="iWRsYqbwV4EJgJvU8QjLe00CptZc5jBVH3FMo5i6n9mVdOSoUurpyBTmst1Z"
    new_proxy_address=f"http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/"
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ray.is_initialized():
        # 初始化本地ray集群
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true", 
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN", 
                    "MKL_THREADING_LAYER": "GNU",
                    "RAY_DEBUG_POST_MORTEM": "1",
                    "RAY_DEBUG": "1",
                    # "https_proxy": new_proxy_address,
                    # "HTTPS_PROXY": new_proxy_address,
                    }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    # 创建远程任务运行器并执行
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # 确保主任务不在头节点上调度
class TaskRunner:
    """Ray远程任务运行器类"""
    
    def run(self, config):
        """
        运行训练任务
        
        参数:
            config: 训练配置
        """
        # 打印初始配置
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True会计算符号值
        OmegaConf.resolve(config)

        # 从HDFS下载检查点到本地
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        max_pixels_string = config.actor_rollout_ref.model.max_pixels
        max_pixels = eval(max_pixels_string)

        # 实例化tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True, max_pixels=max_pixels)

        # 根据策略定义工作器类
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            # 全分片数据并行(Fully Sharded Data Parallel)策略
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            # Megatron-LM策略
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # 角色与工作器的映射关系
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        # 资源池配置
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # 多源奖励函数设置:
        # - 对于基于规则的RM，直接调用奖励分数
        # - 对于基于模型的RM，调用模型
        # - 对于代码相关提示，如果有测试用例则发送到沙箱
        # - 最后，将所有奖励组合在一起
        # - 奖励类型取决于数据的标签
        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # 使用参考模型
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # 选择奖励管理器；原始选择的是naive
        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == "naive":
            from verl.workers.reward_manager import NaiveRewardManager
            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == "prime":
            from verl.workers.reward_manager import PrimeRewardManager
            reward_manager_cls = PrimeRewardManager
        elif reward_manager_name == "batch":
            from verl.workers.reward_manager import BatchRewardManager
            reward_manager_cls = BatchRewardManager
        elif reward_manager_name == "dapo":
            from verl.workers.reward_manager import DAPORewardManager
            reward_manager_cls = DAPORewardManager
        else:
            raise NotImplementedError

        # 获取自定义奖励函数
        compute_score = get_custom_reward_fn(config)
        reward_kwargs = dict(config.reward_model.get("reward_kwargs", {}))
        
        # 创建训练用奖励函数
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
            **reward_kwargs,
        )

        # 创建验证用奖励函数（始终使用基于函数的RM进行验证）
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer, 
            num_examine=1, 
            compute_score=compute_score, 
            reward_fn_key=config.data.reward_fn_key
        )
        
        # 创建资源池管理器
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # 创建并初始化PPO训练器
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()  # 初始化工作器
        trainer.fit()  # 开始训练


if __name__ == "__main__":
    main()
