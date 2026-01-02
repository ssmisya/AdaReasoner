import re
import io
import json
import time
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from verl import DataProto
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.dataset.vision_utils import process_image
from verl.utils.torch_functional import pad_2d_list_to_length
from PIL import Image
from copy import deepcopy
from io import BytesIO
import base64
# 给我用ToolManager！！！
from tool_server.tool_workers.tool_manager.base_manager import ToolManager

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))

def base64_to_pil(b64_str: str) -> Image.Image:
    """
    Convert a base64 encoded string into a PIL image.
    
    Args:
        b64_str (str): The base64 encoded image string.
        
    Returns:
        Image.Image: The resulting PIL image.
    """
    # Remove the data URI scheme if present
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split("base64,")[-1]
    return load_image_from_base64(b64_str)

def pil_to_base64(img: Image.Image, url_format = False) -> str:
    """
    Convert a PIL image to a base64 encoded string.
    
    Args:
        img (Image.Image): The PIL image to convert.
        
    Returns:
        str: Base64 encoded string representation of the image.
    """
    with BytesIO() as buffered:
        # 确保图像是RGB模式，如果是RGBA或其他模式则转换
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            # 需要先转换为RGB模式
            with Image.new('RGB', img.size, (255, 255, 255)) as background:
                background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)  # 使用alpha通道作为mask
                background.save(buffered, format="JPEG")
        elif img.mode != 'RGB':
            with img.convert('RGB') as converted_img:
                converted_img.save(buffered, format="JPEG")
        else:
            img.save(buffered, format="JPEG")
        
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        if url_format:
            img_str = f"data:image/jpeg;base64,{img_str}"
        return img_str


# 在文件开头添加工具统计类
class ToolCallStats:
    def __init__(self):
        self.total_calls = 0
        self.successful_calls = 0
        self.tool_calls = {}  # 格式: {tool_name: total_calls}
        self.tool_success = {}  # 格式: {tool_name: successful_calls}
        
    def add_call(self, success: bool, tool_name: str = None):
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        
        if tool_name:
            if tool_name not in self.tool_calls:
                self.tool_calls[tool_name] = 0
                self.tool_success[tool_name] = 0
            
            self.tool_calls[tool_name] += 1
            if success:
                self.tool_success[tool_name] += 1
    
    def get_stats_dict(self):
        stats = {
            "tool_total_calls": self.total_calls,
            "tool_successful_calls": self.successful_calls,
            "tool_success_rate": self.successful_calls / max(self.total_calls, 1),
        }
        
        # 添加各个工具的统计数据
        for tool_name in self.tool_calls:
            stats[f"tool_{tool_name}_calls"] = self.tool_calls[tool_name]
            stats[f"tool_{tool_name}_success"] = self.tool_success[tool_name]
            stats[f"tool_{tool_name}_success_rate"] = self.tool_success[tool_name] / max(self.tool_calls[tool_name], 1)
            
        return stats
    
    def reset(self):
        self.total_calls = 0
        self.successful_calls = 0
        self.tool_calls = {}
        self.tool_success = {}

def _concat_vllm_input(prompt_token_ids, response_token_ids, tokenizer=None):
    # NOTE: temporarily fix qwen-base oov issue
    if tokenizer is not None:
        max_token_id = max(tokenizer.get_vocab().values())
        tokenizer_size = len(tokenizer)
        max_token_id = max(max_token_id, tokenizer_size)
        valid_token_mask = torch.le(response_token_ids, max_token_id)
        response_token_ids = torch.masked_select(response_token_ids, valid_token_mask)

    if isinstance(prompt_token_ids, torch.Tensor):
        output_tensor = torch.cat([
            prompt_token_ids,
            response_token_ids.to(prompt_token_ids.device),
        ], dim=-1)
        return output_tensor.cpu().numpy().flatten().tolist()
    else:
        output_array = np.concatenate([
            prompt_token_ids,
            response_token_ids.cpu().numpy(),
        ], axis=-1)
        return output_array.flatten().tolist()


def _merge_multi_modal_inputs(mm_input, other):
    if not mm_input and not other:
        return {}
    elif len(mm_input) == 0 and len(other) > 0:
        return other
    elif len(mm_input) > 0 and len(other) == 0:
        return mm_input

    output_dict = {}
    for key in mm_input.keys():
        if key not in other.keys():
            output_dict[key] = mm_input[key]
            continue

        mm_value = mm_input[key]
        other_value = other.pop(key)
        if isinstance(mm_value, np.ndarray) and isinstance(other_value, np.ndarray):
            merged_value = np.concatenate([mm_value, other_value], axis=0)
        elif isinstance(mm_value, torch.Tensor) and isinstance(other_value, torch.Tensor):
            merged_value = torch.cat([mm_value, other_value], dim=0)
        else:
            raise ValueError(f"Invalid {type(mm_value)=}, {type(other_value)=}")

        output_dict[key] = merged_value
    return dict(**output_dict, **other)


def _preprocess_multi_modal_inputs(prompt_str, processor, **kwargs):
    if processor is None or "multi_modal_data" not in kwargs:
        return prompt_str, prompt_str, {}

    vllm_input_prompt = prompt_str.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
    input_mm_data = kwargs.get("multi_modal_data", {"image": []})
    
    image_info_list = []
    for img in input_mm_data["image"]:
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        png_bytes = buf.getvalue()
        buf.close()
        img_info = {"bytes": png_bytes}
        image_info_list.append(img_info)

    input_mm_data["image"] = [process_image(img) for img in image_info_list]
    model_inputs = processor(text=[vllm_input_prompt], images=input_mm_data["image"], return_tensors="pt")
    input_ids = model_inputs.pop("input_ids")[0]
    attention_mask = model_inputs.pop("attention_mask")[0]

    if "second_per_grid_ts" in model_inputs:
        model_inputs.pop("second_per_grid_ts")

    mm_inputs = dict(model_inputs)
    return vllm_input_prompt, input_ids, mm_inputs


def agent_rollout_loop(config, vllm_engine, vllm_inputs, prompts, multi_modal_inputs, sampling_params):
    from vllm.distributed import parallel_state as vllm_ps

    agent_sampling_params = sampling_params.clone()
    agent_sampling_params.detokenize = True
    agent_sampling_params.skip_special_tokens = False
    agent_sampling_params.spaces_between_special_tokens = False
    agent_sampling_params.n = 1
    agent_sampling_params.include_stop_str_in_output = True
    max_generated_tokens = min(config.agent.single_response_max_tokens, config.response_length)
    agent_sampling_params.max_tokens = max_generated_tokens

    # support custom stop specified in dataset, like </search>, ```, etc.
    custom_stop = list(config.agent.custom_stop)
    if custom_stop:
        prev_stop = sampling_params.stop if sampling_params.stop else []
        agent_sampling_params.stop = prev_stop + custom_stop
        print(f' [DEBUG stop] {type(prev_stop)=}, {type(custom_stop)=}, {type(agent_sampling_params.stop)=}')

    tokenizer = hf_tokenizer(config.agent.vl_model_path)
    processor = hf_processor(config.agent.vl_model_path)

    if multi_modal_inputs is not None:
        multi_modal_inputs = multi_modal_inputs.tolist()
    else:
        multi_modal_inputs = [{}] * len(vllm_inputs)

    batch_size = len(vllm_inputs)
    vllm_input_list = []
    running_states = []
    running_action_masks = []
    running_attn_masks = []
    active_mask = []
    mm_input_list = []
    
    # 修改工具调用统计 - 使用新的统计类
    tool_stats = ToolCallStats()
    tool_rewards_list = []  # 保留tool_rewards跟踪

    # env是有用的
    env = ParallelEnv(config.agent, tokenizer, processor)
    # env.reset(prompts, vllm_inputs, n=sampling_params.n)

    # interleaving inputs if sampling_params.n > 1
    for i in range(batch_size):
        for sample_idx in range(sampling_params.n):
            vllm_input_list.append(deepcopy(vllm_inputs[i]))
            prompt_ids = prompts.batch['input_ids'][i, :].clone()
            running_states.append(prompt_ids)
            prompt_mask = prompts.batch['attention_mask'][i, :].clone()
            running_action_masks.append(prompt_mask)
            running_attn_masks.append(prompt_mask)
            active_mask.append(True)
            mm_input_list.append(deepcopy(multi_modal_inputs[i]))
            tool_rewards_list.append([])  # 为每个样本初始化tool_rewards列表
            
            # 初始化图像历史
            current_idx = i * sampling_params.n + sample_idx
            env.image_history[current_idx] = {}
            
            # 修改：从vllm_inputs中获取原始图像数据
            if 'multi_modal_data' in vllm_inputs[i] and 'image' in vllm_inputs[i]['multi_modal_data']:
                images = vllm_inputs[i]['multi_modal_data']['image']
                if not isinstance(images, list):
                    images = [images]
                # 存储初始图像为img_1, img_2, ...
                for img_idx, img in enumerate(images, start=1):
                    env.image_history[current_idx][f"img_{img_idx}"] = img
                    # print(f"啊啊啊啊啊啊[DEBUG] Stored image img_{img_idx} for item {current_idx}, image type: {type(img)}")
                    # if hasattr(img, 'size'):
                    #     print(f"啊啊啊啊啊啊[DEBUG] Image size: {img.size}")
            else:
                # 报错，中断训练
                raise ValueError(f"啊啊啊啊啊啊[DEBUG] No image data found in vllm_inputs[{i}]")

    # 获取VLLM分布式系统中的张量并行组
    pg = vllm_ps.get_tp_group()
    max_total_length = config.prompt_length + config.response_length
    # 开始对话逻辑
    for step in range(config.agent.max_turns):
        print(f' [DEBUG 000] {step=}, total={batch_size}, n={sampling_params.n}, num_active={sum(active_mask)}')
        if sum(active_mask) == 0:
            break

        # 获取所有需要处理的输入索引，要选择active的
        active_indices = [idx for idx, is_active in enumerate(active_mask) if is_active]
        # 获取所有需要处理的输入
        active_vllm_inputs = [vinput for vinput, is_active in zip(vllm_input_list, active_mask) if is_active]
        # actions就是模型的输出，形状和active_vllm_inputs一致
        actions = vllm_engine.generate(
            prompts=active_vllm_inputs,
            sampling_params=agent_sampling_params,
            use_tqdm=False
        )

        if pg.is_first_rank:
            obs_results = env.step(active_indices, actions)
        else:
            obs_results = None

        obs_results = pg.broadcast_object(obs_results)
        observations, dones, info = obs_results
        step_tool_rewards = info.get("tool_rewards", [0.0] * len(observations))
        step_tool_stats = info.get("tool_stats", {})
        
        # 更新全局工具统计
        total_tool_calls = step_tool_stats.get("total_tool_calls", 0)
        successful_tool_calls = step_tool_stats.get("successful_tool_calls", 0)

        tool_stats.total_calls += total_tool_calls
        tool_stats.successful_calls += successful_tool_calls

        print(f"[DEBUG] Step tool stats: total={total_tool_calls}, successful={successful_tool_calls}")
        print(f"[DEBUG] Global tool stats: total={tool_stats.total_calls}, successful={tool_stats.successful_calls}")

        # 更新各工具的统计
        for tool_name, calls in step_tool_stats.get("tool_calls_by_name", {}).items():
            if tool_name not in tool_stats.tool_calls:
                tool_stats.tool_calls[tool_name] = 0
                tool_stats.tool_success[tool_name] = 0
            tool_stats.tool_calls[tool_name] += calls
            tool_stats.tool_success[tool_name] += step_tool_stats.get("tool_success_by_name", {}).get(tool_name, 0)
            
            print(f"[DEBUG] Tool {tool_name}: calls={tool_stats.tool_calls[tool_name]}, success={tool_stats.tool_success[tool_name]}")

        # 在每个对话回合中处理模型输出和工具观察结果的核心逻辑
        for idx, obs, act, done, tool_rew in zip(active_indices, observations, actions, dones, step_tool_rewards):
            # 记录当前轮次的工具奖励分数到历史列表中
            # tool_rewards_list[idx] 保存第idx个样本所有轮次的工具奖励
            tool_rewards_list[idx].append(tool_rew)

            # ========== 处理模型生成的回复tokens ==========
            # 将模型本轮生成的token_ids转换为tensor格式
            response_token_ids = torch.tensor(act.outputs[0].token_ids, dtype=torch.int64, device=running_states[idx].device)
            
            # 将本轮模型回复的tokens追加到running_states（完整对话历史）的末尾
            # running_states[idx] 维护着第idx个样本从开始到现在的完整token序列
            running_states[idx] = torch.cat([running_states[idx], response_token_ids])
            
            # 更新vllm_input_list，将模型回复合并到prompt_token_ids中
            # 这是为下一轮推理准备的输入，包含了到目前为止的所有对话内容
            vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(
                vllm_input_list[idx]['prompt_token_ids'],  # 之前的所有tokens
                response_token_ids,                        # 本轮模型回复的tokens
                tokenizer=tokenizer,
            )

            # ========== 更新attention mask ==========
            # 为模型回复的tokens创建action_mask（全1，表示这些是模型生成的tokens）
            action_mask = torch.ones_like(response_token_ids, dtype=torch.int64, device=running_action_masks[idx].device)
            # 将action_mask追加到running_action_masks，用于区分哪些tokens是模型生成的
            running_action_masks[idx] = torch.cat([running_action_masks[idx], action_mask])
            
            # 为模型回复的tokens创建attention_mask（全1，表示这些tokens需要被注意到）
            attn_mask = torch.ones_like(response_token_ids, dtype=torch.int64, device=running_attn_masks[idx].device)
            running_attn_masks[idx] = torch.cat([running_attn_masks[idx], attn_mask])

            # ========== 检查序列长度限制 ==========
            # 如果token序列长度超过最大限制，则停止这个样本的对话
            if running_states[idx].shape[-1] >= max_total_length or len(vllm_input_list[idx]['prompt_token_ids']) >= max_total_length:
                active_mask[idx] = False  # 标记为非活跃状态
                continue

            # 如果对话已完成或达到最大轮数，停止这个样本的对话
            if done or step == config.agent.max_turns - 1:
                active_mask[idx] = False
                continue


            # ========== 处理工具观察结果 ==========
            # 检查是否有工具观察结果需要处理
            if 'prompt_token_ids_vllm' in obs.keys() and 'prompt_token_ids_model' in obs.keys():
                # 获取工具观察结果的tokens（用于VLLM推理引擎）
                obs_token_ids_vllm = obs['prompt_token_ids_vllm']
                # 获取工具观察结果的tokens（用于模型训练，移到正确设备）
                obs_token_ids_model = obs['prompt_token_ids_model'].to(running_states[idx].device)

                # 再次检查添加观察结果后是否会超过长度限制
                if len(vllm_input_list[idx]['prompt_token_ids']) + len(obs_token_ids_vllm) >= max_total_length:
                    active_mask[idx] = False
                    continue
                if running_states[idx].shape[-1] + len(obs_token_ids_model) >= max_total_length:
                    active_mask[idx] = False
                    continue

                # 将工具观察结果的tokens添加到VLLM输入中
                # 这样下一轮推理时模型就能看到工具的返回结果
                vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(
                    vllm_input_list[idx]['prompt_token_ids'], 
                    obs_token_ids_vllm,                      # 工具观察结果的tokens
                    tokenizer=tokenizer,
                )

                # 将工具观察结果tokens添加到完整对话历史中
                running_states[idx] = torch.cat([running_states[idx], obs_token_ids_model])

                # 为工具观察结果创建mask
                # obs_mask设为0，表示这些tokens不是模型生成的（是环境/工具生成的）
                obs_mask = torch.zeros(len(obs_token_ids_model), dtype=torch.int64, device=running_action_masks[idx].device)
                running_action_masks[idx] = torch.cat([running_action_masks[idx], obs_mask])
                
                # attn_mask设为1，表示模型需要注意到这些工具观察结果
                attn_mask = torch.ones(len(obs_token_ids_model), dtype=torch.int64, device=running_attn_masks[idx].device)
                running_attn_masks[idx] = torch.cat([running_attn_masks[idx], attn_mask])

                # ========== 处理多模态数据（图像） ==========
                # 检查工具观察结果中是否包含新的图像数据
                mm_data = obs.get('multi_modal_data', {})
                if 'image' in mm_data.keys():
                    # 如果vllm输入中还没有multi_modal_data字段，则初始化
                    if 'multi_modal_data' not in vllm_input_list[idx].keys():
                        vllm_input_list[idx]['multi_modal_data'] = {"image": []}
                    # 将新的图像添加到图像列表中
                    # 这样下一轮推理时模型就能看到所有历史图像（包括工具生成的新图像）
                    vllm_input_list[idx]['multi_modal_data']['image'] += mm_data['image']

                # 处理多模态输入的其他元数据（如image_grid_thw等）
                mm_input = obs.get('multi_modal_inputs', {})
                if mm_input:
                    # 合并多模态输入的元数据，用于Qwen2-VL等模型的位置编码
                    mm_input_list[idx] = _merge_multi_modal_inputs(mm_input_list[idx], mm_input)

            # 最终检查：如果序列长度仍然超限，停止对话
            if running_states[idx].shape[-1] >= max_total_length or len(vllm_input_list[idx]['prompt_token_ids']) >= max_total_length:
                active_mask[idx] = False

    # ========== 对话结束后的最终处理 ==========
    env.close()  # 关闭环境

    # 获取目标设备（通常是GPU）
    target_device = prompts.batch['input_ids'].device

    # 截取所有序列到最大长度，确保不超限
    running_states = [state[: max_total_length] for state in running_states]
    # 将所有样本的token序列padding到相同长度，形成batch tensor
    state_tensor = pad_2d_list_to_length(running_states, tokenizer.pad_token_id, max_total_length).to(target_device)

    # 处理action_mask（标识哪些tokens是模型生成的）
    running_action_masks = [mask[: max_total_length] for mask in running_action_masks]
    action_mask_tensor = pad_2d_list_to_length(running_action_masks, 0, max_total_length).to(target_device)

    # 处理attention_mask（标识哪些tokens需要被注意）
    running_attn_masks = [mask[: max_total_length] for mask in running_attn_masks]
    attn_mask_tensor = pad_2d_list_to_length(running_attn_masks, 0, max_total_length).to(target_device)

    # ========== 为不同模型类型生成position_ids ==========
    if processor is not None and processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
        # 对于Qwen2-VL模型，需要特殊的position encoding来处理图像tokens
        # 生成3D position_ids: (batch_size * sampling_params.n, 3, seq_len)
        position_ids_list = [
            get_rope_index(
                processor,
                input_ids=state_tensor[i, :],                                    # 输入token序列
                image_grid_thw=mm_input_list[i].get("image_grid_thw", None),    # 图像网格时间-高度-宽度信息
                video_grid_thw=mm_input_list[i].get("video_grid_thw", None),    # 视频网格信息
                second_per_grid_ts=mm_input_list[i].get("second_per_grid_ts", None),  # 时间戳信息
                attention_mask=attn_mask_tensor[i, :],                          # attention mask
            ) for i in range(batch_size * sampling_params.n)
        ]
        position_ids_tensor = torch.stack(position_ids_list, dim=0)
    else:
        # 对于普通语言模型，使用标准的1D position encoding
        # 生成2D position_ids: (batch_size * sampling_params.n, seq_len)
        position_ids_tensor = compute_position_id_with_mask(attn_mask_tensor)


    # 修改填充函数
    def pad_list_to_same_length(list_of_lists, pad_value=-1, fixed_length=12): # fix_length是在调用的时候指定的
        """
        将列表中的每个子列表填充到指定的固定长度
        
        Args:
            list_of_lists: 要填充的列表的列表
            pad_value: 填充值
            fixed_length: 固定的目标长度，确保所有GPU上的数组都填充到相同长度
        
        Returns:
            填充后的列表的列表
        """
        if not list_of_lists:
            return []
        
        # 使用固定长度而不是计算最大长度
        padded_lists = [sublist + [pad_value] * (fixed_length - len(sublist)) for sublist in list_of_lists]
        
        print(f"Original list shapes: {[len(sublist) for sublist in list_of_lists]}")
        print(f"Padded to fixed length {fixed_length}: {[len(sublist) for sublist in padded_lists]}")
        
        return padded_lists

    # 在返回部分使用固定长度
    padded_tool_rewards = pad_list_to_same_length(tool_rewards_list, fixed_length=10)
    # 转换为numpy数组以便于后续处理
    padded_tool_rewards_array = np.array(padded_tool_rewards)

    # 打印调试信息
    print(f"padded_tool_rewards_array shape: {padded_tool_rewards_array.shape}")

    # ========== 返回最终结果 ==========
    return DataProto.from_dict(
        tensors={
            # 只返回response部分的tokens（用于训练/评估）
            "response": state_tensor[:, -config.response_length: ],
            # action mask标识哪些tokens是模型生成的（用于loss计算）
            "action_mask": action_mask_tensor,
            # attention mask标识有效tokens（用于模型attention）
            "attention_mask": attn_mask_tensor,
            # position ids用于位置编码（处理图像-文本混合序列）
            "position_ids": position_ids_tensor,
        },
        non_tensors={
            # 确保这里的所有值都是np.ndarray类型
            "multi_modal_inputs": mm_input_list if processor is not None else None,
            "tool_rewards": padded_tool_rewards_array,  # np.ndarray
            "tool_call_counts": np.array([len(rewards) for rewards in tool_rewards_list]),  # np.ndarray
        },
        meta_info={
            # 字典类型的数据放在meta_info中
            "tool_stats": tool_stats.get_stats_dict()
        }
    )



class ParallelEnv:
    """
    The interface is designed to be the similar to : https://github.com/openai/gym
    """
    # 初始话没什么问题，会先进行初始化
    def __init__(self, env_config, tokenizer, processor, **kwargs):
        self.config = env_config
        self.tokenizer = tokenizer
        self.processor = processor

        # 初始化工具管理器，需要两个参数，一个是地址，一个是工具列表
        self.tool_manager = ToolManager(self.config.tool_manager.controller_addr, self.config.tool_manager.tools)
        
        # 添加图像历史，用于存储每个对话项的图像
        self.image_history = {}

        self.available_tools = self.tool_manager.available_tools

        print("啊啊啊啊啊啊啊啊[DEBUG] self.available_tools",self.available_tools)
    

    def extract_tool_call(self, text: str):
        """
        从模型响应文本中提取<tool_call>标签内的工具调用信息
        
        参数:
            text (str): 包含tool_call的模型响应文本
            
        返回:
            Optional[List[Dict]]: 解析后的工具调用列表，如果提取失败则返回None
        """
        try:
            # 使用正则表达式查找<tool_call>标签内的内容
            tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
            tool_call_match = re.search(tool_call_pattern, text, re.DOTALL)
            
            if not tool_call_match:
                return None
                
            tool_call_content = tool_call_match.group(1).strip()
            
            # 尝试解析整个JSON数组
            try:
                # 首先尝试解析整个内容为JSON数组
                if tool_call_content.startswith('[') and tool_call_content.endswith(']'):
                    json_array = json.loads(tool_call_content)
                    if isinstance(json_array, list):
                        valid_objects = []
                        for obj in json_array:
                            if isinstance(obj, dict) and "name" in obj and "parameters" in obj:
                                valid_objects.append(obj)
                        if valid_objects:
                            return valid_objects
                
                # 如果不是JSON数组，尝试解析为单个JSON对象
                if (tool_call_content.startswith('{') and tool_call_content.endswith('}')):
                    json_obj = json.loads(tool_call_content)
                    if "name" in json_obj and "parameters" in json_obj:
                        return [json_obj]
            except json.JSONDecodeError:
                pass
            
            # 如果上述方法失败，尝试提取单个JSON对象
            json_objects = []
            # 使用正则表达式匹配所有JSON对象
            json_pattern = r'({[^{}]*(?:{[^{}]*}[^{}]*)*})'
            matches = re.finditer(json_pattern, tool_call_content, re.DOTALL)
            
            for match in matches:
                try:
                    json_obj = json.loads(match.group(1))
                    if isinstance(json_obj, dict) and "name" in json_obj and "parameters" in json_obj:
                        json_objects.append(json_obj)
                except json.JSONDecodeError:
                    continue
            
            if not json_objects:
                return None
                
            return json_objects
            
        except Exception as e:
            print(f"Error extracting tool call: {e}")
            return None

    def call_tool_with_retry(
        self, 
        api_name: str, 
        api_params: dict, 
        item_idx: int,  # 添加item_idx参数用于访问图像历史
        tool_retry_num: int = 3, 
        tool_retry_interval: int = 5, 
        tool_timeout_sec: int = 600
    ) -> dict:
        """
        调用工具并处理重试
        """
        
        # 确保每次调用都被统计
        print(f"啊啊啊啊啊[DEBUG] About to call tool {api_name}")
        
        # 处理图像参数
        if "image" in api_params:
            img_key = api_params.get("image", "")
            # print(f"[DEBUG] Requested image key: {img_key}")
            
            # 如果是img_n格式，则从图像历史中获取对应图像
            if isinstance(img_key, str) and img_key.startswith("img_"):
                if item_idx in self.image_history and img_key in self.image_history[item_idx]:
                    try:
                        image = self.image_history[item_idx][img_key]
                        # print(f"[DEBUG] Found image {img_key}, type: {type(image)}")
                        # if hasattr(image, 'size'):
                        #     print(f"[DEBUG] Image size: {image.size}")
                        
                        # 转换为base64格式
                        image_base64 = pil_to_base64(image, url_format=False)
                        # 更新参数中的图像
                        api_params["image"] = image_base64
                        # print(f"[DEBUG] Successfully converted image {img_key} to base64 (length: {len(image_base64)})")
                    except Exception as e:
                        print(f"[DEBUG] Error converting image {img_key} to base64: {e}")
                        # 保留传入None的逻辑
                        api_params["image"] = None
                else:
                    # 保留传入None的逻辑，如果找不到指定图像
                    available_images = list(self.image_history.get(item_idx, {}).keys())
                    print(f"啊啊啊啊啊啊啊[DEBUG] Image {img_key} not found. Available images: {available_images}")
                    api_params["image"] = None
            else:
                # 如果不是img_n格式，也设置成None
                print(f"啊啊啊啊啊啊[DEBUG] Invalid image parameter format: {img_key}")
                api_params["image"] = None
        
        tool_result = {"text": f"Failed to call tool {api_name}", "error_code": 1}
        
        for attempt in range(tool_retry_num):
            try:
                # 直接调用工具，不设置超时
                tool_result = self.tool_manager.call_tool(api_name, api_params)
                print(f"啊啊啊啊啊[DEBUG] Tool call result for {api_name}: status={tool_result.get('status', 'unknown')}, error_code={tool_result.get('error_code', -1)}")
                
                # 确保每次调用都进入成功判断逻辑
                success = self._determine_tool_success(tool_result, api_params)
                print(f"啊啊啊啊啊[DEBUG] Tool success determination: status='{tool_result.get('status', 'unknown')}', error_code={tool_result.get('error_code', -1)}, success={success}")
                
                return tool_result
                
            except Exception as e:
                print(f"啊啊啊啊啊[DEBUG] Tool call exception for {api_name}: {str(e)}")
                success = False
                print(f"啊啊啊啊啊[DEBUG] Tool success determination: status='failed', error_code=-1, success={success}")
                return tool_result

    def _determine_tool_success(self, tool_result: dict, api_params: dict) -> bool:
        """
        根据工具调用结果和参数判断工具是否成功。
        
        参数:
            tool_result (dict): 工具调用结果，包含status和error_code。
            api_params (dict): 工具调用参数，包含image。
            
        返回:
            bool: 工具是否成功。
        """
        success = False
        status = tool_result.get("status", "")
        error_code = tool_result.get("error_code", -1)
        
        # 修复成功判断逻辑：必须同时满足status="success"且error_code=0
        success = (status == "success" and error_code == 0)
        
        # 如果工具需要图像参数但传入了None，视为失败
        if "image" in api_params and api_params["image"] is None:
            success = False
            print(f"啊啊啊啊啊[DEBUG] Tool call failed due to missing image parameter")
        
        return success

    def step(self, active_indices, actions):
        """
        Input:
        - actions: vllm.RequestOutput

        Output:
        - observations: List[Dict], content like {"prompt_token_ids": ..., "multi_modal_data": ...}, 
                multi_modal_data only appears when there are images/videos in obs
        - rewards: List[ float ].
                each time after an action being executed, procedure rewards can be assigned to 
                the last valid token of model outputs. This might be useful for ..., 
                e.g., invalid action, code execution error, format error,
                or video game envs where immediate feedback is available.
        - dones: List[ Boolean ]
        - infos: Dict, for debugging only
        """
        obs_list = [{}] * len(actions)
        tool_reward_list = [0.0] * len(actions)  # 添加tool_reward跟踪
        done_list = []
        valid_indices = []
        real_indices = []
        valid_actions = []
        
        # 添加工具调用统计
        step_tool_stats = {
            "total_tool_calls": 0,
            "successful_tool_calls": 0,
            "tool_calls_by_name": {},
            "tool_success_by_name": {}
        }
        
        # 1. filtering valid actions
        for i, (idx, act) in enumerate(zip(active_indices, actions)):
            if act.outputs[0].finish_reason == 'length':
                done_list.append(True)
                continue

            if len(act.outputs[0].token_ids) == 0:
                done_list.append(True)
                continue

            done_list.append(False)
            real_indices.append(i)
            valid_indices.append(idx)
            valid_actions.append(act.outputs[0].text)

        # 2. executing actions using tool_manager
        for i, idx, action_text in zip(real_indices, valid_indices, valid_actions):
            tool_reward = 0.0
            
            try:
                # 检查action_text中是否包含<response>，如果包含则标记为完成
                if "<response>" in action_text:
                    obs_list[i] = {}
                    tool_reward_list[i] = tool_reward
                    done_list[i] = True
                    continue
                
                # 提取工具调用信息
                tool_calls = self.extract_tool_call(action_text)
                
                if tool_calls is not None and len(tool_calls) > 0:
                    # 统计工具调用
                    step_tool_stats["total_tool_calls"] += 1
                    tool_reward = 1.0  # 提取到tool_call给1分
                    
                    # 只使用第一个工具调用
                    tool_call = tool_calls[0]
                    tool_name = tool_call['name']
                    tool_params = tool_call['parameters']
                    
                    # 统计具体工具调用
                    if tool_name not in step_tool_stats["tool_calls_by_name"]:
                        step_tool_stats["tool_calls_by_name"][tool_name] = 0
                        step_tool_stats["tool_success_by_name"][tool_name] = 0
                    step_tool_stats["tool_calls_by_name"][tool_name] += 1
                    
                    # 工具名称白名单检查
                    tool_name_list = self.available_tools
                    if tool_name not in tool_name_list:
                        # 工具不在白名单中，返回错误信息
                        error_text = f"Tool {tool_name} is not available"
                        obs_token_ids = self.tokenizer.encode(error_text, add_special_tokens=False)
                        obs_list[i] = {
                            "prompt_token_ids_vllm": torch.tensor(obs_token_ids),
                            "prompt_token_ids_model": torch.tensor(obs_token_ids),
                        }
                        tool_reward_list[i] = tool_reward
                        continue
                    
                    tool_reward = 2.0  # 工具在列表中给2分
                    
                    # 调用工具，传入idx用于访问图像历史
                    tool_result = self.call_tool_with_retry(tool_name, tool_params, idx)
                    
                    # 添加详细的调试信息
                    print(f"啊啊啊啊啊[DEBUG] Tool call result for {tool_name}: status={tool_result.get('status', 'unknown')}, error_code={tool_result.get('error_code', 'unknown')}")
                    if 'message' in tool_result:
                        print(f"啊啊啊啊啊[DEBUG] Tool message: {tool_result['message']}")

                    # 检查工具调用是否成功
                    success = False
                    if tool_result:
                        status = tool_result.get("status", "")
                        error_code = tool_result.get("error_code", -1)
                        
                        # 修复成功判断逻辑：必须同时满足status="success"且error_code=0
                        success = (status == "success" and error_code == 0)
                        
                        # 如果工具需要图像参数但传入了None，视为失败
                        if "image" in tool_params and tool_params["image"] is None:
                            success = False
                            print(f"啊啊啊啊啊[DEBUG] Tool call failed due to missing image parameter")
                        
                        print(f"啊啊啊啊啊[DEBUG] Tool success determination: status='{status}', error_code={error_code}, success={success}")
                        
                        if success:
                            step_tool_stats["successful_tool_calls"] += 1
                            step_tool_stats["tool_success_by_name"][tool_name] += 1
                    
                    # 检查工具结果中是否有tool_reward
                    if tool_result and "tool_reward" in tool_result:
                        tool_reward = tool_result["tool_reward"]
                        tool_result.pop("tool_reward")
                    
                    # 处理工具结果，包含图像编辑逻辑
                    if tool_result:
                        edited_image = None
                        
                        # 工具返回结果只会是字典形式
                        if isinstance(tool_result, dict):
                            # 处理edited_image
                            if "edited_image" in tool_result:
                                try:
                                    # 从结果中获取编辑后的图像
                                    edited_image_base64 = tool_result.get("edited_image")
                                    if edited_image_base64 is not None:
                                        # 将base64字符串转换为PIL图像
                                        edited_image = base64_to_pil(edited_image_base64)
                                        
                                        # 添加到图像历史中
                                        if idx in self.image_history:
                                            next_img_idx = len(self.image_history[idx]) + 1
                                            new_img_key = f"img_{next_img_idx}"
                                            self.image_history[idx][new_img_key] = edited_image
                                except Exception as e:
                                    print(f"啊啊啊啊啊啊[DEBUG] Error processing edited_image: {e}")
                                    edited_image = None
                            
                            # 构建观察文本，移除edited_image字段避免过长的base64字符串
                            tool_result_display = tool_result.copy()
                            if "edited_image" in tool_result_display:
                                tool_result_display["edited_image"] = "<image_data>"  # 用占位符替换
                            
                            obs_text = f"OBSERVATION:\n{tool_name} tool output: {tool_result_display}\n"
                            
                            # 如果有编辑的图像，添加图像信息到观察结果
                            if edited_image is not None:
                                obs_text += f"New image generated and saved as: img_{len(self.image_history[idx])}\n"
                            
                            obs_token_ids = self.tokenizer.encode(obs_text, add_special_tokens=False)
                            
                            # 当有图像数据时，需要不同的处理
                            if edited_image is not None:
                                # 为包含图像的观察文本进行特殊处理
                                obs_text_with_image = f"OBSERVATION:\n{tool_name} tool output: {tool_result_display}\n<image>\nNew image generated and saved as: img_{len(self.image_history[idx])}\n"
                                
                                # 使用_preprocess_multi_modal_inputs处理
                                prompt_str_vllm, obs_token_ids_model, mm_inputs = _preprocess_multi_modal_inputs(
                                    obs_text_with_image, 
                                    self.processor, 
                                    multi_modal_data={"image": [edited_image]}
                                )
                                obs_token_ids_vllm = self.tokenizer.encode(prompt_str_vllm, add_special_tokens=False)
                                
                                obs_result = {
                                    "prompt_token_ids_vllm": torch.tensor(obs_token_ids_vllm),
                                    "prompt_token_ids_model": obs_token_ids_model,
                                    "multi_modal_data": {"image": [edited_image]}
                                }
                                if mm_inputs:
                                    obs_result["multi_modal_inputs"] = mm_inputs
                            else:
                                obs_result = {
                                    "prompt_token_ids_vllm": torch.tensor(obs_token_ids),
                                    "prompt_token_ids_model": torch.tensor(obs_token_ids),  # 完全相同！
                                }
                            
                            obs_list[i] = obs_result
                        
                        else:
                            # 如果不是字典格式（理论上不应该发生），构建默认观察文本
                            obs_text = f"OBSERVATION:\n{tool_name} tool output: {tool_result}\n"
                            obs_token_ids = self.tokenizer.encode(obs_text, add_special_tokens=False)
                            obs_list[i] = {
                                "prompt_token_ids_vllm": torch.tensor(obs_token_ids),
                                "prompt_token_ids_model": torch.tensor(obs_token_ids),
                            }
                
                    else:
                        # 工具调用失败
                        error_text = f"Tool {tool_name} call failed"
                        obs_token_ids = self.tokenizer.encode(error_text, add_special_tokens=False)
                        obs_list[i] = {
                            "prompt_token_ids_vllm": torch.tensor(obs_token_ids),
                            "prompt_token_ids_model": torch.tensor(obs_token_ids),
                        }
                
                else:
                    # 没有找到工具调用，返回原始提示
                    obs_text = "Please continue with your response or call a tool."
                    obs_token_ids = self.tokenizer.encode(obs_text, add_special_tokens=False)
                    obs_list[i] = {
                        "prompt_token_ids_vllm": torch.tensor(obs_token_ids),
                        "prompt_token_ids_model": torch.tensor(obs_token_ids),
                    }
                
                tool_reward_list[i] = tool_reward
                    
            except Exception as e:
                # 处理异常情况
                error_text = f"Error processing action: {str(e)}"
                obs_token_ids = self.tokenizer.encode(error_text, add_special_tokens=False)
                obs_list[i] = {
                    "prompt_token_ids_vllm": torch.tensor(obs_token_ids),
                    "prompt_token_ids_model": torch.tensor(obs_token_ids),
                }
                tool_reward_list[i] = tool_reward

        return obs_list, done_list, {
            "tool_rewards": tool_reward_list,
            "tool_stats": step_tool_stats  # 确保这里包含了正确的统计信息
        }

    def close(self):
        pass
