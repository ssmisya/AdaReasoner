import re
import json
import torch
import signal
import time
import fcntl
import os

from vllm import LLM, SamplingParams
from typing import List, Union, Dict, Optional
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from PIL import Image
from contextlib import contextmanager
from copy import deepcopy
from tqdm import tqdm as tqdm_rank0
import threading

from ....utils.utils import pil_to_base64, base64_to_pil, append_jsonl

# 定义超时异常类
class TimeoutException(Exception): pass

# 定义超时处理函数
def _timeout_handler(signum, frame):
    raise TimeoutException("chat() timed out.")
signal.signal(signal.SIGALRM, _timeout_handler)



class VllmToolInferencer(object):
    """
    VLLM工具推理器类，用于处理模型推理和工具调用
    """
    def __init__(
        self,
        vllm_model: LLM = None,
        model_name: str = None,
        model_configs: dict = None,
        tool_controller_addr: str = None,
        batch_size: int = 1,
        model_mode: str = "general",
        max_rounds: int = 5,
        stop_token: str = "<stop>",
    ):
        """
        初始化VLLM工具推理器
        
        参数:
            vllm_model: 预加载的VLLM模型实例
            model_name: 模型名称
            model_configs: 模型配置参数
            tool_controller_addr: 工具控制器地址
            batch_size: 批处理大小
            model_mode: 模型模式，支持general和llava_plus
            max_rounds: 最大对话轮数
            stop_token: 停止标记
        """
        # 初始化VLLM模型
        # 如果有了vllm_model，则直接使用vllm_model，不需要model_name和model_configs
        if vllm_model is not None:
            self.vllm_model = vllm_model
        elif model_name is not None:
            if model_configs is not None:
                self.vllm_model = LLM(model=model_name, **model_configs)
            else:
                self.vllm_model = LLM(model=model_name)
        else:
            raise ValueError("Either vllm_model or model_name must be provided.")
        
        # 初始化工具管理器
        tool_manager = ToolManager(tool_controller_addr)
        tool_controller_addr_display = tool_controller_addr if tool_controller_addr else "Auto"
        print(f"controller_addr: {tool_controller_addr_display}")
        print(f"Avaliable tools are {tool_manager.available_tools}")
        
        self.tool_manager = tool_manager
        self.batch_size = batch_size
        self.model_mode = model_mode
        self.max_rounds = max_rounds
        self.stop_token = stop_token
        
        # 添加图像字典，用于存储每个对话项的图像
        self.image_history = {}
    
    def append_conversation_fn(
        self,
        conversation, 
        text: str, 
        image=None, 
        role: str = "user",
    ):
        """
        将新消息追加到对话历史中
        
        参数:
            conversation (list): 当前对话历史
            text (str): 要追加的文本消息
            image: (可选) 要包含的图像
            role (str): 发送者角色 (默认为 "user")
            
        返回:
            更新后的对话列表
        """
        if image:
            image_base64 = pil_to_base64(image, url_format=True)
            new_messages = [
                {
                    "role": role,
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_base64}
                        },
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
        else:
            new_messages = [
                {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
        
        conversation.extend(new_messages)
        return conversation

    def handle_tool_result(
        self,
        cfg,
        tool_result,
        conversations,
        model_mode: str = "general",
        original_prompt: Optional[str] = None,
        input_data_item: Dict = None,
    ):
        """
        处理工具结果，更新对话历史，生成新的提示
        
        参数:
            cfg: 工具配置
            tool_result: 工具调用返回的结果
            conversations: 当前对话历史
            model_mode (str): 生成更新提示的模式
            original_prompt (Optional[str]): 原始用户提示
            input_data_item (Dict): 包含图像和其他信息的输入数据项
            
        返回:
            更新后的对话历史
        """
        edited_image = None
        new_round_prompt = original_prompt
        image_key_info = ""
        item_id = id(input_data_item) if input_data_item else None

        if tool_result is not None:
            try:
                # 如果工具结果中包含"edited_image"，则处理图像编辑
                if "edited_image" in tool_result:
                    # 从结果中移除编辑后的图像并添加到历史
                    edited_image_base64 = tool_result.pop("edited_image")
                    # 将base64字符串转换为PIL图像
                    edited_image = base64_to_pil(edited_image_base64)
                    
                    # 添加到图像历史中
                    if item_id and item_id in self.image_history:
                        next_img_idx = len(self.image_history[item_id]) + 1
                        new_img_key = f"img_{next_img_idx}"
                        self.image_history[item_id][new_img_key] = edited_image
                        # 在工具结果中添加图像索引信息
                        image_key_info = f"\nNew image available as: {new_img_key}"
                        if isinstance(tool_result, dict):
                            tool_result["image_key"] = new_img_key
                else:
                    edited_image = None

                    
                # 从工具结果中提取文本输出
                if "edited_image" in tool_result:
                    # 将tool_result中的"edited_image"删除
                    tool_result.pop("edited_image", None)
                    tool_response_text = tool_result
                else:
                    tool_response_text = tool_result

                # 从结果中获取API名称（支持多个键名）
                api_name = cfg.get("API_name", cfg.get("api_name", ""))

                new_response = f"OBSERVATION:\n{api_name} tool output: {tool_response_text}\n"
                new_round_prompt = (
                    f"{new_response}{image_key_info}\nPlease summarize the tool output content and answer my question."
                )

            except Exception as e:
                # 如果出现错误，恢复为原始提示
                print(f"Error in handle_tool_result: {e}")
                edited_image = None
                new_round_prompt = original_prompt

        # 获取最新图像
        latest_image = None
        if item_id and item_id in self.image_history:
            latest_keys = sorted([k for k in self.image_history[item_id].keys() if k.startswith("img_")], 
                                key=lambda x: int(x.split('_')[1]))
            if latest_keys:
                latest_image = self.image_history[item_id][latest_keys[-1]]
        
        # 将新消息（包含文本和可选图像）追加到对话历史中
        updated_conversations = self.append_conversation_fn(
            conversation=conversations, 
            text=new_round_prompt, 
            image=edited_image if edited_image else latest_image, 
            role="user"
        )

        return updated_conversations
    
    def get_repr_of_conversation(self, conversation):
        """
        获取对话的字符串表示
        
        参数:
            conversation: 对话列表
            
        返回:
            对话的字符串表示
        """
        conversation_str = ""
        for message in conversation:
            role = message["role"]
            # content = [item for item in message["content"] if isinstance(item, str) or item["type"] == "text"]
            content = message["content"]
            c_res = ""
            if isinstance(content, str):
                c_res = content
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type","") == "text":
                        c_res += str(c["text"])
            else:
                raise ValueError(f"Unknown content type: {type(content)}")
            conversation_str += f"{role}: {c_res}\n"
        return conversation_str.strip()
        
    def log_item_into_file(self, item, tool_log_file):
        """
        将对话项记录到文件中
        
        参数:
            item: 对话项
            tool_log_file: 日志文件路径
        """
        conversations = item["conversations"]
        conversations_str = self.get_repr_of_conversation(conversations)
        append_jsonl(conversations_str, tool_log_file)

    # def batch_inference(self, dataset):
    #     """
    #     批量推理函数，处理BaseEvalDataset数据集中的所有项目
        
    #     参数:
    #         dataset: 要处理的BaseEvalDataset数据集
    #     """
    #     # 创建数据加载器，批大小为1，工作线程数为2，使用collate_fn确保每次返回单个数据项
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset, 
    #         batch_size=1, 
    #         num_workers=2, 
    #         collate_fn=lambda x: x[0]  # 确保每次返回一个数据项
    #     )
        
    #     # 创建进度条
    #     progress_bar = tqdm_rank0(len(dataloader), desc="Model Responding")
        
    #     # 处理数据集中的每个项目
    #     for idx, item in enumerate(dataloader):
    #         # 更新进度条
    #         progress_bar.update(1)
            
    #         # 准备输入数据
    #         input_data = {
    #             "conversation": item.get("conversation", None),
    #             "prompt": item.get("text", ""),
    #             "images": [item.get("image")] if item.get("image") is not None else []
    #         }
            
    #         if "system" in item:
    #             input_data["system"] = item["system"]
                
    #         # 执行推理
    #         results = self.inference(
    #             inputs=[input_data],
    #             max_rounds=self.max_rounds,
    #             do_sample=False
    #         )
            
    #         # 处理结果并存储到数据集中
    #         # 所以不需要return
    #         for result in results:
    #             idx = item["idx"]
    #             # 获取最后一个样本的结果
    #             sample = result["samples"][0]
    #             # 存储结果
    #             dataset.store_results(dict(idx=idx, results=sample))

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

    def inference(
        self,
        inputs: List[Dict],
        # 不是必须要设定的参数
        sampling_params: SamplingParams = None,
        max_rounds: int = 5,
        do_sample: bool = True,
        sample_num: int = 1,
        vllm_retry_num: int = 3,
        vllm_retry_interval: int = 5,
        vllm_timeout_sec: int = 500,
        tool_retry_num: int = 3,
        tool_retry_interval: int = 5,
        tool_timeout_sec: int = 60,
        tool_log_file: str = None,
    ) -> str:
        """
        执行模型推理和工具调用
        
        参数:
            inputs: 输入数据列表
            sampling_params: 采样参数
            max_rounds: 最大对话轮数
            do_sample: 是否进行采样
            sample_num: 采样数量
            vllm_retry_num: VLLM重试次数
            vllm_retry_interval: VLLM重试间隔时间(秒)
            vllm_timeout_sec: VLLM超时时间(秒)
            tool_retry_num: 工具调用重试次数
            tool_retry_interval: 工具调用重试间隔时间(秒)
            tool_timeout_sec: 工具调用超时时间(秒)
            tool_log_file: 工具日志文件路径
            
        返回:
            推理结果列表
        """
        
        # 设置采样参数
        if not do_sample:
            kwargs = {
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "detokenize": True,
            }
        else:
            kwargs = {
                "n": 1,
                "detokenize": True,
            }
        new_sampling_params = deepcopy(sampling_params)
        for k,v in kwargs.items():
            setattr(new_sampling_params, k, v)
            
        
        # 构建输入数据
        input_data = []
        for input_item_idx,item in enumerate(inputs):
            # 处理图像输入
            if "images" in item:
                images = item["images"]
                if not isinstance(images, list):
                    images = [images]
                    
            # 处理对话输入
            if "conversation" in item:
                conversation = list(item["conversation"])
                first_content = conversation[0]["content"]
                new_first_content = []
                img_idx = 0
                # 处理第一条消息中的文本和图像
                for c in first_content:
                    if c["type"] == "text":
                        new_first_content.append(c)
                    elif c["type"] == "image":
                        image = images[img_idx]
                        image_base64 = pil_to_base64(image, url_format=True)
                        new_first_content.append({"type": "image_url", "image_url": {"url": image_base64}})
                        img_idx += 1
                prompt = new_first_content[-1]["text"]
                conversation[0]["content"] = new_first_content
                initial_user_messages = conversation
            else:
                # 如果没有conversation字段，则使用prompt字段
                prompt = item["prompt"]
                initial_user_messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }]
                # 添加图像到用户消息中
                if "images" in item:
                    for current_image in images:
                        current_image_base64 = pil_to_base64(current_image, url_format=True)
                        initial_user_messages[0]["content"].append({"type": "image_url", "image_url": {"url": current_image_base64}})

            # 添加系统消息（如果存在）
            if "system" in item:
                initial_user_messages.insert(0, {"role": "system", "content": item["system"]})
            
            # 根据是否采样创建数据实例
            if do_sample:
                for _ in range(sample_num):
                    data_instance = dict(
                        input_item_idx=input_item_idx,
                        conversations=initial_user_messages.copy(),
                        status="processing",
                        model_outputs=[],
                        model_output_ids=[],
                        tool_cfgs=[],
                        tool_outputs=[],
                        new_round_input=[],
                        prompt=prompt
                    )
                    input_data.append(data_instance)
                    # 初始化该数据实例的图像历史
                    item_id = id(data_instance)
                    self.image_history[item_id] = {}
                    # 存储初始图像为img_1
                    if len(images) > 0:
                        self.image_history[item_id]["img_1"] = images[0]
                        # 添加其他图像（如果有）
                        for idx, img in enumerate(images[1:], start=2):
                            self.image_history[item_id][f"img_{idx}"] = img
            else:
                data_instance = dict(
                    input_item_idx=input_item_idx,
                    conversations=initial_user_messages.copy(),
                    status="processing",
                    model_outputs=[],
                    model_output_ids=[],
                    tool_cfgs=[],
                    tool_outputs=[],
                    new_round_input=[],
                    prompt=prompt
                )
                input_data.append(data_instance)
                # 初始化该数据实例的图像历史
                item_id = id(data_instance)
                self.image_history[item_id] = {}
                # 存储初始图像为img_1
                if len(images) > 0:
                    self.image_history[item_id]["img_1"] = images[0]
                    # 添加其他图像（如果有）
                    for idx, img in enumerate(images[1:], start=2):
                        self.image_history[item_id][f"img_{idx}"] = img
            
        # 执行多轮对话
        for round_num in range(max_rounds):
            # 获取状态为"processing"的对话和索引
            input_conversations = [item["conversations"] for item in input_data if item["status"] == "processing"]
            input_idxs = [idx for idx, item in enumerate(input_data) if item["status"] == "processing"]
            
            # 如果没有需要处理的对话，则退出循环
            if len(input_conversations) == 0:
                break
            
            # 模型推理，带重试机制
            outputs = None
            for attempt in range(vllm_retry_num):
                try:
                    # 设置超时警报
                    signal.alarm(vllm_timeout_sec)
                    outputs = self.vllm_model.chat(
                        input_conversations, sampling_params=new_sampling_params, use_tqdm = True,
                    )
                    # 取消超时警报
                    signal.alarm(0)
                    break
                except TimeoutException as te:
                    print(f"[VLLM] Timeout during chat(), retrying {attempt + 1}/{vllm_retry_num}")
                except Exception as e:
                    print(f"[VLLM] Unexpected exception during chat(): {e}, retrying {attempt + 1}/{vllm_retry_num}")
                finally:
                    # 确保取消超时警报
                    signal.alarm(0)
                    if attempt < vllm_retry_num - 1:
                        time.sleep(vllm_retry_interval)
                
            # 处理模型输出
            if outputs is None:
                # 如果模型生成失败，使用错误消息
                output_texts = ["Model generation error"] * len(input_conversations)
                output_idss = [(1712, 9471, 1465, 151645)] * len(input_conversations)
            else:
                output_texts = [output.outputs[0].text for output in outputs]
                output_idss = [output.outputs[0].token_ids for output in outputs]

            # 处理每个输出
            for input_idx, output_text, output_ids in zip(input_idxs, output_texts, output_idss):
                # 记录模型输出
                input_data[input_idx]["model_outputs"].append(output_text)
                input_data[input_idx]["model_output_ids"].append(output_ids)
                # 将模型回复添加到对话中
                input_data[input_idx]["conversations"] = self.append_conversation_fn(conversation=input_data[input_idx]["conversations"], text=output_text, role="assistant")
                
                # 如果回复中包含"<response>....</response>"，则标记为已完成并继续处理下一个
                if "<response>" in output_text and "</response>" in output_text:
                    input_data[input_idx]["status"] = "finished"
                    continue

                item_id = id(input_data[input_idx])
                # 获取当前数据项的最新图像
                newest_image = None
                if item_id in self.image_history:
                    image_keys = sorted([k for k in self.image_history[item_id].keys() if k.startswith("img_")], 
                                      key=lambda x: int(x.split('_')[1]))
                    if image_keys:
                        newest_image = self.image_history[item_id][image_keys[-1]]
                
                # 提取工具调用信息
                tool_calls = self.extract_tool_call(output_text)
                if tool_calls is not None and len(tool_calls) > 0:
                    # 只使用第一个工具调用
                    tool_call = tool_calls[0]
                    tool_name = tool_call['name']
                    tool_params = tool_call['parameters']
                    
                    # 构建工具配置格式
                    tool_cfg = [{
                        "API_name": tool_name,
                        "API_params": tool_params
                    }]
                else:
                    tool_cfg = None
                
                # 如果没有工具配置，则将原始提示添加到对话中
                if not tool_cfg:
                    input_data[input_idx]["conversations"] = self.append_conversation_fn(conversation=input_data[input_idx]["conversations"], text=input_data[input_idx]["prompt"], role="user")
                
                else:
                    # 记录工具配置
                    input_data[input_idx]["tool_cfgs"].append(tool_cfg)
                    try:
                        # 获取工具名称和参数
                        original_api_name = tool_cfg[0].get("API_name").lower() 
                        api_params = tool_cfg[0].get("API_params", {})
                        
                        # 检查是否有图像参数，并处理img_n格式的图像引用
                        if "image" in api_params:
                            img_key = api_params.get("image", "")
                            # 如果是img_n格式，则从图像历史中获取对应图像
                            if isinstance(img_key, str) and img_key.startswith("img_"):
                                if item_id in self.image_history and img_key in self.image_history[item_id]:
                                    image = self.image_history[item_id][img_key]
                                    # 转换为base64格式
                                    image_base64 = pil_to_base64(image, url_format=False)
                                    # 更新参数中的图像
                                    api_params["image"] = image_base64
                                else:
                                    print(f"Image key {img_key} not found in image history, using latest image")
                                    # 如果找不到指定图像，使用最新图像
                                    image = newest_image
                                    image_base64 = pil_to_base64(image, url_format=False)
                                    api_params["image"] = image_base64
                            else:
                                # 如果不是img_n格式，使用最新图像
                                image = newest_image
                                image_base64 = pil_to_base64(image, url_format=False)
                                api_params["image"] = image_base64
                        
                        # 工具名称映射表
                        tool_name_mapping = {
                            'segmentregionaroundpoint': 'SegmentRegionAroundPoint',
                            'point': 'Point',
                            'ocr': 'OCR',
                            'drawline': 'DrawLine',
                            'crop': 'Crop',
                            'groundingdino': 'GroundingDINO',
                            'languagemodel': 'LanguageModel',
                            'drawline': 'DrawLine',
                            'drawshape': 'DrawShape',
                            'highlight': 'Highlight',
                            'maskbox': 'MaskBox',
                            'getsubplotinfo': 'GetSubplotInfo',
                            'getbarinfo': 'GetBarInfo',
                        }
                        api_name = tool_name_mapping.get(original_api_name, original_api_name)
                    except Exception as e:
                        print(f"Tool config error: {e}")
                        api_name = "None"
                        api_params = {}

                    # 打印工具参数（如果存在）
                    if "param" in api_params:
                        p = api_params["param"]
                        print(f"Tool name: {api_name}, params: {p}")
                    
                    # 调用工具，带重试机制
                    tool_result = {"text": f"Failed to call tool {api_name}", "error_code": 1}
                    for attempt in range(tool_retry_num):
                        try:
                            # 设置超时警报
                            signal.alarm(tool_timeout_sec)
                            tool_result = self.tool_manager.call_tool(api_name, api_params)
                            # 取消超时警报
                            signal.alarm(0)
                            break
                        except TimeoutException as te:
                            print(f"[Tool] Timeout during tool call, retrying {attempt + 1}/{tool_retry_num}")
                            tool_result = {"text": f"Failed to call tool {api_name}: {te}", "error_code": 1}
                        except Exception as e:
                            print(f"[Tool] Unexpected exception during tool call: {e}, retrying {attempt + 1}/{tool_retry_num}")
                            tool_result = {"text": f"Failed to call tool {api_name}: {e}", "error_code": 1}
                        finally:
                            # 确保取消超时警报
                            signal.alarm(0)
                            if attempt < tool_retry_num - 1:
                                time.sleep(tool_retry_interval)
                    
                    # 记录工具输出
                    input_data[input_idx]["tool_outputs"].append(tool_result)
                    
                    # 处理工具结果并更新对话
                    input_data[input_idx]["conversations"] = self.handle_tool_result(
                        cfg = tool_cfg[0],
                        tool_result = tool_result,
                        conversations = input_data[input_idx]["conversations"],
                        model_mode = "general",
                        original_prompt = input_data[input_idx]["prompt"],
                        input_data_item = input_data[input_idx]
                    )
                    
        # 记录工具输出到日志文件
        if tool_log_file is not None:
            for item in input_data:
                self.log_item_into_file(item, tool_log_file)
                
        # 收集最终结果
        results = []
        # 按输入项索引分组结果
        grouped_results = {}
        for item in input_data:
            input_item_idx = item["input_item_idx"]
            if grouped_results.get(input_item_idx) is None:
                grouped_results[input_item_idx] = []
            grouped_results[input_item_idx].append(item)

            item_id = id(item)
            if item_id in self.image_history:
                del self.image_history[item_id]
        
        # 将分组结果转换为最终输出格式
        for k,v in grouped_results.items():
            results.append({"item_idx": k, "samples": v})
        return results

_lock_lock = threading.Lock()
_file_locks = {}

def _get_file_lock(filepath):
    """Get a lock for a specific file path."""
    with _lock_lock:
        if filepath not in _file_locks:
            _file_locks[filepath] = threading.Lock()
        return _file_locks[filepath]

def append_jsonl(data, filename):
    '''
        追加数据到jsonl文件，线程安全
    '''
    with _get_file_lock(filename):
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')