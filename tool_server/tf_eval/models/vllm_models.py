from .abstract_model import tp_model
import uuid,requests,time
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from typing import List
import torch
from vllm import LLM, SamplingParams


# from .template_instruct import *
from ..utils.utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem
from ...utils.prompts import *

from ..utils.log_utils import get_logger
inferencer_id = str(uuid.uuid4())[:6]  # 生成唯一的推理器ID
logger = get_logger("vllm_models",)    # 获取日志记录器

class VllmModels(tp_model):
    """
    VLLM模型类，用于处理多模态（文本+图像）输入的大语言模型推理
    继承自抽象模型类tp_model
    """
    def __init__(
      self,  
      pretrained : str = None,        # 预训练模型路径
      tensor_parallel: str = "1",     # 张量并行数量
      limit_mm_per_prompt: str = "1",  # 每限制的是单个消息中的图片数量，而不是整个对话历史中的图片总数
      enable_tool: bool = True,          # 是否使用工具
    ):
        tensor_parallel = eval(tensor_parallel)  # 将字符串转换为数值
        self.model = LLM(
            model=pretrained,                   # 模型路径
            tensor_parallel_size=tensor_parallel,  # 并行数
            limit_mm_per_prompt={"image": int(limit_mm_per_prompt)}  # 限制每个prompt的图像数量
        )
        
        if enable_tool.lower() == "true":
            self.enable_tool = True
        else:
            self.enable_tool = False
        
        # print(f"初始化后的enable_tool: {self.enable_tool}, 类型: {type(self.enable_tool)}")

    def generate_conversation_fn(
        self,
        text,                # 输入文本
        image,               # 输入图像
        role = "user",       # 对话角色，默认为用户
    ):  
        """
        生成对话格式的函数，将文本和图像组合成模型可接受的消息格式
        
        参数:
            text: 用户输入的文本
            image: 用户提供的图像
            role: 对话角色，默认为"user"
            
        返回:
            messages: 格式化的对话消息列表
        """
        text = "Question: " + text  # 在文本前添加"Question:"前缀

        # print(f"DEBUG 1: self.enable_tool = {self.enable_tool}, type = {type(self.enable_tool)}")

        if self.enable_tool == True:  # 明确比较
            system_prompt = tool_planning_model_prompt_one_tool_call
            # print(f"DEBUG 2: 进入True分支, system_prompt = {system_prompt[-20:]}...")
        else:
            system_prompt = tool_planning_model_prompt_no_tool_call
            # print(f"DEBUG 3: 进入False分支, system_prompt = {system_prompt[-20:]}...")

        # print(f"DEBUG 4: 最终 system_prompt = {system_prompt[-20:]}...")
        # print(f"DEBUG 5: tool_planning_model_prompt_one_tool_call = {tool_planning_model_prompt_one_tool_call[-20:]}...")
        # print(f"DEBUG 6: tool_planning_model_prompt_no_tool_call = {tool_planning_model_prompt_no_tool_call[-20:]}...")
        
        image = pil_to_base64(image)  # 将PIL图像转换为base64编码
        messages = [
            {
                "role": "system",  # 系统角色
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,  # prompt.py中定义的评估提示
                    },
                ],
            },
            {
                "role": "user",    # 用户角色
                "content": [
                    {
                        "type": "text",
                        "text": text,  # 用户输入的文本
                    },
                    {
                        "type": "image_url",  # 图像URL类型
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"  # 以base64格式提供图像
                        },
                    },
                ],
            }
        ]

        return messages
    
    
    def append_conversation_fn(
        self, 
        conversation,  # 现有对话历史
        text,          # 要添加的文本
        image,         # 要添加的图像（可选）
        role           # 对话角色
    ):
        """
        向现有对话中追加新的消息
        
        参数:
            conversation: 现有对话历史
            text: 要添加的文本内容
            image: 要添加的图像（如果有）
            role: 发言角色（"user"或"assistant"）
            
        返回:
            更新后的对话历史
        """
        if image:  # 如果提供了图像
            new_messages=[
                {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}"
                            },
                        },
                    ],
                }
            ]
        else:  # 只有文本，没有图像
            new_messages=[
                {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                        }
                    ],
                }
            ]
        
        conversation.extend(new_messages)  # 将新消息添加到对话历史中

        return conversation
    
    
    def form_input_from_dynamic_batch(self, batch: List[DynamicBatchItem]):
        """
        从动态批次项中提取对话内容
        
        参数:
            batch: 动态批次项列表
            
        返回:
            messages: 提取的消息列表，用于模型输入
        """
        if len(batch) == 0:  # 如果批次为空
            return None
        messages = []
        for item in batch:
            messages.append(item.conversation)  # 收集每个批次项的对话
        return messages
    
    def generate(self, batch):
        """
        批量生成回复
        
        参数:
            batch: 包含多个对话的批次
            
        功能:
            为批次中的每个对话生成模型回复，并更新对话历史
        """
        if not batch or len(batch) == 0:  # 如果批次为空
            return
        max_new_tokens = self.generation_config.get("max_new_tokens", 2048)  # 获取最大生成token数，默认2048
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.6)  # 设置采样参数
        
        inputs = self.form_input_from_dynamic_batch(batch)  # 从批次中提取输入
        # breakpoint()  # 调试断点
        response = self.model.chat(inputs, sampling_params)  # 调用模型并行生成回复，VLLM库的并行推理


        for item, output_item in zip(batch, response):
            output_text = output_item.outputs[0].text  # 获取生成的文本
            item.model_response.append(output_text)  # 将回复添加到模型响应列表
            self.append_conversation_fn(
                item.conversation, output_text, None, "assistant"  # 将模型回复添加到对话历史
            )
            
    def to(self, *args, **kwargs):
        """
        兼容PyTorch的设备迁移方法
        
        返回:
            self: 模型实例本身
        """
        return self
    
    def eval(self):
        """
        兼容PyTorch的评估模式设置方法
        
        返回:
            self: 模型实例本身
        """
        return self
            
        
        
