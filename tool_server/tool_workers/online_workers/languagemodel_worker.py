"""
A model worker executes the model.
"""

import torch
import numpy as np
from PIL import Image
import base64
import uuid
import os
import traceback
import re
from io import BytesIO
import sys
from pathlib import Path

from transformers import HfArgumentParser, AutoProcessor
from dataclasses import dataclass, field
from typing import Optional
from vllm import LLM, SamplingParams

from tool_server.utils.server_utils import build_logger
from tool_server.utils.utils import load_image, pil_to_base64
from tool_server.utils.worker_arguments import WorkerArguments
from tool_server.utils.error_codes import *
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"languagemodel_worker_{worker_id}.log")


@dataclass
class LanguageModelArguments(WorkerArguments):
    max_tokens: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum number of tokens to generate"}
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "Temperature for sampling"}
    )
    tensor_parallel_size: Optional[int] = field(
        default=1,
        metadata={"help": "Number of GPUs to use in parallel"}
    )


class LanguageModelWorker(BaseToolWorker):
    def __init__(self, worker_arguments: LanguageModelArguments = None):
        # 在调用父类初始化前先设置模型名称
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "LanguageModel"
        
        # 保存参数到实例变量
        self.max_tokens = worker_arguments.max_tokens if worker_arguments else 4096
        self.temperature = worker_arguments.temperature if worker_arguments else 0.7
        self.tensor_parallel_size = worker_arguments.tensor_parallel_size if worker_arguments else 1
        
        super().__init__(worker_arguments)
            
        self.instruction = {
            "type": "function",
            "function": {
                "name": "LanguageModel",
                "description": 
                    "Reason based on the prompt and image, and return the Language Model's inference text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier of the image to analyze, e.g., 'img_1'."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to analyze."
                        }
                    },
                    "required": ["image", "prompt"]
                }
            }
        }
        
    def init_model(self):
        logger.info(f"初始化模型 {self.model_name}...")
        logger.info(f"CUDA 可用: {torch.cuda.is_available()}, GPU 数量: {torch.cuda.device_count()}")
        
        try:
            # 加载模型
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=32768,
                limit_mm_per_prompt={"image": 1, "video": 0},
                enforce_eager=True
            )
            
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load model: {str(e)}")
        
    def process_vision_info(self, messages):
        """处理消息中的图像信息"""
        image_inputs = None
        video_inputs = None
        
        for message in messages:
            for content in message.get("content", []):
                if content.get("type") == "image" and "image" in content:
                    if image_inputs is None:
                        image_inputs = []
                    image_inputs.append(content["image"])
                    
        return image_inputs, video_inputs
        
    def generate(self, params):
        tool_reward = 2.0
        # 计算Parameter Name Matching
        param_keys = set(params.keys())
        required_keys = set(self.instruction["function"]["parameters"]["required"])
        parameter_name_match_reward = len(param_keys & required_keys) / len(required_keys | param_keys)
        tool_reward = tool_reward + parameter_name_match_reward
        # 参数名称没有完全匹配，直接返回
        if parameter_name_match_reward < 1:
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": "Invalid parameters: expected keys: image, prompt.",
                "error_code": INVALID_PARAMETERS,
                "tool_reward": tool_reward
            }
        
        required_keys_num = len(required_keys)
        # 初始化参数合规计数器
        correct_param_content_num = 0
        
        # 提取输入参数
        image_path = params["image"]
        prompt = params.get("prompt")
        
        try:
            # 加载图像
            try:
                # 处理base64编码的图像
                image = Image.open(BytesIO(base64.b64decode(image_path))).convert("RGB")
                correct_param_content_num += 1
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Failed to load image: {str(e)}",
                    "error_code": CANNOT_LOAD_IMAGE,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                }
                return pred_dict
            
            # prompt参数验证通过
            correct_param_content_num += 1
            
            # 构建消息
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            # 处理消息
            chat_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # 处理图像输入
            image_inputs, video_inputs = self.process_vision_info(messages)
            
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            
            # 准备LLM输入
            llm_inputs = {
                "prompt": chat_prompt,
                "multi_modal_data": mm_data,
            }
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # 生成响应
            outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            # 返回结果
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "success",
                "response": response,
                "message": "Successfully generated response",
                "error_code": SUCCESS,
                "tool_reward": tool_reward+correct_param_content_num/required_keys_num
            }
            
            return pred_dict
            
        except Exception as e:
            logger.error(f"推理过程中出错: {e}")
            logger.error(traceback.format_exc())
            
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "error_code": TOOL_RUN_FAILED,
                "message": f"Error: {str(e)}\nTraceback:{traceback.format_exc()}\n",
                "tool_reward": tool_reward+(correct_param_content_num/required_keys_num if required_keys_num > 0 else 0)
            }
            return pred_dict
    
    def get_tool_instruction(self):
        return self.instruction



if __name__ == "__main__":
    parser = HfArgumentParser((LanguageModelArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")

    worker = LanguageModelWorker(
        worker_arguments=args
    )
    worker.run()