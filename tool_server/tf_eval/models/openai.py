from .abstract_model import tp_model
import uuid
import time
import random
import json
import os
import base64
from typing import List
from PIL import Image
from openai import OpenAI
from ..utils.utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem
from .template_instruct import *
from ..utils.log_utils import get_logger
from tool_server.utils.prompts import *
from tool_server.utils.utils import *

inferencer_id = str(uuid.uuid4())[:6]
logger = get_logger(__name__)


class OpenaiModels(tp_model):
    def __init__(
        self,  
        pretrained: str = "gpt-4o",  # OpenAI模型名称
        tensor_parallel: int = 1,  # 用不到，保持接口兼容
        limit_mm_per_prompt: int = 10,  # 用不到，保持接口兼容
        max_retry: int = 15,  # 最大重试次数
        temperature: float = None,
        custom_system_prompt: str = None,
        **kwargs 
    ):
        
        self.model_name = pretrained
        self.max_retry = max_retry
        self.temperature = temperature if temperature is not None else 0.0
        
        # 支持多个API密钥进行负载均衡和容错
        self.api_keys = [
            os.environ.get("OPENAI_API_KEY"),
            # 可以添加更多API密钥
        ]
        # 过滤掉None值
        self.api_keys = [key for key in self.api_keys if key is not None]
        
        if not self.api_keys:
            raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY environment variable.")

        self.base_url = os.environ.get("BASE_URL")
        
        self.api_key = random.choice(self.api_keys)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        
        # 设置生成配置
        self.generation_config = {
            "max_new_tokens": 2048,
            "temperature": self.temperature
        }
        # remote_breakpoint(port=7119)
        
        
    def to(self, *args, **kwargs):
        pass

    def eval(self):
        pass
    
    def generate_conversation_fn(self, text, images, role="user"):
        # 与VLLM保持一致：强制要求system_prompt必须先设置
        assert self.system_prompt, "System prompt must be set before generating conversation."
        
        # 添加 "Question: " 前缀，与 VLLM 保持一致
        text = "Question: " + text
        
        try:
            # 构建消息列表
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                }
            ]
            
            # 构建用户消息内容
            user_content = [
                {
                    "type": "text",
                    "text": text
                }
            ]
            
            # 处理图像
            if images:
                for image in images:
                    image_data = self._process_image(image)
                    if image_data:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        })
            
            messages.append({
                "role": role,
                "content": user_content
            })
            
            return messages
            
        except Exception as e:
            print(f"ERROR: generate_conversation_fn失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _process_image(self, image):
        """处理各种格式的图像输入，返回base64编码的字符串"""
        try:
            if isinstance(image, bytes):
                # 直接使用bytes数据
                return base64.b64encode(image).decode('utf-8')
            elif isinstance(image, str):
                # 假设是base64编码的字符串
                try:
                    # 验证是否为有效的base64
                    base64.b64decode(image)
                    return image
                except:
                    # 可能是文件路径
                    if os.path.exists(image):
                        with open(image, 'rb') as f:
                            return base64.b64encode(f.read()).decode('utf-8')
                    else:
                        print(f"ERROR: 无法处理图像字符串: {image}")
                        return None
            elif isinstance(image, Image.Image):
                # PIL Image对象
                image_base64 = pil_to_base64(image)
                return image_base64
            else:
                print(f"ERROR: 不支持的图像类型: {type(image)}")
                return None
        except Exception as e:
            print(f"ERROR: 处理图像失败: {e}")
            return None
    
    def append_conversation_fn(self, conversation, text, image, role):
        """向对话历史中添加新消息"""
        content = [
            {
                "type": "text", 
                "text": text
            }
        ]
        
        # 处理图像
        if image:
            image_data = self._process_image(image)
            if image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                })
        
        conversation.append({
            "role": role,
            "content": content
        })
        
        return conversation

    def form_input_from_dynamic_batch(self, batch: List[DynamicBatchItem]):
        """从动态批次中提取对话内容"""
        if len(batch) == 0:
            return None
        conversations = []
        for item in batch:
            conversations.append(item.conversation)
        return conversations
    
    def generate(self, batch):
        """批量生成响应"""
        if not batch or len(batch) == 0:
            return
            
        max_new_tokens = self.generation_config.get("max_new_tokens", 2048)
        conversations = self.form_input_from_dynamic_batch(batch)
        
        output_texts = []
        for conv_idx, conversation in enumerate(conversations):
            fail_times = 1
            fail_flag = False
            base_sleeptime = 15
            
            while fail_times <= self.max_retry:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=conversation,
                        max_tokens=max_new_tokens,
                        temperature=self.temperature,
                        timeout=300  # 设置超时时间
                    )
                    
                    response_text = response.choices[0].message.content
                    if not response_text:
                        response_text = "OpenAI failed to generate response."
                    
                    output_texts.append(response_text)
                    fail_flag = False
                    break
                    
                except Exception as e:
                    # 切换API密钥
                    if len(self.api_keys) > 1:
                        new_api = random.choice(self.api_keys)
                        while new_api == self.api_key and len(self.api_keys) > 1:
                            new_api = random.choice(self.api_keys)
                        self.api_key = new_api
                        self.client = OpenAI(api_key=self.api_key)
                    
                    logger.error(
                        f"Error: {e}, retrying in {fail_times * base_sleeptime} seconds"
                    )
                    fail_times += 1
                    fail_flag = True
                    
                    if fail_times <= self.max_retry:
                        time.sleep(fail_times * base_sleeptime)
            
            if fail_flag:
                logger.error(f"Failed to generate response after {self.max_retry} attempts")
                if len(output_texts) < conv_idx + 1:
                    output_texts.append("OpenAI failed to generate response after multiple attempts.")
                    
        # 更新批次项目的响应
        for item, output_text in zip(batch, output_texts):
            item.model_response.append(output_text)
            # OpenAI使用 "assistant" 角色
            self.append_conversation_fn(
                item.conversation, output_text, None, "assistant"
            )
