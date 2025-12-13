from .abstract_model import tp_model
import uuid
import time
import random
import json
import os
from typing import List
from PIL import Image
from google import genai
from google.genai import types
from ..utils.utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem
from .template_instruct import *
from ..utils.log_utils import get_logger
from tool_server.utils.prompts import *
from tool_server.utils.utils import *
from tool_server.utils.debug import remote_breakpoint

inferencer_id = str(uuid.uuid4())[:6]
logger = get_logger(__name__)


class GeminiModels(tp_model):
    def __init__(
      self,  
      pretrained: str = "gemini-2.5-flash",
      tensor_parallel: int = 1, # 用不到
      limit_mm_per_prompt: int = 10, # 用不到
      max_retry: int = 15, # 最大重试次数
      temperature: float = None,
      custom_system_prompt: str = None,
      **kwargs 
    ):
        
        self.model_name = pretrained
        self.max_retry = max_retry
        self.temperature = temperature
        self.api_keys = [
            os.environ.get("GEMINI_API_KEY")
        ]
        self.api_key = random.choice(self.api_keys)
        self.client = genai.Client(api_key=self.api_key)
        self.generation_config = {
            "max_new_tokens": 2048,
            "temperature": temperature if temperature is not None else 0.0
        }
        # remote_breakpoint(port=7119)
        
        
    def to(self, *args, **kwargs):
        pass

    def eval(self):
        pass
    
    def generate_conversation_fn(self, text, images, role="user",**kwargs):
        # 与VLLM保持一致：强制要求system_prompt必须先设置
        assert self.system_prompt, "System prompt must be set before generating conversation."
        
        # 添加调试信息
        
        # 添加 "Question: " 前缀，与 VLLM 保持一致
        text = "Question: " + text
            
        try:
            contents = [
                types.Content(
                    role="user",  # 系统提示词作为用户消息发送
                    parts=[
                        types.Part.from_text(text=self.system_prompt)
                    ]
                )
            ]
            
            first_round_parts = [types.Part.from_text(text=text)]

            # 处理图像 - 修复：添加对bytes类型的支持
            if images:
                for image in images:
                    # 修复：添加对bytes类型的处理
                    if isinstance(image, bytes):
                        try:
                            # 直接使用bytes数据
                            part = types.Part.from_bytes(
                                mime_type="image/jpeg",
                                data=image
                            )
                            first_round_parts.append(part)
                        except Exception as e:
                            print(f"ERROR: 处理bytes图像失败: {e}")
                    elif isinstance(image, str):
                        try:
                            part = types.Part.from_bytes(
                                mime_type="image/jpeg",
                                data=base64.b64decode(image)
                            )
                            first_round_parts.append(part)
                        except Exception as e:
                            print(f"ERROR: 处理base64图像失败: {e}")
                    elif isinstance(image, Image.Image):
                        try:
                            image = pil_to_base64(image)
                            part = types.Part.from_bytes(
                                mime_type="image/jpeg",
                                data=base64.b64decode(image)
                            )
                            first_round_parts.append(part)
                        except Exception as e:
                            print(f"ERROR: 处理PIL图像失败: {e}")
                    else:
                        print(f"ERROR: 不支持的图像类型: {type(image)}")
                        
            contents.append(
                types.Content(
                    role=role,
                    parts=first_round_parts
                )
            )
            return contents
            
        except Exception as e:
            print(f"ERROR: generate_conversation_fn失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def append_conversation_fn(self, conversation, text, image, role,**kwargs):
        parts = [types.Part.from_text(text=text)]
        
        # 处理图像 - 修复：添加对bytes类型的支持
        if image:
            if isinstance(image, bytes):
                try:
                    part = types.Part.from_bytes(
                        mime_type="image/jpeg",
                        data=image
                    )
                    parts.append(part)
                except Exception as e:
                    print(f"ERROR: append_conversation_fn处理bytes图像失败: {e}")
            elif isinstance(image, str):
                try:
                    part = types.Part.from_bytes(
                        mime_type="image/jpeg",
                        data=base64.b64decode(image)
                    )
                    parts.append(part)
                except Exception as e:
                    print(f"ERROR: append_conversation_fn处理base64图像失败: {e}")
            elif isinstance(image, Image.Image):
                try:
                    image = pil_to_base64(image)
                    part = types.Part.from_bytes(
                        mime_type="image/jpeg",
                        data=base64.b64decode(image)
                    )
                    parts.append(part)
                except Exception as e:
                    print(f"ERROR: append_conversation_fn处理PIL图像失败: {e}")
            else:
                print(f"ERROR: append_conversation_fn不支持的图像类型: {type(image)}")
        
        conversation.append(
            types.Content(
                role=role,
                parts=parts
            )
        )
        
        return conversation

    
    def form_input_from_dynamic_batch(self, batch: List[DynamicBatchItem]):
        if len(batch) == 0:
            return None
        contents_list = []
        for item in batch:
            contents_list.append(item.conversation)
        return contents_list
    
    def generate(self, batch):
        if not batch or len(batch) == 0:
            return
            
        max_new_tokens = self.generation_config.get("max_new_tokens", 2048)
        inputs = self.form_input_from_dynamic_batch(batch)
        
        generate_content_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=max_new_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=-1), # 不使用思考
            response_mime_type="text/plain",
        )

        output_texts = []
        for conv_idx, conversation in enumerate(inputs):
            fail_times = 1
            fail_flag = False
            base_sleeptime = 15
            while fail_times < int(self.max_retry):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=conversation,
                        config=generate_content_config,
                    )
                    response_text = response.text
                    if not response_text:
                        response_text = "Gemini failed to generate response."
                    output_texts.append(response_text)
                    fail_flag = False
                    break  
                except Exception as e:
                    new_api = random.choice(self.api_keys)
                    while new_api == self.api_key and len(self.api_keys) > 1:
                        new_api = random.choice(self.api_keys)
                    self.api_key = new_api
                    self.client = genai.Client(api_key=self.api_key)
                    
                    logger.error(
                        f"Error: {e}, retrying in {fail_times * base_sleeptime} seconds"
                    )
                    fail_times += 1
                    fail_flag = True
                    time.sleep(fail_times * base_sleeptime)
            
            if fail_flag:
                logger.error(f"Failed to generate response after {self.max_retry} attempts")
                if len(output_texts) < conv_idx + 1:
                    output_texts.append("Gemini failed to generate response after multiple attempts.")
                    
        for item, output_text in zip(batch, output_texts):
            item.model_response.append(output_text)
            # 修复：Gemini API使用 "model" 而不是 "assistant"
            self.append_conversation_fn(
                item.conversation, output_text, None, "model"
            )