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

inferencer_id = str(uuid.uuid4())[:6]
logger = get_logger(__name__)

class GeminiModels(tp_model):
    def __init__(
      self,  
      model_name: str = "gemini-2.5-flash",
      max_retry: int = 5,
      temperature: float = None,
      enable_tool: bool = True, 
    ):
        self.model_name = model_name
        self.max_retry = max_retry
        self.temperature = temperature
        self.api_keys = [os.environ.get("GEMINI_API_KEY")]  # You can add more keys to the list
        self.api_key = random.choice(self.api_keys)
        self.client = genai.Client(api_key=self.api_key)
        self.generation_config = {
            "max_new_tokens": 2048,
            "temperature": temperature if temperature is not None else 0.0
        }
        if enable_tool.lower() == "true":
            self.enable_tool = True
        else:
            self.enable_tool = False

    def to(self, *args, **kwargs):
        pass

    def eval(self):
        pass
    
    def generate_conversation_fn(self, text, image, role="user"):
        if self.enable_tool == True:  # 明确比较
            system_prompt = tool_planning_model_prompt_one_tool_call
            # print(f"DEBUG 2: 进入True分支, system_prompt = {system_prompt[-20:]}...")
        else:
            system_prompt = tool_planning_model_prompt_no_tool_call
            
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=system_prompt)
                ]
            )
        ]
        first_round_parts = [types.Part.from_text(text=text) ]

        if image:
            if isinstance(image, list):  # Handle multiple images
                for img in image:
                    if isinstance(img, str):
                        part = types.Part.from_bytes(
                            mime_type="image/jpeg",
                            data=base64.b64decode(img)
                        )
                        first_round_parts.append(part)
                        
                    elif isinstance(img, Image.Image):
                        img = pil_to_base64(img)
                        part = types.Part.from_bytes(
                            mime_type="image/jpeg",
                            data=base64.b64decode(img)
                        )
                        first_round_parts.append(part)
                    else:
                        raise ValueError("Unsupported image type in list")
            elif isinstance(image, str):  # Handle a single image
                part = types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=base64.b64decode(image)
                )
                first_round_parts.append(part)
            elif isinstance(image, Image.Image):  # Handle PIL Image
                image = pil_to_base64(image)
                part = types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=base64.b64decode(image)
                )
                first_round_parts.append(part)
            else:
                raise ValueError("Unsupported image type")
            
        contents.append(
            types.Content(
                role=role,
                parts=first_round_parts
            )
        )


        return contents
    
    def append_conversation_fn(self, conversation, text, image, role):
        parts = [types.Part.from_text(text=text)]
        
        # Add image(s) if provided
        if image:
            if isinstance(image, list):  # Handle multiple images
                for img in image:
                    if isinstance(img, str):
                        part = types.Part.from_bytes(
                            mime_type="image/jpeg",
                            data=base64.b64decode(img)
                        )
                        parts.append(part)
                    elif isinstance(img, Image.Image):
                        img = pil_to_base64(img)
                        part = types.Part.from_bytes(
                            mime_type="image/jpeg",
                            data=base64.b64decode(img)
                        )
                        parts.append(part)
                    else:
                        raise ValueError("Unsupported image type in list")
            elif isinstance(image, str):  # Handle a single image
                part = types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=base64.b64decode(image)
                )
                parts.append(part)
            elif isinstance(image, Image.Image):  # Handle PIL Image
                image = pil_to_base64(image)
                part = types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=base64.b64decode(image)
                )
                parts.append(part)
            else:
                raise ValueError("Unsupported image type")
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
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            response_mime_type="text/plain",
        )

        output_texts = []
        for conv_idx,conversation in enumerate(inputs):
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
                if len(output_texts) < conv_idx +1 :
                    output_texts.append("Gemini failed to generate response after multiple attempts.")
                    
        for item, output_text in zip(batch, output_texts):
            item.model_response.append(output_text)
            self.append_conversation_fn(
                item.conversation, output_text, None, "model"
            )