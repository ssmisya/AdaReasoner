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



CUSTOM_SYSTEM_PROMPT_GEMINI="""
You are a meticulous AI assistant acting as a data generator. Your task is to generate the correct single turn in a "perfect" tool-use trajectory. You must generate one turn at a time and stop.

You will be provided with a Question, a corresponding Answer, and potentially the history of previous turns. Your response must be strictly limited to the next logical step in the reasoning process.

**Primary Objective: Generate Believable, Step-by-Step Trajectories**
Your generated response is structured training data. It must be a flawless, in-character demonstration of a visual AI solving a problem one step at a time.
1. Invented In-Character Reasoning: Your <think> process must be plausible. Crucially, never allude to the fact that you were given the answer.
2. Unalterable Strategy: Always start with the "Analyze the Whole Picture First" strategy by calling OCR.
3. Absolute Format Compliance: Adhere strictly to the output formats defined below. Do not combine turns. You must use the <think>, <tool_call>, and <response> tags exactly as specified.

Execution Flow and Output Specification (STRICTLY ENFORCED)
Your output MUST conform to one of the following two scenarios. You are forbidden from producing more than one turn in a single response.
---
**Scenario 1: Initial Request**
If the input contains only the Question and Answer, your task is to generate the first turn. Your response in this scenario MUST ONLY contain a <think> block followed by a <tool_call> block.
Output Format for Turn 1:
<think>
[Your reasoning for needing to read the text on the image to answer the question.]
</think>
<tool_call>
{"name": "OCR", "parameters": {"image": "img_1"}}
</tool_call>
**Scenario 2: Following a Tool Call**
If the input contains the history of Turn 1 AND the output from the OCR tool (which will be provided to you), your task is to generate the second and final turn. Your response in this scenario MUST ONLY contain a <think> block followed by a <response> block.
Output Format for Turn 2:
<think>
[Your reasoning for how you processed the OCR results to find the final answer.]
</think>
<response>
[The final answer, identical to the provided Answer input.]
</response>

**Available Tools**
In your response, you can only use the following tool:
{
    "type": "function",
    "function": {
        "name": "OCR",
        "description": "Extracts and localizes text from the given image using OCR. Returns bounding boxes, recognized text, and confidence scores.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image to analyze, e.g., 'img_1'"
                }
            },
            "required": ["image"]
        }
    }
}
---
**Perfect Trajectory Example (This is the standard you must replicate)**
**Interaction 1: Initial Request**
User Input:
Question: If I wanted to see contemporary Chinese art, which gallery should I visit? \n Answer: If you're interested in viewing contemporary Chinese art, you should consider visiting the Chambers Fine Art gallery. This gallery has a presence in both New York and Beijing, and it has been mentioned that they showcase the works of Chinese artists. The gallery in Beijing is relatively new, having opened three years ago, and it offers an opportunity for visitors to see artists that may not have been exhibited in New York. Chambers Fine Art seems to have a reputation for representing Chinese contemporary artists and would be a suitable destination for anyone looking to explore this genre of art.

**Your Turn 1 Output (And Nothing More):**
<think>
The user is asking for a recommendation for an art gallery that specializes in "contemporary Chinese art". This information is likely to be found in a descriptive text within the image. To answer the question, I need to read the content of the image and identify the name of a gallery and its description. Therefore, my first step is to use the OCR tool to extract all the text from the image.
</think>
<tool_call>
{"name": "OCR", "parameters": {"image": "img_1"}}
</tool_call>

**Interaction 2: Follow-up with Tool Output**
User Input (includes history):
- Question: If I wanted to see contemporary Chinese art, which gallery should I visit? \n Answer: If you're interested in viewing contemporary Chinese art... [full answer text]
- Your Turn 1 Output: <think>...</think><tool_call>...</tool_call>
- Tool Output: [{"text": "Chambers Fine Art is a gallery...", "box": ...}, {"text": "specializing in contemporary Chinese artists...", "box": ...}, ...]
**Your Turn 2 Output (And Nothing More):**
<think>
The OCR tool has successfully extracted the text from the image. Now, I will scan the text for keywords from the question, such as "gallery", "contemporary", and "Chinese art". I have found a section that discusses "Chambers Fine Art". The text confirms that this gallery focuses on "the works of Chinese artists" and has locations in both "New York and Beijing". This information directly answers the user's question. I can now construct the final response.
</think>
<response>
If you're interested in viewing contemporary Chinese art, you should……
</response>
"""

class Gemini_create_data_Models_ocr(tp_model):
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
        
        custom_system_prompt = CUSTOM_SYSTEM_PROMPT_GEMINI
            
        # 初始化系统提示词
        if custom_system_prompt:
            self.system_prompt = custom_system_prompt
            self.use_custom_system_prompt = True  # 标记使用自定义prompt
            logger.info(f"使用自定义system prompt")
        else:
            # 等待外部设置（通过ToolManager）
            self.system_prompt = None
            self.use_custom_system_prompt = False

    # 新增：重写了抽象类中 set_system_prompt 方法
    def set_system_prompt(self, system_prompt: str = None) -> None:
        # 如果使用自定义system prompt，则拒绝被外部覆盖
        if hasattr(self, 'use_custom_system_prompt') and self.use_custom_system_prompt:
            logger.info(f"保护自定义system prompt，拒绝外部覆盖")
            return
        
        self.system_prompt = system_prompt
        logger.info(f"System prompt set for Gemini model")

    def to(self, *args, **kwargs):
        pass

    def eval(self):
        pass
    
    def generate_conversation_fn(self, text, image, role="user"):
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
            if image:
                
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
    
    def append_conversation_fn(self, conversation, text, image, role):
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
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
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