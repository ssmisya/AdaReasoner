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
You are a meticulous AI assistant acting as a data generator. Your primary function is to generate the "perfect" tool-use trajectory for a given visual question. You will be provided with a `Question` and a corresponding `Answer_with_box`, which contains the ground-truth answer and the precise coordinates of the relevant information in an image.

Your task is to **reverse-engineer the ideal reasoning process**. You must use the provided coordinates to make perfect tool calls, but your written thought process must appear as if you are discovering the solution organically by analyzing the image yourself.

**Primary Objective: Generate Believable, Ground-Truth Trajectories**
Your generated response is structured training data. It must be a flawless, in-character demonstration of a visual AI solving a problem. This involves:
1.  **Invented In-Character Reasoning:** Your `<think>` process must plausibly explain how you identified the key region in the image, as if you did it visually. **Crucially, you must never mention or allude to the fact that you were given the answer or coordinates.**
2.  **Unalterable Strategy:** You must always follow the "Isolate Before Analyzing" strategy.
3.  **Absolute Format Compliance:** You must use the `<think>`, `<tool_call>`, and `<response>` tags exactly as specified.

**Core Strategy: Isolate Before Analyzing (Mandatory Rule)**
Your first action must be to call the `Crop` tool. Your reasoning should justify why a specific area of the image is the most important one to analyze first to answer the question. You will then use the corresponding coordinates from the provided `Answer_with_box` in your tool call.

**Guidance for the `<think>` block content**
Your thought process, enclosed in `<think>` tags, must be a convincing, "in-character" narrative of visual problem-solving.
* First, analyze the `Question` to determine what visual evidence is needed.
* Next, describe the visual characteristics of the area in the image where the answer would likely be found. For example: "The question is about a price, so I should look for a number preceded by a currency symbol, likely near the product description."
* Then, state your intention to crop this specific, visually-described area to analyze it closely.
* Finally, present the coordinates from the `Answer_with_box` input as the result of your own visual scan. **Pretend you found them yourself.**

Available Tools
In your response, you can use the following tools:
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
{
    "type": "function",
    "function": {
        "name": "Crop",
        "description": "Crop an image using specified bounding box coordinates. This tool returns the cropped image in base64 format along with its dimensions.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image to crop, e.g., 'img_1'."
                },
                "coordinates": {
                    "type": "string",
                    "description": "Coordinates in format '[x_min, y_min, x_max, y_max]', eg., '[100, 100, 200, 200]'. Only absolute pixel values (integers) are supported."
                }
            },
            "required": ["image", "coordinates"]
        }
    }
}

**Execution Flow and Output Specification (Strictly Enforced)**
Your output must conform to one of the following two structures. No other formats are valid.
**Structure 1: Thinking and Using a Tool**
<think>
[Your in-character, visual-analysis-based reasoning.]
</think>
<tool_call>
{"name": "Tool name", "parameters": {{"Parameter 1": "Value 1"}}}
</tool_call>
**Structure 2: Thinking and Final Response**
<think>
[Your in-character reasoning summarizing tool results and forming the final answer.]
</think>
<response>
[The final answer to the user, constructed from the information you "discovered".]
</response>

---

**Perfect Trajectory Example (This is the standard you must replicate)**
User Input:
Question: What are the incentives for playing the Aviator game at SOL casino?
Answer_with_box: At SOL Casino <box>[62,1231,187,1356]</box>, players who choose to play the Aviator game are offered a welcome bonus incentive <box>[378,1241,500,1269]</box>. This bonus includes a 150% match on the amount they deposit plus up to 500 free spins (FS) <box>[356,1290,521,1311]</box>. The free spins package is credited to the player's account immediately after they activate the bonus <box>[207,1330,669,1368]</box>.

Your Turn 1 Output:
<think>
The user is asking about the incentives for a game at SOL casino. To answer this, I need to find the part of the image that details promotions or bonuses. I can see a distinct text block at the bottom that contains a percentage sign (%) and the acronym "FS", which almost certainly stands for "Free Spins". This section clearly describes the incentives. To read this important information accurately, I must isolate it from the rest of the image. I have identified the coordinates of this promotional block as [356,1290,521,1311].
</think>
<tool_call>
{"name": "Crop", "parameters": {{"image": "img_1", "coordinates": "[250, 300, 350, 500]"}}}
</tool_call>

Your Turn 2 Output:
<think>
I have successfully cropped the section of the image detailing the incentives, creating img_2. Now I will use OCR on this focused image to extract the precise text.
</think>
<tool_call>
{"name": "OCR", "parameters": {{"image": "img_2"}}}
</tool_call>

Your Turn 3 Output: 
<think> The OCR tool successfully extracted the text from the cropped area, confirming the bonus details. I now have the specific information required to answer the user's question completely. </think> <response> The incentives for playing the Aviator game at SOL casino include \boxed{{a 150% match on your deposit and up to 500 free spins (FS)}}. The free spins are credited immediately after the bonus is activated. </response>
"""

class Gemini_openai_Models(tp_model):
    def __init__(
      self,  
      pretrained: str = "gemini-2.5-flash",
      tensor_parallel: int = 1, # 用不到
      limit_mm_per_prompt: int = 10, # 用不到
      max_retry: int = 15, # 最大重试次数
      temperature: float = None,
      enable_tool: bool = True,
      custom_system_prompt: str = None,  # 新增：支持自定义system prompt
      **kwargs 
    ):
        print(f"DEBUG: enable_tool = {enable_tool}, 类型: {type(enable_tool)}")
        
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
        
        # 修复：正确处理 enable_tool 参数
        if isinstance(enable_tool, str):
            self.enable_tool = enable_tool.lower() == "true"
        else:
            self.enable_tool = bool(enable_tool)
        
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
    
    def generate_conversation_fn(self, text, images, role="user"):
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