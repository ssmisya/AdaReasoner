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
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
inferencer_id = str(uuid.uuid4())[:6]  # Generate unique inferencer ID
logger = get_logger("vllm_models",)    # Get logger

class VllmModels(tp_model):
    """
    VLLM model class for handling multimodal (text+image) input for large language model inference
    Inherits from the abstract model class tp_model
    """
    def __init__(
      self,  
      pretrained : str = None,        # Pretrained model path
      tensor_parallel: str = "1",     # Number of tensor parallel
      limit_mm_per_prompt: str = "1",  # Limit on number of images per single message, not the total in conversation history
      enable_tool: bool = True,          # Whether to enable tools
    ):
        tensor_parallel = eval(tensor_parallel)  # Convert string to numeric value
        self.model = LLM(
            model=pretrained,                   # Model path
            tensor_parallel_size=tensor_parallel,  # Parallel size
            limit_mm_per_prompt={"image": int(limit_mm_per_prompt)}  # Limit number of images per prompt
        )
        
        if enable_tool.lower() == "true":
            self.enable_tool = True
        else:
            self.enable_tool = False
        
        self.system_prompt = None

    def generate_conversation_fn(
        self,
        text,                # Input text
        image,               # Input image
        role = "user",       # Conversation role, default is user
    ):  
        """
        Function to generate conversation format, combines text and image into message format accepted by the model
        
        Parameters:
            text: User input text
            image: User provided image
            role: Conversation role, default is "user"
            
        Returns:
            messages: Formatted conversation message list
        """
        text = "Question: " + text  # Add "Question:" prefix to the text

        assert self.system_prompt, "System prompt must be set before generating conversation."  # Ensure system prompt is set

        image = pil_to_base64(image)  # Convert PIL image to base64 encoding
        messages = [
            {
                "role": "system",  # System role
                "content": [
                    {
                        "type": "text",
                        "text": self.system_prompt,  # Evaluation prompt defined in prompt.py
                    },
                ],
            },
            {
                "role": "user",    # User role
                "content": [
                    {
                        "type": "text",
                        "text": text,  # User input text
                    },
                    {
                        "type": "image_url",  # Image URL type
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"  # Provide image in base64 format
                        },
                    },
                ],
            }
        ]

        return messages
    
    
    def append_conversation_fn(
        self, 
        conversation,  # Existing conversation history
        text,          # Text to add
        image,         # Image to add (optional)
        role           # Conversation role
    ):
        """
        Append new message to existing conversation
        
        Parameters:
            conversation: Existing conversation history
            text: Text content to add
            image: Image to add (if any)
            role: Speaking role ("user" or "assistant")
            
        Returns:
            Updated conversation history
        """
        if image:  # If image is provided
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
        else:  # Only text, no image
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
        
        conversation.extend(new_messages)  # Add new messages to conversation history

        return conversation
    
    
    def form_input_from_dynamic_batch(self, batch: List[DynamicBatchItem]):
        """
        Extract conversation content from dynamic batch items
        
        Parameters:
            batch: List of dynamic batch items
            
        Returns:
            messages: Extracted message list for model input
        """
        if len(batch) == 0:  # If batch is empty
            return None
        messages = []
        for item in batch:
            messages.append(item.conversation)  # Collect conversation from each batch item
        return messages
    
    def generate(self, batch):
        """
        Generate responses in batch
        
        Parameters:
            batch: Batch containing multiple conversations
            
        Functionality:
            Generate model responses for each conversation in the batch and update conversation history
        """
        if not batch or len(batch) == 0:  # If batch is empty
            return
        max_new_tokens = self.generation_config.get("max_new_tokens", 2048)  # Get maximum token generation count, default 2048
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.6)  # Set sampling parameters
        
        inputs = self.form_input_from_dynamic_batch(batch)  # Extract inputs from batch
        # breakpoint()  # Debug breakpoint
        response = self.model.chat(inputs, sampling_params)  # Call model for parallel response generation, VLLM library's parallel inference


        for item, output_item in zip(batch, response):
            output_text = output_item.outputs[0].text  # Get generated text
            item.model_response.append(output_text)  # Add response to model response list
            self.append_conversation_fn(
                item.conversation, output_text, None, "assistant"  # 将模型回复添加到对话历史
            )
            
    def to(self, *args, **kwargs):
        return self
    
    def eval(self):
        return self
            
        
        
