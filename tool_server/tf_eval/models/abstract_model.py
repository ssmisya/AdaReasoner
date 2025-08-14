import abc
from typing import List, Optional, Tuple, Type, TypeVar, Union


from functools import partial

from ...utils.utils import *
from ...utils.server_utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem
from PIL import Image as PILImage
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from ...utils.prompts import tool_planning_model_prompt_no_tool_call

import uuid

from ..utils.log_utils import get_logger

inferencer_id = str(uuid.uuid4())[:6]
logger = get_logger("abstract_model",)

R = partial(round, ndigits=2)
T = TypeVar("T", bound="tp_model")


class tp_model(abc.ABC):
    def __init__(
        self,
        enable_tool: bool = True,
    ):
        self.enable_tool = enable_tool
        pass
    
    def to(self, *args, **kwargs):
        # import pdb; pdb.set_trace()
        self.model = self.model.to(*args, **kwargs)
        return self
    
    def eval(self):
        self.model = self.model.eval()
    
    def generate_conversation_fn(
        self,
        text,
        image, 
        role = "user",
    ):
        raise NotImplementedError
    
    def append_conversation_fn(
        self, 
        conversation, 
        text, 
        image, 
        role
    ):
        raise  NotImplementedError
    
    def generate(
        self,
        batch: List[DynamicBatchItem],
    ):
        raise NotImplementedError
    
    def getitem_fn(self, meta_data, idx):
        item = meta_data[idx]
        if "image" in item:
            image = item["image"]
        elif "image_path" in item:
            image = PILImage.open(item["image_path"])
        else:
            raise ValueError("Item must contain 'image' or 'image_path' key.")
        
        text = item["text"]
        item_idx = item["idx"]
        res = dict(image=image, text=text, idx=item_idx)
        return res
    
    def set_generation_config(self, generation_configs: dict = None) -> None:
        if generation_configs is not None:
            self.generation_config = generation_configs
        else:
            self.generation_config = {}
            
    def set_enable_tool(self, enable_tool: bool = True) -> None:
        """
        Set whether the model should use tools.
        
        Parameters:
            enable_tool (bool): Whether to enable tool usage.
        """
        self.enable_tool = enable_tool
        if self.enable_tool:
            logger.info("Tool usage is enabled.")
        else:
            logger.info("Tool usage is disabled.")
            
    def set_system_prompt(self, system_prompt: str = None) -> None:
        self.system_prompt = system_prompt