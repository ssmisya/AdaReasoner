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
        images, 
        role = "user",
        **kwargs
    ):
        raise NotImplementedError
    
    def append_conversation_fn(
        self, 
        conversation, 
        text, 
        image, 
        role,
        **kwargs
    ):
        raise  NotImplementedError
    
    def generate(
        self,
        batch: List[DynamicBatchItem],
    ):
        raise NotImplementedError
    
    def getitem_fn(self, meta_data, idx):
        item = meta_data[idx]
        images = []

        if "images" in item:
            # 处理 "images" 列表
            raw_images = item["images"]
            for i, img_data in enumerate(raw_images):
                img = img_data
                
                # [关键修复]：检查是否为 bytes，如果是则转换
                if isinstance(img, bytes):
                    try:
                        img = PILImage.open(io.BytesIO(img))
                    except Exception as e:
                        # 如果转换失败，抛出更详细的错误
                        raise ValueError(f"Failed to open image {i} from bytes for idx {idx}. Error: {e}")
                
                # 断言检查
                assert isinstance(img, PILImage.Image), f"Each image in 'images' must be a PIL Image, but got {type(img)} for image {i}, idx {idx}"
                images.append(img)

        elif "image" in item:
            # 处理单个 "image"
            image = item["image"]
            
            # [关键修复]：检查是否为 bytes，如果是则转换
            if isinstance(image, bytes):
                try:
                    image = PILImage.open(io.BytesIO(image))
                except Exception as e:
                    raise ValueError(f"Failed to open image from bytes for idx {idx}. Error: {e}")

            # 断言检查
            assert isinstance(image, PILImage.Image), f"'image' must be a PIL Image, but got {type(image)} for idx {idx}"
            images = [image]

        elif "image_path" in item:
            # 处理 "image_path"
            try:
                image = PILImage.open(item["image_path"])
                images = [image]
            except Exception as e:
                raise ValueError(f"Failed to open image from path: {item['image_path']} for idx {idx}. Error: {e}")
        
        else:
            raise ValueError(f"Item (idx {idx}) must contain 'image', 'images', or 'image_path' key.")
        
        text = item["text"]
        item_idx = item["idx"]
        res = dict(images=images, text=text, idx=item_idx)
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