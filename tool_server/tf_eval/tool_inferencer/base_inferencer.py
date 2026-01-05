'''
Adapted tool interface
Implemented indexing using img_1
But round should only use one tool
'''

import torch
from torch.utils.data import DataLoader,Dataset
from accelerate import Accelerator
import requests
import re
import copy
import json

from typing import List, Optional, Tuple, Type, TypeVar, Union

from tool_server.utils.debug import remote_breakpoint
from ..models.abstract_model import tp_model
from .dynamic_batch_manager import DynamicBatchManager
from ..utils.utils import *
from ..utils.log_utils import get_logger
from ...tool_workers.tool_manager.base_manager_randomize import ToolManager
import torch.distributed as dist
from dataclasses import asdict
from PIL import Image
import io
import base64


logger = get_logger(__name__)

class BaseToolInferencer(object):
    """
    Base tool inferencer class
    Used to manage the base class for tool interaction during model inference
    """
    def __init__(
        self,
        tp_model: tp_model = None,  # Text-image processing model
        # dataset: Dataset = None,
        batch_size: int = 1,  # Batch size
        model_mode: str = "general",  # Model mode, supports general and llava_plus
        max_rounds: int = 3,  # Maximum conversation rounds
        stop_token: str = "<stop>",  # Stop token
        controller_addr: str = None,  # Controller address
        if_use_tool: bool = True,  # Whether to use tools
        if_randomize_tool: bool = False,
        min_image_size: int = 30,  # Minimum image size
        max_image_size: int = 9000,  # Maximum image size (for ratio limits when resizing images)
        max_ratio = 150,  # Maximum ratio (for ratio limits when resizing images)
        
    ):
        # Initialize accelerator
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.max_ratio = max_ratio
        self.accelerator = Accelerator()
        self.tp_model = tp_model
        self.model_mode = model_mode # Model mode, supports general and llava_plus, but generally just general
        # Get model's conversation generation function and append conversation function
        self.generate_conversation_fn = self.tp_model.generate_conversation_fn
        self.append_conversation_fn = self.tp_model.append_conversation_fn
        
        # If distributed training is enabled and using CUDA but not vllm model, move model to current device and convert to bfloat16 format
        if dist.is_initialized() and self.accelerator.device.type == "cuda" and not 'vllm_models' in str(type(self.tp_model)):
            self.tp_model = self.tp_model.to(self.accelerator.device)
            self.tp_model = self.tp_model.to(torch.bfloat16)

        self.batch_size = batch_size
        self.if_use_tool = if_use_tool
        self.if_randomize_tool = if_randomize_tool
        # When not using tools, set max_rounds to 1 to ensure completion after generating one response
        self.max_rounds = 1 if not if_use_tool else max_rounds
        self.stop_token = stop_token
        self.controller_addr = controller_addr
        # remote_breakpoint()
        
        # Initialize dynamic batch manager
        self.manager = DynamicBatchManager(
            batch_size=self.batch_size, 
            max_rounds=self.max_rounds, 
            stop_token=self.stop_token,
            generate_conversation_fn = self.tp_model.generate_conversation_fn,
            if_use_tool=self.if_use_tool,  # Pass if_use_tool parameter to DynamicBatchManager
        )
        # Initialize tool manager
        self.tool_manager = None
        
        self.image_keys = ["image","base_image","image_to_insert"]
        self.original_image_keys = self.image_keys.copy()
        
    
        
        

    def batch_tool_response_to_next_round_input(self):
        """
        Convert tool responses to next round input
        Process each item in the current batch, adding tool responses to the conversation
        """
        current_batch = self.manager.get_current_batch()
        
        for idx,item in enumerate(current_batch):
            # Skip unprocessed items or items with status not "processing"
            if item.model_response is None or item.status != "processing":
                continue
            
            tool_cfg = item.tool_cfg[item.current_round-1]
            tool_response = item.tool_response[item.current_round-1]
            # Ensure the number of tool configs and responses matches the current round
            assert len(item.tool_cfg) == item.current_round 
            assert len(item.tool_response) == item.current_round 
            original_prompt = item.meta_data.get("text", "")
            
            if tool_response is not None:
                try:
                    # If tool response contains edited image, update current image
                    if "edited_image" in tool_response:
                        edited_image = tool_response.pop("edited_image")
                        # Ensure edited image isn't too small (minimum dimension of 30px)
                        try:
                            pil_edited_image = base64_to_pil(edited_image)
                            
                            # Check dimensions and resize if necessary
                            width, height = pil_edited_image.size
                            resized = False
                            
                            if width < self.min_image_size or height < self.min_image_size:
                                # Calculate new dimensions while preserving aspect ratio
                                if width < height:
                                    new_width = 30
                                    ratio = 30 / width 
                                    ratio = ratio if ratio < self.max_ratio else self.max_ratio
                                    new_height = int(height * ratio)
                                    new_height = new_height if new_height < self.max_image_size else self.max_image_size
                                else:
                                    new_height = 30
                                    ratio = 30 / height
                                    ratio = ratio if ratio < self.max_ratio else self.max_ratio
                                    new_width = int(width * ratio)
                                    new_width = new_width if new_width < self.max_image_size else self.max_image_size
                                
                                # Resize the image
                                pil_edited_image = pil_edited_image.resize((new_width, new_height), Image.LANCZOS)
                                resized = True
                            
                            # If resized, encode back to base64
                            if resized:
                                edited_image = pil_to_base64(pil_edited_image)
                        except Exception as e:
                            logger.warning(f"Failed to resize image: {e}")
                            
                        item.current_image = edited_image

                        
               
                        # Add new edited image to history
                        assert item.image_history is not None, "item.image_history should not be None."
                        img_idx = len(item.image_history) + 1
                        img_key = f"img_{img_idx}"
                        item.image_history[img_key] = edited_image
                        
                        # Process image format based on model mode
                        if self.model_mode == "llava_plus": 
                            edited_image = base64_to_pil(edited_image)
                        if self.model_mode == "general": 
                            edited_image = edited_image

                    else:
                        edited_image = None
                    
                    # Get tool response text
                    # If tool has "edited_image", remove it and keep remaining content as tool_response_text
                    tool_response.pop("tool_reward", None)  
                    if "edited_image" in tool_response:
                        # Remove "edited_image" from tool_response
                        tool_response.pop("edited_image", None)
                        tool_response_text = f"{tool_response} New image generated and saved as: img_{len(item.image_history)}."
                    else:
                        tool_response_text = tool_response
                
                    # Build new response text based on the obtained response
                    new_round_prompt = f"{tool_response_text}\n"
                except:
                    # Exception handling: if error occurs while processing tool response, use original prompt
                    edited_image = None
                    new_round_prompt = original_prompt
            else:
                # If no tool response, use original prompt
                edited_image = None
                new_round_prompt = "Please continue with your response or call a tool."
            
            # Create new round input and add to item
            # Also add image to new_round_input
            new_round_input = dict(text=new_round_prompt,image=edited_image)
            item.new_round_input.append(new_round_input)
            # Add new input to conversation
            item.conversation = self.append_conversation_fn(
                conversation=item.conversation, text=new_round_prompt, image=edited_image, role="user"
            )

    
    def batch_get_tool_response(self):
        """
        Batch get tool responses
        Process each item in the current batch, call corresponding tools to get responses
        """
        current_batch = self.manager.get_current_batch()
        for item in current_batch:
            # Skip unprocessed items or items with status not "processing"
            if item.model_response is None or item.status != "processing":
                continue
            
            tool_cfg = item.tool_cfg[item.current_round-1]
            assert len(item.tool_cfg) == item.current_round

            # Ensure this item has image history
            item_id = item.meta_data["idx"]
            assert item.image_history

            # If tool config exists, call the corresponding tool
            if tool_cfg is not None and len(tool_cfg) > 0:
                assert item.status == "processing"
                try:
                    # Currently only supports one tool
                    assert len(tool_cfg) == 1, "Only one tool is supported for now, but got: {}".format(tool_cfg)

                    # Get API name
                    api_name = tool_cfg[0].get("API_name", tool_cfg[0].get("api_name", ""))

                    # Check if API is in available models list
                    if api_name not in self.available_models:
                        # Log error and add error response
                        logger.error(f"API_name {api_name} not in available models, {self.available_models}")
                        item.tool_response.append(dict(text=f"There is no tool names {api_name}.",status="failed"))
                        continue

                    # Get API parameters
                    api_params = tool_cfg[0].get("api_params", tool_cfg[0].get("API_params", {}))
                    
                    # Process image parameters
                    image_param = None
                    
                    # Check if parameters contain image parameter and if it matches img_n format
                    continue_flag = False
                    for image_key in self.image_keys:
                        
                        if image_key in api_params:
                            img_key = api_params[image_key]
                            image = item.image_history.get(img_key, None)
                            if image is not None:
                                image = load_image(image)
                                image = pil_to_base64(image)
                                # Update image in parameters
                                api_params[image_key] = image
                            else:
                                
                                continue_flag = True
                                
                    if continue_flag:
                        # If requested image not found, log error
                        logger.error(f"Image {img_key} not found in history for item {item_id}")
                        item.tool_response.append(dict(text=f"Image {img_key} not found in history.",status="failed"))
                        continue
                    
                    # Set default parameters and user-provided parameters
                    api_paras = {
                        **api_params,
                    }
                    
                    # Call tool to get response
                    tool_response = self.tool_manager.call_tool(api_name, api_paras)
                    tool_response_clone = copy.deepcopy(tool_response)

                    # Log tool call result
                    if tool_response['status'] == "success":
                        logger.info(f"The {api_name} calls successfully!")
                    else:
                        logger.info(f"The {api_name} calls failed!")
                    
                    # Add tool response to item
                    item.tool_response.append(tool_response_clone)
                    continue
                except Exception as e:
                    # Exception handling: if error occurs when calling tool, add error response
                    logger.info(f"Tool {api_name} failed to answer the question, tool_cfg is {tool_cfg}, error: {str(e)}")
                    item.tool_response.append(dict(text=f"Tool {api_name} failed to answer the question: {str(e)}",status="failed"))
                    continue
            else:
                # If no tool config, add empty response
                item.tool_response.append(None)
                continue

    def extract_tool_call(self, text: str):
        """
        Extract tool call information from <tool_call> tags in model response text
        
        Args:
            text (str): Model response text containing tool_call
            
        Returns:
            Optional[List[Dict]]: Parsed tool call list, returns None if extraction fails
        """
        try:
            # Use regex to find content within <tool_call> tags
            tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
            tool_call_match = re.search(tool_call_pattern, text, re.DOTALL)
            
            if not tool_call_match:
                return None
                
            tool_call_content = tool_call_match.group(1).strip()
            
            # Try to parse entire JSON array
            try:
                # First try to parse entire content as JSON array
                if tool_call_content.startswith('[') and tool_call_content.endswith(']'):
                    json_array = json.loads(tool_call_content)
                    if isinstance(json_array, list):
                        valid_objects = []
                        for obj in json_array:
                            if isinstance(obj, dict) and "name" in obj and "parameters" in obj:
                                valid_objects.append(obj)
                        if valid_objects:
                            return valid_objects
                
                # If not JSON array, try to parse as single JSON object
                if (tool_call_content.startswith('{') and tool_call_content.endswith('}')):
                    json_obj = json.loads(tool_call_content)
                    if "name" in json_obj and "parameters" in json_obj:
                        return [json_obj]
            except json.JSONDecodeError as e:
                pass
            
            # If above methods fail, try to extract single JSON object
            json_objects = []
            # Use regex to match all JSON objects
            json_pattern = r'({[^{}]*(?:{[^{}]*}[^{}]*)*})'
            matches = re.finditer(json_pattern, tool_call_content, re.DOTALL)
            
            for match in matches:
                try:
                    json_obj = json.loads(match.group(1))
                    if isinstance(json_obj, dict) and "name" in json_obj and "parameters" in json_obj:
                        json_objects.append(json_obj)
                except json.JSONDecodeError:
                    continue
            
            if not json_objects:
                return None
                
            return json_objects
            
        except Exception as e:
            logger.error(f"Error extracting tool call: {e}")
            return None
       
    def batch_parse_tool_config(self):
        """
        Batch parse tool configurations
        Extract tool configuration information from model responses
        """
        current_batch = self.manager.get_current_batch()
        for item in current_batch:
            model_response = item.model_response[item.current_round-1]
            assert len(item.model_response) == item.current_round
            
            # Skip unprocessed items or items with status not "processing"
            if model_response is None or item.status != "processing":
                continue
            
            try:
                # Parse tool config based on model mode
                if self.model_mode == "general":
                    # Add debug information
                    
                    # Extract tool call information
                    tool_calls = self.extract_tool_call(model_response)
                    
                    if tool_calls is not None and len(tool_calls) > 0:
                        # Only use the first tool call, one operation per time step
                        tool_call = tool_calls[0]
                        # Ensure tool_call contains name and parameters fields
                        assert 'name' in tool_call and 'parameters' in tool_call, "missing 'name' or 'parameters' in the parsed tool_call."
                        
                        # Build tool configuration
                        tool_name = tool_call['name']
                        tool_params = tool_call['parameters']
                        
                        # Build general tool configuration
                        tool_cfg = [{'API_name': tool_name,
                                    'API_params': tool_params}]
                    else:
                        # If no tool call extracted, set tool config to None
                        tool_cfg = None
            except Exception as e:
                # Exception handling: if error occurs while parsing tool config, log error and set tool config to None
                logger.info(f"Failed to parse tool config: {e}.")
                tool_cfg = None
                
            # Add tool configuration to data item
            item.tool_cfg.append(tool_cfg)
            
    def pop_qualified_items(self):
        """
        Pop qualified items
        Return items that have completed processing and remove them from current batch
        Also clean up corresponding image_history
        """
        res = []
        new_batch = []
        removed_item_ids = []
        
        for idx, item in enumerate(self.manager.get_current_batch()):
            if item.status == "finished":
                image_history = item.image_history
                item_dict = asdict(item)
                item_dict = remove_pil_objects(item_dict)
                item_dict = remove_non_serializable(item_dict)
                item_id = item_dict["meta_data"].get("idx", str(id(item)))
                
                final_model_output = item_dict["model_response"][-1]
                final_answer = self.manager.extract_final_answer(final_model_output, task_name=self.dataset.task_name)
                item_dict["final_answer"] = final_answer
                item_dict["image_history"] = image_history
                item_dict.pop("current_image", None) 
                
                # Record item_id to be removed
                removed_item_ids.append(item_id)
                
                res.append(item_dict)
            else:
                new_batch.append(item)
        
        
        self.manager.dynamic_batch = new_batch
        return res
    
    def batch_inference(self, dataset):
        """
        Batch inference function
        Process all items in the dataset, execute model inference and tool calls
        
        Args:
            dataset: Dataset to process
        """
        self.dataset = dataset
        # Create data loader with batch size of 1, 2 worker threads, using collate_fn to ensure single data item is returned each time
        self.dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            num_workers=2, 
            collate_fn=lambda x: x[0]  # Ensure one data item is returned each time
        )
        
        # If distributed training is enabled and not using vLLM model, prepare data loader with accelerator
        if dist.is_initialized() and not 'vllm_models' in str(type(self.tp_model)):
            self.dataloader = self.accelerator.prepare(self.dataloader)
            
        # Convert data loader to iterator and set model to evaluation mode
        self.dataloader_iter = iter(self.dataloader)
        self.tp_model.eval()
        # Create progress bar
        progress_bar = tqdm_rank0(len(self.dataloader), desc="Model Responding")

        # If data loader is empty and not using vLLM model, wait for all processes to complete and return
        if len(self.dataloader) == 0 and not 'vllm_models' in str(type(self.tp_model)):
            self.accelerator.wait_for_everyone()
            return
            
        # Add data items from data loader to manager and show progress with progress bar
        self.manager.append_item_to_full(self.dataloader_iter, progress_bar=progress_bar)

        # Get current batch and generate responses using model
        current_batch = self.manager.get_current_batch()
        self.tp_model.generate(current_batch) # Batch responses obtained
        # Update status in manager
        self.manager.update_item_status()
        
        # Main loop: process all batches
        while len(current_batch) > 0:
            try:
                # Pop all items that have completed processing
                results = self.pop_qualified_items()
                # Store results in dataset
                for res in results:
                    idx = res["meta_data"]["idx"]
                    self.dataset.store_results(dict(idx=idx,results=res))

                # If not using tools, directly process next batch of data
                if not self.if_use_tool:
                    # Refill current batch
                    self.manager.append_item_to_full(self.dataloader_iter, progress_bar=progress_bar)
                    
                    # Get updated current batch and generate new responses
                    current_batch = self.manager.get_current_batch()
                    if len(current_batch) > 0:
                        self.tp_model.generate(current_batch)
                        # Update status
                        self.manager.update_item_status()
                    continue
                
                # Below is the workflow when using tools
                # Parse tool configuration
                self.batch_parse_tool_config()
                # Get tool responses
                self.batch_get_tool_response()
                # Convert tool responses to next round input
                self.batch_tool_response_to_next_round_input()
                
                # Refill current batch
                self.manager.append_item_to_full(self.dataloader_iter, progress_bar=progress_bar)
                
                # Get updated current batch and generate new responses
                current_batch = self.manager.get_current_batch()
                if len(current_batch) > 0:
                    self.tp_model.generate(current_batch)
                    # Update status, should update current_batch until completion
                    self.manager.update_item_status()

            except StopIteration:
                # Exit loop when iterator is exhausted
                break
                
        # Ensure all items have been processed
        assert len(self.manager.get_current_batch()) == 0
        # If not using vLLM model, wait for all processes to complete
        if not 'vllm_models' in str(type(self.tp_model)):
            self.accelerator.wait_for_everyone()
    
    def set_tool_selection(self, tool_selection: Union[List, str, None]) -> None:
        if isinstance(tool_selection, List):
            self.tool_selection = tool_selection
        elif isinstance(tool_selection, str):
            self.tool_selection = tool_selection.split(",")
        # elif tool_selection is None:
        #     self.tool_selection = None
        else:
            raise ValueError("tool_selection should be a dictionary or a string.")
        self.tool_manager = ToolManager(controller_url_location=self.controller_addr, tools=self.tool_selection, randomize=self.if_randomize_tool)
        
        self.available_models = self.tool_manager.available_tools
        
        self.system_prompt = self.tool_manager.get_tool_prompt(prompt_type="one_tool_call")
        self.tp_model.set_system_prompt(self.system_prompt)
        
        if self.if_randomize_tool:
            self.original_to_randomized = self.tool_manager.original_to_randomized
            self.image_keys = [self.original_to_randomized.get(k, k) for k in self.original_image_keys]
            self.available_models = [self.original_to_randomized.get(k, k) for k in self.available_models]
        else:
            self.original_to_randomized = None
