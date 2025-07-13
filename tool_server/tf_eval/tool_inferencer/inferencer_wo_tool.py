'''
Inference for models that cannot call tools
--Similar to lmms-eval
'''

import torch
from torch.utils.data import DataLoader,Dataset
from accelerate import Accelerator
import requests
import re
import copy
import json


from tool_server.utils.debug import remote_breakpoint
from ..models.abstract_model import tp_model
from .dynamic_batch_manager import DynamicBatchManager
from ..utils.utils import *
from ..utils.log_utils import get_logger
from ...tool_workers.tool_manager.base_manager import ToolManager
import torch.distributed as dist
from dataclasses import asdict

logger = get_logger(__name__)

class BaseInferencer(object):
    """
    Inferencer class that handles batch inference without using tools.
    """
    def __init__(
        self,
        tp_model: tp_model = None,  
        batch_size: int = 1, 
        if_use_tool: bool = False,  
        **kwargs
    ):
        # Initialize accelerator
        self.accelerator = Accelerator()
        self.tp_model = tp_model
        # Get model's conversation generation function and append conversation function
        self.generate_conversation_fn = self.tp_model.generate_conversation_fn
        self.append_conversation_fn = self.tp_model.append_conversation_fn

        # If distributed training is enabled and using CUDA but not vllm model, move model to current device and convert to bfloat16
        if dist.is_initialized() and self.accelerator.device.type == "cuda" and not 'vllm_models' in str(type(self.tp_model)):
            self.tp_model = self.tp_model.to(self.accelerator.device)
            self.tp_model = self.tp_model.to(torch.bfloat16)

        self.batch_size = batch_size
        self.if_use_tool = if_use_tool
        assert self.if_use_tool == False, "BaseInferencer is designed for models that do not use tools. Please use BaseToolInferencer for models that use tools."
        print(f"Initialized self.if_use_tool: {self.if_use_tool}, Type: {type(self.if_use_tool)}")

        
        # Initialize dynamic batch manager
        self.manager = DynamicBatchManager(
            batch_size=self.batch_size, 
            max_rounds=1, 
            stop_token=None,
            generate_conversation_fn = self.tp_model.generate_conversation_fn,
            if_use_tool = False,  # Pass if_use_tool parameter to DynamicBatchManager
        )
  
            
    def pop_qualified_items(self):
        """
        Pop qualified items
        Return completed items and remove them from the current batch
        Also clean up corresponding image_history
        """
        res = []
        new_batch = []
        removed_item_ids = []
        
        for idx, item in enumerate(self.manager.get_current_batch()):
            if item.status == "finished":
                item_dict = asdict(item)
                item_dict = remove_pil_objects(item_dict)
                item_dict = remove_non_serializable(item_dict)
                
                final_model_output = item_dict["model_response"][-1]
                final_answer = self.manager.extract_final_answer(final_model_output)
                item_dict["final_answer"] = final_answer
                
                # Record item_id to be removed
                item_id = item_dict["meta_data"].get("idx", str(id(item)))
                removed_item_ids.append(item_id)
                
                res.append(item_dict)
            else:
                new_batch.append(item)
        
        
        self.manager.dynamic_batch = new_batch
        return res
    
    def batch_inference(self, dataset):
        """
        Batch inference function - simplified version without tool usage
        Process all items in the dataset, perform single-round model inference and return results
        
        Parameters:
            dataset: Dataset to be processed
        """
        self.dataset = dataset
        # Create data loader with batch size 1, 2 worker threads, using collate_fn to ensure single data item is returned each time
        self.dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            num_workers=2, 
            collate_fn=lambda x: x[0]  # Ensure a single data item is returned each time
        )
        
        # If distributed training is enabled and not using vLLM model, prepare the data loader with accelerator
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
            
        # Add items from data loader to manager and display progress with progress bar
        self.manager.append_item_to_full(self.dataloader_iter, progress_bar=progress_bar)

        # Main loop: process all batches
        while True:
            # Get current batch
            current_batch = self.manager.get_current_batch()
            
            # If no more batches, exit the loop
            if len(current_batch) == 0:
                break
                
            # Generate single-round response using the model
            self.tp_model.generate(current_batch)
            # Update status in the manager
            self.manager.update_item_status()
            
            # Pop all completed items
            results = self.pop_qualified_items()
            # Store results in the dataset
            for res in results:
                idx = res["meta_data"]["idx"]
                self.dataset.store_results(dict(idx=idx, results=res))
                
            # Fill in the next batch of data
            try:
                self.manager.append_item_to_full(self.dataloader_iter, progress_bar=progress_bar)
            except StopIteration:
                # Continue processing remaining batches when the iterator is exhausted
                pass
        
        # Ensure all items have been processed
        assert len(self.manager.get_current_batch()) == 0
        # If not using vLLM model, wait for all processes to complete
        if not 'vllm_models' in str(type(self.tp_model)):
            self.accelerator.wait_for_everyone()
