import torch
import argparse
import datetime
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoTokenizer
from copy import deepcopy
from torch.utils.data import DataLoader,Dataset
from dataclasses import dataclass
from typing import Dict, Sequence
from accelerate import PartialState  
from tqdm import tqdm
import os
import torch.distributed as dist

from ...utils.utils import gather_dict_lists, append_jsonl, process_jsonl, is_vllm_environment
from ...utils.log_utils import get_logger

from tool_server.utils.utils import *

logger = get_logger(__name__)



class BaseEvalDataset(Dataset):
    '''
    The API dataset class
    Initialize: load_data_function is provided by task
                evaluate_function is provided by task
                getitem_function is provided by model
    data storage: self.meta_data
    results storage: self.results
    '''
    def __init__(
        self,
        load_data_function,
        getitem_function=None,
        evaluate_function=None, 
        task_config=None,
        task_args=None,
        model_args=None,
    ) -> None:
        self.load_data_function = load_data_function
        self.getitem_function = getitem_function
        self.evaluate_function = evaluate_function
        self.task_config = task_config
        self.task_args = task_args
        self.model_args = model_args
        self.task_name = task_config["task_name"]  
        self.model_name = model_args.model 
        
        if self.task_config and "generation_config" in self.task_config:
            self.set_gen_kwargs(self.task_config["generation_config"])
        else:
            self.set_gen_kwargs({})
        
        self.results = []
        
        self.full_data = self.load_data_function()
        logger.debug(f"DEBUG: load_data_function returned data length: {len(self.full_data)}")
        if len(self.full_data) > 0:
            logger.debug(f"DEBUG: First data item keys: {self.full_data[0].keys()}")
            logger.debug(f"DEBUG: First data item idx: {self.full_data[0].get('idx', 'N/A')}")
        self.meta_data = deepcopy(self.full_data)
        logger.debug(f"DEBUG: meta_data length: {len(self.meta_data)}")
        self.load_ckpt_path = None
        self.save_ckpt_path = None
        self.middle_images_save_dir = None
        self.tool_selection_dict = None
        
        if task_args.resume_from_ckpt and self.task_name in self.task_args.resume_from_ckpt:
            self.load_ckpt_path = self.task_args.resume_from_ckpt[self.task_name]
            logger.debug(f"DEBUG: Starting to resume from checkpoint, path: {self.load_ckpt_path}")
            self.resume_from_ckpt(self.load_ckpt_path)
            logger.debug(f"DEBUG: Checkpoint resume completed, remaining meta_data length: {len(self.meta_data)}")

        
        if task_args.save_to_ckpt and self.task_name in self.task_args.save_to_ckpt:
            logger.info(f"save to ckpt path: {self.task_args.save_to_ckpt}")
            self.save_ckpt_path = self.task_args.save_to_ckpt[self.task_name]
            os.makedirs(os.path.dirname(self.save_ckpt_path), exist_ok=True)
        
        # Add debug prints
        logger.debug(f"DEBUG: task_args.middle_images_save_dir = {task_args.middle_images_save_dir}")
        logger.debug(f"DEBUG: type(task_args.middle_images_save_dir) = {type(task_args.middle_images_save_dir)}")
        logger.debug(f"DEBUG: self.task_name = {self.task_name}")
        logger.debug(f"DEBUG: task_args.middle_images_save_dir is truthy: {bool(task_args.middle_images_save_dir)}")
        if task_args.middle_images_save_dir:
            logger.debug(f"DEBUG: self.task_name in task_args.middle_images_save_dir = {self.task_name in task_args.middle_images_save_dir}")
            # Print key list instead of full content
            if hasattr(task_args.middle_images_save_dir, 'keys'):
                logger.debug(f"DEBUG: task_args.middle_images_save_dir keys: {list(task_args.middle_images_save_dir.keys())}")
        
        if task_args.middle_images_save_dir and self.task_name in self.task_args.middle_images_save_dir:
            logger.info(f"middle images save dir: {self.task_args.middle_images_save_dir}")
            self.middle_images_save_dir = self.task_args.middle_images_save_dir[self.task_name]
            logger.debug(f"DEBUG: Set middle_images_save_dir to: {self.middle_images_save_dir}")
            if not os.path.exists(self.middle_images_save_dir):
                os.makedirs(self.middle_images_save_dir, exist_ok=True)
                logger.debug(f"DEBUG: Created directory: {self.middle_images_save_dir}")
            else:
                logger.debug(f"DEBUG: Directory already exists: {self.middle_images_save_dir}")
        else:
            logger.debug(f"DEBUG: middle_images_save_dir condition check failed")
        
        if task_args.tool_selection_dict and self.task_name in self.task_args.tool_selection_dict:
            logger.info(f"tool selection dict: {self.task_args.tool_selection_dict}")
            self.tool_selection = self.task_args.tool_selection_dict[self.task_name]
        elif self.task_args.tool_selection:
            self.tool_selection = task_args.tool_selection
        else:
            self.tool_selection = None
        
        if isinstance(self.tool_selection, str):
            self.tool_selection = self.tool_selection.split(",")
                
        
        if self.model_name in ["qwen_qwq"]:
            logger.info("Generation model detected, setting padding side to left")
            self.padding_side = "left"
        else:
            self.padding_side = "right"
            
        if dist.is_available() and dist.is_initialized() and not 'vllm' in self.model_name:
            dist.barrier()
            
    
    def __getitem__(self, index):
        return self.getitem_function(self.meta_data,index)
    
    def __len__(self):
        return len(self.meta_data)
    
    def store_results(self,result):
        logger.debug(f"DEBUG: store_results called, result keys: {result.keys()}")
        image_history = result["results"].pop("image_history", {})
        
        # Add original data's answer information to results
        result_idx = result["idx"]
        # Find corresponding original data from full_data
        original_data = None
        for item in self.full_data:
            if item.get("idx") == result_idx:
                original_data = item
                break
        
        if original_data:
            # Add important information from original data to results
            result["results"]["original_question"] = original_data.get("text", "")
            result["results"]["ground_truth_answer"] = original_data.get("answer", "")
            # Add other useful fields if available
            if "question" in original_data:
                result["results"]["original_question"] = original_data["question"]
        else:
            logger.debug(f"DEBUG: Original data not found for idx={result_idx}")
        
        # Modified: Only print image history keys and type info, not full content
        if image_history:
            image_info = {}
            for key, value in image_history.items():
                if isinstance(value, str) and len(value) > 100:
                    image_info[key] = f"base64_string(length:{len(value)})"
                elif isinstance(value, bytes):
                    image_info[key] = f"bytes(length:{len(value)})"
                else:
                    image_info[key] = f"{type(value).__name__}({str(value)[:50]}...)"
            logger.debug(f"DEBUG: image_history info: {image_info}")
        else:
            logger.debug(f"DEBUG: image_history: empty")
        
        logger.debug(f"DEBUG: self.middle_images_save_dir: {self.middle_images_save_dir}")
        self.results.append(result)
        
        if self.middle_images_save_dir:
            logger.debug(f"DEBUG: Entering image save logic")
            item_id  = result["idx"]
            sub_save_dir = os.path.join(self.middle_images_save_dir, str(item_id))
            logger.debug(f"DEBUG: Subdirectory path: {sub_save_dir}")
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir, exist_ok=True)
                logger.debug(f"DEBUG: Created subdirectory: {sub_save_dir}")
                
            for image_key, image_value in image_history.items():
                logger.debug(f"DEBUG: Saving image {image_key} (type: {type(image_value).__name__})")
                try:
                    # Improved: Use different processing methods for different image data types
                    if isinstance(image_value, bytes):
                        # For bytes type, open directly with PIL
                        from io import BytesIO
                        image = Image.open(BytesIO(image_value))
                        logger.debug(f"DEBUG: Successfully loaded image {image_key} from bytes, size: {image.size}")
                    elif isinstance(image_value, str):
                        # For string type (base64), use existing load_image function
                        image = load_image(image_value)
                        logger.debug(f"DEBUG: Successfully loaded image {image_key} from string, size: {image.size}")
                    elif isinstance(image_value, Image.Image):
                        # If already a PIL Image, use directly
                        image = image_value
                        logger.debug(f"DEBUG: Using PIL image {image_key} directly, size: {image.size}")
                    else:
                        # For other types, try using load_image function
                        image = load_image(image_value)
                        logger.debug(f"DEBUG: Loaded image {image_key} using load_image, size: {image.size}")
                    
                    # Ensure it's a PIL Image
                    assert isinstance(image, Image.Image), f"Image value for {image_key} should be a PIL Image, but got {type(image)}"
                    
                    # If image is RGBA mode, convert to RGB
                    if image.mode in ("RGBA", "LA", "P"):
                        image = image.convert("RGB")
                        logger.debug(f"DEBUG: Converted image {image_key} from {image.mode} to RGB")
                    
                    save_path = os.path.join(sub_save_dir, f"{image_key}.png")
                    image.save(save_path)
                    logger.debug(f"DEBUG: Image saved to: {save_path}")
                    
                except Exception as e:
                    logger.debug(f"DEBUG: Failed to save image {image_key}: {e}")
                    # Print more detailed error information
                    import traceback
                    traceback.print_exc()
        else:
            print(f"DEBUG: self.middle_images_save_dir is empty, skipping image save")
            
        if self.save_ckpt_path:
            self.save_item_into_ckpt_file(result)
        
    def fetch_results(self):
        return self.results
    
    def evaluate(self):
        self.collect_results_from_multi_process()
        res = self.evaluate_function(results=self.results, meta_data=self.full_data)
        res["task_name"] = self.task_name
        res["model_name"] = self.model_name
        return res
    
    def save_result_item_into_log(self,result_item, save_path):
        append_jsonl(result_item, save_path)
        
    def resume_from_ckpt(self,ckpt_path):
        if os.path.exists(ckpt_path):
            logger.info(f"loading results from {ckpt_path}")
            ckpt_data = process_jsonl(ckpt_path)
            logger.debug(f"DEBUG: Checkpoint file contains {len(ckpt_data)} records")
            self.processed_id = {}
            
            for ckpt_item in ckpt_data:
                # assert ckpt_item["task_name"] == self.task_name, f"ckpt task name {ckpt_item['task']} not match with current task name {self.task_name}"
                assert ckpt_item["model_name"] == self.model_name, f"ckpt model name {ckpt_item['model_name']} not match with current model name {self.model_name}"
                if "results" in ckpt_item and isinstance(ckpt_item["results"],dict): 
                    # and ("validity" not in ckpt_item["results"] or ckpt_item["results"]["validity"] == True):
                    self.results.append(ckpt_item["results"])
                    self.processed_id[ckpt_item["results"]["idx"]] = 1
                    
            logger.debug(f"DEBUG: Recovered {len(self.results)} results from checkpoint")
            logger.debug(f"DEBUG: processed_id keys sample: {list(self.processed_id.keys())[:5]}")
            original_meta_data_len = len(self.meta_data)
            self.meta_data = [item for item in self.meta_data if item["idx"] not in self.processed_id]
            logger.debug(f"DEBUG: meta_data length before filtering: {original_meta_data_len}, after filtering: {len(self.meta_data)}")
            logger.info(f"Total items: {len(self.full_data)}, processed items: {len(self.results)}, remaining items: {len(self.meta_data)}")
        else:
            logger.info(f"ckpt path {ckpt_path} not found")
            logger.debug(f"DEBUG: Checkpoint file does not exist: {ckpt_path}")
            pass

    
    def save_item_into_ckpt_file(self,result_item):
        write_item = dict(task_name=self.task_name,model_name=self.model_name,results=result_item)
        append_jsonl(write_item, self.save_ckpt_path)
        
        
    def collect_results_from_multi_process(self):
        # breakpoint()
        if dist.is_available() and dist.is_initialized() and not 'vllm' in self.model_name:
            dist.barrier()
        self.results = gather_dict_lists(self.results)
        results_dict = {}
        renewed_results = []
        for item in self.results:
            if item["idx"] not in results_dict:
                results_dict[item["idx"]] = 1
                renewed_results.append(item)
        self.results = renewed_results
    
    def set_gen_kwargs(self, config):
        if isinstance(config, dict):
            self.gen_kwargs = config
        else:
            self.gen_kwargs = {}



@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for Evaluation."""
    tokenizer: AutoTokenizer
    max_length: int = 512
    padding_side: str = "left"

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        if "model_type" in instances[0] and instances[0]["model_type"] == "openai":
            assert len(instances) == 1
            return instances[0]
        
        idx = [instance["idx"] for instance in instances]
        input_ids = [instance["input_ids"] for instance in instances]
        
        assert isinstance(input_ids[0], torch.Tensor) 
        for input_id in input_ids:
            while input_id.ndim > 1:
                input_id = input_id[0]
                
        lengths  = [input_id.shape[0] for input_id in input_ids]
        max_length = max(lengths)
        max_length = min(max_length, self.max_length)
        pad_token_id = self.tokenizer.pad_token_id
        
        padded_batch = torch.zeros((len(input_ids), max_length), dtype=torch.long)
        attention_mask = torch.zeros((len(input_ids), max_length), dtype=torch.long)

        # Fill tensors and generate attention masks
        for i, input_id in enumerate(input_ids):
            assert isinstance(input_id, torch.Tensor)
            if input_id.shape[0] > max_length:
                input_id = input_id[:max_length]
            length = min(input_id.shape[0], max_length)
            if self.padding_side == "right":
                padded_batch[i, :length] = input_id
                if length < max_length:
                    padded_batch[i, length:] = pad_token_id
                attention_mask[i, :length] = 1
            elif self.padding_side == "left":
                padded_batch[i, -length:] = input_id
                if length < max_length:
                    padded_batch[i, :-length] = pad_token_id
                attention_mask[i, -length:] = 1

        batch = dict(
            idx=idx,
            input_ids=padded_batch,
            attention_mask=attention_mask,
        )
        
        instance0 = instances[0]
        other_keys = [key for key in instance0.keys() if key not in ["idx","input_ids"]]
        if len(other_keys) > 0:
            for key in other_keys:
                values = [instance[key] for instance in instances]
                batch[key] = values
        return batch
