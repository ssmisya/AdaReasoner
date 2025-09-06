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
        print(f"DEBUG: load_data_function返回数据长度: {len(self.full_data)}")
        if len(self.full_data) > 0:
            print(f"DEBUG: 第一个数据项keys: {self.full_data[0].keys()}")
            print(f"DEBUG: 第一个数据项idx: {self.full_data[0].get('idx', 'N/A')}")
        self.meta_data = deepcopy(self.full_data)
        print(f"DEBUG: meta_data长度: {len(self.meta_data)}")
        self.load_ckpt_path = None
        self.save_ckpt_path = None
        self.middle_images_save_dir = None
        self.tool_selection_dict = None
        
        if task_args.resume_from_ckpt and self.task_name in self.task_args.resume_from_ckpt:
            self.load_ckpt_path = self.task_args.resume_from_ckpt[self.task_name]
            print(f"DEBUG: 开始从checkpoint恢复，路径: {self.load_ckpt_path}")
            self.resume_from_ckpt(self.load_ckpt_path)
            print(f"DEBUG: checkpoint恢复完成，剩余meta_data长度: {len(self.meta_data)}")
        
        if task_args.save_to_ckpt and self.task_name in self.task_args.save_to_ckpt:
            logger.info(f"save to ckpt path: {self.task_args.save_to_ckpt}")
            self.save_ckpt_path = self.task_args.save_to_ckpt[self.task_name]
        #     logger.info(f"设置检查点保存路径: {self.save_ckpt_path}")
        # else:
        #     logger.info(f"未设置检查点保存路径 - task_args.save_to_ckpt: {getattr(task_args, 'save_to_ckpt', None)}, task_name: {self.task_name}")
        #     self.save_ckpt_path = None
        
        # 添加调试打印
        print(f"DEBUG: task_args.middle_images_save_dir = {task_args.middle_images_save_dir}")
        print(f"DEBUG: type(task_args.middle_images_save_dir) = {type(task_args.middle_images_save_dir)}")
        print(f"DEBUG: self.task_name = {self.task_name}")
        print(f"DEBUG: task_args.middle_images_save_dir 是否为真值: {bool(task_args.middle_images_save_dir)}")
        if task_args.middle_images_save_dir:
            print(f"DEBUG: self.task_name in task_args.middle_images_save_dir = {self.task_name in task_args.middle_images_save_dir}")
            # 打印键列表而不是完整内容
            if hasattr(task_args.middle_images_save_dir, 'keys'):
                print(f"DEBUG: task_args.middle_images_save_dir 的键: {list(task_args.middle_images_save_dir.keys())}")
        
        if task_args.middle_images_save_dir and self.task_name in self.task_args.middle_images_save_dir:
            logger.info(f"middle images save dir: {self.task_args.middle_images_save_dir}")
            self.middle_images_save_dir = self.task_args.middle_images_save_dir[self.task_name]
            print(f"DEBUG: 设置 middle_images_save_dir 为: {self.middle_images_save_dir}")
            if not os.path.exists(self.middle_images_save_dir):
                os.makedirs(self.middle_images_save_dir, exist_ok=True)
                print(f"DEBUG: 创建目录: {self.middle_images_save_dir}")
            else:
                print(f"DEBUG: 目录已存在: {self.middle_images_save_dir}")
        else:
            print(f"DEBUG: middle_images_save_dir 条件判断失败")
        
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
        print(f"DEBUG: store_results 被调用，result keys: {result.keys()}")
        image_history = result["results"].pop("image_history", {})
        
        # 新增：在结果中添加原始数据的答案信息
        result_idx = result["idx"]
        # 从 full_data 中找到对应的原始数据
        original_data = None
        for item in self.full_data:
            if item.get("idx") == result_idx:
                original_data = item
                break
        
        if original_data:
            # 将原始数据中的重要信息添加到结果中
            result["results"]["original_question"] = original_data.get("text", "")
            result["results"]["ground_truth_answer"] = original_data.get("answer", "")
            # 如果还有其他有用的字段，也可以添加
            if "question" in original_data:
                result["results"]["original_question"] = original_data["question"]
            # print(f"DEBUG: 添加原始答案到结果中: {original_data.get('answer', 'N/A')}")
        else:
            print(f"DEBUG: 未找到 idx={result_idx} 的原始数据")
        
        # 修改：只打印图像历史的键和类型信息，不打印完整内容
        if image_history:
            image_info = {}
            for key, value in image_history.items():
                if isinstance(value, str) and len(value) > 100:
                    image_info[key] = f"base64_string(长度:{len(value)})"
                elif isinstance(value, bytes):
                    image_info[key] = f"bytes(长度:{len(value)})"
                else:
                    image_info[key] = f"{type(value).__name__}({str(value)[:50]}...)"
            print(f"DEBUG: image_history 信息: {image_info}")
        else:
            print(f"DEBUG: image_history: 空")
        
        print(f"DEBUG: self.middle_images_save_dir: {self.middle_images_save_dir}")
        self.results.append(result)
        
        if self.middle_images_save_dir:
            print(f"DEBUG: 进入图像保存逻辑")
            item_id  = result["idx"]
            sub_save_dir = os.path.join(self.middle_images_save_dir, str(item_id))
            print(f"DEBUG: 子目录路径: {sub_save_dir}")
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir, exist_ok=True)
                print(f"DEBUG: 创建子目录: {sub_save_dir}")
                
            for image_key, image_value in image_history.items():
                print(f"DEBUG: 保存图像 {image_key} (类型: {type(image_value).__name__})")
                try:
                    # 改进：针对不同类型的图像数据使用不同的处理方式
                    if isinstance(image_value, bytes):
                        # 对于 bytes 类型，直接使用 PIL 打开
                        from io import BytesIO
                        image = Image.open(BytesIO(image_value))
                        print(f"DEBUG: 成功从 bytes 加载图像 {image_key}, 尺寸: {image.size}")
                    elif isinstance(image_value, str):
                        # 对于字符串类型（base64），使用现有的 load_image 函数
                        image = load_image(image_value)
                        print(f"DEBUG: 成功从字符串加载图像 {image_key}, 尺寸: {image.size}")
                    elif isinstance(image_value, Image.Image):
                        # 如果已经是 PIL Image，直接使用
                        image = image_value
                        print(f"DEBUG: 直接使用 PIL 图像 {image_key}, 尺寸: {image.size}")
                    else:
                        # 其他类型，尝试使用 load_image 函数
                        image = load_image(image_value)
                        print(f"DEBUG: 使用 load_image 加载图像 {image_key}, 尺寸: {image.size}")
                    
                    # 确保是 PIL Image
                    assert isinstance(image, Image.Image), f"Image value for {image_key} should be a PIL Image, but got {type(image)}"
                    
                    # 如果图像是 RGBA 模式，转换为 RGB
                    if image.mode in ("RGBA", "LA", "P"):
                        image = image.convert("RGB")
                        print(f"DEBUG: 将图像 {image_key} 从 {image.mode} 转换为 RGB")
                    
                    save_path = os.path.join(sub_save_dir, f"{image_key}.png")
                    image.save(save_path)
                    print(f"DEBUG: 图像已保存到: {save_path}")
                    
                except Exception as e:
                    print(f"DEBUG: 保存图像 {image_key} 失败: {e}")
                    # 打印更详细的错误信息
                    import traceback
                    traceback.print_exc()
        else:
            print(f"DEBUG: self.middle_images_save_dir 为空，跳过图像保存")
            
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
            print(f"DEBUG: checkpoint文件中有{len(ckpt_data)}条记录")
            self.processed_id = {}
            
            for ckpt_item in ckpt_data:
                # assert ckpt_item["task_name"] == self.task_name, f"ckpt task name {ckpt_item['task']} not match with current task name {self.task_name}"
                assert ckpt_item["model_name"] == self.model_name, f"ckpt model name {ckpt_item['model_name']} not match with current model name {self.model_name}"
                if "results" in ckpt_item and isinstance(ckpt_item["results"],dict): 
                    # and ("validity" not in ckpt_item["results"] or ckpt_item["results"]["validity"] == True):
                    self.results.append(ckpt_item["results"])
                    self.processed_id[ckpt_item["results"]["idx"]] = 1
                    
            print(f"DEBUG: 从checkpoint中恢复了{len(self.results)}个结果")
            print(f"DEBUG: processed_id keys sample: {list(self.processed_id.keys())[:5]}")
            original_meta_data_len = len(self.meta_data)
            self.meta_data = [item for item in self.meta_data if item["idx"] not in self.processed_id]
            print(f"DEBUG: 过滤前meta_data长度: {original_meta_data_len}, 过滤后长度: {len(self.meta_data)}")
            logger.info(f"Total items: {len(self.full_data)}, processed items: {len(self.results)}, remaining items: {len(self.meta_data)}")
        else:
            logger.info(f"ckpt path {ckpt_path} not found")
            print(f"DEBUG: checkpoint文件不存在: {ckpt_path}")
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

        # 填充张量并生成注意力掩码
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

