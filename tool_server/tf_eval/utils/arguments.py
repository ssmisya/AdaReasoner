import os

import json
import copy
import random
import logging
import argparse
import numpy as np
from PIL import Image
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Sequence

import torch
from torch.utils.data import Dataset

import transformers
from transformers import TrainerCallback
from transformers import HfArgumentParser, TrainingArguments
from box import Box

from .utils import *

@dataclass
class ModelArguments:
    model: Optional[str] = field(default="qwen2vl")
    model_args: Optional[str] = field(default="pretrained=/mnt/petrelfs/share_data/quxiaoye/models/Qwen2-VL-72B-Instruct")
    model_mode: Optional[str] = field(default="opensource")
    batch_size: Optional[int] = field(default=1)
    stop_token: Optional[str] = field(default=None)
    max_rounds: Optional[int] = field(default=3)

@dataclass
class TaskArguments:
    task_name: Optional[str] = field(default="charxiv")
    
    resume_from_ckpt: Optional[Dict[str, str]] = field(default=None,)
    save_to_ckpt: Optional[Dict[str, str]] = field(default=None,)
    # 修改了定义，不然会报错
    middle_images_save_dir: Optional[Dict[str, str]] = field(default=None)
    tool_selection: Optional[str] = field(default=None,)
    tool_selection_dict: Optional[str] = field(default=None)
                            
    def __post_init__(self):
        """初始化后处理所有需要转换为Box的字段"""
        print(f"DEBUG: TaskArguments __post_init__ 开始")
        print(f"DEBUG: 原始 middle_images_save_dir = {self.middle_images_save_dir}")
        print(f"DEBUG: 原始 middle_images_save_dir 类型 = {type(self.middle_images_save_dir)}")
        
        box_fields = [
            "resume_from_ckpt", 
            "save_to_ckpt", 
            "middle_images_save_dir", 
            "tool_selection_dict"
        ]
        
        for field_name in box_fields:
            field_value = getattr(self, field_name)
            # 避免打印过长的内容
            if isinstance(field_value, dict) and len(str(field_value)) > 200:
                print(f"DEBUG: 处理字段 {field_name}, 值: <长字典内容>, 类型: {type(field_value)}")
            else:
                print(f"DEBUG: 处理字段 {field_name}, 值: {field_value}, 类型: {type(field_value)}")
            
            if field_value is None:
                setattr(self, field_name, Box())
                print(f"DEBUG: {field_name} 设置为空 Box()")
            elif isinstance(field_value, dict):
                setattr(self, field_name, Box(field_value))
                print(f"DEBUG: {field_name} 转换为 Box，键: {list(field_value.keys())}")
            else:
                print(f"DEBUG: {field_name} 不是字典，抛出错误")
                raise ValueError(f"{field_name} should be a dictionary.")
        
        print(f"DEBUG: 最终 middle_images_save_dir = {self.middle_images_save_dir}")
        print(f"DEBUG: 最终 middle_images_save_dir 类型 = {type(self.middle_images_save_dir)}")
        if hasattr(self.middle_images_save_dir, 'keys'):
            print(f"DEBUG: 最终 middle_images_save_dir 的键: {list(self.middle_images_save_dir.keys())}")


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the Evaluation script.
    """
    config: Optional[str] = field(default=None)
    verbosity: Optional[str] = field(default="INFO")
    wandb_args: Optional[str] = field(default="project=mr_eval,entity=mr_eval")
    output_path: Optional[str] = field(default="output")
    controller_addr: Optional[str] = field(default=None)
    if_use_tool: Optional[bool] = field(default=True)
    if_randomize_tool: Optional[bool] = field(default=False)

def parse_str_into_dict(args_str: str) -> Dict:
    """
    Parse a string of comma-separated key-value pairs into a dictionary.
    """
    args_dict = {}
    for arg in args_str.split(","):
        key, value = arg.split("=")
        args_dict[key] = value
    return args_dict

def parse_str_into_list(args_str: str) -> List:
    """
    Parse a string of comma-separated values into a list.
    """
    # import pdb; pdb.set_trace()
    return args_str.split(",")

def parse_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TaskArguments, ScriptArguments))
    # breakpoint()
    model_args, task_args, script_args = parser.parse_args_into_dataclasses()
    
    if script_args.config:
        if script_args.config.endswith(".json"):
            config = load_json_file(script_args.config)
        elif script_args.config.endswith(".yaml"):
            config = load_yaml_file(script_args.config)
        else:
            raise ValueError("Config file should be either a json or yaml file.")
        
        print(f"DEBUG: 配置文件已加载")
        task_config = config.get('task_args', {})
        print(f"DEBUG: task_args 部分键: {list(task_config.keys())}")
        print(f"DEBUG: middle_images_save_dir 在配置中: {task_config.get('middle_images_save_dir', 'NOT_FOUND')}")
        
        if isinstance(config, dict):
            model_args = ModelArguments(**config["model_args"])
            task_args = TaskArguments(**config["task_args"])
            script_args = ScriptArguments(**config["script_args"])
            print(f"DEBUG: 创建 TaskArguments 后，middle_images_save_dir = {task_args.middle_images_save_dir}")
        elif isinstance(config, list):
            model_args = ModelArguments(**config[0]["model_args"])
            task_args = TaskArguments(**config[0]["task_args"])
            script_args = ScriptArguments(**config[0]["script_args"])
            print(f"DEBUG: 创建 TaskArguments 后（列表模式），middle_images_save_dir = {task_args.middle_images_save_dir}")
        else:
            raise ValueError("Config file should be either a dict or list of dicts.")
    else:
        config = None
        
    # import pdb; pdb.set_trace() 
    script_args.config = config
    task_args.task_name = parse_str_into_list(task_args.task_name)
    if isinstance(model_args.model_args, str):
        model_args.model_args = parse_str_into_dict(model_args.model_args)
    if isinstance(script_args.wandb_args, str):
        script_args.wandb_args = parse_str_into_dict(script_args.wandb_args)
    
    print(f"DEBUG: 最终返回的 task_args.middle_images_save_dir = {task_args.middle_images_save_dir}")
    if hasattr(task_args.middle_images_save_dir, 'keys'):
        print(f"DEBUG: 最终返回的键: {list(task_args.middle_images_save_dir.keys())}")
    return dict(model_args=model_args, task_args=task_args, script_args=script_args)
    
