# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import re
import json
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    The dataset should be pre-randomized with randomized_to_original field included
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        general_config: DictConfig = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        
        self.general_config = general_config
        
        # 新增：从配置中读取是否使用预随机化的数据
        self.use_pre_randomized = config.get("use_pre_randomized", True)
        
        # 如果不使用预随机化数据，则需要在运行时随机化（保留原有逻辑）
        if not self.use_pre_randomized and self.general_config is not None:
            self.randomize = self.general_config.actor_rollout_ref.rollout.agent.tool_manager.randomize
            if self.randomize:
                from tool_server.tool_workers.tool_manager.base_manager_randomize import ToolManager
                self.controller_addr = self.general_config.actor_rollout_ref.rollout.agent.tool_manager.controller_addr
                self.tools = self.general_config.actor_rollout_ref.rollout.agent.tool_manager.tools
                print("⚠️  使用运行时随机化模式（不推荐，会影响性能）")
        else:
            self.randomize = False
            print("✓ 使用预随机化数据模式（推荐）")

        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()
        
        # 验证数据集是否包含必要的字段
        if self.use_pre_randomized:
            self._verify_pre_randomized_data()

    def _verify_pre_randomized_data(self):
        """验证数据集是否包含预随机化所需的字段"""
        if len(self.dataframe) == 0:
            print("⚠️  数据集为空，跳过验证")
            return
        
        first_item = self.dataframe[0]
        
        # 检查是否包含 randomized_to_original 字段
        if 'randomized_to_original' not in first_item:
            print("❌ 错误：数据集缺少 'randomized_to_original' 字段")
            print("   请先使用 randomize_tool_prompt.py 脚本预处理数据")
            raise ValueError("Dataset missing 'randomized_to_original' field. Please pre-process the data first.")
        
        # 检查字段是否有效
        randomized_to_original = first_item['randomized_to_original']
        
        if randomized_to_original is None or (isinstance(randomized_to_original, dict) and len(randomized_to_original) == 0):
            print("⚠️  警告：'randomized_to_original' 字段为空，可能数据未正确预处理")
        else:
            print(f"✓ 数据集验证通过，包含 {len(randomized_to_original)} 个工具映射")
            if isinstance(randomized_to_original, dict):
                print(f"  示例映射: {list(randomized_to_original.items())[:3]}")

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")
        
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))
                <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")
        


    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
            if self.use_pre_randomized:
                self._verify_pre_randomized_data()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict, tool_manager=None):
        """
        构建消息列表
        
        注意：如果使用预随机化数据，则直接使用数据中的prompt，不需要再次随机化
        如果不使用预随机化数据，则需要在运行时随机化
        """
        messages: list = example.pop(self.prompt_key)

        # 仅在不使用预随机化数据且需要随机化时才进行运行时随机化
        if not self.use_pre_randomized and self.randomize:
            assert tool_manager is not None, "运行时随机化需要 tool_manager"
            new_system_prompt = tool_manager.get_tool_prompt()
            messages[0]["content"] = new_system_prompt
        # 如果使用预随机化数据，messages 中的 system prompt 已经是随机化后的，无需修改
        
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        
        关键变化：
        1. 如果使用预随机化数据，直接从数据中读取 randomized_to_original
        2. 如果不使用预随机化数据，则在运行时创建 ToolManager 并生成映射
        """
        row_dict: dict = self.dataframe[item]
        
        # 根据配置决定如何获取 randomized_to_original
        if self.use_pre_randomized:
            # 从预处理的数据中直接读取
            randomized_to_original = row_dict.get('randomized_to_original', None)
            tool_manager = None
            randomized_to_original = json.loads(randomized_to_original) if isinstance(randomized_to_original, str) else randomized_to_original
            # 验证数据完整性
            assert randomized_to_original is not None, f"Record {item} missing randomized_to_original field."

        else:
            # 运行时随机化（原有逻辑）
            if self.randomize:
                from tool_server.tool_workers.tool_manager.base_manager_randomize import ToolManager
                tool_manager = ToolManager(
                    controller_url_location=self.controller_addr,
                    tools=self.tools,
                    randomize=self.randomize,
                )
                randomized_to_original = tool_manager.randomized_to_original
            else:
                tool_manager = None
                randomized_to_original = None
            
        row_dict["randomized_to_original"] = randomized_to_original
        messages = self._build_messages(row_dict, tool_manager)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_raw_image, process_video
            
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}
            origin_multi_modal_data = {}

            images = None
            if self.image_key in row_dict:
                origin_images = [process_raw_image(image) for image in row_dict.get(self.image_key)]
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images
                origin_multi_modal_data["image"] = origin_images

            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict['origin_multi_modal_data'] = origin_multi_modal_data
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()