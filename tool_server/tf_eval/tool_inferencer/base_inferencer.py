'''
适配了工具接口
实现了使用img_1进行索引
但是应该round只能使用一个工具
'''

import torch
from torch.utils.data import DataLoader,Dataset
from accelerate import Accelerator
import requests
import re
import copy
import json


from ..models.abstract_model import tp_model
from .dynamic_batch_manager import DynamicBatchManager
from ..utils.utils import *
from ..utils.log_utils import get_logger
from ...tool_workers.tool_manager.base_manager import ToolManager
import torch.distributed as dist
from dataclasses import asdict

logger = get_logger(__name__)

class BaseToolInferencer(object):
    """
    基础工具推理器类
    用于管理模型推理过程中与工具交互的基础类
    """
    def __init__(
        self,
        tp_model: tp_model = None,  # 文本-图像处理模型
        # dataset: Dataset = None,
        batch_size: int = 1,  # 批处理大小
        model_mode: str = "general",  # 模型模式，支持general和llava_plus
        max_rounds: int = 3,  # 最大对话轮数
        stop_token: str = "<stop>",  # 停止标记
        controller_addr: str = None,  # 控制器地址
        if_use_tool: bool = True,  # 是否使用工具
    ):
        # 初始化加速器
        self.accelerator = Accelerator()
        self.tp_model = tp_model
        self.model_mode = model_mode # 模型模式，支持general和llava_plus，但是一般就是general
        # 获取模型的对话生成函数和追加对话函数
        self.generate_conversation_fn = self.tp_model.generate_conversation_fn
        self.append_conversation_fn = self.tp_model.append_conversation_fn

        # 如果启用分布式训练且使用CUDA但不是vllm模型，则将模型移至当前设备并转换为bfloat16格式
        if dist.is_initialized() and self.accelerator.device.type == "cuda" and not 'vllm_models' in str(type(self.tp_model)):
            self.tp_model = self.tp_model.to(self.accelerator.device)
            self.tp_model = self.tp_model.to(torch.bfloat16)

        self.batch_size = batch_size
        self.if_use_tool = if_use_tool
        print(f"初始化后的self.if_use_tool: {self.if_use_tool}, 类型: {type(self.if_use_tool)}")
        # 不使用工具时，将max_rounds设置为1，以确保在生成一次响应后完成
        self.max_rounds = 1 if not if_use_tool else max_rounds
        self.stop_token = stop_token
        self.controlller_addr = controller_addr
        
        # 初始化动态批处理管理器
        self.manager = DynamicBatchManager(
            batch_size=self.batch_size, 
            max_rounds=self.max_rounds, 
            stop_token=self.stop_token,
            generate_conversation_fn = self.tp_model.generate_conversation_fn,
            if_use_tool=self.if_use_tool,  # 将 if_use_tool 参数传递给 DynamicBatchManager
        )
        # 初始化工具管理器
        self.tool_manager = ToolManager(controller_url_location=self.controlller_addr)
        # 获取可用工具列表
        # 我把offline工具给去除了
        self.available_models = self.tool_manager.available_tools
        
        # 初始化图像历史字典，用于存储每个项目的图像历史
        self.image_history = {}

    def batch_tool_response_to_next_round_input(self):
        """
        将工具响应转换为下一轮输入
        处理当前批次中的每个项目，将工具的响应添加到对话中
        """
        current_batch = self.manager.get_current_batch()
        
        for idx,item in enumerate(current_batch):
            # 跳过未处理或状态不是processing的项目
            if item.model_response is None or item.status != "processing":
                continue
            
            tool_cfg = item.tool_cfg[item.current_round-1]
            tool_response = item.tool_response[item.current_round-1]
            # 确保工具配置和响应的数量与当前轮数一致
            assert len(item.tool_cfg) == item.current_round 
            assert len(item.tool_response) == item.current_round 
            original_prompt = item.meta_data.get("text", "")
            
            if tool_response is not None:
                try:
                    # 如果工具响应包含编辑后的图像，则更新当前图像
                    if "edited_image" in tool_response:
                        edited_image = tool_response.pop("edited_image")
                        item.current_image = edited_image

                        # 确保该项目有图像历史记录
                        item_id = item.meta_data.get("idx", str(id(item)))
                        if item_id not in self.image_history:
                            # 初始化图像历史，原始图像为img_1
                            self.image_history[item_id] = {
                                "img_1": item.meta_data.get("image", None)
                            }
                        
                        # 添加新的编辑后图像到历史
                        img_idx = len(self.image_history[item_id]) + 1
                        img_key = f"img_{img_idx}"
                        self.image_history[item_id][img_key] = edited_image
                        
                        # 根据模型模式处理图像格式
                        if self.model_mode == "llava_plus": 
                            edited_image = base64_to_pil(edited_image)
                        if self.model_mode == "general": 
                            edited_image = edited_image

                    else:
                        edited_image = None
                    
                    # 获取工具响应文本
                    # 如果工具中有"edited_image"，则去除，保留剩余的内容为tool_response_text
                    if "edited_image" in tool_response:
                        # 将tool_response中的"edited_image"删除
                        tool_response.pop("edited_image", None)
                        tool_response_text = tool_response
                    else:
                        tool_response_text = tool_response
                    
                    # 获取API名称
                    api_name = tool_cfg[0].get("API_name", tool_cfg[0].get("api_name", ""))

                    # 根据得到的响应构建新的响应文本
                    new_response = f"OBSERVATION:\n{api_name} tool outputs: {tool_response_text}\n"
                    new_round_prompt = f"{new_response}Please summarize the tool outputs and answer my first question."
                except:
                    # 异常处理：如果处理工具响应时出错，使用原始提示
                    edited_image = None
                    new_round_prompt = original_prompt
            else:
                # 如果没有工具响应，使用原始提示
                edited_image = None
                new_round_prompt = original_prompt
            
            # 创建新轮次输入并添加到项目中
            # 将图片也添加到new_round_input中
            new_round_input = dict(text=new_round_prompt,image=edited_image)
            item.new_round_input.append(new_round_input)
            # 将新输入添加到对话中
            item.conversation = self.append_conversation_fn(
                conversation=item.conversation, text=new_round_prompt, image=edited_image, role="user"
            )

    
    def batch_get_tool_response(self):
        """
        批量获取工具响应
        处理当前批次中的每个项目，调用相应的工具获取响应
        """
        current_batch = self.manager.get_current_batch()
        for item in current_batch:
            # 跳过未处理或状态不是processing的项目
            if item.model_response is None or item.status != "processing":
                continue
            
            tool_cfg = item.tool_cfg[item.current_round-1]
            assert len(item.tool_cfg) == item.current_round

            # 确保该项目有图像历史记录
            item_id = item.meta_data.get("idx", str(id(item)))
            if item_id not in self.image_history:
                # 初始化图像历史，原始图像为img_1
                self.image_history[item_id] = {
                    "img_1": item.meta_data.get("image", None)
                }
                # 如果有current_image，也将其添加到历史
                if item.current_image is not None and "img_2" not in self.image_history[item_id]:
                    self.image_history[item_id]["img_2"] = item.current_image

            # 如果存在工具配置，调用相应的工具
            if tool_cfg is not None and len(tool_cfg) > 0:
                assert item.status == "processing"
                try:
                    # 目前只支持一个工具
                    assert len(tool_cfg) == 1, "Only one tool is supported for now, but got: {}".format(tool_cfg)

                    # 获取API名称
                    api_name = tool_cfg[0].get("API_name", tool_cfg[0].get("api_name", ""))

                    # 检查API是否在可用模型列表中
                    if api_name not in self.available_models:
                        # 记录错误并添加错误响应
                        logger.error(f"API_name {api_name} not in available models, {self.available_models}")
                        item.tool_response.append(dict(text=f"There is no tool names {api_name}.",status="failed"))
                        continue

                    # 获取API参数
                    api_params = tool_cfg[0].get("api_params", tool_cfg[0].get("API_params", {}))
                    
                    # 处理图像参数
                    image_param = None
                    
                    # 检查参数中是否包含image参数，并且是否符合img_n格式
                    if "image" in api_params and isinstance(api_params["image"], str) and api_params["image"].startswith("img_"):
                        img_key = api_params["image"]
                        # 从图像历史中获取对应图像
                        if img_key in self.image_history[item_id]:
                            image = self.image_history[item_id][img_key]
                            # 如果是需要图像的工具，确保图像格式正确
                            if api_name in ["Point","SegmentRegionAroundPoint","Crop","GroundingDINO","DrawLine","OCR","GetSubplotInfo","GetBarInfo", "DrawShape", "HighlightBox", "MaskBox", "LanguageModel"]:
                                if image is not None:
                                    image = load_image(image)
                                    image = pil_to_base64(image)
                                    # 更新参数中的图像
                                    api_params["image"] = image
                        else:
                            # 如果找不到请求的图像，记录错误
                            logger.error(f"Image {img_key} not found in history for item {item_id}")
                            item.tool_response.append(dict(text=f"Image {img_key} not found in history.",status="failed"))
                            continue
                    # 如果没有指定图像或不是img_n格式，使用当前图像或元数据中的图像
                    elif api_name in ["Point","SegmentRegionAroundPoint","Crop","GroundingDINO","DrawLine","OCR","GetSubplotInfo","GetBarInfo", "DrawShape", "HighlightBox", "MaskBox", "LanguageModel"]:
                        # 确定当前使用的图像：优先使用当前图像，否则使用元数据中的图像
                        if item.current_image is not None:
                            image = item.current_image
                        else:
                            image = item.meta_data.get("image", None)
                        
                        if image is not None:
                            image = load_image(image)
                            image = pil_to_base64(image)
                            # 设置图像参数
                            api_params["image"] = image
                        else:
                            # 如果找不到图像，记录错误
                            logger.error(f"No image available for tool {api_name}")
                            item.tool_response.append(dict(text=f"No image available for tool {api_name}.",status="failed"))
                            continue
                    
                    # 设置默认参数和用户提供的参数
                    api_paras = {
                        "box_threshold": 0.3,
                        "text_threshold": 0.25,
                        **api_params,
                    }
                    
                    # 调用工具获取响应
                    tool_response = self.tool_manager.call_tool(api_name, api_paras)
                    tool_response_clone = copy.deepcopy(tool_response)

                    # 记录工具调用结果
                    if tool_response['status'] == "success":
                        logger.info(f"The {api_name} calls successfully!")
                    else:
                        logger.info(f"The {api_name} calls failed!")
                    
                    # 将工具响应添加到项目中
                    item.tool_response.append(tool_response_clone)
                    continue
                except Exception as e:
                    # 异常处理：如果调用工具时出错，添加错误响应
                    logger.info(f"Tool {api_name} failed to answer the question, tool_cfg is {tool_cfg}, error: {str(e)}")
                    item.tool_response.append(dict(text=f"Tool {api_name} failed to answer the question: {str(e)}",status="failed"))
                    continue
            else:
                # 如果没有工具配置，添加空响应
                item.tool_response.append(None)
                continue

    def extract_tool_call(self, text: str):
        """
        从模型响应文本中提取<tool_call>标签内的工具调用信息
        
        参数:
            text (str): 包含tool_call的模型响应文本
            
        返回:
            Optional[List[Dict]]: 解析后的工具调用列表，如果提取失败则返回None
        """
        try:
            # 使用正则表达式查找<tool_call>标签内的内容
            tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
            tool_call_match = re.search(tool_call_pattern, text, re.DOTALL)
            
            if not tool_call_match:
                return None
                
            tool_call_content = tool_call_match.group(1).strip()
            
            # 尝试解析整个JSON数组
            try:
                # 首先尝试解析整个内容为JSON数组
                if tool_call_content.startswith('[') and tool_call_content.endswith(']'):
                    json_array = json.loads(tool_call_content)
                    if isinstance(json_array, list):
                        valid_objects = []
                        for obj in json_array:
                            if isinstance(obj, dict) and "name" in obj and "parameters" in obj:
                                valid_objects.append(obj)
                        if valid_objects:
                            return valid_objects
                
                # 如果不是JSON数组，尝试解析为单个JSON对象
                if (tool_call_content.startswith('{') and tool_call_content.endswith('}')):
                    json_obj = json.loads(tool_call_content)
                    if "name" in json_obj and "parameters" in json_obj:
                        return [json_obj]
            except json.JSONDecodeError as e:
                pass
            
            # 如果上述方法失败，尝试提取单个JSON对象
            json_objects = []
            # 使用正则表达式匹配所有JSON对象
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
        批量解析工具配置
        从模型响应中提取工具配置信息
        """
        current_batch = self.manager.get_current_batch()
        for item in current_batch:
            model_response = item.model_response[item.current_round-1]
            assert len(item.model_response) == item.current_round
            
            # 跳过未处理或状态不是processing的项目
            if model_response is None or item.status != "processing":
                continue
            
            try:
                # 根据模型模式解析工具配置
                if self.model_mode == "general":
                    # 添加调试信息
                    
                    # 提取工具调用信息
                    tool_calls = self.extract_tool_call(model_response)
                    
                    if tool_calls is not None and len(tool_calls) > 0:
                        # 只使用第一个工具调用，每个时间步只做一个操作
                        tool_call = tool_calls[0]
                        # 确保tool_call包含name和parameters字段
                        assert 'name' in tool_call and 'parameters' in tool_call, "missing 'name' or 'parameters' in the parsed tool_call."
                        
                        # 构建工具配置
                        tool_name = tool_call['name']
                        tool_params = tool_call['parameters']
                        
                        # 构建通用工具配置
                        tool_cfg = [{'API_name': tool_name,
                                    'API_params': tool_params}]
                    else:
                        # 如果没有提取到工具调用，设置工具配置为None
                        tool_cfg = None
            except Exception as e:
                # 异常处理：如果解析工具配置时出错，记录错误并设置工具配置为None
                logger.info(f"Failed to parse tool config: {e}.")
                tool_cfg = None
                
            # 将工具配置添加到数据项中
            item.tool_cfg.append(tool_cfg)
            
    def pop_qualified_items(self):
        """
        弹出符合条件的项目
        返回已完成处理的项目，并从当前批次中移除它们
        同时清理对应的image_history
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
                
                # 记录要移除的item_id
                item_id = item_dict["meta_data"].get("idx", str(id(item)))
                removed_item_ids.append(item_id)
                
                res.append(item_dict)
            else:
                new_batch.append(item)
        
        # 清理已完成项目的image_history
        for item_id in removed_item_ids:
            if item_id in self.image_history:
                del self.image_history[item_id]
                # logger.info(f"Cleaned up image history for item {item_id}")
        
        self.manager.dynamic_batch = new_batch
        return res
    
    def batch_inference(self, dataset):
        """
        批量推理函数
        处理数据集中的所有项目，执行模型推理和工具调用
        
        参数:
            dataset: 要处理的数据集
        """
        self.dataset = dataset
        # 创建数据加载器，批大小为1，工作线程数为2，使用collate_fn确保每次返回单个数据项
        self.dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            num_workers=2, 
            collate_fn=lambda x: x[0]  # 确保每次返回一个数据项
        )
        
        # 如果启用分布式训练且不使用vLLM模型，则使用accelerator准备数据加载器
        if dist.is_initialized() and not 'vllm_models' in str(type(self.tp_model)):
            self.dataloader = self.accelerator.prepare(self.dataloader)
            
        # 将数据加载器转换为迭代器并设置模型为评估模式
        self.dataloader_iter = iter(self.dataloader)
        self.tp_model.eval()

        # 创建进度条
        progress_bar = tqdm_rank0(len(self.dataloader), desc="Model Responding")

        # 如果数据加载器为空且不使用vLLM模型，则等待所有进程完成，并返回
        if len(self.dataloader) == 0 and not 'vllm_models' in str(type(self.tp_model)):
            self.accelerator.wait_for_everyone()
            return
            
        # 将数据加载器中的数据项添加到管理器中，并使用进度条显示进度
        self.manager.append_item_to_full(self.dataloader_iter, progress_bar=progress_bar)

        # 获取当前批次并使用模型生成响应
        current_batch = self.manager.get_current_batch()
        self.tp_model.generate(current_batch) # 获得的是批量响应
        # 更新管理器中的状态
        self.manager.update_item_status()
        
        # 主循环：处理所有批次
        while len(current_batch) > 0:
            try:
                # 弹出所有已完成处理的项目
                results = self.pop_qualified_items()
                # 将结果存储到数据集中
                for res in results:
                    idx = res["meta_data"]["idx"]
                    self.dataset.store_results(dict(idx=idx,results=res))

                # 如果不使用工具，直接处理下一批数据
                if not self.if_use_tool:
                    # 重新填充当前批次
                    self.manager.append_item_to_full(self.dataloader_iter, progress_bar=progress_bar)
                    
                    # 获取更新后的当前批次并生成新的响应
                    current_batch = self.manager.get_current_batch()
                    if len(current_batch) > 0:
                        self.tp_model.generate(current_batch)
                        # 更新状态
                        self.manager.update_item_status()
                    continue
                
                # 以下是使用工具的流程
                # 解析工具配置
                self.batch_parse_tool_config()
                # 获取工具响应
                self.batch_get_tool_response()
                # 将工具响应转换为下一轮输入
                self.batch_tool_response_to_next_round_input()
                
                # 重新填充当前批次
                self.manager.append_item_to_full(self.dataloader_iter, progress_bar=progress_bar)
                
                # 获取更新后的当前批次并生成新的响应
                current_batch = self.manager.get_current_batch()
                if len(current_batch) > 0:
                    self.tp_model.generate(current_batch)
                    # 更新状态，应该会更新current_batch，直到结束
                    self.manager.update_item_status()

            except StopIteration:
                # 当迭代器耗尽时退出循环
                break
                
        # 确保所有项目都已处理完毕
        assert len(self.manager.get_current_batch()) == 0
        # 如果不使用vLLM模型，等待所有进程完成
        if not 'vllm_models' in str(type(self.tp_model)):
            self.accelerator.wait_for_everyone()
    