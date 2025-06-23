import re
import json
import torch
import signal
import time
import fcntl
import os

from vllm import LLM, SamplingParams
from typing import List, Union, Dict
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from PIL import Image
from contextlib import contextmanager
from copy import deepcopy

from ....utils.utils import pil_to_base64, base64_to_pil, append_jsonl
from ....utils.tool_utils import parse_tool_config, handle_tool_result, append_conversation_fn

class TimeoutException(Exception): pass

def _timeout_handler(signum, frame):
    raise TimeoutException("chat() timed out.")
signal.signal(signal.SIGALRM, _timeout_handler)




class VllmToolInferencer(object):
    def __init__(
        self,
        vllm_model: LLM = None,
        model_name: str = None,
        model_configs: dict = None,
        tool_controller_addr: str = None,
    ):
        # Init VLLM model
        if vllm_model is not None:
            self.vllm_model = vllm_model
        elif model_name is not None:
            if model_configs is not None:
                self.vllm_model = LLM(model=model_name, **model_configs)
            else:
                self.vllm_model = LLM(model=model_name)
        else:
            raise ValueError("Either vllm_model or model_name must be provided.")
        
        # Init tool manager
        tool_manager = ToolManager(tool_controller_addr)
        tool_manager.available_tools = [tool for tool in tool_manager.available_tools if tool not in ['crop', 'drawline']]
        tool_controller_addr_display = tool_controller_addr if tool_controller_addr else "Auto"
        print(f"controller_addr: {tool_controller_addr_display}")
        print(f"Avaliable tools are {tool_manager.available_tools}")
        
        self.tool_manager = tool_manager
    
    def get_repr_of_conversation(self, conversation):
        conversation_str = ""
        for message in conversation:
            role = message["role"]
            # content = [item for item in message["content"] if isinstance(item, str) or item["type"] == "text"]
            content = message["content"]
            c_res = ""
            if isinstance(content, str):
                c_res = content
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type","") == "text":
                        c_res += str(c["text"])
            else:
                raise ValueError(f"Unknown content type: {type(content)}")
            conversation_str += f"{role}: {c_res}\n"
        return conversation_str.strip()
        
    def log_item_into_file(self, item, tool_log_file):
        conversations = item["conversations"]
        conversations_str = self.get_repr_of_conversation(conversations)
        append_jsonl(conversations_str, tool_log_file)

    def inference(
        self,
        inputs: List[Dict],
        sampling_params: SamplingParams = None,
        max_rounds: int = 5,
        do_sample: bool = True,
        sample_num: int = 1,
        vllm_retry_num: int = 3,
        vllm_retry_interval: int = 5,
        vllm_timeout_sec: int = 500,
        tool_retry_num: int = 3,
        tool_retry_interval: int = 5,
        tool_timeout_sec: int = 60,
        tool_log_file: str = None,
    ) -> str:
        
        # Set the sampling parameters
        if not do_sample:
            kwargs = {
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "detokenize": True,
            }
        else:
            kwargs = {
                "n": 1,
                "detokenize": True,
            }
        new_sampling_params = deepcopy(sampling_params)
        for k,v in kwargs.items():
            setattr(new_sampling_params, k, v)
            
        
        # Build Input data
        input_data = []
        for input_item_idx,item in enumerate(inputs):
            if "images" in item:
                images = item["images"]
                if not isinstance(images, list):
                    images = [images]
                    
            if "conversation" in item:
                conversation = list(item["conversation"])
                first_content = conversation[0]["content"]
                new_first_content = []
                img_idx = 0
                for c in first_content:
                    if c["type"] == "text":
                        new_first_content.append(c)
                    elif c["type"] == "image":
                        image = images[img_idx]
                        image_base64 = pil_to_base64(image, url_format=True)
                        new_first_content.append({"type": "image_url", "image_url": {"url": image_base64}})
                        img_idx += 1
                prompt = new_first_content[-1]["text"]
                conversation[0]["content"] = new_first_content
                initial_user_messages = conversation
            else:
                prompt = item["prompt"]
                initial_user_messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }]
                if "images" in item:
                    for current_image in images:
                        current_image_base64 = pil_to_base64(current_image, url_format=True)
                        initial_user_messages[0]["content"].append({"type": "image_url", "image_url": {"url": current_image_base64}})

            if "system" in item:
                initial_user_messages.insert(0, {"role": "system", "content": item["system"]})
            
            if do_sample:
                for _ in range(sample_num):
                    data_instance = dict(
                        input_item_idx=input_item_idx,
                        conversations=initial_user_messages.copy(),
                        status="processing",
                        model_outputs=[],
                        model_output_ids=[],
                        tool_cfgs=[],
                        tool_outputs=[],
                        new_round_input=[],
                        images=images.copy(),
                        prompt=prompt
                    )
                    input_data.append(data_instance)
            else:
                data_instance = dict(
                    input_item_idx=input_item_idx,
                    conversations=initial_user_messages.copy(),
                    status="processing",
                    model_outputs=[],
                    model_output_ids=[],
                    tool_cfgs=[],
                    tool_outputs=[],
                    new_round_input=[],
                    images=images.copy(),
                    prompt=prompt
                )
                input_data.append(data_instance)
            

        for round_num in range(max_rounds):
            input_conversations = [item["conversations"] for item in input_data if item["status"] == "processing"]
            input_idxs = [idx for idx, item in enumerate(input_data) if item["status"] == "processing"]
            
            if len(input_conversations) == 0:
                break
            
            outputs = None
            for attempt in range(vllm_retry_num):
                try:
                    signal.alarm(vllm_timeout_sec)
                    outputs = self.vllm_model.chat(
                        input_conversations, sampling_params=new_sampling_params, use_tqdm = True,
                    )
                    signal.alarm(0)
                    break
                except TimeoutException as te:
                    print(f"[VLLM] Timeout during chat(), retrying {attempt + 1}/{vllm_retry_num}")
                except Exception as e:
                    print(f"[VLLM] Unexpected exception during chat(): {e}, retrying {attempt + 1}/{vllm_retry_num}")
                finally:
                    signal.alarm(0)
                    if attempt < vllm_retry_num - 1:
                        time.sleep(vllm_retry_interval)
            # breakpoint()       
            if outputs is None:
                output_texts = ["Model generation error"] * len(input_conversations)
                output_idss = [(1712, 9471, 1465, 151645)] * len(input_conversations)
            else:
                output_texts = [output.outputs[0].text for output in outputs]
                output_idss = [output.outputs[0].token_ids for output in outputs]

            for input_idx, output_text, output_ids in zip(input_idxs, output_texts, output_idss):
                input_data[input_idx]["model_outputs"].append(output_text)
                input_data[input_idx]["model_output_ids"].append(output_ids)
                input_data[input_idx]["conversations"] = append_conversation_fn(conversation=input_data[input_idx]["conversations"], text=output_text, role="assistant")
                ## pop qualified data
                if "Terminate" in output_text:
                    input_data[input_idx]["status"] = "finished"
                    continue

                tool_cfg = parse_tool_config(
                        output_text, 
                        model_mode="general", 
                        image_tool_manager=None,
                        newest_image=input_data[input_idx]["images"][-1]
                )
                if not tool_cfg:
                    input_data[input_idx]["conversations"] = append_conversation_fn(conversation=input_data[input_idx]["conversations"], text=input_data[input_idx]["prompt"], role="user")
                
                else:
                    input_data[input_idx]["tool_cfgs"].append(tool_cfg)
                    try:
                        original_api_name = tool_cfg[0].get("API_name").lower() 
                        api_params = tool_cfg[0].get("API_params", {})
                        tool_name_mapping = {
                            'drawhorizontallinebyy': 'DrawHorizontalLineByY',
                            'zoominsubfigure': 'ZoomInSubfigure',
                            'drawverticallinebyx': 'DrawVerticalLineByX',
                            'segmentregionaroundpoint': 'SegmentRegionAroundPoint',
                            'point': 'Point',
                            'ocr': 'OCR'
                        }
                        api_name = tool_name_mapping.get(original_api_name)
                    except:
                        print(f"Tool config error: {tool_cfg}")
                        api_name = "None"
                        api_params = {}

                    if "param" in api_params:
                        p = api_params["param"]
                        print(f"Tool name: {api_name}, params: {p}")
                    
                    tool_result = {"text": f"Failed to call tool {api_name}", "error_code": 1}
                    for attempt in range(tool_retry_num):
                        try:
                            signal.alarm(tool_timeout_sec)
                            tool_result = self.tool_manager.call_tool(api_name, api_params)
                            signal.alarm(0)
                            break
                        except TimeoutException as te:
                            print(f"[Tool] Timeout during tool call, retrying {attempt + 1}/{tool_retry_num}")
                            tool_result = {"text": f"Failed to call tool {api_name}: {te}", "error_code": 1}
                        except Exception as e:
                            print(f"[Tool] Unexpected exception during tool call: {e}, retrying {attempt + 1}/{tool_retry_num}")
                            tool_result = {"text": f"Failed to call tool {api_name}: {e}", "error_code": 1}
                        finally:
                            signal.alarm(0)
                            if attempt < tool_retry_num - 1:
                                time.sleep(tool_retry_interval)
                    
                    input_data[input_idx]["tool_outputs"].append(tool_result)
                    
                    # Process the tool result and update the conversation
                    input_data[input_idx]["conversations"] = handle_tool_result(
                        cfg = tool_cfg[0],
                        tool_result = tool_result,
                        conversations = input_data[input_idx]["conversations"],
                        model_mode = "general",
                        original_prompt = input_data[input_idx]["prompt"],
                        input_data_item = input_data[input_idx]
                    )
                    
        # Log tool outputs
        if tool_log_file is not None:
            for item in input_data:
                self.log_item_into_file(item, tool_log_file)
                
        # Collect final results
        results = []
        grouped_results = {}
        for item in input_data:
            input_item_idx = item["input_item_idx"]
            if grouped_results.get(input_item_idx) is None:
                grouped_results[input_item_idx] = []
            grouped_results[input_item_idx].append(item)
        
        for k,v in grouped_results.items():
            results.append({"item_idx": k, "samples": v})
        return results
        