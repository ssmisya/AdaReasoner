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
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait

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

# ── E3 latency 旁路计时 (rebuttal E3) ─────────────────────────────
# 仅当 env E3_LATENCY_LOG 设置时启用; 累加三段耗时到全局dict, 不改变任何推理逻辑。
import os as _os, time as _time, json as _json, atexit as _atexit
_E3_LOG = _os.environ.get("E3_LATENCY_LOG")
_E3 = {"generation_s": 0.0, "tool_exec_s": 0.0, "orchestration_s": 0.0,
       "gen_calls": 0, "tool_batches": 0, "wall_start": None}
def _e3_add(bucket, dt):
    if _E3_LOG:
        _E3[bucket] = _E3.get(bucket, 0.0) + dt
def _e3_dump():
    if _E3_LOG:
        try:
            _E3["wall_total_s"] = (_time.time() - _E3["wall_start"]) if _E3.get("wall_start") else None
            with open(_E3_LOG, "w") as f:
                _json.dump(_E3, f, indent=2)
        except Exception:
            pass
_atexit.register(_e3_dump)
# ──────────────────────────────────────────────────────────────────

# ── E5 受控故障注入 (rebuttal E5) ──────────────────────────────────
# 仅当 env E5_FAULT 设置时启用。E5_FAULT ∈ {plausible_wrong, missing, malformed, timeout, contradictory}
# E5_FAULT_WHEN ∈ {early, late}: early=第1轮注入, late=最后一轮(current_round==max)注入。
# 注入对象: 工具真实返回后的 tool_response dict, 按类型篡改。不改推理逻辑, 只污染工具返回。
_E5_FAULT = _os.environ.get("E5_FAULT")
_E5_WHEN = _os.environ.get("E5_FAULT_WHEN", "early")
_E5_LOG = _os.environ.get("E5_FAULT_LOG")
_E5 = {"fault_type": _E5_FAULT, "when": _E5_WHEN, "injected": 0}
def _e5_should_inject(cur_round, max_rounds):
    if not _E5_FAULT:
        return False
    if _E5_WHEN == "early":
        return cur_round == 1
    else:  # late
        return cur_round >= max(1, max_rounds - 1)
def _e5_apply(resp):
    """按故障类型篡改一个工具返回dict, 返回(新resp, 是否注入)。"""
    if not isinstance(resp, dict):
        return resp, False
    ft = _E5_FAULT
    import copy as _c
    r = _c.deepcopy(resp)
    if ft == "plausible_wrong":
        # 状态success但内容错误(合理但错): 篡改points/path/bounding_boxes到一个貌似合理的错值
        if "points" in r and isinstance(r["points"], list) and r["points"]:
            for p in r["points"]:
                if isinstance(p, dict):
                    if "x" in p: p["x"] = float(p.get("x", 0)) + 47.0
                    if "y" in p: p["y"] = float(p.get("y", 0)) + 47.0
        if "path" in r and isinstance(r.get("path"), str):
            r["path"] = "R,R,R,R,R"  # 貌似合理的错误路径
        if "bounding_boxes" in r and isinstance(r["bounding_boxes"], list) and r["bounding_boxes"]:
            # 平移bbox一个貌似合理的偏移(仍在图内的错误框)
            def _shift(b):
                try:
                    v = _json.loads(b) if isinstance(b, str) else list(b)
                    v = [float(x) + 40.0 for x in v]
                    return str([int(x) for x in v])
                except Exception:
                    return b
            r["bounding_boxes"] = [_shift(b) for b in r["bounding_boxes"]]
        r["status"] = "success"
    elif ft == "missing":
        # 缺失: 工具无返回
        r = dict(text="", status="success")
    elif ft == "malformed":
        # 畸形: 返回结构损坏(非法字段/截断)
        r = {"tool_response_from": r.get("tool_response_from", "?"),
             "status": "success", "\x00garbage": "�{unterminated",
             "points": "NOT_A_LIST", "bounding_boxes": "NOT_A_LIST"}
    elif ft == "timeout":
        # 超时: 模拟失败状态+超时消息
        r = dict(text="Tool call timed out after 120s.", status="failed", error_code=504)
    elif ft == "contradictory":
        # 与其它工具矛盾: 标注一个和视觉明显冲突的坐标/框
        if "points" in r and isinstance(r["points"], list) and r["points"]:
            for p in r["points"]:
                if isinstance(p, dict):
                    if "x" in p: p["x"] = 0.0
                    if "y" in p: p["y"] = 0.0
        if "bounding_boxes" in r and isinstance(r["bounding_boxes"], list) and r["bounding_boxes"]:
            r["bounding_boxes"] = ["[0, 0, 1, 1]"]  # 退化到左上角1px,与视觉矛盾
        r["message"] = "region at image corner (0,0)"
        r["status"] = "success"
    return r, True
def _e5_dump():
    if _E5_LOG:
        try:
            with open(_E5_LOG, "w") as f:
                _json.dump(_E5, f, indent=2)
        except Exception:
            pass
_atexit.register(_e5_dump)
# ──────────────────────────────────────────────────────────────────

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
            generate_conversation_fn=self.tp_model.generate_conversation_fn,
            if_use_tool=self.if_use_tool,  # Pass if_use_tool parameter to DynamicBatchManager
        )
        self.tool_concurrency = max(
            1, int(os.environ.get("TF_EVAL_TOOL_CONCURRENCY", self.batch_size))
        )
        self.pipeline_enabled = os.environ.get(
            "TF_EVAL_PIPELINE", "0"
        ).lower() in {"1", "true", "yes"}
        self.pipeline_backend = (
            "request"
            if hasattr(self.tp_model, "_generate_one")
            and hasattr(self.tp_model, "apply_generation_result")
            else "batch"
        )
        default_pipeline_depth = max(
            self.batch_size,
            3 * int(getattr(self.tp_model, "request_concurrency", self.batch_size)),
        )
        self.pipeline_max_active = max(
            self.batch_size,
            int(os.environ.get("TF_EVAL_PIPELINE_MAX_ACTIVE", default_pipeline_depth)),
        )
        # Initialize tool manager
        self.tool_manager = None
        
        self.image_keys = ["image","base_image","image_to_insert"]
        self.original_image_keys = self.image_keys.copy()
        
    
        
        

    def _tool_response_to_next_round_input(self, item):
        if item.model_response is None or item.status != "processing":
            return

        tool_response = item.tool_response[item.current_round - 1]
        assert len(item.tool_cfg) == item.current_round
        assert len(item.tool_response) == item.current_round
        original_prompt = item.meta_data.get("text", "")

        if tool_response is not None:
            try:
                if "edited_image" in tool_response:
                    edited_image = tool_response.pop("edited_image")
                    try:
                        pil_edited_image = base64_to_pil(edited_image)
                        width, height = pil_edited_image.size
                        resized = False
                        if width < self.min_image_size or height < self.min_image_size:
                            if width < height:
                                new_width = 30
                                ratio = min(30 / width, self.max_ratio)
                                new_height = min(
                                    int(height * ratio), self.max_image_size
                                )
                            else:
                                new_height = 30
                                ratio = min(30 / height, self.max_ratio)
                                new_width = min(
                                    int(width * ratio), self.max_image_size
                                )
                            pil_edited_image = pil_edited_image.resize(
                                (new_width, new_height), Image.LANCZOS
                            )
                            resized = True
                        if resized:
                            edited_image = pil_to_base64(pil_edited_image)
                    except Exception as exc:
                        logger.warning(f"Failed to resize image: {exc}")

                    item.current_image = edited_image
                    assert item.image_history is not None, (
                        "item.image_history should not be None."
                    )
                    img_idx = len(item.image_history) + 1
                    item.image_history[f"img_{img_idx}"] = edited_image
                    if self.model_mode == "llava_plus":
                        edited_image = base64_to_pil(edited_image)
                else:
                    edited_image = None

                tool_response.pop("tool_reward", None)
                tool_response_text = tool_response
                new_round_prompt = f"{tool_response_text}\n"
            except Exception:
                edited_image = None
                new_round_prompt = original_prompt
        else:
            edited_image = None
            new_round_prompt = "Please continue with your response or call a tool."

        item.new_round_input.append(
            {"text": new_round_prompt, "image": edited_image}
        )
        item.conversation = self.append_conversation_fn(
            conversation=item.conversation,
            text=new_round_prompt,
            image=edited_image,
            role="user",
        )

    def batch_tool_response_to_next_round_input(self):
        """Convert completed tool responses into the next model-round inputs."""
        for item in self.manager.get_current_batch():
            self._tool_response_to_next_round_input(item)

    
    def _prepare_tool_call(self, item):
        tool_cfg = item.tool_cfg[item.current_round - 1]
        assert len(item.tool_cfg) == item.current_round
        assert item.image_history

        if not tool_cfg:
            return None
        assert len(tool_cfg) == 1, (
            f"Only one tool is supported for now, but got: {tool_cfg}"
        )

        api_name = tool_cfg[0].get("API_name", tool_cfg[0].get("api_name", ""))
        if api_name not in self.available_models:
            raise ValueError(
                f"API_name {api_name} not in available models, {self.available_models}"
            )

        # Parse mutates nested tool config objects; use a copy so retries/checkpoints
        # retain the exact model-produced configuration.
        api_params = copy.deepcopy(
            tool_cfg[0].get("api_params", tool_cfg[0].get("API_params", {}))
        )
        for image_key in self.image_keys:
            if image_key not in api_params:
                continue
            img_key = api_params[image_key]
            image = item.image_history.get(img_key)
            if image is None:
                raise ValueError(
                    f"Image {img_key} not found in history for item {item.meta_data['idx']}"
                )
            api_params[image_key] = pil_to_base64(load_image(image))
        return api_name, api_params

    def _call_tool(self, api_name, api_params):
        started_ns = _time.perf_counter_ns()
        try:
            response = self.tool_manager.call_tool(api_name, api_params)
            status = response.get("status", "unknown") if isinstance(response, dict) else "unknown"
            return response, (_time.perf_counter_ns() - started_ns) / 1e9, status, None
        except Exception as exc:
            return None, (_time.perf_counter_ns() - started_ns) / 1e9, "exception", exc

    def _apply_tool_result(self, item, api_name, result):
        tool_response, latency_s, status, error = result
        self.manager.record_tool_timing(item, api_name, latency_s, status)
        if error is not None:
            logger.info(f"Tool {api_name} failed to answer: {error}")
            tool_response = {
                "text": f"Tool {api_name} failed to answer the question: {error}",
                "status": "failed",
            }
        else:
            tool_response = copy.deepcopy(tool_response)

        if _e5_should_inject(item.current_round, self.max_rounds):
            tool_response, injected = _e5_apply(tool_response)
            if injected:
                _E5["injected"] = _E5.get("injected", 0) + 1
                _e5_dump()

        logger.info(
            f"The {api_name} call "
            f"{'succeeded' if status == 'success' else 'failed'}!"
        )
        item.tool_response.append(tool_response)

    def batch_get_tool_response(self):
        """Execute independent tool calls concurrently and preserve batch order."""
        current_batch = self.manager.get_current_batch()
        pending = []

        for item in current_batch:
            if item.model_response is None or item.status != "processing":
                continue
            try:
                prepared = self._prepare_tool_call(item)
                if prepared is None:
                    item.tool_response.append(None)
                else:
                    api_name, api_params = prepared
                    pending.append((item, api_name, api_params))
            except Exception as exc:
                logger.info(f"Failed to prepare tool call: {exc}")
                item.tool_response.append(
                    {"text": f"Tool call could not be prepared: {exc}", "status": "failed"}
                )

        responses = {}
        if pending:
            worker_count = min(self.tool_concurrency, len(pending))
            with ThreadPoolExecutor(
                max_workers=worker_count, thread_name_prefix="tool-call"
            ) as pool:
                futures = {
                    pool.submit(self._call_tool, api_name, api_params): (item, api_name)
                    for item, api_name, api_params in pending
                }
                for future in as_completed(futures):
                    item, api_name = futures[future]
                    responses[id(item)] = (api_name, future.result())

        # Only mutate samples on the main thread so round accounting and
        # checkpoint serialization remain deterministic.
        for item, _, _ in pending:
            api_name, result = responses[id(item)]
            self._apply_tool_result(item, api_name, result)

        for item in current_batch:
            if item.model_response is not None and item.status == "processing":
                assert len(item.tool_response) == item.current_round

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
       
    def _parse_tool_config(self, item):
        model_response = item.model_response[item.current_round - 1]
        assert len(item.model_response) == item.current_round
        if model_response is None or item.status != "processing":
            return

        tool_cfg = None
        try:
            if self.model_mode == "general":
                tool_calls = self.extract_tool_call(model_response)
                if tool_calls:
                    tool_call = tool_calls[0]
                    assert "name" in tool_call and "parameters" in tool_call, (
                        "missing 'name' or 'parameters' in the parsed tool_call."
                    )
                    tool_cfg = [
                        {
                            "API_name": tool_call["name"],
                            "API_params": tool_call["parameters"],
                        }
                    ]
        except Exception as exc:
            logger.info(f"Failed to parse tool config: {exc}.")
        item.tool_cfg.append(tool_cfg)

    def batch_parse_tool_config(self):
        """Extract one tool configuration from each processing sample."""
        for item in self.manager.get_current_batch():
            self._parse_tool_config(item)

    def generate_with_latency(self, current_batch):
        """Run one model round and attach client/backend latency to every item."""
        if not current_batch:
            return
        if _E3_LOG and _E3.get("wall_start") is None:
            _E3["wall_start"] = _time.time()
        self.manager.start_generation_timing(current_batch)
        generation_start_ns = _time.perf_counter_ns()
        self.tp_model.generate(current_batch)
        generation_wall_s = (
            _time.perf_counter_ns() - generation_start_ns
        ) / 1_000_000_000.0
        self.manager.record_generation_timing(current_batch, generation_wall_s)
        _e3_add("generation_s", generation_wall_s)
        _E3["gen_calls"] = _E3.get("gen_calls", 0) + 1
        self.manager.update_item_status(current_batch)
        # A final-answer round has no following tool phase, so it ends here.
        self.manager.finish_round_timing(current_batch, only_status="finished")
            
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
                self.manager.finish_instance_timing(item)
                image_history = item.image_history
                item_dict = asdict(item)
                item_dict.pop("_latency_instance_start_ns", None)
                item_dict.pop("_latency_round_start_ns", None)
                item_dict.pop("_backend_generation_metrics", None)
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

    def _store_completed_results(self):
        for result in self.pop_qualified_items():
            idx = result["meta_data"]["idx"]
            self.dataset.store_results({"idx": idx, "results": result})

    def _batched_pipelined_inference(self, progress_bar):
        """Overlap native batched generation with external tool calls."""
        original_capacity = self.manager.batch_size
        self.manager.batch_size = self.pipeline_max_active
        ready_for_generation = []
        tool_futures = {}
        phase_lock = threading.Lock()
        phase = {
            "last_ns": _time.perf_counter_ns(),
            "generation": 0,
            "tool": 0,
            "generation_busy_s": 0.0,
            "tool_busy_s": 0.0,
            "overlap_s": 0.0,
        }
        scheduler_started_ns = phase["last_ns"]

        if _E3_LOG and _E3.get("wall_start") is None:
            _E3["wall_start"] = _time.time()

        def update_phase(generation_delta=0, tool_delta=0):
            with phase_lock:
                now_ns = _time.perf_counter_ns()
                elapsed_s = (now_ns - phase["last_ns"]) / 1e9
                if phase["generation"]:
                    phase["generation_busy_s"] += elapsed_s
                if phase["tool"]:
                    phase["tool_busy_s"] += elapsed_s
                if phase["generation"] and phase["tool"]:
                    phase["overlap_s"] += elapsed_s
                phase["generation"] += generation_delta
                phase["tool"] += tool_delta
                phase["last_ns"] = now_ns

        def run_tool(api_name, api_params):
            update_phase(tool_delta=1)
            try:
                return self._call_tool(api_name, api_params)
            finally:
                update_phase(tool_delta=-1)

        def submit_tool_or_continue(pool, item):
            try:
                prepared = self._prepare_tool_call(item)
            except Exception as exc:
                logger.info(f"Failed to prepare tool call: {exc}")
                item.tool_response.append(
                    {
                        "text": f"Tool call could not be prepared: {exc}",
                        "status": "failed",
                    }
                )
                prepared = None

            if prepared is not None:
                api_name, api_params = prepared
                future = pool.submit(run_tool, api_name, api_params)
                tool_futures[future] = (item, api_name)
                return

            if len(item.tool_response) < item.current_round:
                item.tool_response.append(None)
            self._tool_response_to_next_round_input(item)
            self.manager.finish_round_timing([item], only_status="processing")
            ready_for_generation.append(item)

        def admit_items():
            existing = {id(item) for item in self.manager.get_current_batch()}
            self.manager.append_item_to_full(
                self.dataloader_iter, progress_bar=progress_bar
            )
            ready_for_generation.extend(
                item
                for item in self.manager.get_current_batch()
                if id(item) not in existing
            )

        def collect_tools(block):
            if not tool_futures:
                return
            if block:
                completed, _ = wait(
                    set(tool_futures), return_when=FIRST_COMPLETED
                )
            else:
                completed = {
                    future for future in tool_futures if future.done()
                }
            for future in completed:
                item, api_name = tool_futures.pop(future)
                self._apply_tool_result(item, api_name, future.result())
                _E3["tool_batches"] = _E3.get("tool_batches", 0) + 1
                orchestration_started = _time.perf_counter()
                self._tool_response_to_next_round_input(item)
                self.manager.finish_round_timing(
                    [item], only_status="processing"
                )
                _e3_add(
                    "orchestration_s",
                    _time.perf_counter() - orchestration_started,
                )
                ready_for_generation.append(item)

        logger.info(
            "Using native batched pipeline: generation_batch=%s, "
            "tool_concurrency=%s, max_active=%s",
            original_capacity,
            self.tool_concurrency,
            self.pipeline_max_active,
        )

        try:
            with ThreadPoolExecutor(
                max_workers=self.tool_concurrency, thread_name_prefix="tool-pipeline"
            ) as tool_pool:
                admit_items()
                while ready_for_generation or tool_futures:
                    collect_tools(block=False)
                    self._store_completed_results()
                    admit_items()

                    if ready_for_generation:
                        generation_batch = ready_for_generation[:original_capacity]
                        del ready_for_generation[:original_capacity]
                        update_phase(generation_delta=1)
                        try:
                            self.generate_with_latency(generation_batch)
                        finally:
                            update_phase(generation_delta=-1)

                        for item in generation_batch:
                            if item.status == "finished":
                                continue
                            orchestration_started = _time.perf_counter()
                            self._parse_tool_config(item)
                            _e3_add(
                                "orchestration_s",
                                _time.perf_counter() - orchestration_started,
                            )
                            submit_tool_or_continue(tool_pool, item)
                        self._store_completed_results()
                        admit_items()
                    else:
                        collect_tools(block=True)
                        self._store_completed_results()
                        admit_items()
        finally:
            update_phase()
            self.manager.batch_size = original_capacity
            scheduler_wall_s = (
                _time.perf_counter_ns() - scheduler_started_ns
            ) / 1e9
            _E3["tool_exec_s"] = _E3.get("tool_exec_s", 0.0) + phase[
                "tool_busy_s"
            ]
            _E3["pipeline_enabled"] = True
            _E3["pipeline_backend"] = "native_batch"
            _E3["pipeline_max_active"] = self.pipeline_max_active
            _E3["pipeline_scheduler_wall_s"] = round(scheduler_wall_s, 6)
            _E3["pipeline_generation_busy_s"] = round(
                phase["generation_busy_s"], 6
            )
            _E3["pipeline_tool_busy_s"] = round(phase["tool_busy_s"], 6)
            _E3["pipeline_overlap_s"] = round(phase["overlap_s"], 6)
            _E3["pipeline_lm_only_s"] = round(
                max(0.0, phase["generation_busy_s"] - phase["overlap_s"]), 6
            )
            _E3["pipeline_tool_only_s"] = round(
                max(0.0, phase["tool_busy_s"] - phase["overlap_s"]), 6
            )
            _e3_dump()

    def _pipelined_batch_inference(self, progress_bar):
        """Continuously overlap per-sample vLLM requests and tool calls."""
        original_capacity = self.manager.batch_size
        self.manager.batch_size = self.pipeline_max_active
        model_workers = max(
            1, int(getattr(self.tp_model, "request_concurrency", self.batch_size))
        )
        generation_futures = {}
        tool_futures = {}
        phase = {
            "last_ns": _time.perf_counter_ns(),
            "generation": 0,
            "tool": 0,
            "generation_busy_s": 0.0,
            "tool_busy_s": 0.0,
            "overlap_s": 0.0,
        }
        scheduler_started_ns = phase["last_ns"]

        if _E3_LOG and _E3.get("wall_start") is None:
            _E3["wall_start"] = _time.time()

        def update_phase(generation_delta=0, tool_delta=0):
            now_ns = _time.perf_counter_ns()
            elapsed_s = (now_ns - phase["last_ns"]) / 1e9
            if phase["generation"]:
                phase["generation_busy_s"] += elapsed_s
            if phase["tool"]:
                phase["tool_busy_s"] += elapsed_s
            if phase["generation"] and phase["tool"]:
                phase["overlap_s"] += elapsed_s
            phase["generation"] += generation_delta
            phase["tool"] += tool_delta
            phase["last_ns"] = now_ns

        def submit_generation(pool, item):
            self.manager.start_generation_timing([item])
            item.latency["generation_attribution"] = (
                "per-request client wall time, including the local request-executor queue"
            )
            started_ns = _time.perf_counter_ns()
            future = pool.submit(self.tp_model._generate_one, item.conversation)
            generation_futures[future] = (item, started_ns)
            update_phase(generation_delta=1)

        def submit_tool_or_continue(tool_pool, model_pool, item):
            try:
                prepared = self._prepare_tool_call(item)
            except Exception as exc:
                logger.info(f"Failed to prepare tool call: {exc}")
                item.tool_response.append(
                    {
                        "text": f"Tool call could not be prepared: {exc}",
                        "status": "failed",
                    }
                )
                prepared = None

            if prepared is not None:
                api_name, api_params = prepared
                future = tool_pool.submit(self._call_tool, api_name, api_params)
                tool_futures[future] = (item, api_name)
                update_phase(tool_delta=1)
                return

            if len(item.tool_response) < item.current_round:
                item.tool_response.append(None)
            self._tool_response_to_next_round_input(item)
            self.manager.finish_round_timing([item], only_status="processing")
            submit_generation(model_pool, item)

        def admit_items(model_pool):
            existing = {id(item) for item in self.manager.get_current_batch()}
            self.manager.append_item_to_full(
                self.dataloader_iter, progress_bar=progress_bar
            )
            for item in self.manager.get_current_batch():
                if id(item) not in existing:
                    submit_generation(model_pool, item)

        logger.info(
            "Using pipelined tool evaluation: model_concurrency=%s, "
            "tool_concurrency=%s, max_active=%s",
            model_workers,
            self.tool_concurrency,
            self.pipeline_max_active,
        )

        try:
            with ThreadPoolExecutor(
                max_workers=model_workers, thread_name_prefix="vllm-pipeline"
            ) as model_pool, ThreadPoolExecutor(
                max_workers=self.tool_concurrency, thread_name_prefix="tool-pipeline"
            ) as tool_pool:
                admit_items(model_pool)
                while generation_futures or tool_futures:
                    pending = set(generation_futures) | set(tool_futures)
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)

                    for future in completed:
                        if future in generation_futures:
                            item, started_ns = generation_futures.pop(future)
                            update_phase(generation_delta=-1)
                            generation_wall_s = (
                                _time.perf_counter_ns() - started_ns
                            ) / 1e9
                            try:
                                generation_result = future.result()
                            except Exception as exc:
                                logger.error(
                                    f"Error during pipelined model inference: {exc}"
                                )
                                generation_result = self.tp_model.generation_failure_result(
                                    exc
                                )
                            self.tp_model.apply_generation_result(
                                item, generation_result
                            )
                            self.manager.record_generation_timing(
                                [item], generation_wall_s
                            )
                            self.manager.update_item_status([item])
                            _E3["gen_calls"] = _E3.get("gen_calls", 0) + 1

                            if item.status == "finished":
                                self.manager.finish_round_timing(
                                    [item], only_status="finished"
                                )
                            else:
                                orchestration_started = _time.perf_counter()
                                self._parse_tool_config(item)
                                _e3_add(
                                    "orchestration_s",
                                    _time.perf_counter() - orchestration_started,
                                )
                                submit_tool_or_continue(
                                    tool_pool, model_pool, item
                                )
                        else:
                            item, api_name = tool_futures.pop(future)
                            update_phase(tool_delta=-1)
                            self._apply_tool_result(
                                item, api_name, future.result()
                            )
                            _E3["tool_batches"] = _E3.get("tool_batches", 0) + 1
                            orchestration_started = _time.perf_counter()
                            self._tool_response_to_next_round_input(item)
                            self.manager.finish_round_timing(
                                [item], only_status="processing"
                            )
                            _e3_add(
                                "orchestration_s",
                                _time.perf_counter() - orchestration_started,
                            )
                            submit_generation(model_pool, item)

                    self._store_completed_results()
                    admit_items(model_pool)
        finally:
            update_phase()
            self.manager.batch_size = original_capacity
            scheduler_wall_s = (
                _time.perf_counter_ns() - scheduler_started_ns
            ) / 1e9
            generation_busy_s = phase["generation_busy_s"]
            tool_busy_s = phase["tool_busy_s"]
            overlap_s = phase["overlap_s"]
            _E3["generation_s"] = _E3.get("generation_s", 0.0) + generation_busy_s
            _E3["tool_exec_s"] = _E3.get("tool_exec_s", 0.0) + tool_busy_s
            _E3["pipeline_enabled"] = True
            _E3["pipeline_backend"] = "openai_request"
            _E3["pipeline_max_active"] = self.pipeline_max_active
            _E3["pipeline_scheduler_wall_s"] = round(scheduler_wall_s, 6)
            _E3["pipeline_generation_busy_s"] = round(generation_busy_s, 6)
            _E3["pipeline_tool_busy_s"] = round(tool_busy_s, 6)
            _E3["pipeline_overlap_s"] = round(overlap_s, 6)
            _E3["pipeline_lm_only_s"] = round(
                max(0.0, generation_busy_s - overlap_s), 6
            )
            _E3["pipeline_tool_only_s"] = round(
                max(0.0, tool_busy_s - overlap_s), 6
            )
            _e3_dump()

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
            
        if self.pipeline_enabled and self.if_use_tool:
            if self.pipeline_backend == "request":
                self._pipelined_batch_inference(progress_bar)
            else:
                self._batched_pipelined_inference(progress_bar)
            assert len(self.manager.get_current_batch()) == 0
            if not 'vllm_models' in str(type(self.tp_model)):
                self.accelerator.wait_for_everyone()
            return

        # Add data items from data loader to manager and show progress with progress bar
        self.manager.append_item_to_full(self.dataloader_iter, progress_bar=progress_bar)

        # Get current batch and generate responses using model
        current_batch = self.manager.get_current_batch()
        self.generate_with_latency(current_batch)
        
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
                        self.generate_with_latency(current_batch)
                    continue
                
                # Below is the workflow when using tools
                # Parse tool configuration
                _t = _time.time()
                self.batch_parse_tool_config()
                _e3_add("orchestration_s", _time.time() - _t)
                # Get tool responses
                _t = _time.time()
                self.batch_get_tool_response()
                _e3_add("tool_exec_s", _time.time() - _t); _E3["tool_batches"] = _E3.get("tool_batches",0)+1
                # Convert tool responses to next round input
                _t = _time.time()
                self.batch_tool_response_to_next_round_input()
                _e3_add("orchestration_s", _time.time() - _t)
                # Tool-using rounds end after the tool result is converted into
                # the next-round input; this is the observed round E2E latency.
                self.manager.finish_round_timing(
                    self.manager.get_current_batch(), only_status="processing"
                )

                # Refill current batch
                self.manager.append_item_to_full(self.dataloader_iter, progress_bar=progress_bar)

                # Get updated current batch and generate new responses
                current_batch = self.manager.get_current_batch()
                if len(current_batch) > 0:
                    self.generate_with_latency(current_batch)

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
