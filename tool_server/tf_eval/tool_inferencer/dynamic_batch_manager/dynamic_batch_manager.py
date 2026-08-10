from dataclasses import dataclass, field, asdict
from typing import Dict, Sequence, Optional,List
from tool_server.tf_eval.utils.log_utils import get_logger
from ...utils.utils import *
from PIL import Image
import time

logger = get_logger(__name__)

@dataclass
class DynamicBatchItem:
    max_rounds: int
    current_round : int
    status: str = "pending" # pending, processing, finished
    meta_data: Dict = field(default = None)
    conversation: object = field(default = None)
    model_response: List[str] = field(default_factory=list)
    tool_cfg :  List[str] = field(default_factory=list)
    tool_response :  List[str] = field(default_factory=list)
    new_round_input :  List[str] = field(default_factory=list)
    current_image : Image = field(default=None)
    image_history: Dict = field(default_factory=list)
    latency: Dict = field(default_factory=dict)
    _latency_instance_start_ns: Optional[int] = field(default=None, repr=False)
    _latency_round_start_ns: Optional[int] = field(default=None, repr=False)
    _backend_generation_metrics: Dict = field(default_factory=dict, repr=False)


class DynamicBatchManager():
    def __init__(
        self,
        batch_size: int,
        stop_token: str = "<stop>",
        max_rounds: int = 3,
        generate_conversation_fn = None,
        if_use_tool: bool = True,
    ):
        self.dynamic_batch = []
        self.batch_size = batch_size
        self.stop_token = stop_token
        self.max_rounds = max_rounds
        self.generate_conversation_fn = generate_conversation_fn
        self.if_use_tool = if_use_tool
    
    @staticmethod
    def _seconds(start_ns, end_ns=None):
        if start_ns is None:
            return None
        end_ns = time.perf_counter_ns() if end_ns is None else end_ns
        return round((end_ns - start_ns) / 1_000_000_000.0, 6)

    def start_generation_timing(self, batch):
        now = time.perf_counter_ns()
        for item in batch:
            if item._latency_round_start_ns is None:
                item._latency_round_start_ns = now

    def record_generation_timing(self, batch, generation_batch_wall_s):
        for item in batch:
            round_record = {
                "round_index": len(item.model_response),
                "generation_batch_wall_s": round(generation_batch_wall_s, 6),
                "generation_batch_size": len(batch),
                "tool_calls": [],
                "tool_latency_s": 0.0,
                "round_e2e_s": None,
            }
            backend = dict(item._backend_generation_metrics or {})
            if backend:
                round_record["backend_request_metrics"] = backend
            item._backend_generation_metrics = {}
            item.latency.setdefault("rounds", []).append(round_record)

    def record_tool_timing(self, item, tool_name, latency_s, status):
        rounds = item.latency.setdefault("rounds", [])
        if not rounds:
            return
        record = {
            "tool_name": tool_name,
            "latency_s": round(latency_s, 6),
            "status": status,
        }
        rounds[-1].setdefault("tool_calls", []).append(record)
        rounds[-1]["tool_latency_s"] = round(
            sum(call["latency_s"] for call in rounds[-1]["tool_calls"]), 6
        )

    def finish_round_timing(self, items, only_status=None):
        now = time.perf_counter_ns()
        for item in items:
            if only_status is not None and item.status != only_status:
                continue
            if item._latency_round_start_ns is None:
                continue
            rounds = item.latency.get("rounds", [])
            if rounds and rounds[-1].get("round_e2e_s") is None:
                round_e2e_s = self._seconds(item._latency_round_start_ns, now)
                generation_s = float(
                    rounds[-1].get("generation_batch_wall_s", 0.0) or 0.0
                )
                tool_s = float(rounds[-1].get("tool_latency_s", 0.0) or 0.0)
                rounds[-1]["round_e2e_s"] = round_e2e_s
                rounds[-1]["orchestration_and_queue_s"] = round(
                    max(0.0, round_e2e_s - generation_s - tool_s), 6
                )
            item._latency_round_start_ns = None

    def finish_instance_timing(self, item):
        self.finish_round_timing([item])
        item.latency["instance_e2e_s"] = self._seconds(
            item._latency_instance_start_ns
        )
        item.latency["round_count"] = len(item.latency.get("rounds", []))
        item.latency["model_generation_batch_wall_s"] = round(
            sum(
                record.get("generation_batch_wall_s", 0.0)
                for record in item.latency.get("rounds", [])
            ),
            6,
        )
        item.latency["tool_latency_s"] = round(
            sum(
                record.get("tool_latency_s", 0.0)
                for record in item.latency.get("rounds", [])
            ),
            6,
        )

    def extract_final_answer(self, final_response: str, task_name: str):
        # According to the new prompt format, the final answer is in the <response> tag
        response_content = final_response
        if "<response>" in final_response and "</response>" in final_response:
            # Extract content from <response> tag
            response_content = final_response.split("<response>")[-1].split("</response>")[0].strip()
        else:
            response_content = final_response.strip()
        # If it's a web task, don't extract boxed content
        logger.debug(f"DEBUG: task_name: {task_name}")
        if "web" not in task_name:
            logger.debug(f"DEBUG: task_name is not web, extracting boxed content")
            if "\\boxed{" in response_content:
                # If contains \boxed{}, extract the content inside
                response_content = response_content.split("\\boxed{")[-1].split("}")[0].strip()
                return response_content
            else:
                return response_content
        else:
            logger.debug(f"DEBUG: task_name is web, not extracting boxed content")
            return response_content
        
    
    def append_item(self, meta_data: Dict):
        # breakpoint()
        # print(f"DEBUG: append_item called, meta_data idx: {meta_data.get('idx', 'N/A')}")
        if len(self.dynamic_batch) < self.batch_size:
            instance_start_ns = time.perf_counter_ns()
            # breakpoint()
            candidate_item = DynamicBatchItem(
                max_rounds=self.max_rounds,
                current_round=0,
                meta_data=meta_data,
                status="pending",
                latency={
                    "schema_version": 1,
                    "clock": "time.perf_counter_ns",
                    "definition": "client-observed wall-clock latency",
                    "instance_definition": "from admission into the dynamic batch until the final answer is stored",
                    "round_definition": "from the start of a generation round through tool execution and next-round orchestration, or through final-answer detection",
                    "generation_attribution": "the synchronous batch generation wall time is assigned to every active instance in that batch",
                    "instance_e2e_s": None,
                    "round_count": 0,
                    "rounds": [],
                },
                _latency_instance_start_ns=instance_start_ns,
            )
            # print(f"DEBUG: Starting to generate conversation")
            if self.if_use_tool:
                few_shot = meta_data.get("tool_few_shot", None)
            else:
                few_shot = None
                
            candidate_item.conversation = self.generate_conversation_fn(
                text = meta_data["text"], 
                images = meta_data["images"],
                role = "user",
                few_shot = few_shot,
            )
            
            image_history = {}
            for idx,image in enumerate(meta_data["images"]):
                image_history[f"img_{idx+1}"] = image
                
            candidate_item.image_history = image_history
                
            # print(f"DEBUG: conversation generated successfully")
            self.dynamic_batch.append(candidate_item)
            # print(f"DEBUG: Successfully added to dynamic_batch, current length: {len(self.dynamic_batch)}")
        else:
            raise ValueError("Batch is full")
    
    
    def append_item_to_full(self, dataloader, progress_bar=None):
        logger.debug(
            "append_item_to_full started: current=%s limit=%s",
            len(self.dynamic_batch),
            self.batch_size,
        )
        items_added = 0
        while len(self.dynamic_batch) < self.batch_size:
            try:
                data_item = next(dataloader)
            except StopIteration:
                logger.debug(
                    "dataloader completed after adding %s items", items_added
                )
                break

            try:
                self.append_item(data_item)
            except Exception:
                item_idx = (
                    data_item.get("idx", "N/A")
                    if isinstance(data_item, dict)
                    else "N/A"
                )
                logger.exception(
                    "Failed to admit dataset item idx=%s; aborting instead of "
                    "silently dropping it",
                    item_idx,
                )
                raise

            items_added += 1
            if progress_bar:
                progress_bar.update(1)

        logger.debug(
            "append_item_to_full completed: current=%s", len(self.dynamic_batch)
        )
        
    

    def get_current_batch(self):
        return self.dynamic_batch
    
    
    # Caution: call this only after model generation has completed for the items.
    def update_item_status(self, items=None):
        target_items = self.dynamic_batch if items is None else items
        for item in target_items:
            has_response_tag = False

            if item.model_response and "<response>" in item.model_response[-1]:
                has_response_tag = True
            
            old_status = item.status
            old_round = item.current_round
                
            if item.status == "pending":
                # model_response already contains the just-finished generation.
                # Stop when that generation reaches max_rounds; do not run a hidden
                # max_rounds+1 generation because current_round starts at zero.
                generated_rounds = len(item.model_response)
                if not self.if_use_tool or generated_rounds >= item.max_rounds or has_response_tag:
                    item.status = "finished"
                else:
                    item.current_round += 1
                    item.status = "processing"
            elif item.status == "processing":
                generated_rounds = len(item.model_response)
                if not self.if_use_tool or generated_rounds >= item.max_rounds or has_response_tag:
                    item.status = "finished"
                else:
                    item.current_round += 1
            elif item.status == "finished":
                pass
            else:
                raise ValueError(f"Invalid status {item.status}")
            
            # Add status change log
            if old_status != item.status or old_round != item.current_round:
                logger.debug(f"DEBUG: Item {item.meta_data.get('idx', 'N/A')} status updated: {old_status}({old_round}) -> {item.status}({item.current_round}), has_response_tag={has_response_tag}, if_use_tool={self.if_use_tool}")
        
    
     