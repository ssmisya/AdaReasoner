from dataclasses import dataclass, field, asdict
from typing import Dict, Sequence, Optional,List
from tool_server.tf_eval.utils.log_utils import get_logger
from ...utils.utils import *
from PIL import Image

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
            # breakpoint()
            candidate_item = DynamicBatchItem(
                max_rounds=self.max_rounds,
                current_round=0,
                meta_data=meta_data,
                status="pending"
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
        print(f"DEBUG: append_item_to_full started, current batch size: {len(self.dynamic_batch)}, batch_size limit: {self.batch_size}")
        items_added = 0
        while len(self.dynamic_batch) < self.batch_size:
            try:
                data_item = next(dataloader)
                # print(f"DEBUG: Successfully retrieved data item from dataloader, idx: {data_item.get('idx', 'N/A')}")
                self.append_item(data_item)
                items_added += 1
                # print(f"DEBUG: Successfully added data item, current batch size: {len(self.dynamic_batch)}")
                if progress_bar:
                    progress_bar.update(1)
            except StopIteration as e:
                logger.debug(f"DEBUG: dataloader iteration completed, total items added: {items_added}")
                break
            except Exception as e:
                logger.debug(f"DEBUG: Exception occurred in append_item_to_full: {e}")
                break
        logger.debug(f"DEBUG: append_item_to_full completed, final batch size: {len(self.dynamic_batch)}")
        
    

    def get_current_batch(self):
        return self.dynamic_batch
    
    
    # Caution: Only model.generate can call this function
    def update_item_status(self):
        for i, item in enumerate(self.dynamic_batch):
            has_response_tag = False

            if item.model_response and "<response>" in item.model_response[-1]:
                has_response_tag = True
            
            old_status = item.status
            old_round = item.current_round
                
            if item.status == "pending":
                # If not using tools, or reached max_rounds, or response contains <response>...</response>, set to finished
                if not self.if_use_tool or item.current_round == item.max_rounds or has_response_tag:
                    item.status = "finished"
                else:
                    item.current_round += 1
                    item.status = "processing"
            elif item.status == "processing":
                # If not using tools, or reached max_rounds, or response contains <response>...</response>, set to finished
                if not self.if_use_tool or item.current_round == item.max_rounds or has_response_tag:
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
        
    
     