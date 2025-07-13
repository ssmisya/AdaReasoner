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
    
    def extract_final_answer(self, final_response: str):
        # 根据新的prompt格式，最终答案在<response>标签中
        response_content = None
        if "<response>" in final_response and "</response>" in final_response:
            # 提取<response>标签中的内容
            response_content = final_response.split("<response>")[-1].split("</response>")[0].strip()
        else:
            response_content = final_response.strip()
        
        if "\\boxed{" in response_content:
            # 如果包含\boxed{}，则提取其中的内容
            response_content = response_content.split("\\boxed{")[-1].split("}")[0].strip()
        
        return response_content
    
        
    def pop_qualified_items(self):
        res = []
        new_batch = []
        for idx,item in enumerate(self.dynamic_batch):
            if item.status == "finished":
                item = asdict(item)
                item = remove_pil_objects(item)
                
                final_model_output = item["model_response"][-1]
                final_answer = self.extract_final_answer(final_model_output)
                item["final_answer"] = final_answer
                
                res.append(item)
            else:
                new_batch.append(item)
        self.dynamic_batch = new_batch
        return res
    
    def append_item(self, meta_data: Dict):
        # breakpoint()
        if len(self.dynamic_batch) < self.batch_size:
            # breakpoint()
            candidate_item = DynamicBatchItem(
                max_rounds=self.max_rounds,
                current_round=0,
                meta_data=meta_data,
                status="pending"
            )
            candidate_item.conversation = self.generate_conversation_fn(
                text = meta_data["text"], 
                image = meta_data["image"],
                role = "user"
            )
            
            self.dynamic_batch.append(candidate_item)
        else:
            raise ValueError("Batch is full")
    
    
    def append_item_to_full(self, dataloader, progress_bar=None):
        while len(self.dynamic_batch) < self.batch_size:
            try:
                # breakpoint()
                self.append_item(next(dataloader))
                if progress_bar:
                    progress_bar.update(1)
            except:
                break
        
    

    def get_current_batch(self):
        return self.dynamic_batch
    
    
    # Caution: Only model.generate can call this function
    def update_item_status(self):
        for item in self.dynamic_batch:
            has_response_tag = False

            if item.model_response and "<response>" in item.model_response[-1]:
                has_response_tag = True
                
            if item.status == "pending":
                # 如果不使用工具，或者达到max_rounds，或者回答中含有<response>....</response>，则设为finished
                if not self.if_use_tool or item.current_round == item.max_rounds or has_response_tag:
                    item.status = "finished"
                else:
                    item.current_round += 1
                    item.status = "processing"
            elif item.status == "processing":
                # 如果不使用工具，或者达到max_rounds，或者回答中含有<response>....</response>，则设为finished
                if not self.if_use_tool or item.current_round == item.max_rounds or has_response_tag:
                    item.status = "finished"
                else:
                    item.current_round += 1
            elif item.status == "finished":
                pass
            else:
                raise ValueError(f"Invalid status {item.status}")
        
    
     