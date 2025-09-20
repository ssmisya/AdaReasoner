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
    
    def extract_final_answer(self, final_response: str, task_name: str):
        # 根据新的prompt格式，最终答案在<response>标签中
        response_content = final_response
        if "<response>" in final_response and "</response>" in final_response:
            # 提取<response>标签中的内容
            response_content = final_response.split("<response>")[-1].split("</response>")[0].strip()
        else:
            response_content = final_response.strip()
        # 如果是web的任务，则不提取boxed内容
        print(f"啊啊啊啊啊啊DEBUG: task_name: {task_name}")
        if "web" not in task_name:
            print(f"啊啊啊啊啊啊DEBUG: task_name不是web，提取boxed内容")
            if "\\boxed{" in response_content:
                # 如果包含\boxed{}，则提取其中的内容
                response_content = response_content.split("\\boxed{")[-1].split("}")[0].strip()
                return response_content
            else:
                return response_content
        else:
            print(f"啊啊啊啊啊啊DEBUG: task_name是web，不提取boxed内容")
            return response_content
        
    
    # 这个好像从来没有被使用过，因为inferencer中都有这个方法
    # def pop_qualified_items(self):
    #     res = []
    #     new_batch = []
    #     for idx,item in enumerate(self.dynamic_batch):
    #         if item.status == "finished":
    #             item = asdict(item)
    #             item = remove_pil_objects(item)
                
    #             final_model_output = item["model_response"][-1]
    #             # 我感觉task_name=item["meta_data"]["task_name"]不对，是cursor写的
    #             final_answer = self.extract_final_answer(final_model_output, task_name=item["meta_data"]["task_name"])
    #             item["final_answer"] = final_answer
                
    #             res.append(item)
    #         else:
    #             new_batch.append(item)
    #     self.dynamic_batch = new_batch
    #     return res
    
    def append_item(self, meta_data: Dict):
        # breakpoint()
        # print(f"DEBUG: append_item被调用，meta_data idx: {meta_data.get('idx', 'N/A')}")
        if len(self.dynamic_batch) < self.batch_size:
            # breakpoint()
            candidate_item = DynamicBatchItem(
                max_rounds=self.max_rounds,
                current_round=0,
                meta_data=meta_data,
                status="pending"
            )
            # print(f"DEBUG: 开始生成conversation")
            candidate_item.conversation = self.generate_conversation_fn(
                text = meta_data["text"], 
                image = meta_data["image"],
                role = "user"
            )
            # print(f"DEBUG: conversation生成成功")
            self.dynamic_batch.append(candidate_item)
            # print(f"DEBUG: 成功添加到dynamic_batch，当前长度: {len(self.dynamic_batch)}")
        else:
            raise ValueError("Batch is full")
    
    
    def append_item_to_full(self, dataloader, progress_bar=None):
        print(f"DEBUG: append_item_to_full开始，当前batch大小: {len(self.dynamic_batch)}, batch_size限制: {self.batch_size}")
        items_added = 0
        while len(self.dynamic_batch) < self.batch_size:
            try:
                data_item = next(dataloader)
                # print(f"DEBUG: 成功从dataloader获取数据项，idx: {data_item.get('idx', 'N/A')}")
                self.append_item(data_item)
                items_added += 1
                # print(f"DEBUG: 成功添加数据项，当前batch大小: {len(self.dynamic_batch)}")
                if progress_bar:
                    progress_bar.update(1)
            except StopIteration as e:
                print(f"DEBUG: dataloader迭代完毕，共添加了{items_added}个数据项")
                break
            except Exception as e:
                print(f"DEBUG: append_item_to_full出现异常: {e}")
                break
        print(f"DEBUG: append_item_to_full完成，最终batch大小: {len(self.dynamic_batch)}")
        
    

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
            
            # 添加状态变化日志
            if old_status != item.status or old_round != item.current_round:
                print(f"DEBUG: 项目 {item.meta_data.get('idx', 'N/A')} 状态更新: {old_status}({old_round}) -> {item.status}({item.current_round}), has_response_tag={has_response_tag}, if_use_tool={self.if_use_tool}")
        
    
     