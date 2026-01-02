# gemini_point_random.py
import os
import json
import uuid
import argparse
import random
import base64
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from tool_server.utils.utils import pil_to_base64,base64_to_pil,setup_openai_proxy, append_jsonl
from copy import deepcopy


def write_to_txt(text, file_path):
    try:
        with open(file_path, 'w') as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False
    
# Import prompt templates
from prompts import PATH_FINDING_FINAL_PROMPT, PATH_VERIFY_FINAL_PROMPT, PATH_FINDING_TASK_INSTRUCTION, PATH_VERIFY_TASK_INSTRUCTION, TOOL_PROMPTS
import time
from fewshot import (
    PATH_FINDING_TASK_FS1_Q, 
    PATH_FINDING_TASK_FS1_A, 
    PATH_VERIFY_TASK_FS1_Q, 
    PATH_VERIFY_TASK_FS1_A
)

# Setup environment variables and constants
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
assert GEMINI_API_KEY, "GEMINI_API_KEY environment variable not set"

# Define task types
TASK_TYPES = {
    "path_finding": "Find the optimal path from start to goal",
    "path_validation": "Validate if the given path is valid"
}


def check_image_exists(image_path):
    """验证图像文件是否存在"""
    return os.path.isfile(image_path) and os.path.getsize(image_path) > 0


def is_valid_json(json_str):
    """检查字符串是否是有效的JSON格式"""
    try:
        json_obj = json.loads(json_str)
        return True, json_obj
    except json.JSONDecodeError:
        return False, None

def validate_conversation(conversation,image_dir):
    """
    验证对话格式是否合法
    检查项目:
    1. 对话包含用户和助手角色
    2. 对话内容存在
    3. 对话格式正确
    4. 确保图片路径存在
    """
    # 检查对话基本结构
    if not isinstance(conversation, list):
        return False, "Conversation is not a list"
    
    if len(conversation) < 2:  # 至少需要用户和助手的对话
        return False, "Conversation too short"
    
    # 验证每个消息
    for message in conversation:
        # 检查必要字段
        if not isinstance(message, dict):
            return False, "Message is not a dictionary"
            
        if 'role' not in message:
            return False, "Message missing 'role'"
            
        if 'content' not in message:
            return False, "Message missing 'content'"
        
        # 检查角色是否有效
        if message['role'] not in ['user', 'assistant']:
            return False, f"Invalid role: {message['role']}"
        
        # 检查内容格式
        if not isinstance(message['content'], list):
            return False, "Message content is not a list"
            
        for content_item in message['content']:
            if not isinstance(content_item, dict):
                return False, "Content item is not a dictionary"
                
            if 'type' not in content_item:
                return False, "Content item missing 'type'"
                
            # 检查内容类型
            if content_item['type'] == 'text':
                if 'text' not in content_item:
                    return False, "Text content missing 'text' field"
            
            elif content_item['type'] == 'image':
                if 'image' not in content_item:
                    return False, "Image content missing 'image' field"
                    
                # 检查图片路径是否存在
                image_path = os.path.join(image_dir, content_item['image'])
                if not check_image_exists(image_path):
                    return False, f"Image does not exist: {image_path}"
            
            else:
                return False, f"Invalid content type: {content_item['type']}"
    try:
        last_message = conversation[-1]["content"]
        last_message_str = [item["text"] for item in last_message if item["type"] == "text"]
        last_message_str = "".join(last_message_str).strip()
        if "\\boxed{" not in last_message_str:
            return False, "Last message does not contain '\\boxed{'"
    except:
        return False, "Last message format is incorrect"
    
    return True, "Valid conversation"

def convert_conversation_into_sharegpt(conversation,image_dir,item_id):
    """
    将对话转换为ShareGPT格式
    Args:
        conversation: 对话内容列表
    Returns:
        ShareGPT格式的对话
    """
    first_round = {
        "from":"system",
        "value":TOOL_PROMPTS
    }
    
    sharegpt_conversation = [deepcopy(first_round)]
    image_list = []
    
    for message in conversation:
        role = "human" if message['role'] == 'user' else "gpt"
        content = message['content']
        new_content_str = ""
        
        for content_item in content:
            if content_item['type'] == 'text':
                text = content_item['text']
                text.replace("<image>","")
                new_content_str += text
                
            elif content_item['type'] == 'image':
                image_path = content_item['image']
                new_content_str += "<image>\n"
                real_image_path = os.path.join(image_dir, image_path)
                assert check_image_exists(real_image_path), f"Image file does not exist: {real_image_path}"
                image_list.append(real_image_path)
            else:
                raise ValueError(f"Unsupported content type: {content_item['type']}")
        
        new_message = {
            "from": role,
            "value": new_content_str.strip()
        }
        sharegpt_conversation.append(new_message)
    
    res = {
        "qid": item_id,
        "conversations": sharegpt_conversation,
        "images": image_list
    }
    return res

@dataclass
class DatasetItem:
    """Dataset item"""
    id: str
    image_path: str
    task_type: str
    env_description: str
    input_message: str
    output_message: Optional[str] = None
    additional_data: Optional[Dict] = None

@dataclass
class GenerationArgs:
    """Generation parameters"""
    input_path: str
    output_path: str
    model_name: str = "gemini-2.5-flash"
    tasks_per_image: int = 2
    max_samples: int = 100
    use_cached: bool = True
    valid_path_ratio: float = 0.5  # Ratio of valid paths in path validation tasks
    max_retry: int = 3  # Maximum retries for API calls
    retry_interval: int = 5  # Interval between retries in seconds
    image_dir: str = "/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation"

class GeminiDataGenerator:
    def __init__(self, args: GenerationArgs):
        """Initialize data generator"""
        self.args = args
        self.model_name = args.model_name
        self.dataset_path = args.input_path
        self.output_path = args.output_path
        self.use_cached = args.use_cached
        self.valid_path_ratio = args.valid_path_ratio
        self.max_retry = args.max_retry
        self.retry_interval = args.retry_interval
        self.image_dir = args.image_dir
        # Initialize Google Gemini API
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Load dataset
        self.load_data()
        
        # Create output directory
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load the original dataset"""
        print(f"Loading dataset from {self.dataset_path}...")
        
        # Read JSONL file
        self.dataset = []
        with open(os.path.join(self.dataset_path, "dataset.jsonl"), 'r') as f:
            for line in f:
                self.dataset.append(json.loads(line))
        
        # Filter out entries without valid A* paths
        self.valid_dataset = []
        for item in self.dataset:
            if item.get("astar_path") and item.get("astar_path").get("path"):
                self.valid_dataset.append(item)
                
        print(f"Loaded {len(self.dataset)} entries, {len(self.valid_dataset)} of which are valid")
        
        # Load processed IDs from existing output files
        self.processed_ids = set()
        if os.path.exists(self.output_path) and self.use_cached:
            for filename in os.listdir(self.output_path):
                if filename.endswith(".jsonl"):
                    with open(os.path.join(self.output_path, filename), 'r') as f:
                        for line in f:
                            data = json.loads(line)
                            self.processed_ids.add(data.get("id", ""))
                            
        print(f"Skipping {len(self.processed_ids)} already processed entries")
    
    def prepare_path_finding_task(self, item):
        """Prepare prompt and related data for path finding task"""
        env_description = item["description"]
        
        
        # Build base context
        context = {
            "id": item["id"] + "_path_finding",
            "image_path": item["image_path"],
            "task_type": "path_finding",
            "additional_data": {
                "start_coords": item["start_coords"],
                "goal_coords": item["goal_coords"],
                "obstacle_coords": item["obstacle_coords"],
                "astar_path": item["astar_path"]["path"] if item["astar_path"].get("path") else "",
                "point_tools": item["point_tools"],
                "text_map": item["text_map"] if "text_map" in item else None
            }
        }
        
        # Build trajectory data
        trajectory_data = self.build_trajectory_data_for_path_finding(item)
        context["trajectory_data"] = trajectory_data
        
        return context
    


    def prepare_path_validation_task(self, item):
        """Prepare prompt and related data for path validation task"""
        env_description = item["description"]
        
        # Decide whether to use valid path or random path
        use_valid_path = random.random() < self.valid_path_ratio
        
        # Get path
        if use_valid_path and item.get("astar_path") and item["astar_path"].get("path"):
            # Use valid A* path
            path = item["astar_path"]["path"]
            is_valid = True
            path_type = "astar"
        elif not use_valid_path and item.get("path_drawings") and item["path_drawings"].get("random") and item["path_drawings"]["random"].get("path"):
            # Use random path
            path = item["path_drawings"]["random"]["path"]
            is_valid = item["path_drawings"]["random"].get("is_valid", False)
            path_type = "random"
        else:
            # Default to using A* path (if available)
            if item.get("astar_path") and item["astar_path"].get("path"):
                path = item["astar_path"]["path"]
                is_valid = True
                path_type = "astar"
            else:
                # Skip this entry if no paths are available
                return None
            
        
        # Build base context
        context = {
            "id": item["id"] + "_path_validation_" + path_type,
            "image_path": item["image_path"],
            "task_type": "path_validation",
            "additional_data": {
                "start_coords": item["start_coords"],
                "path": path,
                "is_valid": is_valid,
                "path_type": path_type
            }
        }
        
        # Build trajectory data
        trajectory_data = self.build_trajectory_data_for_path_validation(item, path, path_type)
        context["trajectory_data"] = trajectory_data
        
        return context
    
    def build_trajectory_data_for_path_finding(self, item):
        """构建路径查找任务的轨迹数据框架，让Gemini补充思考过程"""
        trajectory_data = []
        
        # 1. 添加用户问题
        trajectory_data.append({
            "role": "user",
            "content": [
                {"type": "text", "text": PATH_FINDING_TASK_INSTRUCTION},
                {"type": "image", "image": item["image_path"]}
            ]
        })
        
        # 2. 第一次工具调用 - Point工具定位Elf(起点)
        if "elf" in item["point_tools"]:
            elf_data = item["point_tools"]["elf"]
            
            # 助手思考过程和工具调用 - 让Gemini补充思考
            trajectory_data.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<think>[You need to implement this. Example: First, I need to use the Point tool to locate the Elf (the starting point) in the image. The Elf typically appears as a small sprite or player character in the Frozen Lake environment.]</think>\n\n<tool_call>\n{\"name\": \"Point\", \"parameters\": {\"image\": \"img_1\", \"description\": \"Elf\"}}\n</tool_call>"}
                ]
            })
            
            # 工具返回结果
            tool_response = {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(elf_data["output"], indent=2)},
                ]
            }
            
            # 如果有工具生成的图像，添加到结果中
            if "image_path" in elf_data and elf_data["image_path"]:
                tool_response["content"].append({
                    "type": "image", 
                    "image": elf_data["image_path"]
                })
                    
            trajectory_data.append(tool_response)
        
        # 3. 第二次工具调用 - Point工具定位Gift(终点)
        if "gift" in item["point_tools"]:
            gift_data = item["point_tools"]["gift"]
            
            # 助手思考过程和工具调用 - 让Gemini补充思考
            trajectory_data.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<think>[You need to implement this. Example: Now I need to locate the Gift (the goal) in the image. The Gift should be the endpoint that the Elf needs to reach, typically represented by a gift box or similar icon.]</think>\n\n<tool_call>\n{\"name\": \"Point\", \"parameters\": {\"image\": \"img_1\", \"description\": \"Gift\"}}\n</tool_call>"}
                ]
            })
            
            # 工具返回结果
            tool_response = {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(gift_data["output"], indent=2)},
                ]
            }
            
            # 如果有工具生成的图像，添加到结果中
            if "image_path" in gift_data and gift_data["image_path"]:
                tool_response["content"].append({
                    "type": "image", 
                    "image": gift_data["image_path"]
                })
                    
            trajectory_data.append(tool_response)
        
        # 4. 第三次工具调用 - Point工具定位Ice Holes(冰洞)
        if "ice_holes" in item["point_tools"]:
            holes_data = item["point_tools"]["ice_holes"]
            
            # 助手思考过程和工具调用 - 让Gemini补充思考
            trajectory_data.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<think>[You need to implement this. Example: Next, I need to use the Point tool to locate the obstacles in the image that should be avoided. The obstacles appear to resemble ice holes that would be dangerous for the Elf to step on.]</think>\n\n<tool_call>\n{\"name\": \"Point\", \"parameters\": {\"image\": \"img_1\", \"description\": \"Ice Holes\"}}\n</tool_call>"}
                ]
            })
            
            # 工具返回结果
            tool_response = {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(holes_data["output"], indent=2)},
                ]
            }
            
            # 如果有工具生成的图像，添加到结果中
            if "image_path" in holes_data and holes_data["image_path"]:
                tool_response["content"].append({
                    "type": "image", 
                    "image": holes_data["image_path"]
                })
                    
            trajectory_data.append(tool_response)
        
        # 5. 获取实际坐标（从工具输出或默认坐标）
        start_coords = item["start_coords"]
        goal_coords = item["goal_coords"]
        holes_coords = item["obstacle_coords"]
        
        if "elf" in item["point_tools"] and "output" in item["point_tools"]["elf"] and "points" in item["point_tools"]["elf"]["output"] and len(item["point_tools"]["elf"]["output"]["points"]) > 0:
            point = item["point_tools"]["elf"]["output"]["points"][0]
            start_coords = [point["x"], point["y"]]
            
        if "gift" in item["point_tools"] and "output" in item["point_tools"]["gift"] and "points" in item["point_tools"]["gift"]["output"] and len(item["point_tools"]["gift"]["output"]["points"]) > 0:
            point = item["point_tools"]["gift"]["output"]["points"][0]
            goal_coords = [point["x"], point["y"]]
            
        if "ice_holes" in item["point_tools"] and "output" in item["point_tools"]["ice_holes"] and "points" in item["point_tools"]["ice_holes"]["output"]:
            holes_coords = []
            for point in item["point_tools"]["ice_holes"]["output"]["points"]:
                holes_coords.append([point["x"], point["y"]])
        
        # 6. 最后工具调用 - AStarWithPixelCoordinate找最优路径
        if "astar_path" in item and item["astar_path"]:
            astar_data = item["astar_path"]
            
            # 构建A*调用的参数
            astar_params = {
                "start": start_coords,
                "goal": goal_coords,
                "obstacles": holes_coords
            }
            
            # 助手思考过程和工具调用 - 让Gemini补充思考
            trajectory_data.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"<think>[You need to implement this. Example: Now that I have located the start point ({start_coords}), goal ({goal_coords}), and all obstacles ({holes_coords}), I need to use the A* algorithm to find the optimal path that avoids all ice holes.]</think>\n\n<tool_call>\n{{\"name\": \"AStarWithPixelCoordinate\", \"parameters\": {astar_params}}}\n</tool_call>"}
                ]
            })
            
            # 工具返回结果
            trajectory_data.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(astar_data["output"], indent=2)},
                ]
            })
            
            # 助手最终回答 - 让Gemini补充思考和回答
            astar_path = astar_data["path"] if "path" in astar_data else ""
            trajectory_data.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"<think>[You need to implement this. Example: I've successfully completed all the necessary tool calls. I identified the locations of the Elf (starting point), Gift (goal), and Ice Holes (obstacles). Then I used the A* algorithm to compute the optimal path from start to goal while avoiding all obstacles. Now I can provide the complete path directions.]</think>\n\n<response>[You need to implement this. Example: Based on my analysis, I've found the optimal path from the Elf (start) to the Gift (goal) while avoiding all the dangerous ice holes.\n\nThe best path is: {astar_path}\n\nThis path will safely guide you from the starting point to the goal without falling into any ice holes.\n\n\\boxed{{{astar_path}}}]</response>"}
                ]
            })
        
        return trajectory_data

    def build_trajectory_data_for_path_validation(self, item, path, path_type):
        """构建路径验证任务的轨迹数据框架，让Gemini补充思考过程"""
        trajectory_data = []
        
        # 1. 添加用户问题
        trajectory_data.append({
            "role": "user",
            "content": [
                {"type": "text", "text": PATH_VERIFY_TASK_INSTRUCTION},
                {"type": "image", "image": item["image_path"]}
            ]
        })
        
        # 2. 工具调用 - Draw2DPath绘制路径
        draw_path_data = None
        if path_type == "astar" and "astar" in item.get("path_drawings", {}):
            draw_path_data = item["path_drawings"]["astar"]
        elif path_type == "random" and "random" in item.get("path_drawings", {}):
            draw_path_data = item["path_drawings"]["random"]
        
        if draw_path_data:
            # 获取起点坐标
            start_coords = item["start_coords"]
            if "elf" in item["point_tools"] and "output" in item["point_tools"]["elf"] and "points" in item["point_tools"]["elf"]["output"] and len(item["point_tools"]["elf"]["output"]["points"]) > 0:
                point = item["point_tools"]["elf"]["output"]["points"][0]
                start_coords = [point["x"], point["y"]]
            
            # 构建Draw2DPath调用的参数
            draw_params = {
                "image": "img_1",
                "start_point": start_coords,
                "directions": path,
            }
            
            # 助手思考过程和工具调用 - 让Gemini补充思考
            trajectory_data.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"<think>[**You need to implement this** Example: I need to verify if the given path is valid. First, I'll use the Draw2DPath tool to visualize the path on the map, starting from the Elf's position ({start_coords}) and following the provided directions ({path}). Then I can analyze if this path safely reaches the goal without falling into ice holes.]</think>\n\n<tool_call>\n{{\"name\": \"Draw2DPath\", \"parameters\": {draw_params}}}\n</tool_call>"}
                ]
            })
            
            # 工具返回结果
            tool_response = {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(draw_path_data["output"], indent=2)},
                ]
            }
            
            # 如果有工具生成的图像，添加到结果中
            if "image_path" in draw_path_data and draw_path_data["image_path"]:
                tool_response["content"].append({
                    "type": "image", 
                    "image": draw_path_data["image_path"]
                })
                    
            trajectory_data.append(tool_response)
            
            # 助手最终回答 - 让Gemini补充思考和回答
            is_valid = draw_path_data.get("is_valid", False)
            valid_text = "Yes" if is_valid else "No"
            reason_template = "This path would cause the player to fall into an ice hole or fail to reach the endpoint." if not is_valid else "This path safely leads from the starting point to the endpoint without falling into any ice holes."
            
            trajectory_data.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"<think>[You need to implement this. Example: Now that I've drawn the path on the map, I can analyze whether it's valid. I need to check if the path avoids all ice holes and successfully reaches the goal from the start position. Based on the drawn path visualization, I can determine if this is a valid route.]</think>\n\n<response>[You need to implement this. Example: I've analyzed the path you provided: {path}\n\nAfter drawing this path on the map, I can determine that this path is **{valid_text}**.\n\n{reason_template}\n\n\\boxed{{ {valid_text}}}]</response>"}
                ]
            })
        
        return trajectory_data
    
    def call_gemini_meta(self, system_prompt, context, fs):
        
        for _ in range(self.max_retry):
            try:
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=system_prompt)]
                    ),
                    # types.Content(
                    #     role="user",
                    #     parts=[types.Part.from_text(text=fs[0])]
                    # ),
                    # types.Content(
                    #     role="model",
                    #     parts=[types.Part.from_text(text=fs[1])]
                    # ),
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=f"{json.dumps(context['trajectory_data'], indent=2)}"),
                            # image_part
                        ]
                    )
                ]
                
                # Call API
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        thinking_config = types.ThinkingConfig(
                            thinking_budget=-1,
                        ),
                        response_mime_type="text/plain",
                    )
                )
                response_text = response.text
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                # Add result to context
                item_id = context["id"]
                whether_valid_json,valid_json = is_valid_json(response_text)
                if not whether_valid_json:
                    print(f"Invalid JSON format in item {item_id}")
                    continue
            
                whether_valid_conversation, valid_conversation_message = validate_conversation(valid_json, self.image_dir)
                if not whether_valid_conversation:
                    print(f"Invalid conversation format in item {item_id}: {valid_conversation_message}")
                    continue
                
                sharegpt_item = convert_conversation_into_sharegpt(valid_json, self.image_dir, item_id)
                
                context["output_message"] = response_text
                context["sharegpt_instance"] = sharegpt_item
                return context
            
            except Exception as e:
                print(f"API call failed: {e}")
                time.sleep(self.retry_interval)  # Sleep for retry_interval seconds before trying again
                continue
        
        print(f"API call failed with maximum retries.")
        return None

    def call_gemini_for_path_finding(self, context):
        """Call Gemini API to generate response for path finding task"""
        
        if context["id"] in self.processed_ids:
            print(f"Skipping already processed entry: {context['id']}")
            return None
        start_time = time.time()
        # Prepare API call
        fs = [PATH_FINDING_TASK_FS1_Q, PATH_FINDING_TASK_FS1_A]
        context = self.call_gemini_meta(
            system_prompt=PATH_FINDING_FINAL_PROMPT,
            context=context,
            fs=fs
        )
        consumed_time = time.time() - start_time
        output_file = "./gemini_path_finding_time.jsonl"
        append_jsonl(consumed_time, output_file)
        return context
            
        
    
    def call_gemini_for_path_validation(self, context):
        """Call Gemini API to generate response for path validation task"""
        if context["id"] in self.processed_ids:
            print(f"Skipping already processed entry: {context['id']}")
            return None
        start_time = time.time()
        fs = [PATH_VERIFY_TASK_FS1_Q, PATH_VERIFY_TASK_FS1_A]
        context = self.call_gemini_meta(
            system_prompt=PATH_VERIFY_FINAL_PROMPT,
            context=context,
            fs = fs,
        )
        consumed_time = time.time() - start_time
        output_file = "./gemini_path_verify_time.jsonl"
        append_jsonl(consumed_time, output_file)
        return context
    
    def generate_dataset(self):
        """Generate the entire dataset"""
        # Create output file
        output_file = os.path.join(self.output_path, f"sft_data_t12_new.jsonl")
        
        # Determine sample count
        num_samples = min(len(self.valid_dataset), self.args.max_samples)
        sampled_data = random.sample(self.valid_dataset, num_samples)
        
        # Progress bar
        with tqdm(total=num_samples * self.args.tasks_per_image, desc="Generating SFT data") as pbar:
            for item in sampled_data:
                # For each image, generate specified number of tasks
                task_counter = 0
                
                # Try to generate path finding task
                if task_counter < self.args.tasks_per_image:
                    path_finding_context = self.prepare_path_finding_task(item)
                    result = self.call_gemini_for_path_finding(path_finding_context)
                    
                    # If generation successful, save result
                    if result:
                        with open(output_file, 'a') as f:
                            f.write(json.dumps(result) + "\n")
                        self.processed_ids.add(result["id"])
                        task_counter += 1
                        pbar.update(1)
                
                # Try to generate path validation task
                if task_counter < self.args.tasks_per_image:
                    path_validation_context = self.prepare_path_validation_task(item)
                    
                    # Ensure path is available
                    if path_validation_context:
                        result = self.call_gemini_for_path_validation(path_validation_context)
                        
                        # If generation successful, save result
                        if result:
                            with open(output_file, 'a') as f:
                                f.write(json.dumps(result) + "\n")
                            self.processed_ids.add(result["id"])
                            task_counter += 1
                            pbar.update(1)
        
        print(f"Dataset generation complete, saved to {output_file}")
        
        # Statistics
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                lines = f.readlines()
                task_counts = {}
                for line in lines:
                    data = json.loads(line)
                    task_type = data.get("task_type", "unknown")
                    task_counts[task_type] = task_counts.get(task_type, 0) + 1
                    
                print("Dataset statistics:")
                for task, count in task_counts.items():
                    print(f"  - {task}: {count} entries")

    def extract_message_list(self, output_text):
        """Extract message list from Gemini output"""
        try:
            # Try to extract JSON part
            json_start = output_text.find("```json")
            json_end = output_text.rfind("```")
            
            if json_start >= 0 and json_end > json_start:
                json_text = output_text[json_start + 7:json_end].strip()
                message_list = json.loads(json_text)
                return message_list
            else:
                print("Cannot extract JSON from output")
                return None
        except Exception as e:
            print(f"Error extracting message list: {e}")
            return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate FrozenLake SFT dataset')
    parser.add_argument('--input_path', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_point_dataset", help='Directory containing original dataset')
    parser.add_argument('--output_path', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_point_dataset_sft", help='Directory for output SFT dataset')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash', help='Gemini model name')
    parser.add_argument('--tasks_per_image', type=int, default=2, help='Number of tasks to generate per image')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of samples')
    parser.add_argument('--no_cache', action='store_false', dest='use_cached', 
                        help='Do not use cache, reprocess all data')
    parser.add_argument('--valid_path_ratio', type=float, default=0.5,
                        help='Ratio of valid paths in path validation tasks (0-1)')
    
    args = parser.parse_args()
    
    # Create generator parameters
    gen_args = GenerationArgs(
        input_path=args.input_path,
        output_path=args.output_path,
        model_name=args.model,
        tasks_per_image=args.tasks_per_image,
        max_samples=args.max_samples,
        use_cached=args.use_cached,
        valid_path_ratio=args.valid_path_ratio
    )
    
    # Create generator and generate dataset
    generator = GeminiDataGenerator(gen_args)
    generator.generate_dataset()

if __name__ == "__main__":
    setup_openai_proxy()
    main()