# rl/path_navigation_stage2.py
import os
import json
import random
import time
import re
import argparse
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from tool_server.utils.utils import *
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from frozen_lake.data_curation.sft_data_curation.prompts import PATH_FINDING_TASK_INSTRUCTION_SHORT, PATH_FIDING_REASONING_REPHRASE_FINAL_PROMPT
from copy import deepcopy
import gymnasium as gym

def write_to_txt(text, file_path):
    """将文本写入文件"""
    try:
        with open(file_path, 'w') as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False

def check_image_exists(image_path):
    """验证图像文件是否存在"""
    return os.path.isfile(image_path) and os.path.getsize(image_path) > 0

def extract_think_tags(response_text):
    """
    从Gemini响应中提取所有<think>标签之间的内容
    """
    think_blocks = []
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, response_text, re.DOTALL)
    return matches

def convert_to_gym_map(item):
    """将数据项转换为gym环境可用的地图格式"""
    # 首先从text_map中提取文本地图
    if "text_map" in item and "output" in item["text_map"] and "text_map" in item["text_map"]["output"]:
        text_map_str = item["text_map"]["output"]["text_map"]
        # 解析文本地图
        rows = []
        for line in text_map_str.split('\n'):
            if '|' in line and ('Row' in line or 'Col' in line):
                continue  # 跳过表头行
            
            row_cells = []
            cells = line.split('|')
            for cell in cells[1:]:  # 跳过第一个空元素
                if not cell.strip():
                    continue
                
                cell_value = cell.strip()
                if cell_value == '_':  # 空格表示安全区域
                    row_cells.append('F')
                elif cell_value == '#':  # '#' 表示冰洞
                    row_cells.append('H')
                elif cell_value == '@':  # '@' 表示起点
                    row_cells.append('S')
                elif cell_value == '*':  # '*' 表示终点
                    row_cells.append('G')
            
            if row_cells:  # 如果行不为空
                rows.append(row_cells)
        
        # 检查地图是否有效
        if rows and all(len(row) == len(rows[0]) for row in rows):
            return rows
    
    # 如果无法从text_map中提取，则从坐标信息中构建
    size = item["size"]
    cell_size = 64  # 假设每个单元格是64像素
    
    # 创建一个全是安全区域的地图
    map_data = [['F' for _ in range(size)] for _ in range(size)]
    
    # 设置起点
    start_x = int(item["start_coords"][0] / cell_size)
    start_y = int(item["start_coords"][1] / cell_size)
    map_data[start_y][start_x] = 'S'
    
    # 设置终点
    goal_x = int(item["goal_coords"][0] / cell_size)
    goal_y = int(item["goal_coords"][1] / cell_size)
    map_data[goal_y][goal_x] = 'G'
    
    # 设置障碍物
    for obs in item["obstacle_coords"]:
        obs_x = int(obs[0] / cell_size)
        obs_y = int(obs[1] / cell_size)
        # 确保坐标在有效范围内
        if 0 <= obs_y < size and 0 <= obs_x < size:
            map_data[obs_y][obs_x] = 'H'
    
    return map_data

def verify_path(gym_map, path_string, verbose=False):
    """
    验证路径是否安全且能到达目标
    """
    try:
        # 创建环境
        env = gym.make('FrozenLake-v1', desc=gym_map, render_mode="ansi" if verbose else None, is_slippery=False)
        obs, _ = env.reset()
        
        # 字典，将方向映射到FrozenLake预期的动作
        direction_to_action = {
            'L': 0,  # LEFT
            'D': 1,  # DOWN
            'R': 2,  # RIGHT
            'U': 3,  # UP
        }
        
        # 解析路径字符串
        directions = path_string.split(',')
        
        # 跟随路径，检查是否到达目标
        is_safe = True
        goal_reached = False
        step_count = 0
        
        for direction in directions:
            direction = direction.strip()  # 移除可能的空格
            action = direction_to_action.get(direction)
            if action is None:
                if verbose:
                    print(f"无效的方向: {direction}")
                is_safe = False
                break
                
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            if verbose:
                print(f"Step {step_count}: {direction}, reward={reward}, terminated={terminated}")
                print(env.render())
            
            # 如果reward是1，说明到达了目标
            if reward == 1.0:
                goal_reached = True
                if verbose:
                    print(f"成功到达目标，用了 {step_count} 步")
                break
            
            # 如果terminated且reward不是1，说明掉进了冰洞
            if terminated and reward != 1.0:
                is_safe = False
                if verbose:
                    print(f"在第 {step_count} 步掉进了冰洞")
                break
        
        env.close()
        
        # 路径有效：安全且到达目标
        is_valid = is_safe and goal_reached
        
        return is_valid, is_safe, goal_reached
    
    except Exception as e:
        print(f"验证路径时出错: {e}")
        return False, False, False

def generate_wrong_path_perturbation(correct_path, map_size=4):
    """生成错误路径的扰动版本"""
    directions = correct_path.split(',')
    perturbed = directions.copy()
    
    # 随机选择一个位置进行扰动
    if len(perturbed) > 0:
        pos = random.randint(0, len(perturbed) - 1)
        current_dir = perturbed[pos]
        
        # 选择一个不同的方向
        all_dirs = ['U', 'D', 'L', 'R']
        other_dirs = [d for d in all_dirs if d != current_dir]
        perturbed[pos] = random.choice(other_dirs)
    
    # 有时添加额外的错误步骤
    if random.random() < 0.3:
        extra_dir = random.choice(['U', 'D', 'L', 'R'])
        insert_pos = random.randint(0, len(perturbed))
        perturbed.insert(insert_pos, extra_dir)
    
    return ','.join(perturbed)

def convert_conversation_into_sharegpt(conversation, image_dir, item_id):
    """
    将对话转换为ShareGPT格式
    """
    # 基础system message，包含tool prompts
    tool_manager = ToolManager(["Point", "Draw2DPath"])
    TOOL_PROMPTS = tool_manager.get_tool_prompt(prompt_type="one_tool_call")


    first_round = {
        "from": "system",
        "value": TOOL_PROMPTS
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
                text = text.replace("<image>", "")
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
class GenerationArgs:
    """生成参数"""
    input_path: str
    output_path: str
    model_name: str = "gemini-2.5-flash"
    max_samples_a: int = 250
    max_samples_b: int = 200
    max_samples_c: int = 100
    max_retry: int = 3
    retry_interval: int = 5
    image_dir: str = "./frozen_lake_metadata"
    seed: int = 42

class PathNavigationStage2Generator:
    def __init__(self, args: GenerationArgs):
        """初始化数据生成器"""
        self.args = args
        self.model_name = args.model_name
        self.dataset_path = args.input_path
        self.output_path = args.output_path
        self.max_retry = args.max_retry
        self.retry_interval = args.retry_interval
        self.image_dir = args.image_dir
        
        # 设置随机种子
        random.seed(args.seed)
        
        # 初始化Gemini客户端
        self.api_key = os.environ.get("GEMINI_API_KEY")
        assert self.api_key, "GEMINI_API_KEY环境变量未设置"
        self.client = genai.Client(api_key=self.api_key)
        
        # 输出文件
        self.output_file_a = os.path.join(self.output_path, "navigation_stage2_data_a.jsonl")
        self.output_file_b = os.path.join(self.output_path, "navigation_stage2_data_b.jsonl")
        self.output_file_c = os.path.join(self.output_path, "navigation_stage2_data_c.jsonl")
        
        # 加载数据集
        self.load_data()
        
        # 创建输出目录
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """加载Stage1生成的数据"""
        print(f"从 {self.dataset_path} 加载Stage1数据...")
        
        # 读取JSONL文件
        self.dataset = process_jsonl(self.dataset_path)
        print(f"加载完成，共 {len(self.dataset)} 条记录")
        
        # 分离valid和invalid数据
        self.valid_data = [item for item in self.dataset if item.get("is_valid", False)]
        self.invalid_data = [item for item in self.dataset if not item.get("is_valid", False)]
        
        print(f"有效数据: {len(self.valid_data)} 条")
        print(f"无效数据: {len(self.invalid_data)} 条")
        
        # 随机打乱数据
        random.shuffle(self.valid_data)
        random.shuffle(self.invalid_data)
        random.shuffle(self.dataset)

    def build_base_conversation(self, item):
        """构建前三轮的基础对话（获取起点、终点、冰洞）"""
        conversation = []
        
        # 第一轮：用户问题
        conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": PATH_FINDING_TASK_INSTRUCTION_SHORT
                },
                {"type": "image", "image": item["image_path"]}
            ]
        })
        
        # 第二轮：识别Elf
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": "<think>[**You need to implement this** Example: I need to identify the starting position in this frozen lake grid. Looking at the image, I can see there's an elf character that represents where I need to start my journey from.]</think>\n\n<tool_call>\n" + json.dumps({
                        "name": "Point",
                        "parameters": {
                            "image": "img_1",
                            "description": "Elf"
                        }
                    }, indent=2) + "\n</tool_call>"
                }
            ]
        })
        
        # 第三轮：Elf结果
        elf_output = item["point_tools"]["elf"]["output"]
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(elf_output, indent=2)},
                {"type": "image", "image": item["point_tools"]["elf"]["image_path"]}
            ]
        })
        
        # 第四轮：识别Gift
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": "<think>[**You need to implement this** Example: Now I have the starting position. Next, I need to find the destination - the gift that the elf needs to reach. I should locate where the gift is positioned on the grid.]</think>\n\n<tool_call>\n" + json.dumps({
                        "name": "Point",
                        "parameters": {
                            "image": "img_1",
                            "description": "Gift"
                        }
                    }, indent=2) + "\n</tool_call>"
                }
            ]
        })
        
        # 第五轮：Gift结果
        gift_output = item["point_tools"]["gift"]["output"]
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(gift_output, indent=2)},
                {"type": "image", "image": item["point_tools"]["gift"]["image_path"]}
            ]
        })
        
        # 第六轮：识别Ice Holes
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": "<think>[**You need to implement this** Example: I have both the start and end positions. Now I need to identify all the dangerous ice holes that I must avoid when planning my path. These holes will be obstacles that could end the journey if stepped on.]</think>\n\n<tool_call>\n" + json.dumps({
                        "name": "Point",
                        "parameters": {
                            "image": "img_1",
                            "description": "Ice Holes"
                        }
                    }, indent=2) + "\n</tool_call>"
                }
            ]
        })
        
        # 第七轮：Ice Holes结果
        if "ice_holes" in item["point_tools"]:
            ice_holes_output = item["point_tools"]["ice_holes"]["output"]
            ice_holes_image = item["point_tools"]["ice_holes"]["image_path"]
        else:
            ice_holes_output = {
                "tool_response_from": "Point",
                "status": "success",
                "points": [],
                "image_dimensions_pixels": {"width": 384, "height": 384},
                "error_code": 0
            }
            ice_holes_image = item["image_path"]
            
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(ice_holes_output, indent=2)},
                {"type": "image", "image": ice_holes_image}
            ]
        })
        
        return conversation
    
    def build_conversation_type_a(self, item):
        """构建类型A的对话：直接答对，然后验证发现错误"""
        conversation = self.build_base_conversation(item)
        
        # 获取正确路径
        correct_path = item["optimal_path"]
        start_coords = item["start_coords"]
        goal_coords = item["goal_coords"]
        obstacle_coords = item["obstacle_coords"]
        
        optimal_image_path = item["path_drawings"]["astar"]["image_path"]
        
        # 第八轮：给出正确答案并调用Draw2DPath验证
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[**You need to implement this** Example: Now I have all the coordinates. Let me analyze the positions: Elf at {start_coords}, Gift at {goal_coords}, and ice holes scattered at {obstacle_coords}. I need to find a safe path that avoids all ice holes.\n\nBased on the analysis, I believe the safe path from the elf to the gift is: {correct_path}\n\nLet me verify this path by drawing it:]</think>\n\n<tool_call>\n" + json.dumps({
                        "name": "Draw2DPath",
                        "parameters": {
                            "image": "img_1",
                            "start_point": start_coords,
                            "directions": correct_path,
                            "pixel_coordinate": True,
                        }
                    }, indent=2) + "\n</tool_call>"
                }
            ]
        })
        
        # 第九轮：Draw2DPath结果（假设总是成功）
        draw_output = {
            "tool_response_from": "Draw2DPath",
            "status": "success",
            "message": "Path drawn successfully",
            "execution_time": 0.003,
            "error_code": 0
        }
        
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(draw_output, indent=2)},
                {"type": "image", "image": f"{optimal_image_path}"}
            ]
        })
        
       # 第十轮：确认这个答案是正确的
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[**You need to implement this** Example: Perfect! Looking at the drawn path, I can see that {correct_path} successfully navigates from the elf to the gift while avoiding all the ice holes. The visualization confirms this is a safe and optimal route.]</think>\n\nAfter drawing and examining the path, I can confirm that the path {correct_path} successfully reaches the destination while avoiding all ice holes.\n\n\\boxed{{{correct_path}}}"
                }
            ]
        })
        
        return conversation

    def build_conversation_type_b(self, item):
        """构建类型B的对话：错1轮，然后纠正"""
        conversation = self.build_base_conversation(item)
        
        # 获取正确路径和错误路径
        correct_path = item["optimal_path"]
        wrong_path = item["generated_path"]  # Stage1生成的错误路径
        start_coords = item["start_coords"]
        
        # 第八轮：给出错误答案并验证
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[You need to implement this Example: Looking at the coordinates, I think I can go from the elf position through what seems like a clear path. Let me try this route that appears to avoid the obvious ice holes.]</think>\n\nBased on my analysis of the positions, I think the safe path is: {wrong_path}\n\nLet me verify this path:\n\n<tool_call>\n" + json.dumps({
                        "name": "Draw2DPath",
                        "parameters": {
                            "image": "img_1",
                            "start_point": start_coords,
                            "directions": wrong_path,
                            "pixel_coordinate": True,
                        }
                    }, indent=2) + "\n</tool_call>"
                }
            ]
        })
        
        # 第九轮：Draw2DPath结果
        draw_output = {
            "tool_response_from": "Draw2DPath",
            "status": "success",
            "message": "Path drawn successfully",
            "execution_time": 0.003,
            "error_code": 0
        }
        
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(draw_output, indent=2)},
                {"type": "image", "image": f"{item['image_path'].replace('.png', '_stage2_wrong.png')}"}
            ]
        })
        
        # 第十轮：发现错误，重新思考
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[You need to implement this Example: I can see from the visualization that my path {wrong_path} intersects with an ice hole. I need to be more careful and find an alternative route that completely avoids all the detected ice holes.]</think>\n\nI can see that the path {wrong_path} leads to an ice hole or doesn't reach the destination. Let me reconsider and find the correct path: {correct_path}\n\n<tool_call>\n" + json.dumps({
                        "name": "Draw2DPath",
                        "parameters": {
                            "image": "img_1",
                            "start_point": start_coords,
                            "directions": correct_path,
                            "pixel_coordinate": True,
                        }
                    }, indent=2) + "\n</tool_call>"
                }
            ]
        })
        
        # 第十一轮：正确路径的Draw2DPath结果
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(draw_output, indent=2)},
                {"type": "image", "image": f"{item['image_path'].replace('.png', '_stage2_correct.png')}"}
            ]
        })
        
        # 第十二轮：确认正确答案
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[You need to implement this Example: Excellent! The green path clearly shows {correct_path} successfully navigates from the elf to the gift while avoiding all ice holes. This path is safe and optimal.]</think>\n\nPerfect! Now I can see that the path {correct_path} safely leads from the elf to the gift without hitting any ice holes.\n\n\\boxed{{{correct_path}}}"
                }
            ]
        })
        
        return conversation

    def build_conversation_type_c(self, item):
        """构建类型C的对话：错2轮，然后纠正"""
        conversation = self.build_base_conversation(item)
        
        # 获取路径
        correct_path = item["optimal_path"]
        wrong_path1 = item["generated_path"]  # Stage1生成的错误路径
        wrong_path2 = generate_wrong_path_perturbation(wrong_path1, item["size"])  # 扰动版本
        start_coords = item["start_coords"]
        
        # 第八轮：第一次错误尝试
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[You need to implement this Example: Let me plan a route from the elf to the gift. Looking at the positions, I think I can navigate through what appears to be open spaces to reach the destination.]</think>\n\nLet me analyze the path. I think the route should be: {wrong_path1}\n\n<tool_call>\n" + json.dumps({
                        "name": "Draw2DPath",
                        "parameters": {
                            "image": "img_1",
                            "start_point": start_coords,
                            "directions": wrong_path1,
                            "pixel_coordinate": True,
                        }
                    }, indent=2) + "\n</tool_call>"
                }
            ]
        })
        
        # 第九轮：第一次Draw2DPath结果
        draw_output = {
            "tool_response_from": "Draw2DPath",
            "status": "success",
            "message": "Path drawn successfully",
            "execution_time": 0.003,
            "error_code": 0
        }
        
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(draw_output, indent=2)},
                {"type": "image", "image": f"{item['image_path'].replace('.png', '_stage2_wrong1.png')}"}
            ]
        })
        
        # 第十轮：发现第一次错误，尝试第二次
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[You need to implement this Example: That path clearly hits an ice hole. Let me try a different approach, maybe going around the obstacles in a different direction to avoid the dangerous areas.]</think>\n\nI see that path {wrong_path1} doesn't work. Let me try a different approach: {wrong_path2}\n\n<tool_call>\n" + json.dumps({
                        "name": "Draw2DPath",
                        "parameters": {
                            "image": "img_1",
                            "start_point": start_coords,
                            "directions": wrong_path2,
                            "pixel_coordinate": True,
                        }
                    }, indent=2) + "\n</tool_call>"
                }
            ]
        })
        
        # 第十一轮：第二次Draw2DPath结果
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(draw_output, indent=2)},
                {"type": "image", "image": f"{item['image_path'].replace('.png', '_stage2_wrong2.png')}"}
            ]
        })
        
        # 第十二轮：发现第二次也错误，给出正确答案
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[You need to implement this Example: Both attempts failed. I need to be more systematic and carefully analyze each step to ensure I'm avoiding all ice holes while finding the most direct safe route to the goal.]</think>\n\nBoth previous attempts failed. Let me carefully reconsider the optimal path: {correct_path}\n\n<tool_call>\n" + json.dumps({
                        "name": "Draw2DPath",
                        "parameters": {
                            "image": "img_1",
                            "start_point": start_coords,
                            "directions": correct_path,
                            "pixel_coordinate": True,
                        }
                    }, indent=2) + "\n</tool_call>"
                }
            ]
        })
        
        # 第十三轮：正确路径的Draw2DPath结果
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(draw_output, indent=2)},
                {"type": "image", "image": f"{item['image_path'].replace('.png', '_stage2_final.png')}"}
            ]
        })
        
        # 第十四轮：确认正确答案
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[You need to implement this Example: Perfect! This green path shows that {correct_path} successfully navigates from the elf to the gift while completely avoiding all ice holes. This is the optimal safe route.]</think>\n\nExcellent! The path {correct_path} successfully avoids all ice holes and reaches the gift.\n\n\\boxed{{{correct_path}}}"
                }
            ]
        })
        
        return conversation

    def call_gemini_for_reasoning(self, conversation, item_id, data_type):
        """调用Gemini API改写思考内容"""
        start_time = time.time()
        
        # 提取需要改写的思考部分
        think_placeholders = []
        for message in conversation:
            if message["role"] == "assistant":
                for content_item in message["content"]:
                    if content_item["type"] == "text":
                        text = content_item["text"]
                        # 查找所有PLACEHOLDER_THINK_X
                        import re
                        placeholders = re.findall(r'\[PLACEHOLDER_THINK_\d+\]', text)
                        think_placeholders.extend(placeholders)
        
        # 构建改写提示
        system_prompt =  PATH_FIDING_REASONING_REPHRASE_FINAL_PROMPT

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=system_prompt)]
            )
        ]
        
        for attempt in range(self.max_retry):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=-1),
                        response_mime_type="text/plain",
                    )
                )
                response_text = response.text
                
                # 尝试解析JSON响应
                try:
                    # 提取JSON部分
                    if "{" in response_text and "}" in response_text:
                        json_start = response_text.find("{")
                        json_end = response_text.rfind("}") + 1
                        json_str = response_text[json_start:json_end]
                        think_replacements = json.loads(json_str)
                        
                        # 替换对话中的占位符
                        updated_conversation = deepcopy(conversation)
                        for message in updated_conversation:
                            if message["role"] == "assistant":
                                for content_item in message["content"]:
                                    if content_item["type"] == "text":
                                        text = content_item["text"]
                                        for placeholder, replacement in think_replacements.items():
                                            text = text.replace(f"[{placeholder}]", replacement)
                                        content_item["text"] = text
                        
                        consumed_time = time.time() - start_time
                        print(f"Gemini API 调用耗时: {consumed_time:.2f}秒")
                        
                        return updated_conversation
                    else:
                        print(f"响应中没有找到有效的JSON格式")
                        continue
                        
                except json.JSONDecodeError as e:
                    print(f"JSON解析失败: {e}")
                    continue
                
            except Exception as e:
                print(f"API调用失败: {e}")
                time.sleep(self.retry_interval)
                continue
        
        print(f"API调用超过最大重试次数，返回原始对话")
        return conversation

    def generate_dataset(self):
        """生成三种类型的数据集"""
        # 准备数据
        data_a_source = self.dataset[:self.args.max_samples_a]  # 任意250个
        data_b_source = [item for item in self.invalid_data if not item.get("is_valid", False)][:self.args.max_samples_b]  # 200个invalid
        data_c_source = [item for item in self.invalid_data if not item.get("is_valid", False)][self.args.max_samples_b:self.args.max_samples_b+self.args.max_samples_c]  # 100个invalid
        
        print(f"准备生成数据集:")
        print(f"类型A: {len(data_a_source)} 个样本")
        print(f"类型B: {len(data_b_source)} 个样本") 
        print(f"类型C: {len(data_c_source)} 个样本")
        
        # 生成类型A数据
        print("生成类型A数据...")
        completed_a = []
        for item in tqdm(data_a_source, desc="类型A"):
            try:
                conversation = self.build_conversation_type_a(item)
                updated_conversation = self.call_gemini_for_reasoning(conversation, item["id"], "A")
                
                result = {
                    "id": f"{item['id']}_stage2_a",
                    "original_item": item,
                    "conversation": updated_conversation,
                    "data_type": "A",
                    "sharegpt_instance": convert_conversation_into_sharegpt(updated_conversation, self.image_dir, f"{item['id']}_stage2_a")
                }
                
                completed_a.append(result)
                append_jsonl(result, self.output_file_a)
                
            except Exception as e:
                print(f"处理类型A数据 {item['id']} 时出错: {e}")
        
        # 生成类型B数据
        print("生成类型B数据...")
        completed_b = []
        for item in tqdm(data_b_source, desc="类型B"):
            try:
                conversation = self.build_conversation_type_b(item)
                updated_conversation = self.call_gemini_for_reasoning(conversation, item["id"], "B")
                
                result = {
                    "id": f"{item['id']}_stage2_b",
                    "original_item": item,
                    "conversation": updated_conversation,
                    "data_type": "B",
                    "sharegpt_instance": convert_conversation_into_sharegpt(updated_conversation, self.image_dir, f"{item['id']}_stage2_b")
                }
                
                completed_b.append(result)
                append_jsonl(result, self.output_file_b)
                
            except Exception as e:
                print(f"处理类型B数据 {item['id']} 时出错: {e}")
        
        # 生成类型C数据
        print("生成类型C数据...")
        completed_c = []
        for item in tqdm(data_c_source, desc="类型C"):
            try:
                conversation = self.build_conversation_type_c(item)
                updated_conversation = self.call_gemini_for_reasoning(conversation, item["id"], "C")
                
                result = {
                    "id": f"{item['id']}_stage2_c",
                    "original_item": item,
                    "conversation": updated_conversation,
                    "data_type": "C",
                    "sharegpt_instance": convert_conversation_into_sharegpt(updated_conversation, self.image_dir, f"{item['id']}_stage2_c")
                }
                
                completed_c.append(result)
                append_jsonl(result, self.output_file_c)
                
            except Exception as e:
                print(f"处理类型C数据 {item['id']} 时出错: {e}")
        
        # 生成综合ShareGPT数据
        all_sharegpt = []
        for result in completed_a + completed_b + completed_c:
            if "sharegpt_instance" in result:
                all_sharegpt.append(result["sharegpt_instance"])
        
        sharegpt_output = os.path.join(self.output_path, "navigation_stage2_sharegpt_all.jsonl")
        write_json_file(all_sharegpt, sharegpt_output)
        
        print(f"数据集生成完成:")
        print(f"类型A: {len(completed_a)} 个样本 -> {self.output_file_a}")
        print(f"类型B: {len(completed_b)} 个样本 -> {self.output_file_b}")
        print(f"类型C: {len(completed_c)} 个样本 -> {self.output_file_c}")
        print(f"合并ShareGPT: {len(all_sharegpt)} 个样本 -> {sharegpt_output}")

def main():
    parser = argparse.ArgumentParser(description='生成FrozenLake路径导航任务Stage2数据集')
    parser.add_argument('--input_path', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/path_navigation/path_navigation_data.jsonl", 
                        help='Stage1输出数据路径')
    parser.add_argument('--output_path', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/path_navigation_stage2", 
                        help='Stage2输出目录')
    parser.add_argument('--model', type=str, default="gemini-2.5-flash", 
                        help='Gemini模型名称')
    parser.add_argument('--max_samples_a', type=int, default=250, 
                        help='类型A最大样本数')
    parser.add_argument('--max_samples_b', type=int, default=200, 
                        help='类型B最大样本数')
    parser.add_argument('--max_samples_c', type=int, default=100, 
                        help='类型C最大样本数')
    parser.add_argument('--image_dir', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation", 
                        help='图像目录')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 创建生成器参数
    gen_args = GenerationArgs(
        input_path=args.input_path,
        output_path=args.output_path,
        model_name=args.model,
        max_samples_a=args.max_samples_a,
        max_samples_b=args.max_samples_b,
        max_samples_c=args.max_samples_c,
        image_dir=args.image_dir,
        seed=args.seed
    )
    
    # 创建生成器并生成数据集
    generator = PathNavigationStage2Generator(gen_args)
    generator.generate_dataset()

if __name__ == "__main__":
    setup_openai_proxy()  # 如果需要代理
    main()