# rl/path_navigation_stage2.py
import os
import json
import random
import time
import re
import argparse
import base64
import tempfile
import traceback
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
import gymnasium as gym
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from tool_server.utils.utils import pil_to_base64, base64_to_pil, setup_openai_proxy, append_jsonl, write_json_file, process_jsonl, load_json_file
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from frozen_lake.data_curation.sft_data_curation.prompts import PATH_FINDING_TASK_INSTRUCTION_SHORT, PATH_FIDING_REASONING_REPHRASE_FINAL_PROMPT
from copy import deepcopy

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
    """从Gemini响应中提取所有<think>标签之间的内容"""
    think_blocks = []
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, response_text, re.DOTALL)
    return matches

def get_image_filename_for_path(item_id, suffix, path_str=None):
    """为不同类型的路径生成图像文件名"""
    base_path = f"./frozen_lake_metadata_v2/images_sft/{item_id}"
    
    if suffix.startswith("_stage2_"):
        return f"{base_path}{suffix}.png"
    else:
        # 对于已有图片，使用常规命名
        return f"{base_path}{suffix}.png"

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
    """验证路径是否安全且能到达目标"""
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

def convert_conversation_into_sharegpt(conversation, image_dir, item_id, tool_manager):
    """将对话转换为ShareGPT格式"""
    # 基础system message，包含tool prompts
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


    
# wrong_path_image = self.draw_path_for_stage2(item, wrong_path, color="red", suffix="_stage2_wrong")
def create_path_drawing(tool_manager, image_path, start_point, directions, color="red", step=64):
    """
    使用实际的Draw2DPath工具创建路径图像
    
    Args:
        tool_manager: 工具管理器实例
        image_path: 原始图像路径
        start_point: 起点坐标 [x, y]
        directions: 方向序列 (如 "U,D,L,R")
        color: 线条颜色
        step: 步长
    
    Returns:
        tuple: (生成的图像路径, 工具输出)
    """
    # 读取原始图像
 
    image_pil = Image.open(image_path).convert("RGB")
    image_data = pil_to_base64(image_pil)
    # 创建临时文件用于保存结果
    tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()
    
    # 创建工具输入
    tool_input = {
        "name": "Draw2DPath",
        "parameters": {
            "image": image_data,
            "start_point": start_point,
            "directions": directions,
            "pixel_coordinate": True,
        }
    }
    
    # 调用工具
    output = tool_manager.call_tool("Draw2DPath", tool_input["parameters"])
    

    image_data = base64.b64decode(output["edited_image"].split(",")[-1])
    with open(tmp_path, 'wb') as f:
        f.write(image_data)
    
    return tmp_path, output

@dataclass
class GenerationArgs:
    """生成参数"""
    stage1_path: str    # Stage1的输出路径
    sft_path: str       # SFT的完整数据路径
    output_path: str    # 输出目录
    model_name: str = "gemini-2.5-flash"
    max_samples_a: int = 250
    max_samples_b: int = 200
    max_samples_c: int = 100
    max_retry: int = 3
    retry_interval: int = 5
    image_dir: str = "./frozen_lake_metadata_v2"
    seed: int = 42

class PathNavigationStage2Generator:
    def __init__(self, args: GenerationArgs):
        """初始化数据生成器"""
        self.args = args
        self.model_name = args.model_name
        self.stage1_path = args.stage1_path
        self.sft_path = args.sft_path
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
        
        # 初始化工具管理器
        self.tool_manager = ToolManager(tools=["Point", "Draw2DPath"])
        
        # 输出文件
        self.output_file_a = os.path.join(self.output_path, "navigation_stage2_data_a.jsonl")
        self.output_file_b = os.path.join(self.output_path, "navigation_stage2_data_b.jsonl")
        self.output_file_c = os.path.join(self.output_path, "navigation_stage2_data_c.jsonl")
        
        self.processed_ids_a = set()
        self.processed_ids_b = set()
        self.processed_ids_c = set()
        
        self.ckpt_a = []
        self.ckpt_b = []
        self.ckpt_c = []
        if os.path.exists(self.output_file_a):
            self.ckpt_a = process_jsonl(self.output_file_a)
            self.processed_ids_a.update(item["id"] for item in self.ckpt_a)
        if os.path.exists(self.output_file_b):
            self.ckpt_b = process_jsonl(self.output_file_b)
            self.processed_ids_b.update(item["id"] for item in self.ckpt_b)
        if os.path.exists(self.output_file_c):
            self.ckpt_c = process_jsonl(self.output_file_c)
            self.processed_ids_c.update(item["id"] for item in self.ckpt_c)
        
        # 创建输出目录
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
        # 加载数据集
        self.load_data()
        
    def load_data(self):
        """加载Stage1生成的数据和SFT完整数据"""
        print(f"从 {self.stage1_path} 加载Stage1数据...")
        self.stage1_data = process_jsonl(self.stage1_path)
        print(f"加载完成，共 {len(self.stage1_data)} 条记录")
        
        print(f"从 {self.sft_path} 加载SFT数据...")
        self.sft_data = process_jsonl(self.sft_path)
        print(f"加载完成，共 {len(self.sft_data)} 条记录")
        
        random.shuffle(self.stage1_data)
        
        # 为SFT数据创建ID索引，方便快速查找
        self.sft_data_dict = {item["id"]: item for item in self.sft_data}
        
        # 从Stage1数据中分离有效和无效数据
        self.valid_data = [item for item in self.stage1_data if item["is_valid"]]
        self.invalid_data = [item for item in self.stage1_data if not item["is_valid"] and len(item["generated_path"]) < 30 and len(item["generated_path"]) > 0]
        
        print(f"有效数据: {len(self.valid_data)} 条")
        print(f"无效数据: {len(self.invalid_data)} 条")
        
        
        
        # 找出每个Stage1数据对应的完整SFT数据项
        for item in self.stage1_data:
            item_id = item["id"]
            if item_id in self.sft_data_dict:
                item["full_data"] = self.sft_data_dict[item_id]
            else:
                raise ValueError(f"ID {item_id} not found in SFT data. Please check your data integrity.")
            
    def draw_path_for_stage2(self, item, path, color="red", suffix="_stage2"):
        """
        为Stage2数据集绘制路径图像
        
        Args:
            item: 数据项
            path: 路径字符串（如 "U,D,L,R"）
            color: 线条颜色
            suffix: 图像文件名后缀
        
        Returns:
            str: 生成的图像的相对路径
        """
        # 获取完整数据项
        full_item = item["full_data"]
        
        # 获取原始图像路径
        original_image_path = os.path.join(self.image_dir, full_item["image_path"])
        
        # 获取起点坐标
        start_coords = full_item["start_coords"]
        
        # 生成输出图像路径
        item_id = item["id"]
        output_image_filename = get_image_filename_for_path(item_id, suffix)
        output_image_path = os.path.join(self.image_dir, output_image_filename)
        
        # # 检查是否已经存在路径图像
        # if os.path.exists(output_image_path):
        #     # 如果已存在，返回相对路径
        #     return output_image_filename
        
        # 如果不存在，调用create_path_drawing创建图像
        tmp_path, _ = create_path_drawing(
            self.tool_manager,
            original_image_path, 
            start_coords, 
            path, 
            color=color,
            step=64
        )
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        
        # 复制临时文件到目标位置
        try:
            with open(tmp_path, 'rb') as src_file, open(output_image_path, 'wb') as dst_file:
                dst_file.write(src_file.read())
            
            # 清理临时文件
            os.unlink(tmp_path)
            
            # 返回相对路径
            return output_image_filename
            
        except Exception as e:
            print(f"保存路径图像时出错: {e}")
            return full_item["image_path"]  # 出错时返回原始图像路径

    def build_base_conversation(self, item):
        """构建前三轮的基础对话（获取起点、终点、冰洞）"""
        # 使用完整数据项
        full_item = item.get("full_data", item)
        conversation = []
        
        # 第一轮：用户问题
        conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": PATH_FINDING_TASK_INSTRUCTION_SHORT
                },
                {"type": "image", "image": full_item["image_path"]}
            ]
        })
        
        # 获取坐标信息，用于构造思考内容
        elf_coords = full_item["start_coords"]
        gift_coords = full_item["goal_coords"]
        obstacle_coords = full_item["obstacle_coords"]
        
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
        elf_output = full_item["point_tools"]["elf"]["output"]
        elf_image_path = full_item["point_tools"]["elf"]["image_path"]
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(elf_output, indent=2)},
                {"type": "image", "image": elf_image_path}
            ]
        })
        
        # 第四轮：识别Gift
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": f"<think>[**You need to implement this** Example: Now I have identified the elf at {elf_coords}. Next, I need to find the destination - the gift that the elf needs to reach. I should locate where the gift is positioned on the grid.]</think>\n\n<tool_call>\n" + json.dumps({
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
        gift_output = full_item["point_tools"]["gift"]["output"]
        gift_image_path = full_item["point_tools"]["gift"]["image_path"]
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(gift_output, indent=2)},
                {"type": "image", "image": gift_image_path}
            ]
        })
        
        # 第六轮：识别Ice Holes
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": f"<think>[**You need to implement this** Example: I have both the start position at {elf_coords} and the goal at {gift_coords}. Now I need to identify all the dangerous ice holes that I must avoid when planning my path. These holes will be obstacles that could end the journey if stepped on.]</think>\n\n<tool_call>\n" + json.dumps({
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
        if "ice_holes" in full_item["point_tools"]:
            ice_holes_output = full_item["point_tools"]["ice_holes"]["output"]
            ice_holes_image_path = full_item["point_tools"]["ice_holes"]["image_path"]
        else:
            ice_holes_output = {
                "tool_response_from": "Point",
                "status": "success",
                "points": [],
                "image_dimensions_pixels": {"width": 384, "height": 384},
                "error_code": 0
            }
            ice_holes_image_path = full_item["image_path"]
            
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(ice_holes_output, indent=2)},
                {"type": "image", "image": ice_holes_image_path}
            ]
        })
        
        return conversation

    def build_conversation_type_a(self, item):
        """构建类型A的对话：直接给出正确答案，然后演示"""
        conversation = self.build_base_conversation(item)
        
        # 获取正确路径
        full_item = item.get("full_data", item)
        correct_path = full_item["astar_path"]["path"]
        start_coords = full_item["start_coords"]
        goal_coords = full_item["goal_coords"]
        obstacle_coords = full_item["obstacle_coords"]
        
        # 使用现有的A*路径图像
        optimal_image_path = full_item["path_drawings"]["astar"]["image_path"]
        
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
        
        # 第九轮：Draw2DPath结果（使用已有的图像）
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
                {"type": "image", "image": optimal_image_path}
            ]
        })
        
        # 第十轮：确认这个答案是正确的
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[**You need to implement this** Example: Perfect! Looking at the drawn path, I can see that {correct_path} successfully navigates from the elf at {start_coords} to the gift at {goal_coords} while avoiding all the ice holes. The visualization confirms this is a safe and optimal route.]</think>\n\n<response>After drawing and examining the path, I can confirm that the path {correct_path} successfully reaches the destination while avoiding all ice holes.\n\n\\boxed{{{correct_path}}}</response>"
                }
            ]
        })
        
        return conversation

    def build_conversation_type_b(self, item):
        """构建类型B的对话：错1轮，然后纠正"""
        conversation = self.build_base_conversation(item)
        
        # 获取正确路径和错误路径
        full_item = item.get("full_data", item)
        correct_path = full_item["astar_path"]["path"]
        wrong_path = item["generated_path"]  # Stage1生成的错误路径
        start_coords = full_item["start_coords"]
        goal_coords = full_item["goal_coords"]
        obstacle_coords = full_item["obstacle_coords"]
        
        # 为错误路径生成图像
        wrong_path_image = self.draw_path_for_stage2(item, wrong_path, color="red", suffix="_stage2_b_wrong")
        
        # 为正确路径生成图像（使用已有的A*路径图像）
        correct_path_image = full_item["path_drawings"]["astar"]["image_path"]
        
        # 第八轮：给出错误答案并验证
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[**You need to implement this** Example: Looking at the coordinates, I think I can go from the elf position at {start_coords} to the gift at {goal_coords} through what seems like a clear path. The ice holes are at {obstacle_coords}, and I think path {wrong_path} should safely navigate around them. Based on my analysis of the positions, I think the safe path is: {wrong_path}\n\nLet me verify this path:]</think>\n\n<tool_call>\n" + json.dumps({
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
        
        # 第九轮：Draw2DPath结果（错误路径）
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
                {"type": "image", "image": wrong_path_image}
            ]
        })
        
        # 第十轮：发现错误，重新思考
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[**You need to implement this** Example: I can see from the visualization that my path {wrong_path} intersects with an ice hole or doesn't reach the destination correctly. Looking at the drawn path, I notice it doesn't provide a safe route from {start_coords} to {goal_coords}. I need to be more careful and find an alternative route. I can see that the path {wrong_path} is not ideal. Let me reconsider and find the correct path: {correct_path}\n\n]</think><tool_call>\n" + json.dumps({
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
                {"type": "image", "image": correct_path_image}
            ]
        })
        
        # 第十二轮：确认正确答案
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[**You need to implement this** Example: Excellent! The green path clearly shows {correct_path} successfully navigates from the elf at {start_coords} to the gift at {goal_coords} while avoiding all ice holes. This new path completely avoids the dangers that my previous attempt missed.]</think>\n\n<response>Perfect! Now I can see that the path {correct_path} safely leads from the elf to the gift without hitting any ice holes.\n\n\\boxed{{{correct_path}}}</response>"
                }
            ]
        })
        
        return conversation

    def build_conversation_type_c(self, item):
        """构建类型C的对话：错2轮，然后纠正"""
        conversation = self.build_base_conversation(item)
        
        # 获取路径
        full_item = item.get("full_data", item)
        correct_path = full_item["astar_path"]["path"]
        wrong_path1 = item["generated_path"]  # Stage1生成的错误路径
        wrong_path2 = generate_wrong_path_perturbation(wrong_path1, item["size"])  # 扰动版本
        start_coords = full_item["start_coords"]
        goal_coords = full_item["goal_coords"]
        obstacle_coords = full_item["obstacle_coords"]
        
        # 为错误路径1生成图像
        wrong_path1_image = self.draw_path_for_stage2(item, wrong_path1, color="red", suffix="_stage2_c_wrong1")
        
        # 为错误路径2生成图像
        wrong_path2_image = self.draw_path_for_stage2(item, wrong_path2, color="red", suffix="_stage2_c_wrong2")
        
        # 为正确路径生成图像（使用已有的A*路径图像）
        correct_path_image = full_item["path_drawings"]["astar"]["image_path"]
        
        # 第八轮：第一次错误尝试
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[**You need to implement this** Example: Let me plan a route from the elf at {start_coords} to the gift at {goal_coords}. Looking at the positions and the ice holes at {obstacle_coords}, I think I can navigate through what appears to be open spaces with path {wrong_path1}. Let me analyze the path. I think the route should be: {wrong_path1}]</think>\n\n<tool_call>\n" + json.dumps({
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
                {"type": "image", "image": wrong_path1_image}
            ]
        })
        
        # 第十轮：发现第一次错误，尝试第二次
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[**You need to implement this** Example: That path {wrong_path1} clearly doesn't work - it either hits an ice hole or doesn't reach the destination. Let me try a different approach, with path {wrong_path2}, going around the obstacles in a different direction to avoid the dangerous areas at {obstacle_coords}. I see that path {wrong_path1} doesn't work well. Let me try a different approach: {wrong_path2}]</think>\n\n<tool_call>\n" + json.dumps({
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
                {"type": "image", "image": wrong_path2_image}
            ]
        })
        
        # 第十二轮：发现第二次也错误，给出正确答案
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[**You need to implement this** Example: Both attempts failed. My path {wrong_path1} and {wrong_path2} don't work. I need to be more systematic and carefully analyze each step to ensure I'm avoiding all ice holes at {obstacle_coords} while finding the most direct safe route from {start_coords} to {goal_coords}. Let me try path {correct_path}. Both previous attempts failed to find the optimal path. Let me carefully reconsider the map and find the safest route: {correct_path}]</think>\n\n<tool_call>\n" + json.dumps({
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
                {"type": "image", "image": correct_path_image}
            ]
        })
        
        # 第十四轮：确认正确答案
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[**You need to implement this** Example: Perfect! This green path shows that {correct_path} successfully navigates from the elf at {start_coords} to the gift at {goal_coords} while completely avoiding all ice holes at {obstacle_coords}. After two failed attempts, I've finally found the optimal safe route.]</think>\n\n<response>Excellent! The path {correct_path} successfully avoids all ice holes and reaches the gift efficiently.\n\n\\boxed{{{correct_path}}}</response>"
                }
            ]
        })
        
        return conversation

    def call_gemini_for_reasoning(self, conversation, item_id, data_type):
        """
        调用Gemini API替换思考示例，并在请求中嵌入路径图片
        
        Args:
            conversation: 包含占位符的对话
            item_id: 项目ID
            data_type: 数据类型(A/B/C)
        
        
        Returns:
            更新后的对话，其中占位符已被替换为实际推理
        """
        start_time = time.time()
        
        # 从对话中提取需要改写的思考部分和相关图片
        think_contents = []
        input_parts = []
        # 分析对话，提取占位符内容和对应的图像
        for i, message in enumerate(conversation):
            text_part = types.Part.from_text(text=f"{message}")
            input_parts.append(text_part)
            
            if message["role"] == "assistant":
                for content_item in message["content"]:
                    if content_item["type"] == "text":
                        text = content_item["text"]
                        if "<think>" in text and "</think>" in text:
                            think_contents.append(text)
            
            # 如果是用户消息，查找是否包含图片
            if message["role"] == "user" and (i==0 or i > 6) :  # 跳过第一条消息
                for content_item in message["content"]:
                    if content_item["type"] == "image":
                        # 找到图像路径
                        image_path = os.path.join(self.image_dir, content_item["image"])
                        image_data = Image.open(image_path).convert("RGB")
                        image_data = pil_to_base64(image_data)
                        
                        # 创建图像部分
                        image_part = types.Part.from_bytes(
                            mime_type="image/png", 
                            data=base64.b64decode(image_data),
                        )
                        input_parts.append(image_part)
        
        # 如果没有找到需要改写的思考内容，直接返回
        if not think_contents:
            print("没有找到需要改写的思考内容")
            return conversation
        
        # 为API请求准备内容
        system_prompt = PATH_FIDING_REASONING_REPHRASE_FINAL_PROMPT
        
        # 准备请求内容
        parts = input_parts
        
        
        # 构建API请求内容
        contents = [
            # 第一轮：系统提示
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=system_prompt)]
            ),
            # 第二轮：带图像的对话内容
            types.Content(
                role="user",
                parts=parts
            )
        ]
        
        # 调用API并处理响应
        for _ in range(self.max_retry):
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
                
                # 提取所有<think>标签中的内容
                new_think_blocks = extract_think_tags(response_text)
                
                if len(new_think_blocks) != len(think_contents):
                    print(f"错误: 返回的思考块数量不匹配，期望 {len(think_contents)}，实际 {len(new_think_blocks)}")
                    time.sleep(self.retry_interval)
                    continue
                
                # 用新的思考内容替换旧的思考内容
                updated_conversation = deepcopy(conversation)
                think_index = 0
                
                for i, message in enumerate(updated_conversation):
                    if message["role"] == "assistant":
                        for j, content_item in enumerate(message["content"]):
                            if content_item["type"] == "text" and "<think>" in content_item["text"]:
                                text = content_item["text"]
                                text = re.sub(r'<think>.*?</think>', f'<think>{new_think_blocks[think_index]}</think>', text, flags=re.DOTALL)
                                updated_conversation[i]["content"][j]["text"] = text
                                think_index += 1
                
                consumed_time = time.time() - start_time
                print(f"Gemini API 调用耗时: {consumed_time:.2f}秒，替换了 {think_index} 个思考块")
                
                return updated_conversation
            
            except Exception as e:
                print(f"API调用失败: {e}")
                time.sleep(self.retry_interval)
                continue
        
        print(f"API调用超过最大重试次数，跳过此条目")
        return conversation  # 返回原始对话



    def generate_dataset(self):
        """生成三种类型的数据集"""
        # 准备数据
        data_a_source = self.stage1_data[:self.args.max_samples_a]  # 任意250个
        data_b_source = self.invalid_data[:self.args.max_samples_b]  # 200个invalid
        data_c_source = self.invalid_data[self.args.max_samples_b:self.args.max_samples_b+self.args.max_samples_c]  # 100个invalid
        
        data_a_source = [data for data in data_a_source if data["id"] not in self.processed_ids_a]
        data_b_source = [data for data in data_b_source if data["id"] not in self.processed_ids_b]
        data_c_source = [data for data in data_c_source if data["id"] not in self.processed_ids_c]
        
        print(f"准备生成数据集:")
        print(f"类型A: {len(data_a_source)} 个样本")
        print(f"类型B: {len(data_b_source)} 个样本") 
        print(f"类型C: {len(data_c_source)} 个样本")
        
        # 生成类型A数据
        print("生成类型A数据...")
        completed_a = self.ckpt_a
        for item in tqdm(data_a_source, desc="类型A"):
            try:
                # 检查是否包含完整数据
                if "full_data" not in item:
                    print(f"警告: 项目 {item['id']} 没有关联的完整数据，跳过")
                    continue
                    
                conversation = self.build_conversation_type_a(item)
                updated_conversation = self.call_gemini_for_reasoning(conversation, item["id"], "A")
                
                result = {
                    "id": f"{item['id']}",
                    "original_item": item,
                    "conversation": updated_conversation,
                    "data_type": "A",
                    "sharegpt_instance": convert_conversation_into_sharegpt(updated_conversation, self.image_dir, f"{item['id']}", self.tool_manager)
                }
                
                completed_a.append(result)
                append_jsonl(result, self.output_file_a)
                
            except Exception as e:
                print(f"处理类型A数据 {item['id']} 时出错: {e}")
                traceback.print_exc()
        
        # 生成类型B数据
        print("生成类型B数据...")
        completed_b = self.ckpt_b
        for item in tqdm(data_b_source, desc="类型B"):
            try:
                # 检查是否包含完整数据和错误路径
                if "full_data" not in item or not item.get("generated_path"):
                    print(f"警告: 项目 {item['id']} 没有关联的完整数据或错误路径，跳过")
                    continue
                    
                conversation = self.build_conversation_type_b(item)
                updated_conversation = self.call_gemini_for_reasoning(conversation, item["id"], "B")
                
                result = {
                    "id": f"{item['id']}",
                    "original_item": item,
                    "conversation": updated_conversation,
                    "data_type": "B",
                    "sharegpt_instance": convert_conversation_into_sharegpt(updated_conversation, self.image_dir, f"{item['id']}",self.tool_manager)
                }
                
                completed_b.append(result)
                append_jsonl(result, self.output_file_b)
                
            except Exception as e:
                print(f"处理类型B数据 {item['id']} 时出错: {e}")
                traceback.print_exc()
        
        # 生成类型C数据
        print("生成类型C数据...")
        completed_c = self.ckpt_c
        for item in tqdm(data_c_source, desc="类型C"):
            try:
                # 检查是否包含完整数据和错误路径
                if "full_data" not in item or not item.get("generated_path"):
                    print(f"警告: 项目 {item['id']} 没有关联的完整数据或错误路径，跳过")
                    continue
                    
                conversation = self.build_conversation_type_c(item)
                updated_conversation = self.call_gemini_for_reasoning(conversation, item["id"], "C")
                
                result = {
                    "id": f"{item['id']}",
                    "original_item": item,
                    "conversation": updated_conversation,
                    "data_type": "C",
                    "sharegpt_instance": convert_conversation_into_sharegpt(updated_conversation, self.image_dir, f"{item['id']}", self.tool_manager)
                }
                
                completed_c.append(result)
                append_jsonl(result, self.output_file_c)
                
            except Exception as e:
                print(f"处理类型C数据 {item['id']} 时出错: {e}")
                traceback.print_exc()
        
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
    parser.add_argument('--stage1_path', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/path_navigation/stage1/path_navigation_data.jsonl", 
                        help='Stage1输出数据路径')
    parser.add_argument('--sft_path', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/metadata_split/path_navigation/sft.jsonl", 
                        help='完整SFT数据路径')
    parser.add_argument('--output_path', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/path_navigation/stage2", 
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
        stage1_path=args.stage1_path,
        sft_path=args.sft_path,
        output_path=args.output_path,
        model_name=args.model,
        max_samples_a=args.max_samples_a,
        max_samples_b=args.max_samples_b,
        max_samples_c=args.max_samples_c,
        image_dir=args.image_dir,
        seed=args.seed,
    )
    
    # 创建生成器并生成数据集
    generator = PathNavigationStage2Generator(gen_args)
    generator.generate_dataset()

if __name__ == "__main__":
    setup_openai_proxy()  # 如果需要代理
    main()