# rl/path_navigation_generator.py
import os
import json
import random
import time
import re
import argparse
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import gymnasium as gym
import numpy as np
from google import genai
from google.genai import types
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from tool_server.utils.utils import pil_to_base64, base64_to_pil, setup_openai_proxy, append_jsonl, write_json_file, process_jsonl, load_json_file
from copy import deepcopy
import base64

from frozen_lake.data_curation.sft_data_curation.prompts import PATH_FINDING_STAGE1

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

def extract_path_from_response(response_text):
    """从API响应中提取路径（方向序列）"""
    try:
        # 尝试提取思考过程
        if "<think>" in response_text and "</think>" in response_text:
            think_process = response_text.split("<think>")[-1].split("</think>")[0].strip()
        else:
            think_process = ""
            
        # 尝试提取最终答案
        if "<response>" in response_text and "</response>" in response_text:
            response = response_text.split("<response>")[-1].split("</response>")[0].strip()
            # 提取 \boxed{} 中的内容
            if "\\boxed{" in response:
                response = response.split("\\boxed{")[-1].split("}")[0].strip()
            directions = response
            tool_call_text = ""
        elif "\\boxed{" in response_text:
            # 直接从整个响应中提取 \boxed{} 内容
            response = response_text.split("\\boxed{")[-1].split("}")[0].strip()
            directions = response
            tool_call_text = ""
            think_process = response_text.split("\\boxed{")[0].strip() if not think_process else think_process
        else:
            # 尝试从响应中提取路径模式
            path_pattern = r'([UDLR](,[UDLR])*)'
            path_match = re.search(path_pattern, response_text, re.IGNORECASE)
            if path_match:
                directions = path_match.group(1).upper()
                response = directions
                tool_call_text = ""
                think_process = response_text if not think_process else think_process
            else:
                raise ValueError("Response text does not contain expected path format.")
        
        # 清理方向字符串，只保留有效的方向字符
        directions = ",".join(c for c in directions.upper() if c in "LRUD")
        return directions, think_process, tool_call_text, response
    
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error extracting directions from response: {e}")
        return "", "", "", ""

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
    
    Args:
        gym_map: gym环境的地图表示
        path_string: 逗号分隔的方向字符串
        verbose: 是否打印详细信息
        
    Returns:
        Tuple[bool, bool, bool]: (是否有效, 是否安全, 是否到达目标)
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
        
        # 如果路径执行完毕但未到达目标
        if not terminated and not goal_reached:
            if verbose:
                print(f"路径执行完毕，但未到达终点")
        
        # 路径有效：安全且到达目标
        is_valid = is_safe and goal_reached
        
        return is_valid, is_safe, goal_reached
    
    except Exception as e:
        print(f"验证路径时出错: {e}")
        return False, False, False

@dataclass
class GenerationArgs:
    """生成参数"""
    input_path: str
    output_path: str
    model_name: str = "gemini-2.5-flash"
    api_provider: str = "gemini"  # "gemini" or "openai"
    max_samples: int = 500
    max_retry: int = 3
    retry_interval: int = 5
    image_dir: str = "./frozen_lake_metadata"
    api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    num_threads: int = 4  # OpenAI 并发线程数
    seed: int = 42

class PathNavigationGenerator:
    def __init__(self, args: GenerationArgs):
        """初始化数据生成器"""
        self.args = args
        self.model_name = args.model_name
        self.api_provider = args.api_provider
        self.dataset_path = args.input_path
        self.output_path = args.output_path
        self.max_retry = args.max_retry
        self.retry_interval = args.retry_interval
        self.image_dir = args.image_dir
        self.num_threads = args.num_threads
        
        # 设置随机种子确保可重现性
        random.seed(args.seed)
        np.random.seed(args.seed)
        
        # 初始化API客户端
        if self.api_provider == "gemini":
            self.api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
            assert self.api_key, "GEMINI_API_KEY环境变量或参数未设置"
            self.client = genai.Client(api_key=self.api_key)
        elif self.api_provider == "openai":
            self.openai_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
            self.openai_base_url = args.openai_base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            assert self.openai_api_key, "OPENAI_API_KEY环境变量或参数未设置"
            self.openai_client = openai.OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")
        
        self.output_file = os.path.join(self.output_path, "path_navigation_data.jsonl")
        
        # 加载已完成的任务
        if os.path.exists(self.output_file):
            self.ckpt = process_jsonl(self.output_file)
            self.processed_ids = {item["id"] for item in self.ckpt}
        else:
            self.ckpt = []
            self.processed_ids = set()
            
        # 线程安全的锁
        self.lock = Lock()
        
        # 加载数据集
        self.load_data()
        
        # 创建输出目录
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """加载数据，确保每次加载结果一致"""
        # 设置随机种子确保数据加载的一致性
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        
        # 从指定路径加载数据
        if self.dataset_path.endswith('.jsonl'):
            data = list(process_jsonl(self.dataset_path))
        elif self.dataset_path.endswith('.json'):
            data = load_json_file(self.dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {self.dataset_path}")
        
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.shuffle(data)
        
        for item in data:
            conversation = self.build_navigation_conversation(item)
            item["conversation"] = conversation
        
        # 如果限制了样本数量，确保取样的一致性
        if self.args.max_samples and len(data) > self.args.max_samples:
            data = data[:self.args.max_samples]
        
        print(f"Loaded {len(data)} samples from {self.dataset_path}")
        self.valid_dataset = data

    def build_navigation_conversation(self, item):
        """构建导航任务的对话框架"""
        conversation = []
        
        # 用户问题
        conversation.append({
            "role": "user", 
            "content": [
                {"type": "text", "text": PATH_FINDING_STAGE1},
                {"type": "image", "image": item["image_path"]}
            ]
        })
        
        # 调用Point工具识别起点（Elf）
        conversation.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>I need to identify the key elements in this frozen lake image. First, I'll locate the elf, which represents the starting point.</think>\n\n<tool_call>\n" + json.dumps({
                    "name": "Point",
                    "parameters": {
                        "image": "img_1",
                        "description": "Elf"
                    }
                }, indent=2) + "\n</tool_call>"}
            ]
        })
        
        # Point工具返回起点结果
        elf_output = item["point_tools"]["elf"]["output"]
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(elf_output, indent=2)},
                {"type": "image", "image": item["point_tools"]["elf"]["image_path"]}
            ]
        })
        
        # 调用Point工具识别终点（Gift）
        conversation.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>Now I need to locate the gift, which represents the goal position.</think>\n\n<tool_call>\n" + json.dumps({
                    "name": "Point",
                    "parameters": {
                        "image": "img_1",
                        "description": "Gift"
                    }
                }, indent=2) + "\n</tool_call>"}
            ]
        })
        
        # Point工具返回终点结果
        gift_output = item["point_tools"]["gift"]["output"]
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(gift_output, indent=2)},
                {"type": "image", "image": item["point_tools"]["gift"]["image_path"]}
            ]
        })
        
        # 调用Point工具识别冰洞（Ice Holes）
        conversation.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>Finally, I need to identify all ice holes in the image, which are the obstacles to avoid.</think>\n\n<tool_call>\n" + json.dumps({
                    "name": "Point",
                    "parameters": {
                        "image": "img_1",
                        "description": "Ice Holes"
                    }
                }, indent=2) + "\n</tool_call>"}
            ]
        })
        
        # Point工具返回冰洞结果
        if "ice_holes" not in item["point_tools"]:
            ice_holes_output = "{\n  \"tool_response_from\": \"Point\",\n  \"status\": \"success\",\n  \"points\": [ ],\n  \"image_dimensions_pixels\": {\n    \"width\": 384,\n    \"height\": 384\n  },\n  \"error_code\": 0\n}"
            ice_hole_image = item["image_path"]
        else:
            ice_holes_output = item["point_tools"]["ice_holes"]["output"]
            ice_hole_image = item["point_tools"]["ice_holes"]["image_path"]
            
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(ice_holes_output, indent=2)},
                {"type": "image", "image": ice_hole_image}
            ]
        })
        
        # 模型需要生成的思考过程和路径规划
        thinking_placeholder = "Next, please use the above information to reason out the final path and present it in the following format: <think> reasoning process </think><response>\\boxed{answer}</response>"
        
        # 助手最后的回答模板
        conversation.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{thinking_placeholder}"}
            ]
        })
        
        return conversation

    def call_gemini_api(self, item, conversation):
        """调用Gemini API"""
        # 将对话转换为Gemini API所需的格式
        contents = []
        for message in conversation:
            parts = []
            for content in message["content"]:
                if content["type"] == "text":
                    parts.append(types.Part.from_text(text=content["text"]))
                elif content["type"] == "image":
                    # 加载图像文件
                    image_path = os.path.join(self.image_dir, content["image"])
                    if check_image_exists(image_path):
                        with open(image_path, "rb") as f:
                            image_data = f.read()
                        image_data = Image.open(image_path).convert("RGB")
                        image_data = pil_to_base64(image_data)
                        # 创建图像部分
                        image_part = types.Part.from_bytes(
                            mime_type="image/png", 
                            data=base64.b64decode(image_data),
                        )
                        parts.append(image_part)
                    else:
                        raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
            # 添加到内容列表
            contents.append(types.Content(
                role="user" if message["role"] == "user" else "model",
                parts=parts
            ))
        
        # 构建请求设置，包括执行思考步骤
        generation_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            response_mime_type="text/plain",
        )
        
        # 调用API
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=generation_config
        )
        
        return response.text

    def call_openai_api(self, item, conversation):
        """调用OpenAI兼容API (包括VLLM)"""
        # 将对话转换为OpenAI API所需的格式
        messages = []
        
        for message in conversation:
            content = []
            for msg_content in message["content"]:
                if msg_content["type"] == "text":
                    content.append({
                        "type": "text",
                        "text": msg_content["text"]
                    })
                elif msg_content["type"] == "image":
                    # 加载图像文件
                    image_path = os.path.join(self.image_dir, msg_content["image"])
                    if check_image_exists(image_path):
                        image_data = Image.open(image_path).convert("RGB")
                        image_base64 = pil_to_base64(image_data)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                            }
                        })
                    else:
                        raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
            messages.append({
                "role": message["role"],
                "content": content
            })
        
        # VLLM 优化的参数设置
        api_params = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": 2048,
            "temperature": 0.7,
        }
        
        try:
            # 调用API
            response = self.openai_client.chat.completions.create(**api_params)
            return response.choices[0].message.content
        except Exception as e:
            # VLLM 特定的错误处理
            if "Connection" in str(e):
                raise ConnectionError(f"无法连接到VLLM服务器: {e}")
            elif "timeout" in str(e).lower():
                raise TimeoutError(f"VLLM响应超时: {e}")
            else:
                raise e
       
    def process_single_item(self, item):
        """处理单个数据项"""
        # 如果已经处理过，直接跳过
        if item["id"] in self.processed_ids:
            return None
        
        # 构建对话框架
        conversation = item["conversation"]
        
        start_time = time.time()
        last_error = None
        
        # 尝试调用API，最多重试指定次数
        for attempt in range(self.max_retry):
            try:
                print(f"[{self.api_provider.upper()}] 处理 {item['id']} - 尝试 {attempt+1}/{self.max_retry}")
                
                # 根据API提供商调用对应的API
                if self.api_provider == "gemini":
                    response_text = self.call_gemini_api(item, conversation)
                else:  # openai (包括 VLLM)
                    response_text = self.call_openai_api(item, conversation)
                
                # 提取路径（方向序列）
                path, think_process, tool_call_text, response_token_text = extract_path_from_response(response_text)
                
                # 验证路径是否有效
                if path:
                    # 转换为gym环境可用的地图
                    gym_map = convert_to_gym_map(item)
                    
                    # 验证路径是否安全且能到达目标
                    is_valid, is_safe, reaches_goal = verify_path(gym_map, path)
                    
                    # 获取最优路径（A*算法）
                    optimal_path = item["astar_path"]["path"]
                    is_optimal = (len(path.split(',')) <= len(optimal_path.split(',')))
                    
                    # 构建结果
                    result = {
                        "id": f"{item['id']}",
                        "image_path": item["image_path"],
                        "size": item["size"],
                        "start_coords": item["start_coords"],
                        "goal_coords": item["goal_coords"],
                        "obstacle_coords": item["obstacle_coords"],
                        "generated_path": path,
                        "is_valid": is_valid,
                        "is_safe": is_safe,
                        "reaches_goal": reaches_goal,
                        "optimal_path": optimal_path,
                        "is_optimal": is_optimal,
                        "api_response": response_text,
                        "think_process": think_process,
                        "response_token_text": response_token_text,
                        "conversation": conversation,
                        "api_provider": self.api_provider,
                        "model_name": self.model_name,
                        "attempts": attempt + 1
                    }
                    
                    consumed_time = time.time() - start_time
                    
                    # 线程安全地写入文件和更新处理列表
                    with self.lock:
                        append_jsonl(result, self.output_file)
                        self.processed_ids.add(item["id"])
                    
                    print(f"[{self.api_provider.upper()}] ✓ 处理完成 {item['id']}: 耗时 {consumed_time:.2f}秒，路径: {path}, 有效: {is_valid}")
                    
                    return result
                else:
                    last_error = f"未能从响应中提取路径: {response_text[:100]}..."
                    print(f"[{self.api_provider.upper()}] ⚠ {last_error}")
                    
            except (ConnectionError, TimeoutError) as e:
                last_error = f"连接错误: {e}"
                print(f"[{self.api_provider.upper()}] ⚠ {last_error}")
                # 连接错误等待更长时间
                time.sleep(self.retry_interval * 2)
            except Exception as e:
                last_error = f"API调用失败: {e}"
                print(f"[{self.api_provider.upper()}] ⚠ {last_error}")
                # 一般错误等待标准时间
                time.sleep(self.retry_interval)
        
        print(f"[{self.api_provider.upper()}] ✗ 处理失败 {item['id']}: {last_error}")
        return None
    
    def generate_dataset(self):
        """生成数据集"""
        # 创建输出文件
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # 保存已完成的结果
        completed_items = []
        
        # 过滤未处理的数据
        unprocessed_data = [item for item in self.valid_dataset if item["id"] not in self.processed_ids]
        
        print(f"开始处理 {len(unprocessed_data)} 个未处理的样本")
        
        if self.api_provider == "openai" and self.num_threads > 1:
            # 使用多线程处理OpenAI API调用
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # 提交所有任务
                future_to_item = {executor.submit(self.process_single_item, item): item for item in unprocessed_data}
                
                # 使用tqdm显示进度
                with tqdm(total=len(unprocessed_data), desc=f"生成路径导航数据 ({self.api_provider.upper()})") as pbar:
                    for future in as_completed(future_to_item):
                        result = future.result()
                        if result:
                            completed_items.append(result)
                        pbar.update(1)
        else:
            # 单线程处理（Gemini或OpenAI单线程模式）
            with tqdm(total=len(unprocessed_data), desc=f"生成路径导航数据 ({self.api_provider.upper()})") as pbar:
                for item in unprocessed_data:
                    result = self.process_single_item(item)
                    if result:
                        completed_items.append(result)
                    pbar.update(1)
        
        # 计算统计信息
        total_items = len(completed_items)
        if total_items > 0:
            valid_count = sum(1 for item in completed_items if item["is_valid"])
            safe_count = sum(1 for item in completed_items if item["is_safe"])
            goal_count = sum(1 for item in completed_items if item["reaches_goal"])
            optimal_count = sum(1 for item in completed_items if item["is_optimal"])
            
            # 保存统计信息
            stats = {
                "api_provider": self.api_provider,
                "total_samples": total_items,
                "valid_paths": valid_count,
                "valid_percentage": round(valid_count / total_items * 100, 2),
                "safe_paths": safe_count,
                "safe_percentage": round(safe_count / total_items * 100, 2),
                "goal_reaching_paths": goal_count,
                "goal_percentage": round(goal_count / total_items * 100, 2),
                "optimal_paths": optimal_count,
                "optimal_percentage": round(optimal_count / total_items * 100, 2),
            }
            
            stats_path = os.path.join(self.output_path, "path_navigation_stats.json")
            write_json_file(stats, stats_path)
            
            print(f"数据集生成完成，共生成 {total_items} 个样本")
            print(f"结果保存至 {self.output_file}")
            print(f"统计信息保存至 {stats_path}")
            
            print(f"有效路径: {valid_count} ({stats['valid_percentage']}%)")
            print(f"安全路径: {safe_count} ({stats['safe_percentage']}%)")
            print(f"到达目标的路径: {goal_count} ({stats['goal_percentage']}%)")
            print(f"最优路径: {optimal_count} ({stats['optimal_percentage']}%)")
        else:
            print("没有成功生成任何样本")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成FrozenLake路径导航任务数据集')
    parser.add_argument('--input_path', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/metadata_split/path_navigation/sft.jsonl", 
                        help='输入数据集路径')
    parser.add_argument('--output_path', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/path_navigation", 
                        help='输出数据集目录')
    parser.add_argument('--model', type=str, default="Qwen2.5-VL-7B-Instruct", 
                        help='模型名称 (Gemini: gemini-2.5-flash; OpenAI/VLLM: gpt-4o, Qwen2.5-VL-7B-Instruct, etc.)')
    parser.add_argument('--api_provider', type=str, default="openai", choices=["gemini", "openai"],
                        help='API提供商: gemini 或 openai (VLLM使用openai兼容接口)')
    parser.add_argument('--max_samples', type=int, default=100, 
                        help='最大样本数')
    parser.add_argument('--image_dir', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation", 
                        help='图像目录')
    parser.add_argument('--api_key', type=str, default=None, 
                        help='Gemini API Key (也可通过环境变量GEMINI_API_KEY设置)')
    parser.add_argument('--openai_api_key', type=str, default="EMPTY",
                        help='OpenAI/VLLM API Key (VLLM通常使用EMPTY)')
    parser.add_argument('--openai_base_url', type=str, default="http://SH-IDC1-10-140-37-118:7120/v1",
                        help='OpenAI/VLLM Base URL')
    parser.add_argument('--num_threads', type=int, default=1,
                        help='API并发线程数 (VLLM支持高并发)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--max_retry', type=int, default=3,
                        help='最大重试次数')
    parser.add_argument('--retry_interval', type=int, default=2,
                        help='重试间隔(秒)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    if args.api_provider == "gemini":
        setup_openai_proxy()
    else:
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
    # 创建生成器参数
    gen_args = GenerationArgs(
        input_path=args.input_path,
        output_path=args.output_path,
        model_name=args.model,
        api_provider=args.api_provider,
        max_samples=args.max_samples,
        image_dir=args.image_dir,
        api_key=args.api_key,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        num_threads=args.num_threads,
        seed=args.seed
    )
    
    os.chdir(args.image_dir)
    
    # 创建生成器并生成数据集
    generator = PathNavigationGenerator(gen_args)
    generator.generate_dataset()

if __name__ == "__main__":
    # setup_openai_proxy()  # 如果需要代理
    main()