# gemini_verify_generator.py
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
from tool_server.utils.utils import pil_to_base64, base64_to_pil, setup_openai_proxy, append_jsonl, write_json_file, process_jsonl,load_json_file
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
from prompts import TOOL_PROMPTS, PATH_VERIFY_TASK_INSTRUCTION_SHORT, REASONING_REPHRASE_FINAL_PROMPT

# Setup environment variables and constants
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
assert GEMINI_API_KEY, "GEMINI_API_KEY environment variable not set"

def check_image_exists(image_path):
    """验证图像文件是否存在"""
    return os.path.isfile(image_path) and os.path.getsize(image_path) > 0

def convert_conversation_into_sharegpt(conversation, image_dir, item_id):
    """
    将对话转换为ShareGPT格式
    """
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

def extract_think_tags(response_text):
    """
    从Gemini响应中提取所有<think>标签之间的内容
    """
    think_blocks = []
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, response_text, re.DOTALL)
    return matches

@dataclass
class GenerationArgs:
    """生成参数"""
    input_path: str
    output_path: str
    model_name: str = "gemini-2.5-flash"
    max_samples: int = 2000
    use_cached: bool = True
    max_retry: int = 3
    retry_interval: int = 5
    image_dir: str = "./frozen_lake_metadata"
    selected_ids_file: str = "/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata/sft_output/selected_ids.json"
    re_extract_ids: bool = False  # 是否重新提取ID

class GeminiVerifyGenerator:
    def __init__(self, args: GenerationArgs):
        """初始化数据生成器"""
        self.args = args
        self.model_name = args.model_name
        self.dataset_path = args.input_path
        self.output_path = args.output_path
        self.use_cached = args.use_cached
        self.max_retry = args.max_retry
        self.retry_interval = args.retry_interval
        self.image_dir = args.image_dir
        
        self.re_extract_ids = args.re_extract_ids
        self.output_file = os.path.join(self.output_path, "verify_sft_data.jsonl")
        
        
        self.selected_ids_file = args.selected_ids_file
        
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        
        # 加载数据集
        self.load_data()
        
        # 创建输出目录
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """加载原始数据集"""
        print(f"从 {self.dataset_path} 加载数据集...")
        
        # 读取JSONL文件
        self.dataset = process_jsonl(self.dataset_path)
        
        # 过滤出有随机路径的条目
        self.valid_dataset = []
        safe_paths = []
        unsafe_paths = []
        
        if not self.re_extract_ids:
            self.valid_dataset = load_json_file(self.selected_ids_file)
            print(f"从 {self.selected_ids_file} 加载已选择的ID，共 {len(self.valid_dataset)} 条目")
        else:
            for item in self.dataset:
                if (item.get("path_drawings") and 
                    item["path_drawings"].get("random") and 
                    item["path_drawings"]["random"].get("path")):
                    
                    # 检查路径是否安全
                    is_safe = item["path_drawings"]["random"].get("is_safe", False)
                    
                    # 将路径分为安全和不安全两组
                    if is_safe:
                        safe_paths.append(item)
                    else:
                        unsafe_paths.append(item)
            
            print(f"发现 {len(safe_paths)} 个安全路径和 {len(unsafe_paths)} 个不安全路径")
            
            # 确定要使用的样本数量（安全和不安全路径各占一半）
            target_per_group = min(len(safe_paths), len(unsafe_paths), self.args.max_samples // 2)
            
            # 随机抽样安全和不安全路径
            selected_safe = random.sample(safe_paths, target_per_group)
            selected_unsafe = random.sample(unsafe_paths, target_per_group)
            
            # 合并并打乱数据
            self.valid_dataset = selected_safe + selected_unsafe
            random.shuffle(self.valid_dataset)
            write_json_file(self.valid_dataset, self.selected_ids_file)
            
            print(f"选择了 {len(selected_safe)} 个安全路径和 {len(selected_unsafe)} 个不安全路径，共 {len(self.valid_dataset)} 个有效样本")
        
        # 加载已处理的ID
        self.processed_ids = set()
        if os.path.exists(self.output_file) and self.use_cached:
            ckpt = process_jsonl(self.output_file)
            for item in ckpt:
                if 'id' in item:
                    self.processed_ids.add(item['id'])

        print(f"跳过 {len(self.processed_ids)} 个已处理的条目")

    def build_trajectory_data_for_path_validation(self, item):
        """构建路径验证任务的轨迹数据框架，包含需要Gemini改写的思考部分"""
        trajectory_data = []
        
        # 获取随机路径数据
        random_path_data = item["path_drawings"]["random"]
        path = random_path_data["path"]
        is_safe = random_path_data.get("is_safe", False)
        
        # 获取起点坐标
        start_coords = item["start_coords"]
        
        # 获取Point工具数据（如果存在）
        point_data = item["point_tools"]["elf"]
        point = point_data["output"]["points"][0]
        start_coords = [point["x"], point["y"]]
        
            
        
        # 1. 添加用户问题
        instruction_text = PATH_VERIFY_TASK_INSTRUCTION_SHORT.replace("<ACTION-SEQ>", path)
        trajectory_data.append({
            "role": "user",
            "content": [
                {"type": "text", "text": instruction_text},
                {"type": "image", "image": item["image_path"]}
            ]
        })
        
        # 2. 助手思考过程和Point工具调用
        thinking_placeholder_point = f"[**You need to implement this** Example: I need to determine if the given path is valid. First, I'll use the Point tool to identify the Elf's position (starting point) on the map. Once I know the starting position, I'll visualize the path with Draw2DPath and then analyze if this path is safe.]"
        
        # 构建Point工具调用的参数
        point_params = {
            "image": "img_1",
            "description": "Elf"  # 寻找起点（Elf）
        }
        
        trajectory_data.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"<think>{thinking_placeholder_point}</think>\n\n<tool_call>\n{json.dumps(point_data['input'], indent=2)}\n</tool_call>"}
            ]
        })
        
        # 3. Point工具返回结果
        point_response = {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(item["point_tools"]["elf"]["output"], indent=2)},
            ]
        }
        
        # 如果有Point工具生成的图像，添加到结果中
        if "image_path" in item["point_tools"]["elf"] and item["point_tools"]["elf"]["image_path"]:
            point_response["content"].append({
                "type": "image", 
                "image": item["point_tools"]["elf"]["image_path"]
            })
                
        trajectory_data.append(point_response)
        
        # 4. 助手思考过程和Draw2DPath工具调用
        thinking_placeholder_draw = f"[**You need to implement this** Example: Now that I've identified the Elf's position at coordinates {start_coords}, I'll use the Draw2DPath tool to visualize the path following the provided directions: {path}. This will help me determine if the path is safe or if it crosses any ice holes.]"
        
        # 构建Draw2DPath调用的参数
        draw_params = {
            "image": "img_1",
            "start_point": start_coords,
            "directions": path,
            "step": 64,
            "line_width": 3,
            "line_color": "red",
            "pixel_coordinate": True
        }
        
        trajectory_data.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"<think>{thinking_placeholder_draw}</think>\n\n<tool_call>\n{json.dumps({'name': 'Draw2DPath', 'parameters': draw_params}, indent=2)}\n</tool_call>"}
            ]
        })
        
        # 5. Draw2DPath工具返回结果
        tool_response = {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(random_path_data["output"], indent=2)},
            ]
        }
        
        # 如果有工具生成的图像，添加到结果中
        if "image_path" in random_path_data and random_path_data["image_path"]:
            tool_response["content"].append({
                "type": "image", 
                "image": random_path_data["image_path"]
            })
                
        trajectory_data.append(tool_response)
        
        # 6. 助手最终回答
        valid_text = "Yes" if is_safe else "No"
        reason_template = "This path would cause the player to fall into an ice hole." if not is_safe else "This path safely navigates from the starting point without falling into any ice holes."
        
        thinking_placeholder_final = f"[**You need to implement this** Example: Now that I've drawn the path on the map, I can analyze whether it's safe. I need to check if the path avoids all ice holes. Based on the drawn path visualization, I can see that this path is {'' if is_safe else 'not'} safe.]"
        
        trajectory_data.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"<think>{thinking_placeholder_final}</think>\n\n<response>After analyzing the path {path} from the Elf's position, I can determine that this path is {'safe' if is_safe else 'not safe'}.\n\n{reason_template}\n\n\\boxed{{{valid_text}}}</response>"}
            ]
        })
        
        return trajectory_data

    def prepare_path_validation_task(self, item):
        """准备路径验证任务的上下文"""
        # 构建上下文数据
        context = {
            "id": item["id"],
            "image_path": item["image_path"],
            "task_type": "path_validation",
            "additional_data": {
                "start_coords": item["start_coords"],
                "path": item["path_drawings"]["random"]["path"],
                "is_safe": item["path_drawings"]["random"].get("is_safe", False)
            }
        }
        
        # 构建轨迹数据
        trajectory_data = self.build_trajectory_data_for_path_validation(item)
        context["trajectory_data"] = trajectory_data
        
        return context
    
    def call_gemini_for_reasoning(self, context):
        """调用Gemini API只改写思考内容"""
        if context["id"] in self.processed_ids:
            print(f"跳过已处理的条目: {context['id']}")
            return None
        
        start_time = time.time()
        
        input_string = f"{context['trajectory_data']}"
        # 提取需要改写的思考部分
        think_contents = []
        for message in context["trajectory_data"]:
            if message["role"] == "assistant":
                for content_item in message["content"]:
                    if content_item["type"] == "text":
                        text = content_item["text"]
                        if "<think>" in text and "</think>" in text:
                            think_contents.append(text)

        system_prompt = REASONING_REPHRASE_FINAL_PROMPT
        
        # 添加需要改写的思考内容
        input_prompt = "\n\n# Content to rewrite:\n"+input_string
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
                            types.Part.from_text(text=input_prompt),
                            # image_part
                        ]
                    )
                ]
        for _ in range(self.max_retry):
            try:
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
                
                # 提取所有<think>标签中的内容
                new_think_blocks = extract_think_tags(response_text)
                
                if len(new_think_blocks) != len(think_contents):
                    print(f"错误: 返回的思考块数量不匹配，期望 {len(think_contents)}，实际 {len(new_think_blocks)}")
                    continue
                
                # 用新的思考内容替换旧的思考内容
                updated_trajectory = deepcopy(context["trajectory_data"])
                think_index = 0
                
                for i, message in enumerate(updated_trajectory):
                    if message["role"] == "assistant":
                        for j, content_item in enumerate(message["content"]):
                            if content_item["type"] == "text" and "<think>" in content_item["text"]:
                                text = content_item["text"]
                                text = re.sub(r'<think>.*?</think>', f'<think>{new_think_blocks[think_index]}</think>', text, flags=re.DOTALL)
                                updated_trajectory[i]["content"][j]["text"] = text
                                think_index += 1
                
                # 更新上下文
                context["trajectory_data"] = updated_trajectory
                
                # 生成ShareGPT格式的数据
                sharegpt_item = convert_conversation_into_sharegpt(updated_trajectory, self.image_dir, context["id"])
                context["sharegpt_instance"] = sharegpt_item
                
                consumed_time = time.time() - start_time
                print(f"Gemini API 调用耗时: {consumed_time:.2f}秒")
                
                return context
            
            except Exception as e:
                print(f"API调用失败: {e}")
                time.sleep(self.retry_interval)
                continue
        
        print(f"API调用超过最大重试次数，跳过此条目")
        return None
    
    def generate_dataset(self):
        """生成数据集"""
        # 创建输出文件
        output_file = os.path.join(self.output_path, "verify_sft_data.jsonl")
        sharegpt_output = os.path.join(self.output_path, "verify_sharegpt_data.jsonl")
        
        # 保存已完成的结果
        completed_items = []
        completed_sharegpt = []
        
        # 进度条
        with tqdm(total=len(self.valid_dataset), desc="生成路径验证SFT数据") as pbar:
            for item in self.valid_dataset:
                # 准备路径验证任务
                context = self.prepare_path_validation_task(item)
                
                # 调用Gemini改写思考过程
                result = self.call_gemini_for_reasoning(context)
                
                if result:
                    # 保存JSON结果
                    completed_items.append(result)
                    
                    # 保存ShareGPT格式结果
                    if "sharegpt_instance" in result:
                        completed_sharegpt.append(result["sharegpt_instance"])
                    
                    # 定期写入文件
                    append_jsonl(result, output_file)
                
                pbar.update(1)
        
        all_data = process_jsonl(output_file)
        sharegpt_data = [item["sharegpt_instance"] for item in all_data if "sharegpt_instance" in item]
        write_json_file(sharegpt_data, sharegpt_output)

        print(f"数据集生成完成，共生成 {len(completed_items)} 个样本")
        print(f"结果保存至 {output_file} 和 {sharegpt_output}")
        
        # 统计安全和不安全路径数量
        safe_count = 0
        unsafe_count = 0
        
        for item in completed_items:
            if item["additional_data"]["is_safe"]:
                safe_count += 1
            else:
                unsafe_count += 1
        
        print(f"安全路径数量: {safe_count} ({safe_count/len(completed_items)*100:.1f}%)")
        print(f"不安全路径数量: {unsafe_count} ({unsafe_count/len(completed_items)*100:.1f}%)")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成FrozenLake路径验证任务SFT数据集')
    parser.add_argument('--input_path', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata/dataset.jsonl", 
                        help='输入数据集路径')
    parser.add_argument('--output_path', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata/sft_output", 
                        help='输出SFT数据集目录')
    parser.add_argument('--model', type=str, default="gemini-2.5-flash", 
                        help='Gemini模型名称')
    parser.add_argument('--max_samples', type=int, default=500, 
                        help='最大样本数')
    parser.add_argument('--no_cache', action='store_false', dest='use_cached', 
                        help='不使用缓存，重新处理所有数据')
    parser.add_argument('--image_dir', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation", 
                        help='图像目录')
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument('--selected_ids_file', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata/sft_output/selected_ids.json", 
                        help='已选择的ID文件路径')
    
    args = parser.parse_args()
    
    
    random.seed(args.seed)  # 设置随机种子
    # 创建生成器参数
    gen_args = GenerationArgs(
        input_path=args.input_path,
        output_path=args.output_path,
        model_name=args.model,
        max_samples=args.max_samples,
        use_cached=args.use_cached,
        image_dir=args.image_dir,
        selected_ids_file = args.selected_ids_file
        
    )
    
    # 创建生成器并生成数据集
    generator = GeminiVerifyGenerator(gen_args)
    generator.generate_dataset()

if __name__ == "__main__":
    setup_openai_proxy()  # 如果需要代理
    main()