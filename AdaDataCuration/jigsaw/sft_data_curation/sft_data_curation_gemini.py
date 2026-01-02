# sft_data_curation_gemini.py
import os
import re
import json
import time
import random
import string
import argparse
import tempfile
import base64
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from google import genai
from google.genai import types
from typing import List, Dict, Any, Tuple, Optional, Union

# 导入工具管理器
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from tool_server.utils.utils import pil_to_base64, base64_to_pil, setup_openai_proxy, load_json_file, write_json_file, append_jsonl
from jigsaw.prompts import INSERT_VERIFY_PROMPT, SELF_THINK_PROMPT, TOOL_PROMPTS

def from_file_to_identifier(file_path):
    file_base_name = os.path.basename(file_path)
    identifier = None
    if "question" in file_base_name:
        identifier = "img_1"
    else:
        for id_num in range(1,5):
            if f"choice_{id_num}" in file_base_name:
                identifier = f"img_{id_num+1}"
    assert identifier is not None, f"无法从文件名 {file_base_name} 中提取标识符"
    return identifier
    
def extract_think_tags(response_text):
    """从Gemini响应中提取所有<think>标签之间的内容"""
    think_blocks = []
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, response_text, re.DOTALL)
    return matches

def extract_response_tags(response_text):
    """从Gemini响应中提取所有<response>标签之间的内容"""
    pattern = r'<response>(.*?)</response>'
    matches = re.findall(pattern, response_text, re.DOTALL)
    return matches[0] if matches else ""

def extract_answer(response_text):
    """从响应中提取boxed答案"""
    pattern = r'\\boxed{([A-C])}'
    matches = re.findall(pattern, response_text)
    return matches[0] if matches else None

def prepare_detect_black_area_example(item):
    """
    准备DetectBlackArea工具调用的示例
    
    Args:
        item: 数据项
        
    Returns:
        dict: 格式化的工具调用示例
    """
    tools_data = item.get("tools", {})
    detect_data = tools_data.get("detect_black_area", {})
    
    tool_input = detect_data.get("input", {})
    tool_output = detect_data["output"]
    tool_output.pop("comparison",None)
    
    tool_input_image = tool_input["image"]
    identifier = from_file_to_identifier(tool_input_image)
    
    # 构建工具调用示例
    tool_call = {
        "name": "DetectBlackArea",
        "parameters": {
            "image": identifier,
        }
    }
    
    return {
        "tool_call": tool_call,
        "tool_output": tool_output
    }

def prepare_insert_image_example(item, choice_idx):
    """
    准备InsertImage工具调用的示例
    
    Args:
        item: 数据项
        choice_idx: 选项索引
        
    Returns:
        dict: 格式化的工具调用示例
    """
    tools_data = item.get("tools", {})
    insert_data = tools_data.get("insert_images", [])
    
    if choice_idx < len(insert_data):
        tool_input = insert_data[choice_idx]["input"]
        tool_output = insert_data[choice_idx]["output"]
        
        # 构建工具调用示例
        tool_call = {
            "name": "InsertImage",
            "parameters": {
                "base_image": "img_1",
                "image_to_insert": f"img_{choice_idx + 2}",
                "coordinates": tool_input["coordinates"],
            }
        }
        
        return {
            "tool_call": tool_call,
            "tool_output": tool_output,
            "image_path": item["inserted_images"][choice_idx]
        }
    
    return None

def create_first_attempt_trajectory(item, correct_idx):
    """
    创建第一次尝试就选对答案的trajectory
    
    Args:
        item: 数据项
        correct_idx: 正确答案的索引
        
    Returns:
        list: 轨迹对话列表
    """
    choices = item.get("choices", [])
    question_text = item["question_text"]
    question_image = item["question_image"]
    correct_letter = string.ascii_uppercase[correct_idx]
    
    # 准备工具示例
    detect_example = prepare_detect_black_area_example(item)
    insert_example = prepare_insert_image_example(item, correct_idx)
    
    trajectory = []
    
    # 用户提问
    trajectory.append({
        "role": "user",
        "content": [
            {"type": "text", "text": question_text},
            {"type": "image", "image": question_image}
        ] + [{"type": "image", "image": choice["image"]} for choice in choices]
    })
    
    # 助手回复 - 调用DetectBlackArea
    trajectory.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "<think>[**You need to implement this** Example: I need to identify the missing part in the first image. I can see there's a black area that needs to be filled. Let me first use the DetectBlackArea tool to precisely locate this region.]</think>\n\n<tool_call>\n" + json.dumps(detect_example["tool_call"], indent=2) + "\n</tool_call>"
            }
        ]
    })
    
    # 用户返回DetectBlackArea结果
    trajectory.append({
        "role": "user",
        "content": [
            {"type": "text", "text": json.dumps(detect_example["tool_output"], indent=2)}
        ]
    })
    
    # 助手回复 - 调用InsertImage
    trajectory.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": f"<think>[**You need to implement this** Example: Now that I have identified the black area coordinates, I should try inserting the potential missing parts to see which one fits best. Let me start by trying option {correct_letter}, which looks most promising based on the content and edges.]</think>\n\n<tool_call>\n" + json.dumps(insert_example["tool_call"], indent=2) + "\n</tool_call>"
            }
        ]
    })
    
    # 用户返回InsertImage结果
    trajectory.append({
        "role": "user",
        "content": [
            {"type": "text", "text": json.dumps(insert_example["tool_output"], indent=2)},
            {"type": "image", "image": insert_example["image_path"]}
        ]
    })
    
    # 助手给出最终答案
    trajectory.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": f"<think>[**You need to implement this** Example: After inserting option {correct_letter} into the black area, I can see that it fits perfectly. The colors, patterns and edges align seamlessly with the surrounding parts of the image. This confirms that option {correct_letter} is indeed the correct missing part.]</think>\n\n<response>After analyzing the image with the black area and testing the insertion of option {correct_letter}, I can see that it fits perfectly into the missing region. The edges align seamlessly, and the content flows naturally when this piece is inserted.\n\n\\boxed{{{correct_letter}}}</response>"
            }
        ]
    })
    
    return trajectory

def create_second_attempt_trajectory(item, correct_idx, wrong_idx):
    """
    创建第二次尝试才选对答案的trajectory（先试错一次）
    
    Args:
        item: 数据项
        correct_idx: 正确答案的索引
        wrong_idx: 错误答案的索引
        
    Returns:
        list: 轨迹对话列表
    """
    choices = item.get("choices", [])
    question_text = item.get("question_text", "")
    question_image = item.get("question_image", "")
    correct_letter = string.ascii_uppercase[correct_idx]
    wrong_letter = string.ascii_uppercase[wrong_idx]
    
    # 准备工具示例
    detect_example = prepare_detect_black_area_example(item)
    wrong_insert_example = prepare_insert_image_example(item, wrong_idx)
    correct_insert_example = prepare_insert_image_example(item, correct_idx)
    
    trajectory = []
    
    # 用户提问
    trajectory.append({
        "role": "user",
        "content": [
            {"type": "text", "text": question_text},
            {"type": "image", "image": question_image}
        ] + [{"type": "image", "image": choice["image"]} for choice in choices]
    })
    
    # 助手回复 - 调用DetectBlackArea
    trajectory.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "<think>[**You need to implement this** Example: I need to identify the missing part in the first image. I can see there's a black area that needs to be filled. Let me first use the DetectBlackArea tool to precisely locate this region.]</think>\n\n<tool_call>\n" + json.dumps(detect_example["tool_call"], indent=2) + "\n</tool_call>"
            }
        ]
    })
    
    # 用户返回DetectBlackArea结果
    trajectory.append({
        "role": "user",
        "content": [
            {"type": "text", "text": json.dumps(detect_example["tool_output"], indent=2)}
        ]
    })
    
    # 助手回复 - 调用第一次InsertImage (错误选项)
    trajectory.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": f"<think>[**You need to implement this** Example: Now I have identified the black area coordinates. Let me start by trying option {wrong_letter}, which initially seems like it might fit based on my visual assessment of the content and edges.]</think>\n\n<tool_call>\n" + json.dumps(wrong_insert_example["tool_call"], indent=2) + "\n</tool_call>"
            }
        ]
    })
    
    # 用户返回第一次InsertImage结果
    trajectory.append({
        "role": "user",
        "content": [
            {"type": "text", "text": json.dumps(wrong_insert_example["tool_output"], indent=2)},
            {"type": "image", "image": wrong_insert_example["image_path"]}
        ]
    })
    
    # 助手回复 - 调用第二次InsertImage (正确选项)
    trajectory.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": f"<think>[**You need to implement this** Example: After inserting option {wrong_letter}, I notice that it doesn't fit perfectly. The colors and patterns don't align well with the surrounding areas, and the edges look disjointed. Let me try option {correct_letter} instead to see if it provides a better fit.]</think>\n\n<tool_call>\n" + json.dumps(correct_insert_example["tool_call"], indent=2) + "\n</tool_call>"
            }
        ]
    })
    
    # 用户返回第二次InsertImage结果
    trajectory.append({
        "role": "user",
        "content": [
            {"type": "text", "text": json.dumps(correct_insert_example["tool_output"], indent=2)},
            {"type": "image", "image": correct_insert_example["image_path"]}
        ]
    })
    
    # 助手给出最终答案
    trajectory.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": f"<think>[**You need to implement this** Example: After trying both options, I can clearly see that option {correct_letter} fits much better than option {wrong_letter}. With option {correct_letter}, the colors, patterns, and edges align seamlessly with the surrounding parts of the image. This confirms that option {correct_letter} is the correct missing part.]</think>\n\n<response>After trying both options, I can see that option {wrong_letter} doesn't fit well - the edges don't align properly and the content doesn't flow naturally with the rest of the image.\n\nHowever, option {correct_letter} fits perfectly into the missing region. The edges align seamlessly, and the content flows naturally when this piece is inserted.\n\n\\boxed{{{correct_letter}}}</response>"
            }
        ]
    })
    
    return trajectory

def create_third_attempt_trajectory(item, correct_idx, wrong_idx1, wrong_idx2):
    """
    创建第三次尝试才选对答案的trajectory（先试错两次）
    
    Args:
        item: 数据项
        correct_idx: 正确答案的索引
        wrong_idx1: 第一个错误答案的索引
        wrong_idx2: 第二个错误答案的索引
        
    Returns:
        list: 轨迹对话列表
    """
    choices = item.get("choices", [])
    question_text = item.get("question_text", "")
    question_image = item.get("question_image", "")
    correct_letter = string.ascii_uppercase[correct_idx]
    wrong_letter1 = string.ascii_uppercase[wrong_idx1]
    wrong_letter2 = string.ascii_uppercase[wrong_idx2]
    
    # 准备工具示例
    detect_example = prepare_detect_black_area_example(item)
    wrong_insert_example1 = prepare_insert_image_example(item, wrong_idx1)
    wrong_insert_example2 = prepare_insert_image_example(item, wrong_idx2)
    correct_insert_example = prepare_insert_image_example(item, correct_idx)
    
    trajectory = []
    
    # 用户提问
    trajectory.append({
        "role": "user",
        "content": [
            {"type": "text", "text": question_text},
            {"type": "image", "image": question_image}
        ] + [{"type": "image", "image": choice["image"]} for choice in choices]
    })
    
    # 助手回复 - 调用DetectBlackArea
    trajectory.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "<think>[**You need to implement this** Example: I need to identify the missing part in the first image. I can see there's a black area that needs to be filled. Let me first use the DetectBlackArea tool to precisely locate this region.]</think>\n\n<tool_call>\n" + json.dumps(detect_example["tool_call"], indent=2) + "\n</tool_call>"
            }
        ]
    })
    
    # 用户返回DetectBlackArea结果
    trajectory.append({
        "role": "user",
        "content": [
            {"type": "text", "text": json.dumps(detect_example["tool_output"], indent=2)}
        ]
    })
    
    # 助手回复 - 调用第一次InsertImage (第一个错误选项)
    trajectory.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": f"<think>[**You need to implement this** Example: Now I have identified the black area coordinates. Let me start by trying option {wrong_letter1}, which initially seems like it might fit based on my visual assessment of the content and edges.]</think>\n\n<tool_call>\n" + json.dumps(wrong_insert_example1["tool_call"], indent=2) + "\n</tool_call>"
            }
        ]
    })
    
    # 用户返回第一次InsertImage结果
    trajectory.append({
        "role": "user",
        "content": [
            {"type": "text", "text": json.dumps(wrong_insert_example1["tool_output"], indent=2)},
            {"type": "image", "image": wrong_insert_example1["image_path"]}
        ]
    })
    
    # 助手回复 - 调用第二次InsertImage (第二个错误选项)
    trajectory.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": f"<think>[**You need to implement this** Example: After inserting option {wrong_letter1}, I notice that it doesn't fit well. The colors and patterns don't align with the surrounding areas. Let me try option {wrong_letter2} next to see if it provides a better fit.]</think>\n\n<tool_call>\n" + json.dumps(wrong_insert_example2["tool_call"], indent=2) + "\n</tool_call>"
            }
        ]
    })
    
    # 用户返回第二次InsertImage结果
    trajectory.append({
        "role": "user",
        "content": [
            {"type": "text", "text": json.dumps(wrong_insert_example2["tool_output"], indent=2)},
            {"type": "image", "image": wrong_insert_example2["image_path"]}
        ]
    })
    
    # 助手回复 - 调用第三次InsertImage (正确选项)
    trajectory.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": f"<think>[**You need to implement this** Example: After trying options {wrong_letter1} and {wrong_letter2}, I see that neither provides a good fit. The edges don't align well, and the content doesn't flow naturally with the rest of the image. Let me now try option {correct_letter}, which is the last option to check.]</think>\n\n<tool_call>\n" + json.dumps(correct_insert_example["tool_call"], indent=2) + "\n</tool_call>"
            }
        ]
    })
    
    # 用户返回第三次InsertImage结果
    trajectory.append({
        "role": "user",
        "content": [
            {"type": "text", "text": json.dumps(correct_insert_example["tool_output"], indent=2)},
            {"type": "image", "image": correct_insert_example["image_path"]}
        ]
    })
    
    # 助手给出最终答案
    trajectory.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": f"<think>[**You need to implement this** Example: After trying all three options, I can clearly see that option {correct_letter} fits much better than options {wrong_letter1} and {wrong_letter2}. With option {correct_letter}, the colors, patterns, and edges align seamlessly with the surrounding parts of the image. This confirms that option {correct_letter} is the correct missing part.]</think>\n\n<response>After trying all three options, I can see that options {wrong_letter1} and {wrong_letter2} don't fit well - the edges don't align properly and the content doesn't flow naturally with the rest of the image.\n\nHowever, option {correct_letter} fits perfectly into the missing region. The edges align seamlessly, and the content flows naturally when this piece is inserted.\n\n\\boxed{{{correct_letter}}}</response>"
            }
        ]
    })
    
    return trajectory

def create_failed_tool_call_trajectory(item):
    """
    创建工具调用失败但仍使用工具的trajectory
    
    Args:
        item: 数据项
        
    Returns:
        list: 轨迹对话列表
    """
    choices = item.get("choices", [])
    question_text = item.get("question_text", "")
    question_image = item.get("question_image", "")
    correct_idx = item.get("correct_answer", {}).get("index", 0)
    correct_letter = string.ascii_uppercase[correct_idx]
    
    # 准备工具示例
    detect_example = prepare_detect_black_area_example(item)
    
    # 准备所有选项的InsertImage示例
    insert_examples = []
    for i in range(len(choices)):
        insert_examples.append(prepare_insert_image_example(item, i))
    
    trajectory = []
    
    # 用户提问
    trajectory.append({
        "role": "user",
        "content": [
            {"type": "text", "text": question_text},
            {"type": "image", "image": question_image}
        ] + [{"type": "image", "image": choice["image"]} for choice in choices]
    })
    
    # 助手回复 - 调用DetectBlackArea
    trajectory.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "<think>[**You need to implement this** Example: I need to identify the missing part in the first image. I can see there's a black area that needs to be filled. Let me first use the DetectBlackArea tool to precisely locate this region.]</think>\n\n<tool_call>\n" + json.dumps(detect_example["tool_call"], indent=2) + "\n</tool_call>"
            }
        ]
    })
    
    # 用户返回DetectBlackArea结果
    trajectory.append({
        "role": "user",
        "content": [
            {"type": "text", "text": json.dumps(detect_example["tool_output"], indent=2)}
        ]
    })
    
    # 为每个选项添加InsertImage调用和结果
    for i, example in enumerate(insert_examples):
        option_letter = string.ascii_uppercase[i]
        
        # 助手回复 - 调用InsertImage
        trajectory.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<think>[**You need to implement this** Example: Now let me try inserting option {option_letter} into the black area to see if it fits well.]</think>\n\n<tool_call>\n" + json.dumps(example["tool_call"], indent=2) + "\n</tool_call>"
                }
            ]
        })
        
        # 用户返回InsertImage结果
        trajectory.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(example["tool_output"], indent=2)},
                {"type": "image", "image": example["image_path"]}
            ]
        })
    
    # 助手给出最终答案（尽管工具失败，但仍然可以通过视觉分析得出结论）
    trajectory.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": f"<think>[**You need to implement this** Example: I've tried inserting all options into the black area, but none seem to fit perfectly due to some issues with the tool detection. Despite this technical limitation, I need to make a judgment based on visual analysis of the original images.\n\nLooking at the options and the original image with the black area, I need to determine which piece would most naturally complete the image. I'll analyze the edges, colors, and content patterns of each option compared to the surrounding areas of the black region.\n\nBased on my careful visual analysis, option {correct_letter} appears to be the most suitable match for the missing part. The content and edges of this piece seem to align best with the surrounding context of the image.]</think>\n\n<response>After trying to insert each option using the tools, I notice that the automatic insertion didn't provide perfect results due to some technical limitations. However, by carefully analyzing the original images:\n\nI can see that option {correct_letter} has features that best match the surrounding area of the black region. The colors, patterns, and edges of this piece align most naturally with the rest of the image.\n\nBased on my visual analysis, I can determine that the correct answer is option {correct_letter}.\n\n\\boxed{{{correct_letter}}}</response>"
            }
        ]
    })
    
    return trajectory

def create_no_tool_call_trajectory(item):
    """
    创建不使用工具的trajectory（直接通过视觉分析得出结论）
    
    Args:
        item: 数据项
        
    Returns:
        list: 轨迹对话列表
    """
    choices = item.get("choices", [])
    question_text = item.get("question_text", "")
    question_image = item.get("question_image", "")
    correct_idx = item.get("correct_answer", {}).get("index", 0)
    correct_letter = string.ascii_uppercase[correct_idx]
    
    trajectory = []
    
    # 用户提问
    trajectory.append({
        "role": "user",
        "content": [
            {"type": "text", "text": question_text},
            {"type": "image", "image": question_image}
        ] + [{"type": "image", "image": choice["image"]} for choice in choices]
    })
    
    # 助手直接给出答案（不使用工具）
    trajectory.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": f"<think>[**You need to implement this** Example: I need to determine which of the provided images fits into the black area in the first image. Instead of using tools, I'll carefully analyze the visual features of each option.\n\nThe black area in the main image appears to be at [location in image]. Looking at the edges around this black area, I need to find a piece that would naturally extend the patterns, colors, and content.\n\nOption {correct_letter} seems to have edges that would align well with the surrounding areas of the black region. The colors and patterns in this piece appear to be a natural continuation of what's visible around the black area.\n\nAfter comparing all options, I believe option {correct_letter} is the most suitable match for the missing part. The content and edges of this piece align best with the surrounding context of the image.]</think>\n\n<response>I'll carefully analyze which piece best fits into the black area of the first image.\n\nLooking at the first image, I can see a black area that needs to be filled. By examining the edges, colors, and patterns surrounding this area, I can determine which option would create a coherent complete image.\n\nAfter comparing all options with the original image:\n\nOption {correct_letter} appears to be the correct missing piece. The colors, patterns, and edges of this image would create a natural continuation of the surrounding content when placed in the black area.\n\n\\boxed{{{correct_letter}}}</response>"
            }
        ]
    })
    
    return trajectory

def convert_conversation_into_sharegpt(conversation, item_id):
    """将对话转换为ShareGPT格式"""
    sharegpt_conversation = []
    
    # 添加system prompt
    sharegpt_conversation.append({
        "from": "system",
        "value": TOOL_PROMPTS
    })
    
    sharegpt_images = []
    # 处理对话中的每条消息
    for message in conversation:
        role = "human" if message['role'] == 'user' else "gpt"
        content = message['content']
        new_content_str = ""
        
        # 处理每种内容类型
        for content_item in content:
            if content_item["type"] == "text":
                new_content_str += content_item["text"] + "\n"
            elif content_item["type"] == "image":
                image_path = content_item["image"]
                image_id = f"img_{len(sharegpt_images) + 1}"
                sharegpt_images.append(image_path)
                new_content_str += f"<image>\n"
            else:
                raise ValueError(f"未知的内容类型: {content_item['type']}")
        
        # 创建新消息
        new_message = {
            "from": role,
            "value": new_content_str.strip()
        }
        sharegpt_conversation.append(new_message)
    
    # 创建最终结果
    res = {
        "id": item_id,
        "conversations": sharegpt_conversation,
        "images": sharegpt_images,
    }
    return res

def select_items(dataset, count, condition=None, tool_failed=None, num_choices=None):
    """
    从数据集中选择满足条件的项
    
    Args:
        dataset: 数据集
        count: 要选择的项数
        condition: 额外的条件函数
        tool_failed: 如果指定，则筛选工具失败(True)或成功(False)的项
        num_choices: 如果指定，则筛选有特定选项数的项
        
    Returns:
        list: 选择的项列表
    """
    filtered = []
    
    for item in dataset:
        if tool_failed is not None and item.get("tools", {}).get("tool_failed", False) != tool_failed:
            continue
        
        if num_choices is not None and len(item.get("choices", [])) != num_choices:
            continue
        
        if condition is not None and not condition(item):
            continue
        
        filtered.append(item)
    
    if len(filtered) < count:
        print(f"警告：满足条件的项只有{len(filtered)}个，少于请求的{count}个")
        return filtered
    
    # 随机选择指定数量的项
    return random.sample(filtered, count)

def call_gemini_for_reasoning(client, model_name, conversation, item_id, trajectory_type, max_retries=3, retry_interval=5):
    """
    调用Gemini补充推理过程，参考path_navigation_stage2.py中的方法
    
    Args:
        client: Gemini客户端
        model_name: 使用的模型名称
        conversation: 要增强的对话
        item_id: 数据项ID
        trajectory_type: trajectory类型，用于选择适当的prompt
        max_retries: 最大重试次数
        retry_interval: 重试间隔（秒）
        
    Returns:
        str: Gemini的响应
    """
    start_time = time.time()
    
    # 从对话中提取需要改写的思考部分和相关图片
    think_contents = []
    input_parts = []
    
    # 分析对话，提取占位符内容和对应的图像
    for i, message in enumerate(conversation):
        input_parts.append(types.Part.from_text(text=f"\n{message['role']}:"))
        for content in message['content']:
            if content["type"] == "text":
                input_parts.append(types.Part.from_text(text=content["text"]))
            elif content["type"] == "image":
                try:
                    image_path = content["image"]
                    image = Image.open(image_path).convert("RGB")
                    image_base64 = pil_to_base64(image)
                    
                    input_parts.append(types.Part.from_bytes(data=base64.b64decode(image_base64), mime_type="image/png"))
                except Exception as e:
                    print(f"加载图像 {image_path} 失败: {e}")
                    
            else:
                raise ValueError(f"未知的内容类型: {content['type']}")
    
    # 如果没有找到需要改写的思考内容，直接返回
    if not input_parts:
        print(f"未在对话中找到需要改写的思考部分，跳过")
        return None
    
    # 为API请求准备内容
    # 根据trajectory类型选择适当的prompt
    if "first" in trajectory_type or "second" in trajectory_type or "third" in trajectory_type or "wrong_tool_call" in trajectory_type:
        system_prompt = INSERT_VERIFY_PROMPT  # 使用工具的提示
    else:
        system_prompt = SELF_THINK_PROMPT  # 自我思考的提示
    
    # 构建API请求内容
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=system_prompt)]
        ),
        types.Content(
            role="user",
            parts=input_parts
        )
    ]
    
    # 调用API并处理响应
    for attempt in range(max_retries):
        try:
            # 配置生成参数
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 2048,
                "response_mime_type": "text/plain",
            }
           
            response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=-1),
                        response_mime_type="text/plain",
                    )
            )
            
            # 记录时间和处理响应
            elapsed_time = time.time() - start_time
            print(f"Gemini API 调用耗时: {elapsed_time:.2f} 秒")
            
            return response.text
            
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_interval} 秒后重试...")
                time.sleep(retry_interval)
    
    print(f"API调用超过最大重试次数，跳过此条目")
    return None

def load_checkpoint(checkpoint_path):
    """加载检查点数据
    
    Args:
        checkpoint_path: 检查点文件路径
    
    Returns:
        dict: 包含已处理项目ID的字典
    """
    try:
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {'processed_ids': {}, 'enhanced_ids': {}}

def save_checkpoint(checkpoint_path, processed_ids, enhanced_ids):
    """保存检查点数据
    
    Args:
        checkpoint_path: 检查点文件路径
        processed_ids: 已处理项目ID的字典
        enhanced_ids: 已增强项目ID的字典
    """
    data = {
        'processed_ids': processed_ids,
        'enhanced_ids': enhanced_ids
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f)

def enhance_trajectories_with_gemini(trajectories, client, model_name, output_dir, trajectory_type, max_retries=3, retry_interval=5, checkpoint_path=None):
    """
    使用Gemini增强trajectory中的推理过程，支持断点续传
    
    Args:
        trajectories: 原始trajectory列表
        client: Gemini客户端
        model_name: 使用的Gemini模型名称
        output_dir: 输出目录
        trajectory_type: trajectory类型标识
        max_retries: 最大重试次数
        retry_interval: 重试间隔（秒）
        checkpoint_path: 检查点文件路径
        
    Returns:
        list: 增强后的trajectory列表
    """
    enhanced_trajectories = []
    
    # 加载检查点数据
    if checkpoint_path is None:
        checkpoint_path = os.path.join(output_dir, f"{trajectory_type}_checkpoint.json")
    
    checkpoint_data = load_checkpoint(checkpoint_path)
    processed_ids = checkpoint_data['processed_ids']
    enhanced_ids = checkpoint_data['enhanced_ids']
    
    # 如果轨迹类型不在字典中，初始化为空集合
    if trajectory_type not in processed_ids:
        processed_ids[trajectory_type] = []
    if trajectory_type not in enhanced_ids:
        enhanced_ids[trajectory_type] = []
    
    # 加载已生成的增强轨迹文件（如果存在）
    enhanced_path = os.path.join(output_dir, f"{trajectory_type}_raw.json")
    if os.path.exists(enhanced_path):
        try:
            with open(enhanced_path, 'r') as f:
                existing_enhanced = json.load(f)
                enhanced_trajectories = existing_enhanced
                print(f"已从 {enhanced_path} 加载 {len(enhanced_trajectories)} 条增强轨迹")
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"无法加载现有增强轨迹文件 {enhanced_path}，将创建新文件")
    
    # 为每个trajectory调用Gemini（跳过已处理的）
    for traj_data in tqdm(trajectories, desc=f"增强 {trajectory_type} trajectories"):
        item_id = traj_data["item_id"]
        
        # 检查是否已处理过这个ID
        if item_id in processed_ids[trajectory_type]:
            print(f"跳过已处理的项 {item_id}")
            continue
        
        traj = traj_data["trajectory"]
        
        # 调用Gemini补充推理
        enhanced_text = call_gemini_for_reasoning(
            client=client,
            model_name=model_name,
            conversation=traj,
            item_id=item_id,
            trajectory_type=trajectory_type,
            max_retries=max_retries,
            retry_interval=retry_interval
        )
        
        # 标记此ID为已处理
        processed_ids[trajectory_type].append(item_id)
        
        # 定期保存检查点，防止处理过程中断
        if len(processed_ids[trajectory_type]) % 5 == 0:
            save_checkpoint(checkpoint_path, processed_ids, enhanced_ids)
        
        if not enhanced_text:
            print(f"无法为 {item_id} 生成增强推理，跳过")
            continue
        
        # 提取所有的<think>标签内容
        think_blocks = extract_think_tags(enhanced_text)
        
        if not think_blocks:
            print(f"无法从 {item_id} 的响应中提取<think>标签，跳过")
            continue
        
        # 更新trajectory中的推理部分
        think_index = 0
        for i, message in enumerate(traj):
            if message["role"] == "assistant" and "<think>[**You need to implement this**" in message['content'][0]['text']:
                if think_index < len(think_blocks):
                    # 替换思考部分
                    old_text = message['content'][0]['text']
                    new_text = old_text.replace(
                        old_text.split("<think>")[1].split("</think>")[0],
                        think_blocks[think_index]
                    )
                    message['content'][0]['text'] = new_text
                    think_index += 1
        
        # 添加到增强后的trajectories
        enhanced_trajectories.append({
            "item_id": item_id,
            "trajectory": traj
        })
        
        # 标记此ID为已增强
        enhanced_ids[trajectory_type].append(item_id)
        
        # 每处理10个项目保存一次中间结果
        if len(enhanced_trajectories) % 10 == 0:
            with open(enhanced_path, 'w') as f:
                json.dump(enhanced_trajectories, f, indent=2)
            print(f"已保存 {len(enhanced_trajectories)} 条增强轨迹到 {enhanced_path}")
            # 更新检查点
            save_checkpoint(checkpoint_path, processed_ids, enhanced_ids)
    
    # 保存最终结果
    with open(enhanced_path, 'w') as f:
        json.dump(enhanced_trajectories, f, indent=2)
    
    # 更新检查点
    save_checkpoint(checkpoint_path, processed_ids, enhanced_ids)
    
    return enhanced_trajectories

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="为拼图数据生成SFT轨迹数据")
    parser.add_argument("--input_file", type=str, default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/jigsaw/jigsaw_metadata_v1/splits/sft/dataset_sft.json", help="完整数据集JSON文件")
    parser.add_argument("--output_dir", type=str, default="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/jigsaw/jigsaw_metadata_v1/splits/sft/gemini_enhanced", help="输出目录")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini模型名称")
    parser.add_argument("--correct_2", type=int, default=200, help="2选项正确轨迹数量")
    parser.add_argument("--correct_3", type=int, default=200, help="3选项正确轨迹数量")
    parser.add_argument("--wrong_call", type=int, default=100, help="工具调用失败但仍使用工具的轨迹数量")
    parser.add_argument("--wrong_no_call", type=int, default=100, help="工具调用失败且不使用工具的轨迹数量")
    parser.add_argument("--max_retry", type=int, default=3, help="Gemini API调用最大重试次数")
    parser.add_argument("--retry_interval", type=int, default=5, help="Gemini API调用重试间隔（秒）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--not-resume", action="store_true", help="不从检查点恢复执行")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 检查点路径
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.json")
    
    # 初始化Gemini客户端
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("环境变量 GEMINI_API_KEY 未设置")
        return
    
    client = genai.Client(api_key=api_key)
    
    # 加载检查点数据
    checkpoint_data = load_checkpoint(checkpoint_path)
    processed_ids = checkpoint_data['processed_ids']
    enhanced_ids = checkpoint_data['enhanced_ids']
    
    # 加载数据集
    print(f"从 {args.input_file} 加载数据集...")
    with open(args.input_file, "r") as f:
        dataset = json.load(f)
    print(f"加载完成，共 {len(dataset)} 条记录")
    
    # 筛选SFT分割的数据
    sft_data = [item for item in dataset if item.get("split") == "sft"]
    print(f"SFT分割有 {len(sft_data)} 条记录")
    
    # 创建不同类型的轨迹
    trajectories = {
        "correct_first_2choices": [],   # 2选项，第一次就选对
        "correct_second_2choices": [],  # 2选项，第二次才选对
        "correct_first_3choices": [],   # 3选项，第一次就选对
        "correct_second_3choices": [],  # 3选项，第二次才选对
        "correct_third_3choices": [],   # 3选项，第三次才选对
        "wrong_tool_call": [],          # 工具调用失败，但仍使用工具
        "wrong_no_tool_call": []        # 工具调用失败，不使用工具
    }
    
    # 如果是继续执行，检查每个类别的已处理ID
    if not args.not_resume:
        # 对每种类型，加载已生成的轨迹
        for traj_type in trajectories.keys():
            raw_path = os.path.join(args.output_dir, f"{traj_type}_raw.json")
            if os.path.exists(raw_path):
                try:
                    with open(raw_path, 'r') as f:
                        existing_data = json.load(f)
                        processed_item_ids = [item["item_id"] for item in existing_data]
                        print(f"{traj_type}: 已处理 {len(processed_item_ids)} 个项目")
                        
                        # 如果检查点中没有这些ID，添加它们
                        if traj_type not in processed_ids:
                            processed_ids[traj_type] = []
                        if traj_type not in enhanced_ids:
                            enhanced_ids[traj_type] = []
                            
                        for item_id in processed_item_ids:
                            if item_id not in processed_ids[traj_type]:
                                processed_ids[traj_type].append(item_id)
                            if item_id not in enhanced_ids[traj_type]:
                                enhanced_ids[traj_type].append(item_id)
                except (json.JSONDecodeError, FileNotFoundError):
                    print(f"无法加载轨迹文件 {raw_path}")
    
    # 保存更新的检查点
    save_checkpoint(checkpoint_path, processed_ids, enhanced_ids)
    
    # 1. 选择2选项的正确数据项
    items_2choices = select_items(sft_data, args.correct_2, tool_failed=False, num_choices=2)
    
    # 2. 选择3选项的正确数据项
    items_3choices = select_items(sft_data, args.correct_3, tool_failed=False, num_choices=3)
    
    # 3. 选择工具调用失败但仍使用工具的数据项
    items_wrong_call = select_items(sft_data, args.wrong_call, tool_failed=True)
    
    # 4. 选择工具调用失败且不使用工具的数据项
    items_wrong_no_call = select_items(sft_data, args.wrong_no_call, tool_failed=True)
    
    print(f"选择的2选项数据: {len(items_2choices)}")
    print(f"选择的3选项数据: {len(items_3choices)}")
    print(f"选择的工具调用失败数据 (使用工具): {len(items_wrong_call)}")
    print(f"选择的工具调用失败数据 (不使用工具): {len(items_wrong_no_call)}")
    
    # 如果是继续执行，过滤掉已处理的项
    if args.not_resume:
        print("从上次中断的地方继续执行...")
    
    # 创建2选项的轨迹，跳过已处理的项
    for idx, item in enumerate(items_2choices):
        item_id = item["id"]
        # 检查是否已处理此项
            
        correct_idx = item["correct_answer"]["index"]
        wrong_idx = 1 - correct_idx  # 在2选项中，另一个选项就是错误选项
        
        # 第一次就选对
        if not ("correct_first_2choices" in processed_ids and item_id in processed_ids["correct_first_2choices"]):
            traj = create_first_attempt_trajectory(item, correct_idx)
            trajectories["correct_first_2choices"].append({
                "item_id": item_id,
                "item": item,
                "trajectory": traj
            })
        
        # 第二次才选对
        if not ("correct_second_2choices" in processed_ids and item_id in processed_ids["correct_second_2choices"]):
            traj = create_second_attempt_trajectory(item, correct_idx, wrong_idx)
            trajectories["correct_second_2choices"].append({
                "item_id": item_id,
                "item": item,
                "trajectory": traj
            })
    
    # 创建3选项的轨迹，跳过已处理的项
    items_per_type = args.correct_3
    for idx, item in enumerate(items_3choices):
        item_id = item["id"]

        correct_idx = item["correct_answer"]["index"]
        wrong_indices = [i for i in range(3) if i != correct_idx]
        
    
        # 第一次就选对
        if not ("correct_first_3choices" in processed_ids and item_id in processed_ids["correct_first_3choices"]):
            traj = create_first_attempt_trajectory(item, correct_idx)
            trajectories["correct_first_3choices"].append({
                "item_id": item_id,
                "item": item,
                "trajectory": traj
            })
   
        # 第二次才选对
        if not ("correct_second_3choices" in processed_ids and item_id in processed_ids["correct_second_3choices"]):
            traj = create_second_attempt_trajectory(item, correct_idx, wrong_indices[0])
            trajectories["correct_second_3choices"].append({
                "item_id": item_id,
                "item": item,
                "trajectory": traj
            })

        # 第三次才选对
        if not ("correct_third_3choices" in processed_ids and item_id in processed_ids["correct_third_3choices"]):
            traj = create_third_attempt_trajectory(item, correct_idx, wrong_indices[0], wrong_indices[1])
            trajectories["correct_third_3choices"].append({
                "item_id": item_id,
                "item": item,
                "trajectory": traj
            })
    
    # 创建工具调用失败的轨迹，跳过已处理的项
    for item in items_wrong_call:
        item_id = item["id"]
        # 检查是否已处理此项
        if "wrong_tool_call" in processed_ids and item_id in processed_ids["wrong_tool_call"]:
            continue
            
        traj = create_failed_tool_call_trajectory(item)
        trajectories["wrong_tool_call"].append({
            "item_id": item_id,
            "item": item,
            "trajectory": traj
        })
    
    # 创建不使用工具的轨迹，跳过已处理的项
    for item in items_wrong_no_call:
        item_id = item["id"]
        # 检查是否已处理此项
        if "wrong_no_tool_call" in processed_ids and item_id in processed_ids["wrong_no_tool_call"]:
            continue
            
        traj = create_no_tool_call_trajectory(item)
        trajectories["wrong_no_tool_call"].append({
            "item_id": item_id,
            "item": item,
            "trajectory": traj
        })
    
    # 使用Gemini增强轨迹中的推理过程，支持断点续传
    print("使用Gemini增强轨迹中的推理过程...")
    enhanced_trajectories = {}
    for traj_type, traj_list in trajectories.items():
        if traj_list:
            enhanced = enhance_trajectories_with_gemini(
                traj_list,
                client, 
                args.model, 
                args.output_dir, 
                traj_type,
                args.max_retry,
                args.retry_interval,
                checkpoint_path
            )
            enhanced_trajectories[traj_type] = enhanced
            print(f"{traj_type}: 原始 {len(traj_list)}, 增强后 {len(enhanced)}")
    
    # 转换为ShareGPT格式并保存
    for traj_type in trajectories.keys():
        # 加载增强后的轨迹
        raw_path = os.path.join(args.output_dir, f"{traj_type}_raw.json")
        if os.path.exists(raw_path):
            try:
                with open(raw_path, 'r') as f:
                    traj_list = json.load(f)
                    
                # 转换为ShareGPT格式
                sharegpt_data = []
                for traj_data in traj_list:
                    item_id = traj_data["item_id"]
                    traj = traj_data["trajectory"]
                    sharegpt_item = convert_conversation_into_sharegpt(traj, item_id)
                    sharegpt_data.append(sharegpt_item)
                
                # 保存ShareGPT格式
                sharegpt_path = os.path.join(args.output_dir, f"{traj_type}_sharegpt.json")
                with open(sharegpt_path, 'w') as f:
                    json.dump(sharegpt_data, f, indent=2)
                    
                print(f"已保存 {len(sharegpt_data)} 条 {traj_type} ShareGPT格式轨迹")
                
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"处理 {traj_type} 时出错: {e}")
    
    # 计算并保存统计信息
    stats = {"total": 0, "types": {}}
    for traj_type in trajectories.keys():
        raw_path = os.path.join(args.output_dir, f"{traj_type}_raw.json")
        if os.path.exists(raw_path):
            try:
                with open(raw_path, 'r') as f:
                    traj_list = json.load(f)
                    stats["types"][traj_type] = len(traj_list)
                    stats["total"] += len(traj_list)
            except:
                stats["types"][traj_type] = 0
    
    # 保存统计信息
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n===== 数据生成统计 =====")
    print(f"总共生成轨迹: {stats['total']}")
    for traj_type, count in stats["types"].items():
        print(f"- {traj_type}: {count}")
    print(f"\n数据保存在: {args.output_dir}")

if __name__ == "__main__":
    # 设置代理(如需要)
    setup_openai_proxy()
    
    # 运行主函数
    main()