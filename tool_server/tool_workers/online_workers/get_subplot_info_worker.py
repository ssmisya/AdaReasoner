"""
A model worker executes the model.
"""

import torch
import numpy as np
from PIL import Image
import base64
import uuid
import os
import traceback
import re
from io import BytesIO
import sys
from pathlib import Path
import cv2
import json
import requests
import time
import contextlib
import httpx
import openai

from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

from tool_server.utils.server_utils import build_logger
from tool_server.utils.utils import load_image, pil_to_base64
from tool_server.utils.worker_arguments import WorkerArguments
from tool_server.utils.error_codes import *
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"get_subplot_info_worker_{worker_id}.log")

# vLLM 模型配置
VLLM_API_BASE_URL = "http://SH-IDC1-10-140-37-35:16112/v1"
VLLM_API_KEY = "not-needed"
VLLM_MODEL_NAME = "/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-VL-72B-Instruct"
        
@contextlib.contextmanager
def no_proxy():
    """一个上下文管理器，可以在其作用域内临时禁用代理环境变量。"""
    # 定义需要处理的代理环境变量键名
    proxy_keys = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
    
    # 保存原始的代理设置
    original_proxies = {key: os.environ.get(key) for key in proxy_keys}
    
    # 临时删除当前环境中的代理设置
    for key in proxy_keys:
        if key in os.environ:
            del os.environ[key]
            
    try:
        # yield 关键字将控制权交给 with 代码块
        yield
    finally:
        # with 代码块执行完毕后（无论是否发生异常），恢复原始的代理设置
        for key, value in original_proxies.items():
            if value is not None:
                os.environ[key] = value


class GetSubplotInfoWorker(BaseToolWorker):
    def __init__(self, worker_arguments: WorkerArguments = None):
        # 在调用父类初始化前先设置模型名称
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "GetSubplotInfo"
        super().__init__(worker_arguments)
            
        self.instruction = {
            "type": "function",
            "function": {
                "name": "GetSubplotInfo",
                "description": 
                    "Extract the bounding boxes of each subplot within the image along with their corresponding titles. Returns a dictionary mapping each title to its subplot bounding box.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier of the image to analyze, e.g., 'img_1'"
                        }
                    },
                    "required": ["image"]
                }
            }
        }
        
        self.controller_addr = "http://SH-IDC1-10-140-37-6:21112"
        

        # 初始化 OpenAI 客户端
        self.client = openai.OpenAI(
            api_key=VLLM_API_KEY, 
            base_url=VLLM_API_BASE_URL,
        )
                
    def init_model(self):
        logger.info(f"No need to initialize model {self.model_name}.")
        self.model = None
        
    def extract_subplots_from_image(self, image):
        """
        从图像中提取子图区域
        
        参数:
        image - PIL图像对象或numpy数组
        
        返回:
        subplot_bboxes - 包含所有子图bbox的字典
        """
        # 如果输入是PIL图像，转换为numpy数组
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            # 转换RGB到BGR (OpenCV格式)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_np = image
            
        # 转换为灰度图
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 膨胀边缘，使线条更粗
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        subplot_bboxes = {}
        subplot_count = 0
        
        # 获取图像尺寸
        height, width = image_np.shape[:2]
        min_area = height * width * 0.05  # 最小面积阈值，图像面积的5%
        
        # 遍历轮廓
        for contour in contours:
            # 过滤掉太小的轮廓
            if cv2.contourArea(contour) > min_area:  # 根据图像大小动态调整阈值
                # 获取边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 检查是否为矩形（通过比较轮廓面积与边界框面积的比例）
                rect_area = w * h
                contour_area = cv2.contourArea(contour)
                if contour_area / rect_area > 0.7:  # 可以根据实际情况调整比例
                    subplot_bboxes[f'subplot_{subplot_count}'] = {
                        'bbox': [x, y, x + w, y + h]  # [x_min, y_min, x_max, y_max]
                    }
                    subplot_count += 1
        
        # 如果没有检测到子图，尝试使用霍夫线变换方法
        if not subplot_bboxes:
            # 使用霍夫线变换检测直线
            lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is not None:
                # 创建一个新的图像来绘制线条
                line_image = np.zeros_like(image_np)
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 使用线条图像进行轮廓检测
                gray_lines = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_lines, 10, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 遍历轮廓
                for contour in contours:
                    if cv2.contourArea(contour) > min_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        subplot_bboxes[f'subplot_{subplot_count}'] = {
                            'bbox': [x, y, x + w, y + h]
                        }
                        subplot_count += 1
        
        return subplot_bboxes
    
    def get_worker_address(self, controller_addr, model_name):
        """获取特定工具的worker地址"""
        try:
            response = requests.post(
                controller_addr + "/get_worker_address",
                headers={"User-Agent": "FastChat Client"},
                json={"model": model_name}
            )
            if response.status_code == 200:
                return response.json()["address"]
            else:
                logger.error(f"获取工具地址失败: {response.text}")
                return None
        except Exception as e:
            logger.error(f"获取工具地址出错: {str(e)}")
            return None
    
    def call_ocr_tool(self, image_data):
        """调用OCR工具识别图像中的文本"""
        ocr_worker_addr = self.get_worker_address(self.controller_addr, "OCR")
        if not ocr_worker_addr:
            logger.error("无法获取OCR工具地址")
            return None
        
        try:
            # 准备OCR请求数据
            datas = {"image": image_data}
            
            # 发送请求
            response = requests.post(
                ocr_worker_addr + "/worker_generate",
                headers={"User-Agent": "FastChat Client"},
                json=datas,
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"OCR请求失败: {response.text}")
                return None
        except Exception as e:
            logger.error(f"调用OCR工具出错: {str(e)}")
            return None
    
    def handle_duplicate_keys(self, result_dict):
        """处理字典中的重复键，为重复键添加编号
        
        参数:
        result_dict - 可能包含重复键的字典
        
        返回:
        processed_dict - 处理后的字典，重复键会添加(1)、(2)等编号
        """
        # 创建一个新字典来存储处理后的结果
        processed_dict = {}
        # 记录每个键出现的次数
        key_counts = {}
        
        # 遍历原始字典中的所有键值对
        for key, value in result_dict.items():
            # 检查键是否已经在新字典中
            if key in processed_dict:
                # 获取当前键的计数
                count = key_counts.get(key, 1)
                # 为重复键创建新的键名
                new_key = f"{key}({count})"
                # 更新计数
                key_counts[key] = count + 1
                # 将新键值对添加到处理后的字典
                processed_dict[new_key] = value
            else:
                # 如果键是第一次出现，直接添加到新字典
                processed_dict[key] = value
                # 初始化计数为1
                key_counts[key] = 1
        
        return processed_dict
    
    def call_language_model(self, subplots, ocr_results, image_base64):
        """使用vLLM模型进行结果组合"""
        try:
            # 从OCR结果中提取文本信息
            text_detections = []
            if ocr_results and "detections" in ocr_results:
                for detection in ocr_results["detections"]:
                    if "label" in detection and "pixel_bbox" in detection:
                        text_detections.append({
                            "text": detection["label"],
                            "bbox": [
                                detection["pixel_bbox"]["x_min"],
                                detection["pixel_bbox"]["y_min"],
                                detection["pixel_bbox"]["x_max"],
                                detection["pixel_bbox"]["y_max"]
                            ]
                        })
            
        #     # 准备语言模型请求数据
        #     prompt = f"""
        # I have extracted subplot regions and text from an image. Please help match each subplot with its corresponding title.

        # Subplot information (coordinates in [x_min, y_min, x_max, y_max] format):
        # {json.dumps(subplots, indent=2)}

        # Detected text information:
        # {json.dumps(text_detections, indent=2)}

        # Based on the spatial relationship between text and subplot regions, please generate a dictionary where:
        # - Keys are the subplot titles (use the detected text that most likely represents a title)
        # - Values are the corresponding subplot coordinates [x_min, y_min, x_max, y_max]

        # IMPORTANT: If you find that the provided subplot coordinates are inaccurate or incomplete:
        # 1. Modify the coordinates if they don't properly contain the subplot content
        # 2. Add missing subplots if you can identify them from the text or visual information
        # 3. Merge overlapping or duplicate subplot regions

        # If a subplot doesn't have a clear title, use "subplot_X" as the key.

        # Return ONLY a valid JSON dictionary in this format:
        # {{"subplot_title1": [x_min, y_min, x_max, y_max], "subplot_title2": [x_min, y_min, x_max, y_max], ...}}
            # """

            prompt = f"""
You are an expert in scientific image analysis. Your task is to carefully observe the provided image and identify all subplots.
For each subplot, you must extract its full title and determine the precise pixel coordinates of its bounding box.
**Key features to identify a subplot:**
1.  **Title Pattern:** A subplot almost always has a title. Look for labels that start with an enumeration like `(a)`, `a)`, `(b)`, `b)`, etc., often followed by a descriptive text. The entire string (e.g., "(a) Temperature over time") should be treated as the title.
2.  **Graphical Area:** The coordinates should represent the bounding box that tightly encloses the main graphical content of the subplot (e.g., the plot area with axes, data points, lines, etc.). This bounding box should generally *exclude* the title itself, which is typically located just above or beside the plot.
**Output Format:**
Return ONLY a valid JSON dictionary where each key is the full title of a subplot and its value is a list of four integers representing the bounding box coordinates: `[x_min, y_min, x_max, y_max]`. The origin `(0, 0)` is the top-left corner of the image.
Example format:
`{{"(a) Title of first plot": [x1, y1, x2, y2], "(b) Title of second plot": [x3, y3, x4, y4]}}`
            """
            
            image = image_base64
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                    ]
                }
            ]
            
            # 只在请求的时候关闭代理
            with no_proxy():
                chat_completion = self.client.chat.completions.create(
                    model=VLLM_MODEL_NAME,
                    messages=messages,
                    max_tokens=20000,
                    temperature=0.7,
                )
            
            content = chat_completion.choices[0].message.content.strip()
            
            logger.info(f"Language model response: {content[:100]}...")
            
            # 查找JSON部分
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.info(f"Extracted JSON: {json_str[:100]}...")
                
                # 处理JSON字符串中的重复键问题
                processed_json_str = self.process_json_with_duplicate_keys(json_str)
                
                return json.loads(processed_json_str)
            else:
                # 如果没有找到JSON格式，尝试直接解析整个内容
                try:
                    # 处理可能的重复键
                    processed_content = self.process_json_with_duplicate_keys(content)
                    return json.loads(processed_content)
                except json.JSONDecodeError:
                    # 如果直接解析失败，尝试清理内容后再解析
                    # 移除可能的代码块标记
                    clean_content = re.sub(r'```json|```', '', content).strip()
                    # 处理可能的重复键
                    processed_content = self.process_json_with_duplicate_keys(clean_content)
                    return json.loads(processed_content)
                    
        except Exception as e:
            logger.error(f"调用语言模型工具出错: {str(e)}")
            return None
    
    def process_json_with_duplicate_keys(self, json_str):
        """处理JSON字符串中的重复键问题
        
        参数:
        json_str - 可能包含重复键的JSON字符串
        
        返回:
        processed_json_str - 处理后的JSON字符串，重复键会添加(1)、(2)等编号
        """
        # 使用正则表达式查找所有键值对
        pattern = r'"([^"]+)"\s*:\s*(\[[^\]]+\])'
        matches = re.findall(pattern, json_str)
        
        # 创建一个字典来记录每个键出现的次数
        key_counts = {}
        
        # 创建一个新的JSON字符串
        processed_json_str = "{"
        
        for key, value in matches:
            # 检查键是否已经在字典中
            if key in key_counts:
                # 获取当前键的计数
                count = key_counts[key]
                # 为重复键创建新的键名
                new_key = f"{key}({count})"
                # 更新计数
                key_counts[key] += 1
                # 将新键值对添加到处理后的JSON字符串
                processed_json_str += f'"{new_key}": {value}, '
            else:
                # 如果键是第一次出现，直接添加到JSON字符串
                processed_json_str += f'"{key}": {value}, '
                # 初始化计数为1
                key_counts[key] = 1
        
        # 移除最后的逗号和空格，并添加结束括号
        if processed_json_str.endswith(", "):
            processed_json_str = processed_json_str[:-2]
        processed_json_str += "}"
        
        return processed_json_str
    
    def base64_to_pil(self, base64_str):
        """将base64字符串转换为PIL图像"""
        try:
            image_data = base64.b64decode(base64_str)
            return Image.open(BytesIO(image_data))
        except Exception as e:
            logger.error(f"Base64转PIL图像失败: {str(e)}")
            return None
    
    def pil_to_base64(self, img, format="PNG"):
        """将PIL图像转换为base64字符串"""
        buffered = BytesIO()
        img.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def generate(self, params):
        tool_reward = 2.0
        # 计算Parameter Name Matching
        param_keys = set(params.keys())
        required_keys = set(self.instruction["function"]["parameters"]["required"])
        parameter_name_match_reward = len(param_keys & required_keys) / len(required_keys | param_keys)
        tool_reward = tool_reward + parameter_name_match_reward
        # 参数名称没有完全匹配，直接返回
        if parameter_name_match_reward < 1:
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": "Invalid parameters: expected keys: image.",
                "error_code": INVALID_PARAMETERS,
                "tool_reward": tool_reward
            }
        try:            
            # 加载图像
            try:
                image_data = params["image"]
                image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
                image_base64 = image_data
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Failed to load image: {str(e)}",
                    "error_code": CANNOT_LOAD_IMAGE,
                    "tool_reward": tool_reward
                }
                return pred_dict
            
            # 1. 提取子图区域
            try:
                subplot_bboxes = self.extract_subplots_from_image(image)
                logger.info(f"Detected {len(subplot_bboxes)} subplot regions.")
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Failed to extract subplot regions: {str(e)}",
                    "error_code": TOOL_RUN_FAILED
                }
                return pred_dict
            
            # 2. 调用OCR工具识别文本
            ocr_results = self.call_ocr_tool(image_base64)
            if not ocr_results or ocr_results.get("status") != "success":
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": "OCR detection failed",
                    "error_code": TOOL_RUN_FAILED
                }
                return pred_dict
            
            # 3. 调用语言模型进行结果组合
            try:
                final_result = self.call_language_model(subplot_bboxes, ocr_results, image_base64)
                print("getsubplotinfo_worker的final_result: ", final_result)
            except Exception as e:
                logger.error(f"组合结果失败: {str(e)}")
                final_result = None
            
            # 4. 判断是否成功提取到子图信息
            if (not subplot_bboxes or len(subplot_bboxes) == 0) and (not final_result or len(final_result) == 0):
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": "No subplot regions detected after both extraction methods",
                    "error_code": TOOL_RUN_FAILED,
                    "tool_reward": tool_reward
                }
                return pred_dict
            
            # 5. 如果语言模型返回了结果，使用它，否则使用基本提取结果
            if final_result and len(final_result) > 0:
                result_to_use = final_result
            else:
                # 如果语言模型未返回有效结果，使用基本提取的结果
                logger.warning("Language model returned empty result, using basic extraction results")
                result_to_use = {}
                for subplot_id, subplot_info in subplot_bboxes.items():
                    result_to_use[subplot_id] = subplot_info["bbox"]
                
                # 处理基本提取结果中可能的重复键
                result_to_use = self.handle_duplicate_keys(result_to_use)
            
            tool_reward += 1
            # 6. 返回结果
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "success",
                "message": "Successfully extracted subplot information",
                "subplots": result_to_use,
                "error_code": SUCCESS,
                "tool_reward": tool_reward
            }
            
            return pred_dict
            
        except Exception as e:
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error: {str(e)}\n Traceback: {traceback.format_exc()}\n",
                "error_code": TOOL_RUN_FAILED,
                "tool_reward": tool_reward
            }
            logger.error(f"子图信息提取操作错误: {e}")
            logger.error(traceback.format_exc())
            return pred_dict
    
    def get_tool_instruction(self):
        return self.instruction



if __name__ == "__main__":
    parser = HfArgumentParser((WorkerArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")

    worker = GetSubplotInfoWorker(
        worker_arguments=args
    )
    worker.run()