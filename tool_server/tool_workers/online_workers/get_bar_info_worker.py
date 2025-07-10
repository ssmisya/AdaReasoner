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
import openai
import contextlib
import httpx  # 在文件顶部导入模块

from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

from tool_server.utils.server_utils import build_logger
from tool_server.utils.utils import load_image, pil_to_base64
from tool_server.utils.worker_arguments import WorkerArguments
from tool_server.utils.error_codes import *
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"get_bar_info_worker_{worker_id}.log")

# vLLM 模型配置
VLLM_API_BASE_URL = "http://SH-IDC1-10-140-37-35:16112/v1"
VLLM_API_KEY = "not-needed"
VLLM_MODEL_NAME = "/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-VL-72B-Instruct"

@dataclass
class GetBarInfoArguments(WorkerArguments):
    """
    获取柱状图信息的工作参数
    """
    api_key: Optional[str] = field(default=None, metadata={
        "help": "OpenAI API key for accessing the model."
    })
    api_base_url : Optional[str] = field(default=None, metadata={
        "help": "Base URL for the OpenAI API."
    })
    api_model_name: Optional[str] = field(default=None, metadata={
        "help": "Name of the model to use for processing bar chart information."
    })
        

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

class GetBarInfoWorker(BaseToolWorker):
    def __init__(self, worker_arguments: WorkerArguments = None):
        # 在调用父类初始化前先设置模型名称
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "GetBarInfo"
        super().__init__(worker_arguments)
        self.api_key = worker_arguments.api_key
        self.api_base_url = worker_arguments.api_base_url
        self.api_model_name = worker_arguments.api_model_name
        
        self.instruction = {
            "type": "function",
            "function": {
                "name": "GetBarInfo",
                "description": 
                    "Extract bounding boxes of all bars in the image along with their corresponding axis titles or labels. Returns a dictionary mapping each label to its bounding box.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier of the image to analyze, e.g., 'img_1'."
                        }
                    },
                    "required": ["image"]
                }
            }
        }
        
        # 初始化 OpenAI 客户端
        self.client = openai.OpenAI(
            api_key=self.api_key, 
            base_url=self.api_base_url,
        )
        
    def init_model(self):
        logger.info(f"No need to initialize model {self.model_name}.")
        self.model = None
        
    def extract_bars_from_image(self, image):
        """
        从图像中提取柱状图
        
        参数:
        image - PIL图像对象或numpy数组
        
        返回:
        bar_bboxes - 包含所有柱子bbox的字典
        """
        # 如果输入是PIL图像，转换为numpy数组
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            # 转换RGB到BGR (OpenCV格式)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_np = image
            
        # 转换为HSV颜色空间
        hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        
        # 定义常见柱状图颜色范围
        color_ranges = [
            # 蓝色范围
            (np.array([100, 150, 50]), np.array([140, 255, 255])),
            # 红色范围 (由于HSV中红色在0和180附近，需要两个范围)
            (np.array([0, 150, 50]), np.array([10, 255, 255])),
            (np.array([170, 150, 50]), np.array([180, 255, 255])),
            # 绿色范围
            (np.array([40, 100, 50]), np.array([80, 255, 255])),
            # 黄色范围
            (np.array([20, 100, 50]), np.array([40, 255, 255])),
            # 青色范围
            (np.array([80, 100, 50]), np.array([100, 255, 255])),
            # 粉色范围
            (np.array([140, 100, 50]), np.array([170, 255, 255])),
            # 灰色范围
            (np.array([0, 0, 50]), np.array([180, 30, 200])),
            # 橙色范围
            (np.array([10, 150, 150]), np.array([25, 255, 255]))
        ]
        
        bar_bboxes = {}
        bar_count = 0
        
        # 遍历所有颜色范围
        for lower_hsv, upper_hsv in color_ranges:
            # 创建颜色掩码
            mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
            
            # 形态学操作去除噪点
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 遍历轮廓
            for contour in contours:
                # 过滤掉太小的轮廓
                if cv2.contourArea(contour) > 50:  # 可以根据实际情况调整阈值
                    # 获取边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    bar_bboxes[f'bar_{bar_count}'] = {
                        'bbox': [x, y, x + w, y + h],  # [x_min, y_min, x_max, y_max]
                        'area': int(cv2.contourArea(contour))
                    }
                    bar_count += 1
        
        return bar_bboxes
    
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
    
    def call_language_model(self, bar_bboxes, ocr_results, image: Image.Image):
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
            
            # 准备语言模型请求数据
        #     prompt = f"""
        # I have extracted bar regions and text from a bar chart image. Please help match each bar with its corresponding label or title.

        # Bar information (coordinates in [x_min, y_min, x_max, y_max] format):
        # {json.dumps(bar_bboxes, indent=2)}

        # Detected text information:
        # {json.dumps(text_detections, indent=2)}

        # Based on the spatial relationship between text and bar regions, please generate a dictionary where:
        # - Keys are the bar labels (use the detected text that most likely represents a label for each bar)
        # - Values are the corresponding bar coordinates [x_min, y_min, x_max, y_max]

        # IMPORTANT: If you find that the provided bar coordinates are inaccurate or incomplete:
        # 1. Modify the coordinates if they don't properly contain the bar content
        # 2. Add missing bars if you can identify them from the text or visual information
        # 3. Remove overlapping or duplicate bar regions

        # If a bar doesn't have a clear label, use "bar_X" as the key.

        # Return ONLY a valid JSON dictionary in this format:
        # {{"bar_label1": [x_min, y_min, x_max, y_max], "bar_label2": [x_min, y_min, x_max, y_max], ...}}
        #     """

            prompt = f"""
You are a highly meticulous visual analysis agent tasked with extracting data from bar charts.

Your objective is to generate a JSON object containing the x-axis labels and bounding box coordinates for every bar in the image.

Follow this exact procedure:
1.  **Scan the X-Axis:** First, identify all the text labels present on the chart's x-axis.
2.  **Map Labels to Bars:** For each label found, locate the corresponding bar (or stack of bars) vertically aligned with it.
3.  **Calculate Bounding Box:** Determine the tightest possible bounding box `[x_min, y_min, x_max, y_max]` for the entire bar.
    * **For stacked bars:** The box must enclose the complete stack.
    * **For regular bars:** The box encloses the single rectangle.
    * **Bars in a series usually have uniform widths, and the horizontal spacing between adjacent bars is typically consistent.
4.  **Construct the JSON:** Assemble the extracted data into the specified JSON format.

**Crucial Rules:**
* The keys of the JSON dictionary **must be** the text labels from the x-axis.
* The coordinates must be in pixels, with `(0, 0)` at the top-left corner.
* Return ONLY the final, valid JSON object. Do not include any other text, reasoning, or explanations in your response.

**Example of the process:**
* **Thought:** "I am analyzing a bar chart. First, I'll read the x-axis. I see three labels: 'Group A', 'Group B', 'Group C'.
    * Above 'Group A', there is a blue bar. I will find its bounding box.
    * Above 'Group B', there is a stacked bar (green and orange). I will find the bounding box for the entire stack.
    * Above 'Group C', there is another blue bar. I will find its bounding box.
    * Now, I will format this into the final JSON."

**Output Format:**
Return ONLY a valid JSON dictionary. The keys should be the labels found on the x-axis, and the values should be the bounding box coordinates `[x_min, y_min, x_max, y_max]`. Coordinates must be in pixels, with the origin `(0, 0)` at the top-left corner of the image.

Example format:
`{{"x-axis label for bar 1": [x1, y1, x2, y2], "x-axis label for bar 2": [x3, y3, x4, y4]}}`
            """
            
            # 确保图像正确转换为base64格式
            if isinstance(image, Image.Image):
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            else:
                # 如果已经是base64字符串，直接使用
                base64_image = image
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
            
            # 只在请求的时候关闭代理
            with no_proxy():
                chat_completion = self.client.chat.completions.create(
                    model=self.api_model_name,
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
            
            # 1. 提取柱子区域
            try:
                bar_bboxes = self.extract_bars_from_image(image)
                if not bar_bboxes:
                    logger.warning("No bar regions detected by color-based method, continuing with OCR and language model...")
                else:
                    logger.info(f"Detected {len(bar_bboxes)} bar regions.")
            except Exception as e:
                logger.error(f"Failed to extract bar regions: {str(e)}")
                bar_bboxes = {}  # 设置为空字典，继续执行
            
            # 2. 调用OCR工具识别文本
            ocr_results = self.call_ocr_tool(image_base64)
            if not ocr_results or ocr_results.get("status") != "success":
                logger.warning("OCR detection failed or returned no results")
                ocr_results = {"detections": []}  # 设置为空列表，继续执行
            
            # 3. 调用语言模型进行结果组合
            final_result = {}
            try:
                # 即使前面的步骤没有成功，也尝试使用语言模型
                lm_result = self.call_language_model(bar_bboxes, ocr_results, image)
                if lm_result:
                    final_result = lm_result
                else:
                    logger.warning("Language model returned empty result")
                    # 如果语言模型没有返回结果，但我们有从第一步获取的柱子信息
                    if bar_bboxes:
                        for bar_id, bar_info in bar_bboxes.items():
                            final_result[bar_id] = bar_info["bbox"]
            except Exception as e:
                logger.error(f"组合结果失败: {str(e)}")
                # 如果语言模型处理失败，但我们有从第一步获取的柱子信息
                if bar_bboxes:
                    for bar_id, bar_info in bar_bboxes.items():
                        final_result[bar_id] = bar_info["bbox"]
            
            # 4. 检查最终结果，如果仍然没有提取到柱子，则返回错误
            if not final_result:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": "No bar regions could be detected after all processing steps",
                    "error_code": TOOL_RUN_FAILED,
                    "tool_reward": tool_reward
                }
                return pred_dict
            
            tool_reward += 1
            # 5. 返回成功结果
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "success",
                "message": "Successfully extracted bar information",
                "bars": final_result,
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
            logger.error(f"柱状图信息提取操作错误: {e}")
            logger.error(traceback.format_exc())
            return pred_dict
    
    def get_tool_instruction(self):
        return self.instruction



if __name__ == "__main__":
    parser = HfArgumentParser((GetBarInfoArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")

    worker = GetBarInfoWorker(
        worker_arguments=args
    )
    worker.run()