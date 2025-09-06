"""
A model worker executes the model.
"""

import uuid
import os
import re
import io
import numpy as np
from PIL import Image
import torch
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from tool_server.utils.worker_arguments import WorkerArguments
from tool_server.utils.error_codes import *
import matplotlib.pyplot as plt

from paddleocr import PaddleOCR

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None

np.random.seed(3)

@dataclass
class OCRToolArguments(WorkerArguments):
    """OCR工具工作器的参数"""
    max_concurrency: Optional[int] = field(
        default=5,
        metadata={"help": "Maximum number of concurrent requests to process."}
    )

class OCRToolWorker(BaseToolWorker):
    def __init__(self, worker_arguments: OCRToolArguments = None):
        # 在调用父类初始化前先设置模型名称和并发数
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "ocr"
        
        # 设置更高的并发处理能力
        if hasattr(worker_arguments, "max_concurrency"):
            worker_arguments.limit_model_concurrency = worker_arguments.max_concurrency
        else:
            worker_arguments.limit_model_concurrency = 10  # 默认允许10个并发请求
            
        super().__init__(worker_arguments)
        
        # 创建线程池用于并发处理请求
        self.thread_pool = ThreadPoolExecutor(max_workers=worker_arguments.limit_model_concurrency)
        
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Extracts and localizes text from the given image using OCR. Returns bounding boxes, recognized text, and confidence scores.",
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
        
    def init_model(self):
        logger.info(f"Initializing model {self.model_name} with concurrency {self.args.limit_model_concurrency}...")
        self.ocr_model = PaddleOCR(
            use_doc_orientation_classify=False, 
            use_doc_unwarping=False, 
            use_textline_orientation=False,
            lang='ch'  # 支持中英文
        )
        
    def get_tool_instruction(self):
        return self.instruction
        
    def generate_gate(self, params):
        """覆盖父类方法，使用线程池处理请求"""
        try:
            # 使用线程池提交任务
            future = self.thread_pool.submit(self.generate_impl, params)
            ret = future.result(timeout=12000)  # 设置超时时间
            return ret
        except Exception as e:
            logger.error(f"Error in generate_gate: {e}")
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error in generate_gate: {e}",
                "error_code": TOOL_RUN_FAILED
            }
    
    def generate_impl(self, params):
        """实现实际的生成逻辑"""
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
        
        required_keys_num = len(required_keys)
        # 初始化参数合规计数器
        correct_param_content_num = 0
        
        image = params["image"]
        text_threshold = params.get("text_threshold", 0.25)
        
        # If params are ok, continue
        try:
            try:
                img = base64_to_pil(image).convert("RGB")
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Cannot load image: {str(e)}",
                    "error_code": CANNOT_LOAD_IMAGE,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                }
                return pred_dict
            
            correct_param_content_num += 1

            result = self.ocr_model.predict(np.array(img))
            detections = []

            # PaddleOCR返回的格式是一个列表，包含一个字典
            # 字典有'rec_texts', 'rec_boxes', 'rec_scores'三个key
            if result and len(result) > 0:
                for text, box, confidence in zip(result[0]['rec_texts'], result[0]['rec_boxes'], result[0]['rec_scores']):
                    # Skip results with confidence below threshold
                    if confidence < text_threshold:
                        continue
                        
                    # Round confidence to 2 decimal places
                    confidence = round(float(confidence), 2)
                    
                    # PaddleOCR的box格式是[x_min, y_min, x_max, y_max]的数组
                    box_coords = box.tolist() if hasattr(box, 'tolist') else box
                    logger.info(f"Processing detection - text: '{text}', box: {box_coords}, confidence: {confidence}")
                    
                    # box_coords应该是[x_min, y_min, x_max, y_max]格式
                    if len(box_coords) == 4:
                        x_min, y_min, x_max, y_max = box_coords
                    else:
                        # 如果不是预期格式，跳过这个检测结果
                        logger.warning(f"Unexpected box format: {box_coords}")
                        continue

                    detections.append({
                        "label": text,  # PaddleOCR返回的是text而不是label
                        "confidence": confidence,
                        "pixel_bbox": {
                            "x_min": int(x_min),
                            "y_min": int(y_min),
                            "x_max": int(x_max),
                            "y_max": int(y_max)
                        }
                    })

            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "success",
                "detections": detections,
                "image_dimensions_pixels": {
                    "width": img.width,  # 确保转换为Python int
                    "height": img.height  # 确保转换为Python int
                },
                "error_code": SUCCESS,
                "tool_reward": tool_reward+correct_param_content_num/required_keys_num
            }
            
            # 释放GPU显存缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return pred_dict
            
        except Exception as e:
            logger.error(f"Error when ocr: {e}")
            logger.error(traceback.format_exc())
            
            # 即使出现异常也要清理GPU显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error: {str(e)}\nTraceback:{traceback.format_exc()}\n",
                "error_code": TOOL_RUN_FAILED,
                "tool_reward": tool_reward+(correct_param_content_num/required_keys_num if required_keys_num > 0 else 0)
            }
            return pred_dict
    
    # 使用generate_impl作为generate方法
    def generate(self, params):
        """为了与base_tool_worker兼容，提供一个generate方法"""
        return self.generate_impl(params)
        
    def setup_routes(self):
        """覆盖父类方法，添加新的API端点用于并发处理多个请求"""
        super().setup_routes()  # 调用父类的路由设置
        
        @self.app.post("/worker_generate_batch")
        async def api_generate_batch(request: Request):
            """批量处理多个请求的API端点"""
            batch_params = await request.json()
            if not isinstance(batch_params, list):
                return JSONResponse({"status": "failed", "message": "Expected a list of parameter objects"})
            
            results = []
            futures = []
            
            # 提交所有任务到线程池
            for params in batch_params:
                future = self.thread_pool.submit(self.generate_impl, params)
                futures.append(future)
            
            # 收集所有结果
            for future in futures:
                try:
                    result = future.result(timeout=12000)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing batch item: {e}")
                    results.append({
                        "tool_response_from": self.model_name,
                        "status": "failed",
                        "message": f"Processing error: {str(e)}",
                        "error_code": TOOL_RUN_FAILED
                    })
            
            return JSONResponse(results)

    def __del__(self):
        """析构函数，确保线程池正确关闭"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)


if __name__ == "__main__":
    # Use the new argument parser from transformers
    from transformers import HfArgumentParser
    from dataclasses import dataclass, field
    from typing import Optional
    
    parser = HfArgumentParser(OCRToolArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    logger.info(f"args: {args}")

    worker = OCRToolWorker(worker_arguments=args)
    worker.run()