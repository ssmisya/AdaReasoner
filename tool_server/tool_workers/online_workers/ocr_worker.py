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
import multiprocessing as mp
import queue
import time

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None

np.random.seed(3)

def gpu_worker_process(gpu_id, task_queue, result_queue):
    """GPU工作进程函数"""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 初始化PaddleOCR
    ocr_model = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='ch',
        device=f'gpu:0'  # 在进程内总是使用GPU:0，因为CUDA_VISIBLE_DEVICES已经设置
    )
    
    logger.info(f"GPU {gpu_id} worker process started")
    
    while True:
        try:
            # 获取任务
            task = task_queue.get(timeout=1)
            if task is None:  # 退出信号
                break
                
            task_id, img_array, text_threshold = task
            start_time = time.time()
            
            # 执行OCR推理
            result = ocr_model.predict(img_array)
            detections = []
            
            if result and len(result) > 0:
                first_result = result[0]
                if hasattr(first_result, 'rec_texts'):
                    # OCRResult对象格式
                    texts = first_result.rec_texts
                    boxes = first_result.rec_boxes
                    scores = first_result.rec_scores
                elif isinstance(first_result, dict):
                    # 字典格式
                    texts = first_result['rec_texts']
                    boxes = first_result['rec_boxes']
                    scores = first_result['rec_scores']
                else:
                    logger.warning(f"GPU {gpu_id}: Unexpected result format")
                    texts, boxes, scores = [], [], []
                
                for text, box, confidence in zip(texts, boxes, scores):
                    if confidence < text_threshold:
                        continue
                        
                    confidence = round(float(confidence), 2)
                    box_coords = box.tolist() if hasattr(box, 'tolist') else box
                    
                    if len(box_coords) == 4:
                        x_min, y_min, x_max, y_max = box_coords
                        detections.append({
                            "label": text,
                            "confidence": confidence,
                            "pixel_bbox": {
                                "x_min": int(x_min),
                                "y_min": int(y_min),
                                "x_max": int(x_max),
                                "y_max": int(y_max)
                            }
                        })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 返回结果
            result_data = {
                "status": "success",
                "detections": detections,
                "processing_time": processing_time,
                "gpu_id": gpu_id
            }
            
            result_queue.put((task_id, result_data))
            logger.info(f"GPU {gpu_id} processed task {task_id} in {processing_time:.2f}s, found {len(detections)} texts")
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"GPU {gpu_id} error: {e}")
            error_result = {
                "status": "failed",
                "message": str(e),
                "detections": [],
                "gpu_id": gpu_id
            }
            if 'task_id' in locals():
                result_queue.put((task_id, error_result))

@dataclass
class OCRToolArguments(WorkerArguments):
    """OCR工具工作器的参数"""
    max_concurrency: Optional[int] = field(
        default=5,
        metadata={"help": "Maximum number of concurrent requests to process."}
    )
    gpu_ids: Optional[str] = field(
        default="0",
        metadata={"help": "Comma-separated GPU IDs to use for multi-GPU inference, e.g., '0,1,2,3'"}
    )
    enable_multi_gpu: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable multi-GPU inference"}
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
            worker_arguments.limit_model_concurrency = 5  # 默认允许5个并发请求
        
        # 在调用父类初始化前先初始化多GPU配置
        self.enable_multi_gpu = getattr(worker_arguments, 'enable_multi_gpu', False)
        self.gpu_ids = []
        if hasattr(worker_arguments, 'gpu_ids') and worker_arguments.gpu_ids:
            self.gpu_ids = [int(x.strip()) for x in worker_arguments.gpu_ids.split(',')]
        else:
            self.gpu_ids = [0]  # 默认使用GPU 0
            
        logger.info(f"Multi-GPU enabled: {self.enable_multi_gpu}, GPU IDs: {self.gpu_ids}")
        
        # 多GPU相关变量初始化
        self.gpu_processes = []
        self.task_queues = []
        self.result_queue = None
        self.task_counter = 0
        self.pending_tasks = {}
        
        # 调用父类初始化
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
        
        if self.enable_multi_gpu and len(self.gpu_ids) > 1:
            logger.info(f"Initializing multi-GPU multiprocessing with GPUs: {self.gpu_ids}")
            self._init_multi_gpu_processes()
        else:
            logger.info(f"Initializing single-GPU PaddleOCR with GPU: {self.gpu_ids[0]}")
            self._init_single_gpu()
    
    def _init_single_gpu(self):
        """初始化单GPU模式"""
        device_str = f'gpu:{self.gpu_ids[0]}'
        self.ocr_model = PaddleOCR(
            use_doc_orientation_classify=False, 
            use_doc_unwarping=False, 
            use_textline_orientation=False,
            lang='ch',  # 支持中英文
            device=device_str
        )
        logger.info(f"Single PaddleOCR model loaded on {device_str}")
    
    def _init_multi_gpu_processes(self):
        """使用multiprocessing初始化多GPU进程"""
        # 设置multiprocessing启动方法为spawn
        mp.set_start_method('spawn', force=True)
        
        # 创建共享队列
        self.result_queue = mp.Queue()
        
        # 为每个GPU创建进程和队列
        for gpu_id in self.gpu_ids:
            task_queue = mp.Queue()
            self.task_queues.append(task_queue)
            
            # 创建GPU工作进程
            process = mp.Process(
                target=gpu_worker_process,
                args=(gpu_id, task_queue, self.result_queue)
            )
            process.start()
            self.gpu_processes.append(process)
            logger.info(f"Started GPU worker process for GPU {gpu_id}")
        
        # 启动结果收集线程
        import threading
        self.result_collector_thread = threading.Thread(target=self._collect_results)
        self.result_collector_thread.daemon = True
        self.result_collector_thread.start()
        
        logger.info(f"Multi-GPU setup completed with {len(self.gpu_ids)} GPUs")
    
    def _collect_results(self):
        """收集GPU处理结果的线程"""
        while True:
            try:
                task_id, result = self.result_queue.get(timeout=1)
                if task_id in self.pending_tasks:
                    # 将结果传递给等待的任务
                    self.pending_tasks[task_id].put(result)
                    del self.pending_tasks[task_id]
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error collecting results: {e}")
        
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

            # 根据GPU配置选择处理方式
            if self.enable_multi_gpu and len(self.gpu_ids) > 1:
                ocr_result = self._process_ocr_multi_gpu(np.array(img), text_threshold)
            else:
                ocr_result = self._process_ocr_single_gpu(np.array(img), text_threshold)
            
            # 构建返回结果
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": ocr_result["status"],
                "detections": ocr_result["detections"],
                "image_dimensions_pixels": {
                    "width": img.width,
                    "height": img.height
                },
                "error_code": SUCCESS if ocr_result["status"] == "success" else TOOL_RUN_FAILED,
                "tool_reward": tool_reward+correct_param_content_num/required_keys_num
            }
            
            if ocr_result["status"] == "failed":
                pred_dict["message"] = ocr_result.get("message", "Unknown error")
            
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
    
    def _process_ocr_single_gpu(self, img_array, text_threshold):
        """单GPU OCR处理方法"""
        detections = []
        
        try:
            # 使用PaddleOCR进行推理
            result = self.ocr_model.predict(img_array)
            
            if result and len(result) > 0:
                first_result = result[0]
                if hasattr(first_result, 'rec_texts'):
                    # OCRResult对象格式
                    texts = first_result.rec_texts
                    boxes = first_result.rec_boxes
                    scores = first_result.rec_scores
                elif isinstance(first_result, dict):
                    # 字典格式
                    texts = first_result['rec_texts']
                    boxes = first_result['rec_boxes']
                    scores = first_result['rec_scores']
                else:
                    logger.warning(f"Unexpected result format: {type(first_result)}")
                    return {"status": "failed", "message": "Unexpected result format", "detections": []}
                
                for text, box, confidence in zip(texts, boxes, scores):
                    if confidence < text_threshold:
                        continue
                        
                    confidence = round(float(confidence), 2)
                    box_coords = box.tolist() if hasattr(box, 'tolist') else box
                    logger.info(f"Processing detection - text: '{text}', box: {box_coords}, confidence: {confidence}")
                    
                    if len(box_coords) == 4:
                        x_min, y_min, x_max, y_max = box_coords
                        detections.append({
                            "label": text,
                            "confidence": confidence,
                            "pixel_bbox": {
                                "x_min": int(x_min),
                                "y_min": int(y_min),
                                "x_max": int(x_max),
                                "y_max": int(y_max)
                            }
                        })
                    else:
                        logger.warning(f"Unexpected box format: {box_coords}")
            
            return {
                "status": "success",
                "detections": detections
            }
            
        except Exception as e:
            logger.error(f"Single GPU OCR processing error: {e}")
            return {
                "status": "failed",
                "message": str(e),
                "detections": []
            }
    
    def _process_ocr_multi_gpu(self, img_array, text_threshold):
        """多GPU OCR处理方法（使用负载均衡）"""
        try:
            # 生成任务ID
            self.task_counter += 1
            task_id = f"task_{self.task_counter}_{time.time()}"
            
            # 选择队列最短的GPU（负载均衡）
            queue_sizes = [q.qsize() for q in self.task_queues]
            selected_gpu_idx = queue_sizes.index(min(queue_sizes))
            selected_gpu_id = self.gpu_ids[selected_gpu_idx]
            
            logger.info(f"Task {task_id} assigned to GPU {selected_gpu_id} (queue size: {queue_sizes[selected_gpu_idx]})")
            
            # 创建结果队列
            result_queue = queue.Queue()
            self.pending_tasks[task_id] = result_queue
            
            # 将任务添加到选定的GPU队列
            self.task_queues[selected_gpu_idx].put((task_id, img_array, text_threshold))
            
            # 等待结果
            try:
                result = result_queue.get(timeout=30)  # 30秒超时
                return result
            except queue.Empty:
                logger.error(f"Task {task_id} timeout")
                return {
                    "status": "failed",
                    "message": "Multi-GPU processing timeout",
                    "detections": []
                }
            finally:
                # 清理等待任务
                if task_id in self.pending_tasks:
                    del self.pending_tasks[task_id]
            
        except Exception as e:
            logger.error(f"Multi-GPU OCR processing error: {e}")
            return {
                "status": "failed", 
                "message": str(e),
                "detections": []
            }
    
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
        """析构函数，确保线程池和GPU进程正确关闭"""
        self._cleanup()
    
    def _cleanup(self):
        """清理资源"""
        # 关闭GPU进程
        if hasattr(self, 'gpu_processes') and self.gpu_processes:
            for i, process in enumerate(self.gpu_processes):
                if process.is_alive():
                    # 发送退出信号
                    if i < len(self.task_queues):
                        try:
                            self.task_queues[i].put(None)
                        except:
                            pass
                    process.join(timeout=5)
                    if process.is_alive():
                        process.terminate()
                        
        # 关闭线程池
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