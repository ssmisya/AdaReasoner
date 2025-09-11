"""
A model worker executes the model.
"""

import uuid
import os
import re
import io
import signal
import numpy as np
from PIL import Image
import torch
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
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
        lang='en',
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
    ocr_task_timeout: Optional[int] = field(
        default=60,
        metadata={"help": "Timeout for individual OCR inference tasks in seconds"}
    )
    ocr_request_timeout: Optional[int] = field(
        default=120,
        metadata={"help": "Overall timeout for OCR requests in seconds"}
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
        
        # 首先初始化超时配置
        self.ocr_task_timeout = getattr(worker_arguments, 'ocr_task_timeout', 60)
        self.ocr_request_timeout = getattr(worker_arguments, 'ocr_request_timeout', 120)
        
        # 在调用父类初始化前先初始化多GPU配置
        self.enable_multi_gpu = getattr(worker_arguments, 'enable_multi_gpu', False)
        self.gpu_ids = []
        if hasattr(worker_arguments, 'gpu_ids') and worker_arguments.gpu_ids:
            self.gpu_ids = [int(x.strip()) for x in worker_arguments.gpu_ids.split(',')]
        else:
            self.gpu_ids = [0]  # 默认使用GPU 0
            
        logger.info(f"Multi-GPU enabled: {self.enable_multi_gpu}, GPU IDs: {self.gpu_ids}")
        logger.info(f"OCR timeout settings - Task: {self.ocr_task_timeout}s, Request: {self.ocr_request_timeout}s")
        
        # 多GPU相关变量初始化
        self.gpu_processes = []
        self.task_queues = []
        self.result_queue = None
        self.task_counter = 0
        self.pending_tasks = {}
        
        # 负载均衡相关变量
        self.current_gpu_index = 0  # 用于轮询调度
        self.gpu_task_counts = {}  # 跟踪每个GPU的任务数
        self.gpu_processing_times = {}  # 跟踪每个GPU的处理时间统计
        import threading
        self.load_balancer_lock = threading.Lock()  # 保证线程安全
        
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
            lang='en',
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
            
            # 初始化GPU统计信息
            self.gpu_task_counts[gpu_id] = 0
            self.gpu_processing_times[gpu_id] = []
            
            # 创建GPU工作进程
            process = mp.Process(
                target=gpu_worker_process,
                args=(gpu_id, task_queue, self.result_queue)
            )
            process.start()
            self.gpu_processes.append(process)
            logger.info(f"Started GPU worker process for GPU {gpu_id}")
            
        logger.info(f"Initialized load balancing for {len(self.gpu_ids)} GPUs: {self.gpu_ids}")
        
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
                    # 更新GPU统计信息
                    if 'gpu_id' in result:
                        gpu_id = result['gpu_id']
                        if gpu_id in self.gpu_task_counts:
                            self.gpu_task_counts[gpu_id] += 1
                        if 'processing_time' in result and gpu_id in self.gpu_processing_times:
                            self.gpu_processing_times[gpu_id].append(result['processing_time'])
                            # 只保留最近100个处理时间记录
                            if len(self.gpu_processing_times[gpu_id]) > 100:
                                self.gpu_processing_times[gpu_id] = self.gpu_processing_times[gpu_id][-100:]
                    
                    # 将结果传递给等待的任务
                    self.pending_tasks[task_id].put(result)
                    del self.pending_tasks[task_id]
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error collecting results: {e}")
        
    def get_tool_instruction(self):
        return self.instruction
    
    def update_timeout_settings(self, task_timeout=None, request_timeout=None):
        """动态更新超时设置"""
        if task_timeout is not None:
            old_task_timeout = self.ocr_task_timeout
            self.ocr_task_timeout = task_timeout
            logger.info(f"Updated OCR task timeout from {old_task_timeout}s to {task_timeout}s")
            
        if request_timeout is not None:
            old_request_timeout = self.ocr_request_timeout
            self.ocr_request_timeout = request_timeout
            logger.info(f"Updated OCR request timeout from {old_request_timeout}s to {request_timeout}s")
    
    def get_timeout_settings(self):
        """获取当前超时设置"""
        return {
            "task_timeout": self.ocr_task_timeout,
            "request_timeout": self.ocr_request_timeout
        }
    
    def get_gpu_stats(self):
        """获取GPU统计信息"""
        if not self.enable_multi_gpu or len(self.gpu_ids) <= 1:
            return {"multi_gpu_enabled": False}
            
        stats = {
            "multi_gpu_enabled": True,
            "gpu_ids": self.gpu_ids,
            "current_gpu_index": self.current_gpu_index,
            "task_counts": dict(self.gpu_task_counts),
            "queue_sizes": {gpu_id: self.task_queues[i].qsize() for i, gpu_id in enumerate(self.gpu_ids)},
            "average_processing_times": {}
        }
        
        # 计算平均处理时间
        for gpu_id in self.gpu_ids:
            times = self.gpu_processing_times.get(gpu_id, [])
            if times:
                stats["average_processing_times"][gpu_id] = {
                    "avg": sum(times) / len(times),
                    "count": len(times),
                    "min": min(times),
                    "max": max(times)
                }
            else:
                stats["average_processing_times"][gpu_id] = {
                    "avg": 0.0,
                    "count": 0,
                    "min": 0.0,
                    "max": 0.0
                }
        
        return stats
    
    def reset_gpu_stats(self):
        """重置GPU统计信息"""
        if self.enable_multi_gpu and len(self.gpu_ids) > 1:
            with self.load_balancer_lock:
                for gpu_id in self.gpu_ids:
                    self.gpu_task_counts[gpu_id] = 0
                    self.gpu_processing_times[gpu_id] = []
                logger.info("GPU statistics have been reset")
                return True
        return False
        
    async def generate_gate_async(self, params):
        """异步generate_gate，提供真正的异步处理能力"""
        try:
            # 直接调用异步实现，避免线程池开销
            ret = await self.generate_impl_async(params)
            return ret
        except asyncio.TimeoutError:
            logger.error(f"OCR request timeout after {self.ocr_request_timeout} seconds")
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"OCR request timeout after {self.ocr_request_timeout} seconds",
                "error_code": TOOL_RUN_FAILED
            }
        except Exception as e:
            logger.error(f"Error in generate_gate_async: {e}")
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error in generate_gate_async: {e}",
                "error_code": TOOL_RUN_FAILED
            }
    
    def generate_gate(self, params):
        """覆盖父类方法，使用线程池处理请求并支持超时控制（向后兼容）"""
        try:
            # 使用线程池提交任务，使用可配置的超时时间
            future = self.thread_pool.submit(self.generate_impl, params)
            ret = future.result(timeout=self.ocr_request_timeout)
            return ret
        except FutureTimeoutError:
            logger.error(f"OCR request timeout after {self.ocr_request_timeout} seconds")
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"OCR request timeout after {self.ocr_request_timeout} seconds",
                "error_code": TOOL_RUN_FAILED
            }
        except Exception as e:
            logger.error(f"Error in generate_gate: {e}")
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error in generate_gate: {e}",
                "error_code": TOOL_RUN_FAILED
            }
    
    async def generate_impl_async(self, params):
        """异步实现实际的生成逻辑"""
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

            # 根据GPU配置选择异步处理方式
            if self.enable_multi_gpu and len(self.gpu_ids) > 1:
                ocr_result = await self._process_ocr_multi_gpu_async(np.array(img), text_threshold)
            else:
                ocr_result = await self._process_ocr_single_gpu_async(np.array(img), text_threshold)
            
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
    
    def generate_impl(self, params):
        """同步包装器，实现实际的生成逻辑"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_impl_async(params))
        finally:
            loop.close()
    
    async def _ocr_predict_with_timeout_async(self, img_array, timeout):
        """异步带超时控制的OCR推理函数"""
        import threading
        import queue
        
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def ocr_worker():
            try:
                result = self.ocr_model.predict(img_array)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
        
        # 启动OCR线程
        ocr_thread = threading.Thread(target=ocr_worker)
        ocr_thread.daemon = True
        ocr_thread.start()
        
        # 异步等待结果或超时
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, 
                lambda: result_queue.get(timeout=timeout)
            )
            return result, None
        except queue.Empty:
            logger.warning(f"OCR inference timeout after {timeout} seconds")
            return None, "OCR inference timeout"
        except Exception as e:
            if not exception_queue.empty():
                exception = exception_queue.get()
                return None, str(exception)
            return None, str(e)
    
    def _ocr_predict_with_timeout(self, img_array, timeout):
        """同步包装器，用于向后兼容"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._ocr_predict_with_timeout_async(img_array, timeout))
        finally:
            loop.close()
    
    async def _process_ocr_single_gpu_async(self, img_array, text_threshold):
        """异步单GPU OCR处理方法（带超时保护）"""
        detections = []
        
        try:
            # 使用异步带超时的PaddleOCR推理
            result, error = await self._ocr_predict_with_timeout_async(img_array, self.ocr_task_timeout)
            
            if error:
                return {
                    "status": "failed",
                    "message": error,
                    "detections": []
                }
            
            if not result:
                return {
                    "status": "failed",
                    "message": "OCR inference returned no result",
                    "detections": []
                }
            
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
    
    def _process_ocr_single_gpu(self, img_array, text_threshold):
        """同步包装器，用于向后兼容"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._process_ocr_single_gpu_async(img_array, text_threshold))
        finally:
            loop.close()
    
    def _select_gpu_for_task(self):
        """选择最适合的GPU处理任务"""
        with self.load_balancer_lock:
            # 策略1: 轮询调度 (Round Robin) - 最公平的分配
            selected_gpu_idx = self.current_gpu_index
            self.current_gpu_index = (self.current_gpu_index + 1) % len(self.gpu_ids)
            selected_gpu_id = self.gpu_ids[selected_gpu_idx]
            
            # 获取当前状态用于日志
            queue_sizes = [q.qsize() for q in self.task_queues]
            task_counts = [self.gpu_task_counts.get(gpu_id, 0) for gpu_id in self.gpu_ids]
            
            logger.info(f"Load balancer - Selected GPU {selected_gpu_id} (Round Robin)")
            logger.info(f"Current state - Queue sizes: {dict(zip(self.gpu_ids, queue_sizes))}, Task counts: {dict(zip(self.gpu_ids, task_counts))}")
            
            return selected_gpu_idx, selected_gpu_id
    
    async def _process_ocr_multi_gpu_async(self, img_array, text_threshold):
        """异步多GPU OCR处理方法（使用改进的负载均衡）"""
        try:
            # 生成任务ID
            self.task_counter += 1
            task_id = f"task_{self.task_counter}_{time.time()}"
            
            # 使用改进的负载均衡算法选择GPU
            selected_gpu_idx, selected_gpu_id = self._select_gpu_for_task()
            
            # 创建结果队列
            result_queue = queue.Queue()
            self.pending_tasks[task_id] = result_queue
            
            # 将任务添加到选定的GPU队列（非阻塞）
            self.task_queues[selected_gpu_idx].put((task_id, img_array, text_threshold))
            logger.info(f"Task {task_id} dispatched to GPU {selected_gpu_id} (async)")
            
            # 异步等待结果
            loop = asyncio.get_event_loop()
            try:
                # 使用线程池执行阻塞操作，避免阻塞事件循环
                result = await loop.run_in_executor(
                    None, 
                    lambda: result_queue.get(timeout=self.ocr_task_timeout)
                )
                return result
            except queue.Empty:
                logger.error(f"Task {task_id} timeout after {self.ocr_task_timeout} seconds")
                return {
                    "status": "failed",
                    "message": f"Multi-GPU processing timeout after {self.ocr_task_timeout} seconds",
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
    
    def _process_ocr_multi_gpu(self, img_array, text_threshold):
        """同步包装器，用于向后兼容"""
        # 在线程池中运行异步方法
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._process_ocr_multi_gpu_async(img_array, text_threshold))
        finally:
            loop.close()
    
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
            
            # 创建异步任务列表，实现真正的并发处理
            async_tasks = []
            for params in batch_params:
                task = asyncio.create_task(self.generate_gate_async(params))
                async_tasks.append(task)
            
            # 并发执行所有任务，使用超时控制
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*async_tasks, return_exceptions=True),
                    timeout=self.ocr_request_timeout
                )
                
                # 处理结果和异常
                processed_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch item {i} error: {result}")
                        processed_results.append({
                            "tool_response_from": self.model_name,
                            "status": "failed",
                            "message": f"Processing error: {str(result)}",
                            "error_code": TOOL_RUN_FAILED
                        })
                    else:
                        processed_results.append(result)
                
                results = processed_results
                
            except asyncio.TimeoutError:
                logger.error(f"Batch processing timeout after {self.ocr_request_timeout} seconds")
                # 取消所有未完成的任务
                for task in async_tasks:
                    if not task.done():
                        task.cancel()
                
                results = [{
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Batch processing timeout after {self.ocr_request_timeout} seconds",
                    "error_code": TOOL_RUN_FAILED
                } for _ in batch_params]
            
            return JSONResponse(results)
        
        @self.app.post("/worker_timeout_settings")
        async def api_timeout_settings(request: Request):
            """管理超时设置的API端点"""
            try:
                data = await request.json()
                action = data.get("action", "get")
                
                if action == "get":
                    return JSONResponse({
                        "status": "success",
                        "timeout_settings": self.get_timeout_settings()
                    })
                elif action == "update":
                    task_timeout = data.get("task_timeout")
                    request_timeout = data.get("request_timeout")
                    self.update_timeout_settings(task_timeout, request_timeout)
                    return JSONResponse({
                        "status": "success",
                        "message": "Timeout settings updated",
                        "timeout_settings": self.get_timeout_settings()
                    })
                else:
                    return JSONResponse({
                        "status": "failed",
                        "message": f"Unknown action: {action}"
                    })
            except Exception as e:
                logger.error(f"Error in timeout settings API: {e}")
                return JSONResponse({
                    "status": "failed",
                    "message": f"Error: {str(e)}"
                })
        
        @self.app.post("/worker_gpu_stats")
        async def api_gpu_stats(request: Request):
            """管理GPU统计信息的API端点"""
            try:
                data = await request.json()
                action = data.get("action", "get")
                
                if action == "get":
                    stats = self.get_gpu_stats()
                    return JSONResponse({
                        "status": "success",
                        "gpu_stats": stats
                    })
                elif action == "reset":
                    success = self.reset_gpu_stats()
                    if success:
                        return JSONResponse({
                            "status": "success",
                            "message": "GPU statistics reset successfully"
                        })
                    else:
                        return JSONResponse({
                            "status": "failed",
                            "message": "Multi-GPU not enabled or only one GPU available"
                        })
                else:
                    return JSONResponse({
                        "status": "failed",
                        "message": f"Unknown action: {action}"
                    })
            except Exception as e:
                logger.error(f"Error in GPU stats API: {e}")
                return JSONResponse({
                    "status": "failed",
                    "message": f"Error: {str(e)}"
                })

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