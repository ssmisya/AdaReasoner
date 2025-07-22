"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid
import os

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests
import torch
import uvicorn
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from enum import IntEnum

from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from tool_server.utils.worker_arguments import WorkerArguments

SERVER_ERROR_MSG = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
class ErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303
    TIMEOUT_ERROR = 40304

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("tool_worker", f"base_tool_worker_{worker_id}.log")
model_semaphore = None

class BaseToolWorker:
    def __init__(self, worker_arguments: WorkerArguments = None,):
        self.args = worker_arguments
        self.controller_addr = self.args.controller_addr
        self.port = self.args.port
        self.worker_addr = self.args.worker_addr
        self.no_register = self.args.no_register
        
        assert self.port is not None, "Port must be specified"
        if self.worker_addr == "auto":
            node_name = os.getenv("SLURMD_NODENAME", "Unknown")
            print(f"SLURM Node Name: {node_name}")
            if node_name == "Unknown":
                node_name = "localhost"
            self.worker_addr = f"http://{node_name}:{self.port}"
        else:
            self.worker_addr = self.worker_addr
            
        self.model_path =  self.args.model_path
        self.model_base =  self.args.model_base
        self.model_name = self.args.model_name

        
        if model_semaphore is not None:
            self.model_semaphore = model_semaphore
        else:
            self.model_semaphore = None
        
        self.worker_id = worker_id
        
       
        self.load_8bit = self.args.load_8bit
        self.load_4bit = self.args.load_4bit
        self.device = self.args.device
        self.limit_model_concurrency = self.args.limit_model_concurrency
        self.host = self.args.host
        self.port = self.args.port
        
        self.global_counter = 0
        self.task_timeout = self.args.task_timeout
        self.semaphore_timeout = self.args.wait_timeout
        
        if not self.args.no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=self.heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()
            
        # 初始化线程池用于处理并发请求
        self.thread_pool = ThreadPoolExecutor(max_workers=self.limit_model_concurrency)
        logger.info(f"Initialized thread pool with {self.limit_model_concurrency} workers")
            
        # Set up the routes
        self.app = FastAPI()
        self.setup_routes()
        self.init_model()
        
        self.basic_ret = {
            "tool_response_from": None,
            "status": None,
            "message": None,
        }
        

    ## HTTP Methods    
    def heart_beat_worker(self, controller):
        while True:
            time.sleep(WORKER_HEART_BEAT_INTERVAL)
            controller.send_heart_beat()

        
    def release_model_semaphore(self, fn=None):
        if self.model_semaphore:
            self.model_semaphore.release()
            if fn is not None:
                fn()
    
    async def acquire_model_semaphore(self):
        self.global_counter += 1
        if self.model_semaphore is None:
            # Use a threading semaphore instead of asyncio semaphore for thread safety
            self.model_semaphore = threading.Semaphore(self.limit_model_concurrency)
        # For thread-safe acquisition in async context with timeout
        loop = asyncio.get_event_loop()
        
        future = loop.run_in_executor(None, lambda: self.model_semaphore.acquire(timeout=self.semaphore_timeout))
        try:
            return await asyncio.wait_for(future, timeout=self.semaphore_timeout)
        except asyncio.TimeoutError:
            logger.error(f"Semaphore acquisition timed out after {self.semaphore_timeout}s")
            raise TimeoutError("Model is busy, request timed out while waiting in queue")
                
    def setup_routes(self):
        @self.app.post("/worker_generate")
        async def api_generate(request: Request):
            params = await request.json()
            try:
                await self.acquire_model_semaphore()
                # 使用线程池执行生成任务，避免阻塞FastAPI的事件循环
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    self.thread_pool, 
                    self.generate_gate, 
                    params
                )
                return JSONResponse(output)
            except TimeoutError:
                ret = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": "Request timed out while waiting in queue",
                }
                return JSONResponse(ret, status_code=503)
            finally:
                # Only release if we actually acquired it
                if self.model_semaphore and self.model_semaphore._value < self.limit_model_concurrency:
                    self.release_model_semaphore()
        
        @self.app.post("/worker_generate_batch")
        async def api_generate_batch(request: Request):
            """批量处理多个请求的API端点"""
            batch_params = await request.json()
            if not isinstance(batch_params, list):
                return JSONResponse({"status": "failed", "message": "Expected a list of parameter objects"})
            
            results = []
            tasks = []
            
            # 创建异步任务来处理每个请求
            for params in batch_params:
                task = asyncio.create_task(self._process_single_request(params))
                tasks.append(task)
            
            # 等待所有任务完成
            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for result in completed_tasks:
                if isinstance(result, Exception):
                    logger.error(f"Error processing batch item: {result}")
                    results.append({
                        "tool_response_from": self.model_name,
                        "status": "failed",
                        "message": f"Processing error: {str(result)}",
                    })
                else:
                    results.append(result)
            
            return JSONResponse(results)

        @self.app.post("/worker_get_status")
        async def get_status(request: Request):
            return self.get_status()
        
        @self.app.post("/model_details")
        async def model_details(request: Request):
            pass
        
        @self.app.post("/tool_instruction")
        async def tool_instruction(request: Request):
            
            try:
                tool_instruction = self.get_tool_instruction()
                return JSONResponse({
                    "tool_instruction": tool_instruction,
                    "status": "success",
                    "error_code": 0
                })
            except Exception as e:
                logger.error(f"Error getting tool instruction: {e}")
                return JSONResponse({
                    "tool_instruction": None,
                    "status": "failed",
                    "error_code": ErrorCode.INTERNAL_ERROR
                }, status_code=500)
    
    async def _process_single_request(self, params):
        """处理单个请求的异步方法"""
        try:
            await self.acquire_model_semaphore()
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.thread_pool, self.generate_gate, params)
        finally:
            if self.model_semaphore and self.model_semaphore._value < self.limit_model_concurrency:
                self.release_model_semaphore()
                
    def generate_gate(self, params):
        try:
            ret = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": "Unknown error occurred. Please try again later.",
            }
            ret = self.generate(params)
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message":  f"CUDA OUT OF MEMORY: {e}",
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message":  f"ValueError, RuntimeError: {e}",
            }
        return ret
    
    
    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200, f"Failed to register to controller: {r.text}"

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {self.global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()
    
    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def get_queue_length(self):
        if self.model_semaphore is None:
            return 0
        return self.limit_model_concurrency - self.model_semaphore._value

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

    def init_model(self):
        # 在子类中实现
        pass

    @torch.inference_mode()
    def generate(self, params):
        # 在子类中实现
        pass
    
    def get_tool_instruction(self):
        return self.instruction

    def __del__(self):
        """析构函数，确保线程池正确关闭"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)

    
