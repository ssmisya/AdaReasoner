# POINT_WORKER_PARALLEL.PY
import torch
import numpy as np
from PIL import Image, ImageDraw
import base64
import uuid
import os
import traceback
import re
import json
import time
import threading
from io import BytesIO
import sys
import logging
from pathlib import Path
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import subprocess
import requests
import matplotlib.pyplot as plt

from transformers import HfArgumentParser, AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

from tool_server.utils.server_utils import build_logger
from tool_server.utils.worker_arguments import WorkerArguments
from tool_server.utils.error_codes import *
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.debug import remote_breakpoint

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"point_worker_parallel_{worker_id}.log")

@dataclass
class ParallelPointArguments(WorkerArguments):
    max_length: Optional[int] = field(
        default=120000,
        metadata={"help": "Maximum length for token generation"}
    )
    max_concurrency: Optional[int] = field(
        default=8000,
        metadata={"help": "Maximum number of concurrent requests to process."}
    )
    gpu_count: Optional[int] = field(
        default=None,
        metadata={"help": "Number of GPUs to use (default: auto-detect)"}
    )
    gpu_ids: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of GPU IDs to use (overrides gpu_count)"}
    )
    is_worker: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether this instance is a worker or controller"}
    )
    worker_gpu_id: Optional[int] = field(
        default=0,
        metadata={"help": "GPU ID for this worker"}
    )
    worker_port: Optional[int] = field(
        default=0,
        metadata={"help": "Port for this worker"}
    )


class PointWorkerController(BaseToolWorker):
    """
    控制器实现，负责请求分发和负载均衡
    """
    def __init__(self, args):
        self.args = args
        self.host = args.host
        self.port = args.port
        self.controller_addr = args.controller_addr
        self.model_name = args.model_name or "Point"
        # remote_breakpoint(port=7119)
        
        # 确定GPU数量和ID
        if args.gpu_ids:
            self.gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(",")]
        elif args.gpu_count:
            self.gpu_ids = list(range(min(args.gpu_count, torch.cuda.device_count())))
        else:
            # 自动检测所有可用GPU
            self.gpu_ids = list(range(torch.cuda.device_count()))
        
        # 确保至少有一个GPU
        if not self.gpu_ids:
            self.gpu_ids = [0]
        
        self.worker_count = len(self.gpu_ids)
        logger.info(f"Using {self.worker_count} GPUs: {self.gpu_ids}")
        
        # 为每个worker分配端口
        self.base_worker_port = self.port + 1
        self.worker_ports = [self.base_worker_port + i for i in range(self.worker_count)]
        
       
        
        # 请求分发计数器和锁
        self.current_worker_idx = 0
        self.worker_lock = threading.Lock()
        
        # 设置API服务器
        self.app = FastAPI()
        self.setup_routes()
        
        # 监控worker健康状态
        self.worker_health = [True] * self.worker_count
        self.health_check_interval = 10  # 秒
        self.health_check_thread = None
        
        super().__init__(args)
         # 存储worker进程和URL
        self.worker_processes = []
        
        worker_addr_list = self.worker_addr.split(":")
        if len(worker_addr_list) == 1:
            self.worker_addr_wo_port = worker_addr_list[0]
        elif len(worker_addr_list) == 2 and "http" in worker_addr_list[0]:
            self.worker_addr_wo_port = worker_addr_list[1]
        elif len(worker_addr_list) == 2:
            self.worker_addr_wo_port = worker_addr_list[0]
        elif len(worker_addr_list) == 3:
            self.worker_addr_wo_port = worker_addr_list[0] + ":" + worker_addr_list[1]
        else:
            raise ValueError(f"Invalid worker address format: {self.worker_addr}")
        
        
        self.worker_urls = [f"{self.worker_addr_wo_port}:{port}" for port in self.worker_ports]
        
        # 设置基本工具指令
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Identify a point in the image based on a natural language description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier of the image to edit, e.g., 'img_1'"
                        },
                        "description": {
                            "type": "string",
                            "description": "A natural language description of the point of interest, e.g., 'the dog's nose', 'center of the clock', 'the tallest tree'."
                        }
                    },
                    "required": ["image", "description"]
                }
            }
        }
    
    def start_workers(self):
        """启动所有worker进程"""
        logger.info("Starting worker processes...")
        
        # 获取当前脚本路径
        script_path = os.path.abspath(__file__)
        
        for i, (gpu_id, port) in enumerate(zip(self.gpu_ids, self.worker_ports)):
            # 构建worker进程的命令行参数
            cmd = [
                sys.executable,
                script_path,
                "--is_worker=True",
                f"--worker_gpu_id=0",  # 子进程内部总是使用0
                f"--worker_port={port}",
                f"--host={self.args.host}",
                f"--port={port}",
                f"--model_path={self.args.model_path}",
                f"--model_base={self.args.model_base}",
                f"--model_name={self.args.model_name}",
                f"--max_concurrency={self.args.max_concurrency}",
                f"--load_8bit={self.args.load_8bit}",
                f"--load_4bit={self.args.load_4bit}",
                "--no_register=True"  # 子Worker不需要注册到主控制器
            ]
            
            # 设置环境变量以限制GPU可见性
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            logger.info(f"Starting worker {i} on GPU {gpu_id}, port {port}")
            logger.info(f"Command: {' '.join(cmd)}")
            
            # 启动进程
            process = subprocess.Popen(cmd, env=env)
            self.worker_processes.append(process)
        
        # 等待worker启动完成
        logger.info("Waiting for workers to initialize...")
        time.sleep(5)
        
        # 启动健康检查线程
        self.start_health_check()
    
    def start_health_check(self):
        """启动worker健康检查线程"""
        def health_check_loop():
            while True:
                self.check_worker_health()
                time.sleep(self.health_check_interval)
        
        self.health_check_thread = threading.Thread(
            target=health_check_loop, 
            daemon=True,
            name="HealthCheck"
        )
        self.health_check_thread.start()
    
    def check_worker_health(self):
        """检查所有worker的健康状态"""
        for i, url in enumerate(self.worker_urls):
            try:
                response = requests.post(
                    f"{url}/worker_get_status",
                    timeout=2
                )
                if response.status_code == 200:
                    if not self.worker_health[i]:
                        logger.info(f"Worker {i} is back online")
                    self.worker_health[i] = True
                else:
                    logger.warning(f"Worker {i} health check failed: HTTP {response.status_code}")
                    self.worker_health[i] = False
            except Exception as e:
                logger.warning(f"Worker {i} health check failed: {str(e)}")
                self.worker_health[i] = False
    
    def get_next_healthy_worker(self):
        """获取下一个健康的worker索引"""
        with self.worker_lock:
            original_idx = self.current_worker_idx
            
            # 尝试找到一个健康的worker
            for _ in range(self.worker_count):
                if self.worker_health[self.current_worker_idx]:
                    idx = self.current_worker_idx
                    self.current_worker_idx = (self.current_worker_idx + 1) % self.worker_count
                    return idx
                
                self.current_worker_idx = (self.current_worker_idx + 1) % self.worker_count
            
            # 如果没有健康的worker，回到原来的索引并返回None
            self.current_worker_idx = original_idx
            return None
    
    def setup_routes(self):
        """设置API路由"""
        
        @self.app.post("/worker_generate")
        async def api_generate(request: Request):
            """处理单个请求，转发到一个worker"""
            params = await request.json()
            
            # 获取下一个健康的worker
            worker_idx = self.get_next_healthy_worker()
            
            if worker_idx is None:
                return JSONResponse({
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": "No healthy workers available",
                    "error_code": TOOL_RUN_FAILED
                }, status_code=503)
            
            worker_url = f"{self.worker_urls[worker_idx]}/worker_generate"
            
            # 转发请求并返回响应
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(worker_url, json=params, timeout=600) as response:
                        result = await response.json()
                        return JSONResponse(result)
                except Exception as e:
                    logger.error(f"Error forwarding request to worker: {e}")
                    # 标记worker为不健康
                    self.worker_health[worker_idx] = False
                    return JSONResponse({
                        "tool_response_from": self.model_name,
                        "status": "failed",
                        "message": f"Error communicating with worker: {str(e)}",
                        "error_code": TOOL_RUN_FAILED
                    }, status_code=500)
                    
        # @self.app.post("/worker_generate_batch")
        # async def api_generate_batch(request: Request):
        #     """批量处理多个请求的API端点"""
        #     batch_params = await request.json()
        #     if not isinstance(batch_params, list):
        #         return JSONResponse({"status": "failed", "message": "Expected a list of parameter objects"})
            
        #     results = []
        #     futures = []
            
        #     # 提交所有任务到线程池
        #     for params in batch_params:
        #         future = self.thread_pool.submit(self.process_single_request, params)
        #         futures.append(future)
            
        #     # 收集所有结果
        #     for future in futures:
        #         try:
        #             result = future.result(timeout=120)
        #             results.append(result)
        #         except Exception as e:
        #             logger.error(f"Error processing batch item: {e}")
        #             results.append({
        #                 "tool_response_from": self.model_name,
        #                 "status": "failed",
        #                 "message": f"Processing error: {str(e)}",
        #                 "error_code": TOOL_RUN_FAILED
        #             })
            
        #     return JSONResponse(results)
    
        @self.app.post("/worker_get_status")
        async def get_status(request: Request):
            """获取所有worker状态并合并"""
            status_tasks = []
            async with aiohttp.ClientSession() as session:
                for worker_url in self.worker_urls:
                    status_tasks.append(session.post(f"{worker_url}/worker_get_status", timeout=2))
                
                status_responses = await asyncio.gather(*status_tasks, return_exceptions=True)
            
            # 合并所有状态
            combined_status = {
                "model_names": [],
                "speed": 0,
                "queue_length": 0,
                "healthy_workers": sum(1 for h in self.worker_health if h)
            }
            
            for i, response in enumerate(status_responses):
                if not isinstance(response, Exception) and self.worker_health[i]:
                    try:
                        worker_status = await response.json()
                        if "model_names" in worker_status:
                            combined_status["model_names"].extend(worker_status["model_names"])
                        if "speed" in worker_status:
                            combined_status["speed"] += worker_status["speed"]
                        if "queue_length" in worker_status:
                            combined_status["queue_length"] += worker_status["queue_length"]
                    except Exception as e:
                        logger.error(f"Failed to parse worker status: {e}")
            
            # 去除重复的模型名称
            combined_status["model_names"] = list(set(combined_status["model_names"]))
            
            return JSONResponse(combined_status)
        
        @self.app.post("/tool_instruction")
        async def tool_instruction(request: Request):
            """返回工具指令"""
            return JSONResponse({
                "tool_instruction": self.instruction,
                "status": "success",
                "error_code": 0
            })
    
    def register_to_controller(self):
        """向主控制器注册自己"""
        if self.args.no_register:
            return
        
        logger.info("Registering to controller")
        try:
            url = self.controller_addr + "/register_worker"
            data = {
                "worker_name": f"{self.worker_addr_wo_port}:{self.args.port}",
                "check_heart_beat": True,
                "worker_status": {
                    "model_names": [self.model_name],
                    "speed": self.worker_count,  # 速度是GPU数量的函数
                    "queue_length": 0,
                }
            }
            response = requests.post(url, json=data)
            if response.status_code == 200:
                logger.info("Registered successfully")
            else:
                logger.error(f"Registration failed: {response.text}")
        except Exception as e:
            logger.error(f"Registration failed: {e}")
    
    def run(self):
        """启动控制器和所有workers"""
        import uvicorn
        
        # 首先启动所有worker
        self.start_workers()
        
        # 向主控制器注册
        self.register_to_controller()
        
        # 启动API服务器
        logger.info(f"Starting controller on {self.args.host}:{self.args.port}")
        uvicorn.run(self.app, host=self.args.host, port=self.args.port, log_level="info")
    
    def __del__(self):
        """确保退出时关闭所有worker进程"""
        for process in self.worker_processes:
            try:
                process.terminate()
            except:
                pass


class OptimizedPointWorker(BaseToolWorker):
    """
    优化后的单GPU点检测工作器实现
    """
    def __init__(self, worker_arguments):
        # 如果是worker模式，设置特定的GPU
        if getattr(worker_arguments, "is_worker", False):
            worker_arguments.device = f"cuda:0"
            if worker_arguments.worker_port > 0:
                worker_arguments.port = worker_arguments.worker_port
        
        # 设置模型名称
        if worker_arguments.model_name is None:
            worker_arguments.model_name = "Point"
        
        # 设置并发处理能力
        if hasattr(worker_arguments, "max_concurrency"):
            worker_arguments.limit_model_concurrency = worker_arguments.max_concurrency
            
        # 配置参数        
        self.max_length = getattr(worker_arguments, "max_length", 120000)
        super().__init__(worker_arguments)
        
        # 线程池用于并发处理请求
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        
        # 添加计数器和锁
        self.global_counter = 0
        self.global_counter_lock = threading.Lock()
        
        # 指令定义
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Identify a point in the image based on a natural language description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier of the image to edit, e.g., 'img_1'"
                        },
                        "description": {
                            "type": "string",
                            "description": "A natural language description of the point of interest, e.g., 'the dog's nose', 'center of the clock', 'the tallest tree'."
                        }
                    },
                    "required": ["image", "description"]
                }
            }
        }
        
    def init_model(self):
        """初始化模型"""
        logger.info(f"Initializing model {self.model_name} on {self.device}")
        logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
        logger.info(f"Available GPU count: {torch.cuda.device_count()}")
        
        # 量化配置
        quant_config = None
        if self.load_4bit or self.load_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=self.load_8bit,
                load_in_4bit=self.load_4bit,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            logger.info(f"Using quantization config: {quant_config}")
        
        try:
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype='auto',
            )

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map="auto",  # 自动选择设备
                quantization_config=quant_config
            )
            self.model.eval()
            logger.info(f"Model loaded successfully on device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            logger.error(traceback.format_exc())
            raise

    def extract_points(self, molmo_output, image_w, image_h):
        """从模型输出中提取点坐标"""
        all_points = []
        for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # 无效输出
                    continue
                point /= 100.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)
        return np.array(all_points)
    
    def show_points(self, coords, labels, ax, marker_size=375):
        """在图上展示点"""
        if len(coords) == 0:
            return
        
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        
        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        if len(neg_points) > 0:
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    
    def create_image_with_points(self, image, coords, labels, marker_size=375):
        """创建包含标记点的图像"""
        fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100), dpi=100)
        ax.imshow(image)
        image_format = 'png'
        if image.format is not None:
            image_format = image.format.lower()
            if image_format not in ['png', 'jpeg', 'jpg']:
                image_format = 'png'

        self.show_points(coords, labels, ax, marker_size)

        plt.axis('off')

        buf = BytesIO()
        plt.savefig(buf, format=image_format, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    @torch.inference_mode()
    def process_single_request(self, params):
        """处理单个请求"""
        tool_reward = 2.0
        # 计算Parameter Name Matching
        param_keys = set(params.keys())
        required_keys = set(['image', 'description'])
        parameter_name_match_reward = len(param_keys & required_keys) / len(required_keys | param_keys)
        tool_reward = tool_reward + parameter_name_match_reward
        
        if parameter_name_match_reward < 1:
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": "Invalid parameters: expected keys: image, description.",
                "error_code": INVALID_PARAMETERS,
                "tool_reward": tool_reward
            }
        
        required_keys_num = len(required_keys)
        correct_param_content_num = 0
        
        # 提取输入
        image_data = params.get("image")
        description = params.get("description")
        
        try:
            # 将base64转换为PIL图像
            try:
                image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
                correct_param_content_num += 1
                
                # 创建图像尺寸信息
                image_dimensions = {
                    "width": image.width,
                    "height": image.height
                }
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Failed to load image: {str(e)}",
                    "error_code": CANNOT_LOAD_IMAGE,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                }
                return pred_dict
            
            # 验证description参数
            if description and len(description.strip()) > 0:
                correct_param_content_num += 1
            else:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": "Invalid parameters: description cannot be empty.",
                    "error_code": INVALID_PARAMETERS,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num,
                    "image_dimensions_pixels": image_dimensions
                }
                return pred_dict
            
            text_prompt = f"Point to the {description} in the scene."
            
            # 处理输入并运行模型
            try:
                inputs = self.processor.process(
                    images=[image],
                    text=text_prompt,
                )
                inputs["images"] = inputs["images"].to(torch.bfloat16)
                inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    output = self.model.generate_from_batch(
                        inputs,
                        GenerationConfig(max_new_tokens=self.max_length, stop_strings="<|endoftext|>"),
                        tokenizer=self.processor.tokenizer
                    )
                    
                    # 获取生成的tokens并解码为文本
                    generated_tokens = output[0, inputs['input_ids'].size(1):]
                    response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Model inference failed: {str(e)}",
                    "error_code": TOOL_RUN_FAILED,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num,
                    "image_dimensions_pixels": image_dimensions
                }
                return pred_dict
            
            # 从响应中提取点
            if response:
                points = self.extract_points(response, image.width, image.height)
                
                # 创建点表示
                point_data = []
                if len(points) > 0:
                    for point in points:
                        point_data.append({
                            "x": float(point[0]),
                            "y": float(point[1]),
                        })
                
                # 创建带有标记点的图片
                if len(points) > 0:
                    labels = np.ones(len(points), dtype=np.int32)  # 所有点为正向 (1)
                    image_with_points = self.create_image_with_points(image, points, labels)
                    
                    # 将图片转换为base64字符串
                    buffered = BytesIO()
                    image_format = image.format if image.format else 'PNG'
                    image_with_points.save(buffered, format=image_format)
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    pred_dict = {
                        "tool_response_from": self.model_name,
                        "status": "success",
                        "points": point_data,
                        "edited_image": img_str,
                        "image_dimensions_pixels": image_dimensions,
                        "error_code": SUCCESS,
                        "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                    }
                else:
                    pred_dict = {
                        "tool_response_from": self.model_name,
                        "status": "success",
                        "points": point_data,
                        "message": "No valid points detected in the response.",
                        "image_dimensions_pixels": image_dimensions,
                        "error_code": SUCCESS,
                        "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                    }
                
                return pred_dict
            else:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": "Model did not generate any response.",
                    "error_code": TOOL_RUN_FAILED,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num,
                    "image_dimensions_pixels": image_dimensions
                }
                return pred_dict
                
        except Exception as e:
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error: {str(e)}\n Traceback:{traceback.format_exc()}\n",
                "error_code": TOOL_RUN_FAILED,
                "tool_reward": tool_reward+(correct_param_content_num/required_keys_num if required_keys_num > 0 else 0)
            }
            logger.error(f"Error during Point inference: {e}")
            logger.error(traceback.format_exc())
            return pred_dict

    def generate_gate(self, params):
        """覆盖父类方法，使用线程池处理请求"""
        try:
            # 使用线程池提交任务
            future = self.thread_pool.submit(self.process_single_request, params)
            ret = future.result(timeout=580)  # 设置超时时间
            return ret
        except Exception as e:
            logger.error(f"Error in generate_gate: {e}")
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error in generate_gate: {e}",
                "error_code": TOOL_RUN_FAILED
            }

    def setup_routes(self):
        """设置API路由"""
        
        @self.app.post("/worker_generate")
        async def api_generate(request: Request):
            """处理单个请求"""
            params = await request.json()
            
            try:
                # 添加错误处理，确保即使没有锁也能工作
                if hasattr(self, 'global_counter_lock') and self.global_counter_lock:
                    with self.global_counter_lock:
                        self.global_counter += 1
                        count = self.global_counter
                elif hasattr(self, 'global_counter'):
                    self.global_counter += 1
                    count = self.global_counter
                
                # 不再使用信号量，由线程池管理
                result = self.generate_gate(params)
                
                # 减少计数
                if hasattr(self, 'global_counter_lock') and self.global_counter_lock:
                    with self.global_counter_lock:
                        self.global_counter -= 1
                elif hasattr(self, 'global_counter'):
                    self.global_counter -= 1
                
                return JSONResponse(result)
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                logger.error(traceback.format_exc())
                
                # 减少计数
                if hasattr(self, 'global_counter_lock') and self.global_counter_lock:
                    with self.global_counter_lock:
                        self.global_counter -= 1
                elif hasattr(self, 'global_counter'):
                    self.global_counter -= 1
                
                return JSONResponse({
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Error: {str(e)}",
                    "error_code": TOOL_RUN_FAILED
                }, status_code=500)
        
        @self.app.post("/worker_get_status")
        async def get_status(request: Request):
            """返回状态信息"""
            return JSONResponse(self.get_status())
        
        @self.app.post("/tool_instruction")
        async def tool_instruction(request: Request):
            """返回工具指令"""
            return JSONResponse({
                "tool_instruction": self.instruction,
                "status": "success",
                "error_code": 0
            })
    
    def get_status(self):
        """获取当前状态"""
        queue_length = 0
        
        # 添加错误处理，确保即使没有锁也能工作
        if hasattr(self, 'global_counter_lock') and self.global_counter_lock:
            with self.global_counter_lock:
                queue_length = self.global_counter
        elif hasattr(self, 'global_counter'):
            queue_length = self.global_counter
        
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": queue_length,
        }
    
    def __del__(self):
        """确保资源正确释放"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        
        # 调用父类析构函数
        super().__del__()


if __name__ == "__main__":
    parser = HfArgumentParser((ParallelPointArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")
    
    if args.is_worker:
        # 作为Worker启动
        logger.info(f"Starting as worker on GPU {args.worker_gpu_id}, port {args.worker_port}")
        worker = OptimizedPointWorker(args)
        worker.run()
    else:
        # 作为控制器启动
        logger.info("Starting as controller")
        controller = PointWorkerController(args)
        controller.run()