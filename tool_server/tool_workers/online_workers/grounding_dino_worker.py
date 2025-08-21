import torch
import numpy as np
from PIL import Image
import base64
import uuid
import os
import torchvision
from io import BytesIO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor

from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops

from tool_server.utils.server_utils import build_logger
from tool_server.utils.worker_arguments import WorkerArguments
from tool_server.utils.error_codes import *
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"grounding_dino_worker_{worker_id}.log")

@dataclass
class GroundingDinoArguments(WorkerArguments):
    model_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model config file"}
    )
    max_concurrency: Optional[int] = field(
        default=120000,
        metadata={"help": "Maximum number of concurrent requests to process."}
    )

class GroundingDinoWorker(BaseToolWorker):
    def __init__(self, worker_arguments: GroundingDinoArguments = None):
        # 在调用父类初始化前先设置模型名称和并发数
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "GroundingDINO"
        
        # 设置更高的并发处理能力
        if hasattr(worker_arguments, "max_concurrency"):
            worker_arguments.limit_model_concurrency = worker_arguments.max_concurrency
        else:
            worker_arguments.limit_model_concurrency = 10  # 默认允许10个并发请求
        
        # 保存model_config到实例变量
        self.model_config = worker_arguments.model_config if worker_arguments else None
        
        super().__init__(worker_arguments)
        
        # 创建线程池用于并发处理请求
        self.thread_pool = ThreadPoolExecutor(max_workers=worker_arguments.limit_model_concurrency)
            
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Locate objects in the image based on a natural language description. Returns detected objects with their bounding boxes in absolute pixel coordinates, confidence scores, and an annotated image with visualized detections.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier of the image to analyze, e.g., 'img_1'"
                        },
                        "description": {
                            "type": "string",
                            "description": "A natural language description of the object to locate, e.g., 'a red car', 'a man holding a dog'."
                        }
                    },
                    "required": ["image", "description"]
                }
            }
        }

    def init_model(self):
        logger.info(f"Initializing model {self.model_name} with concurrency {self.args.limit_model_concurrency}...")
        logger.info(f"CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}")
        self.model = load_model(
            model_config_path=self.model_config,
            model_checkpoint_path=self.model_path,
            device=self.device,
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    def load_image(self, image_path: str):
        if os.path.exists(image_path):
            image_source = Image.open(image_path).convert("RGB")
        else:
            # Handle base64 encoded image
            image_source = Image.open(BytesIO(base64.b64decode(image_path))).convert("RGB")

        image = np.asarray(image_source)
        image_transformed, _ = self.transform(image_source, None)
        return image, image_transformed
        
    def nms(self, boxes, logits, phrases):
        iou_threshold = 0.8 # 用于去除重复的框
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
        logger.info(f"Before NMS: {boxes_xyxy.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_xyxy, logits, iou_threshold)

        boxes = boxes[nms_idx]
        logits = logits[nms_idx]
        phrases = [phrases[idx] for idx in nms_idx]
        logger.info(f"After NMS: {boxes.shape[0]} boxes")

        return boxes, logits, phrases
        
    def annotate_image(self, image_np, boxes, logits, phrases):
        """在图像上绘制边界框和标签"""
        # 获取图像的原始尺寸
        h, w, _ = image_np.shape
        
        # 定义输出的DPI (dots per inch)
        dpi = 100
        
        # 根据图像的像素尺寸和DPI，计算出画布的尺寸（英寸）
        # 这样可以确保输出的图像和输入图像像素完全一样
        figsize = w / float(dpi), h / float(dpi)

        # 创建一个指定尺寸和DPI的画布
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # 添加一个完全填充画布的坐标轴，无边距
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off') # 关闭坐标轴显示

        # 在这个精确控制的坐标轴上显示你的图像
        ax.imshow(image_np)
        
        # 为不同类别设置不同颜色 (这部分逻辑保持不变)
        unique_phrases = list(set(phrases))
        colors = plt.cm.hsv(np.linspace(0, 1, len(unique_phrases) if len(unique_phrases) > 0 else 1))
        color_map = {phrase: colors[i % len(colors)] for i, phrase in enumerate(unique_phrases)}
        
        # 绘制每个检测框 (这部分逻辑保持不变)
        for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
            x_min, y_min, x_max, y_max = box
            x_min_px, y_min_px = int(x_min * w), int(y_min * h)
            x_max_px, y_max_px = int(x_max * w), int(y_max * h)
            width, height = x_max_px - x_min_px, y_max_px - y_min_px
            color = color_map.get(phrase, 'red')
            
            rect = patches.Rectangle(
                (x_min_px, y_min_px), width, height, 
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            label = f"{phrase}: {logit:.2f}"
            ax.text(
                x_min_px, y_min_px - 5, label, 
                bbox=dict(facecolor=color, alpha=0.5),
                fontsize=8, color='white'
            )
        
        # 将图形保存到内存缓冲区
        buf = BytesIO()
        # 由于我们已经精确控制了尺寸和边距，不再需要 'bbox_inches' 和 'pad_inches'
        fig.savefig(buf, format='png', dpi=dpi)
        plt.close(fig) # 关闭画布，释放内存
        buf.seek(0)
        
        # 从缓冲区加载图像
        annotated_image = Image.open(buf).convert('RGB')
        return annotated_image
    
    def image_to_base64(self, image):
        """将PIL图像转换为base64编码的字符串"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def generate_gate(self, params):
        """覆盖父类方法，使用线程池处理请求"""
        try:
            # 使用线程池提交任务
            future = self.thread_pool.submit(self._generate_with_torch_inference, params)
            ret = future.result(timeout=600000)
            return ret
        # 在 generate_gate 函数中
        except Exception as e:
            # 记录具体的异常类型和信息
            logger.error(f"Error in generate_gate. Exception type: {type(e).__name__}, Details: {e}")
            # (可选) 记录完整的堆栈跟踪信息，便于调试
            logger.error(traceback.format_exc())
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                # 在返回的消息中也包含更具体的信息
                "message": f"Error in generate_gate: {type(e).__name__}",
                "error_code": TOOL_RUN_FAILED
            }
    
    @torch.inference_mode()
    def _generate_with_torch_inference(self, params):
        """实际的生成函数，包装在torch.inference_mode装饰器中"""
        return self._generate_impl(params)
        
    def _generate_impl(self, params):
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
                "message": "Invalid parameters: expected keys: image, description.",
                "error_code": INVALID_PARAMETERS,
                "tool_reward": tool_reward
            }

        required_keys_num = len(required_keys)
        # 初始化参数合规计数器
        correct_param_content_num = 0    
        
        # Extract inputs
        image_path = params["image"]
        description = params.get("description")
        
        # If the params have been parsed successfully.
        box_threshold = params.get("box_threshold", 0.25)
        text_threshold = params.get("text_threshold", 0.25)
        try:
            # Load image and run model
            try:
                image_np, image = self.load_image(image_path)
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Cannot load image: {str(e)}",
                    "error_code": CANNOT_LOAD_IMAGE,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                }
                return pred_dict
            correct_param_content_num += 1  # 图像加载成功，合规+1
            
            # 获取图像尺寸
            h, w, _ = image_np.shape
                
            boxes, logits, phrases = predict(
                model=self.model, 
                image=image, 
                caption=description, 
                box_threshold=box_threshold, 
                text_threshold=text_threshold,
                device=self.device
            )
###
            # Apply NMS to boxes
            boxes, logits, phrases = self.nms(boxes, logits, phrases)

            # --- 全新修改：过滤掉包含 NaN 或 Inf 的无效框 ---

            # 1. 创建一个布尔掩码 (mask)
            #    torch.isfinite(boxes) 会对每个坐标进行检查，返回 True 或 False
            #    .all(dim=1) 会检查每一行（一个box的所有坐标）是否都为 True
            finite_mask = torch.all(torch.isfinite(boxes), dim=1)

            # 2. 使用掩码过滤掉所有相关的张量和列表
            #    只有掩码中为 True 的行才会被保留下来
            boxes = boxes[finite_mask]
            logits = logits[finite_mask]
            # phrases 是一个列表，需要用列表推导式配合掩码来过滤
            phrases = [p for i, p in enumerate(phrases) if finite_mask[i]]

            if boxes.shape[0] == 0:
                logger.error("检测失败：所有原始边界框都包含 NaN/Inf，过滤后无有效框。")
                return {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": "Detection failed: All generated bounding boxes contained invalid data and were discarded.",
                    "error_code": TOOL_RUN_FAILED,
                    "tool_reward": tool_reward + correct_param_content_num / required_keys_num,
                }

            # 经过上述过滤后，所有数据都是干净的，可以继续后续处理
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)

            # ---------------------------------------------------

            # 验证boxes有没有问题 (这部分逻辑依然有用，用于检查坐标是否在[0,1]范围内)
            original_detection_count = boxes.shape[0]
            valid_boxes = []
            valid_logits = []
            valid_phrases = []

            boxes_list_normalized = boxes.tolist()
            logits_list_normalized = logits.tolist()

            for i, box in enumerate(boxes_list_normalized):
                x_min, y_min, x_max, y_max = box
                # 在归一化坐标上进行验证
                if 0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1:
                    valid_boxes.append(box)
                    valid_logits.append(logits_list_normalized[i])
                    valid_phrases.append(phrases[i])
                else:
                    logger.warning(f"过滤掉范围不正确的边界框: {box}")
            
            # --- 新增：失败条件检查 ---
            # 如果模型有检测结果，但所有结果的坐标都无效，则触发此条件
            if original_detection_count > 0 and len(valid_boxes) == 0:
                logger.error("检测失败：所有检测到的边界框坐标均异常。")
                return {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": "Detection failed: No valid objects were detected because the model generated anomalous results for all bounding boxes.",
                    "error_code": TOOL_RUN_FAILED,
                    "tool_reward": tool_reward + correct_param_content_num / required_keys_num,
                    "image_dimensions_pixels": {"width": w, "height": h}
                }
###
            # --- 使用通过验证的、干净的数据继续执行 ---
            h, w, _ = image_np.shape
            
            detections = []
            # 这个循环现在只处理有效的数据
            for i in range(len(valid_boxes)):
                if valid_logits[i] < box_threshold:
                    continue
                
                x_min = int(valid_boxes[i][0] * w)
                y_min = int(valid_boxes[i][1] * h)
                x_max = int(valid_boxes[i][2] * w)
                y_max = int(valid_boxes[i][3] * h)
                detections.append({
                    "label": valid_phrases[i],
                    "confidence": round(valid_logits[i], 2),
                    "bbox": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max
                    }
                })
            
            # 仅使用有效的检测框来生成标注图像
            annotated_image = self.annotate_image(image_np, valid_boxes, valid_logits, valid_phrases)
            annotated_image_base64 = self.image_to_base64(annotated_image)
            
            correct_param_content_num += 1

            # ... (构建成功的 pred_dict 的其余部分) ...
            message_text = f"成功检测到 {len(detections)} 个物体。"
            if len(detections) == 0:
                # 使用英文
                message_text = "Successfully processed the image, but no objects matching the description were detected."

            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "success",
                "detections": detections,
                "image_dimensions_pixels": {"width": w, "height": h},
                "edited_image": annotated_image_base64,
                "message": message_text,
                "error_code": SUCCESS,
                "tool_reward": tool_reward + correct_param_content_num / required_keys_num
            }
            return pred_dict
            
        except Exception as e:
            # 检查是否已获取图像尺寸
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error: {str(e)}\nTraceback:{traceback.format_exc()}\n",
                "error_code": TOOL_RUN_FAILED,
                "tool_reward": tool_reward+correct_param_content_num/required_keys_num
            }
            
            # 如果在异常发生前已经获取了图像尺寸，则添加到返回结果中
            if 'h' in locals() and 'w' in locals():
                pred_dict["image_dimensions_pixels"] = {"width": w, "height": h}
                
            logger.error(f"Error during GroundingDINO inference: {e}")
            logger.error(traceback.format_exc())
            return pred_dict
    
    # 使用没有torch.inference_mode装饰器的版本作为generate
    def generate(self, params):
        """为了与base_tool_worker兼容，提供一个generate方法"""
        return self._generate_impl(params)
    
    def get_tool_instruction(self):
        return self.instruction
        
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
                future = self.thread_pool.submit(self._generate_with_torch_inference, params)
                futures.append(future)
            
            # 收集所有结果
            for future in futures:
                try:
                    result = future.result(timeout=600000)
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
    parser = HfArgumentParser((GroundingDinoArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")

    worker = GroundingDinoWorker(
        worker_arguments=args
    )
    worker.run()
