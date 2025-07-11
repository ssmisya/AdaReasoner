"""
A model worker executes the model.
"""

import uuid
import os
import re
import io
import argparse
import torch
import numpy as np
from PIL import Image
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
import matplotlib.pyplot as plt
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from io import BytesIO
import base64
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from tool_server.utils.worker_arguments import WorkerArguments
from tool_server.utils.error_codes import *  # Import error codes


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"sam_around_point_worker_{worker_id}.log")
model_semaphore = None

import numpy as np

np.random.seed(3)

@dataclass
class SAMAroundPointArguments(WorkerArguments):
    sam2_checkpoint: str = field(
        default="/mnt/petrelfs/haoyunzhuo/mmtool/checkpoints/sam2_hiera_large.pt",
        metadata={"help": "Path to the SAM2 checkpoint file"}
    )
    sam2_model_cfg: str = field(
        default="sam2_hiera_l.yaml",
        metadata={"help": "Path to the SAM2 model configuration file"}
    )
    max_concurrency: Optional[int] = field(
        default=10,
        metadata={"help": "Maximum number of concurrent requests to process."}
    )

def extract_points(generate_param, image_w, image_h):
    """Extract a single point from the description. Only accepts one point in format 'x=50, y=50'."""
    # 支持的格式：
    # x=50, y=50 (绝对坐标)
    
    # 格式：x=50, y=50 (绝对坐标)
    pattern_absolute = r'x\s*=\s*([0-9]+(?:\.[0-9]*)?)\s*,\s*y\s*=\s*([0-9]+(?:\.[0-9]*)?)' 
  
    try:
        # 尝试匹配绝对坐标格式
        matches = list(re.finditer(pattern_absolute, generate_param))
        if not matches:
            raise ValueError("Invalid point coordinates format. Please use the format 'x=50, y=50' with absolute coordinates.")
        
        # 只取第一个匹配，忽略其他可能的点
        if len(matches) > 1:
            logger.warning(f"Multiple points detected. Only the first point will be used.")
            
        match = matches[0]
        x = float(match.group(1))
        y = float(match.group(2))
        
        # 检查坐标是否在图像范围内
        if 0 <= x < image_w and 0 <= y < image_h:
            return np.array([[x, y]])
        else:
            # 坐标超出图像范围
            raise ValueError(f"Point coordinates ({x}, {y}) out of image bounds ({image_w}x{image_h})")
            
    except Exception as e:
        message = f"Invalid point coordinates format. Please use the format 'x=50, y=50' with absolute coordinates. Error: {str(e)}"
        pred_dict = {
            "tool_response_from": "SegmentRegionAroundPoint",
            "status": "failed",
            "message": message,
            "error_code": INVALID_PARAMETERS
        }
        return pred_dict

    return []

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    fig, ax = plt.subplots()
    ax.imshow(image)
    image_format = image.format.lower() if image.format else 'png'
    if image_format not in ['png', 'jpeg', 'jpg']:
        image_format = 'png'
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        show_mask(mask, ax, random_color=True, borders=borders)
        if len(scores) > 1:
            ax.set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    
    if point_coords is not None and input_labels is not None:
        show_points(point_coords, input_labels, ax)
    
    if box_coords is not None:
        show_box(box_coords, ax)
    
    plt.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format=image_format, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    
    edited_image = Image.open(buf).convert("RGB")
    
    return edited_image

def segment_everything(predictor, img):
    """对整个图像进行自动分割，无需输入点坐标"""
    # 获取图像尺寸
    height, width = np.array(img).shape[:2]
    
    # 自动分割模式
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=None,
        multimask_output=True,  # 返回多个掩码
    )
    
    # 根据分数对掩码进行排序
    if len(masks) > 0:
        indices = np.argsort(-scores)  # 按分数降序排序
        masks = masks[indices]
        scores = scores[indices]
        
        # 限制返回的掩码数量（避免过多）
        max_masks = min(5, len(masks))  # 最多显示5个掩码
        masks = masks[:max_masks]
        scores = scores[:max_masks]
    
    return masks, scores

class SAMAroundPointWorker(BaseToolWorker):
    def __init__(self, worker_arguments: SAMAroundPointArguments = None):
        # 在调用父类初始化前先设置模型名称和必要的属性
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "SegmentRegionAroundPoint"
        
        # 设置更高的并发处理能力
        if hasattr(worker_arguments, "max_concurrency"):
            worker_arguments.limit_model_concurrency = worker_arguments.max_concurrency
        else:
            worker_arguments.limit_model_concurrency = 10  # 默认允许10个并发请求
        
        # 将这两行移到super().__init__之前
        self.sam2_checkpoint = worker_arguments.sam2_checkpoint
        self.sam2_model_cfg = worker_arguments.sam2_model_cfg
        
        super().__init__(worker_arguments)
        
        # 创建线程池用于并发处理请求
        self.thread_pool = ThreadPoolExecutor(max_workers=worker_arguments.limit_model_concurrency)
        
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Segments objects in an image. Can perform automatic segmentation on the entire image or segment a specific object based on a single designated point. Returns the image with segmentation masks and related processing info.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier of the image to edit, e.g., 'img_1'"
                        },
                        "coordinates": {
                            "type": "string",
                            "description": "Optional: Single point coordinates in format 'x=value1, y=value2', eg., 'x=50, y=100'. Using absolute pixel coordinates within image bounds. If not provided, the tool will automatically segment all objects in the image."
                        }
                    },
                    "required": ["image", "coordinates"]
                }
            }
        }



    def init_model(self):
        logger.info(f"Initializing model {self.model_name} with concurrency {self.args.limit_model_concurrency}...")
        logger.info(f"CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info(f"Using device: {device}")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            logger.info(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS."
            )

        self.sam2_model = build_sam2(self.sam2_model_cfg, self.sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
    
    def generate_gate(self, params):
        """覆盖父类方法，使用线程池处理请求"""
        try:
            # 使用线程池提交任务
            future = self.thread_pool.submit(self._generate_with_torch_inference, params)
            ret = future.result(timeout=120)  # 设置超时时间
            return ret
        except Exception as e:
            logger.error(f"Error in generate_gate: {e}")
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error in generate_gate: {e}",
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
        if "coordinates" in param_keys:
            required_keys = set(self.instruction["function"]["parameters"]["required"])
            parameter_name_match_reward = len(param_keys & required_keys) / len(required_keys | param_keys)
            tool_reward = tool_reward + parameter_name_match_reward
            # 参数名称没有完全匹配，直接返回
            if parameter_name_match_reward < 1:
                return {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": "Invalid parameters: expected keys: image, coordinates.",
                    "error_code": INVALID_PARAMETERS,
                    "tool_reward": tool_reward
                }
        else:
            required_keys = set(['image'])
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
        
        try:
            # Extract inputs
            if "coordinates" in param_keys:
                image_data = params["image"]
                point_param = params.get("coordinates", "")
            else:
                image_data = params["image"]
                point_param = None
            
            # Load and process the image
            try:
                img = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
            except Exception as e:
                return {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Error: {str(e)} Traceback:{traceback.format_exc()}",
                    "error_code": CANNOT_LOAD_IMAGE,
                    "tool_reward": tool_reward + correct_param_content_num/required_keys_num
                }
            try:
                correct_param_content_num += 1
                width, height = img.size
                self.predictor.set_image(img)

                # 检查是否有点坐标
                point_result = extract_points(point_param, width, height) if point_param else []
                
                # 如果extract_points返回了错误信息（字典类型）
                if isinstance(point_result, dict) and point_result.get("status") == "failed":
                    point_result["tool_reward"] = tool_reward + correct_param_content_num/required_keys_num
                    return point_result
                
                # 处理提取的点
                points = point_result
                
                # 坐标参数验证通过
                if point_param:
                    correct_param_content_num += 1

                if len(points) > 0:
                    # 有点坐标，执行点周围区域分割
                    input_labels = np.ones(len(points))
                    masks, scores, _ = self.predictor.predict(
                        point_coords=points,
                        point_labels=input_labels,
                        box=None,
                        multimask_output=False,
                    )
                    
                    # 生成可视化结果
                    edited_img = show_masks(img, masks, scores, point_coords=points, input_labels=input_labels)
                else:
                    # 没有点坐标，执行整图分割
                    logger.info("No point coordinates provided, performing automatic segmentation on the entire image.")
                    masks, scores = segment_everything(self.predictor, img)
                    
                    if len(masks) == 0:
                        return {
                            "tool_response_from": self.model_name,
                            "status": "failed",
                            "message": "Automatic segmentation did not produce any valid masks.",
                            "error_code": TOOL_RUN_FAILED,
                            "tool_reward": tool_reward + correct_param_content_num/required_keys_num
                        }
                    
                    # 生成带有所有分割掩码的可视化结果
                    edited_img = show_masks(img, masks, scores, borders=True)
                
                # Prepare the response
                buffered = BytesIO()
                image_format = img.format if img.format else 'PNG'
                edited_img.save(buffered, format=image_format)
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # 准备响应
                response = {
                    "tool_response_from": self.model_name,
                    "status": "success",
                    "edited_image": img_str,
                    "image_dimensions_pixels": {
                        "width": width,
                        "height": height
                    },
                    "error_code": SUCCESS,
                    "tool_reward": tool_reward + correct_param_content_num/required_keys_num
                }
                
                # 添加分割模式信息
                if len(points) > 0:
                    response["segmentation_mode"] = "point_based"
                    
                    # 添加使用的点坐标（绝对坐标）
                    point = points[0]  # 只有一个点
                    response["point_used"] = {
                        "x": float(point[0]),
                        "y": float(point[1])
                    }
                else:
                    response["segmentation_mode"] = "automatic"
                    response["mask_count"] = len(masks)
                
                return response
                
            except Exception as e:
                return {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Error: {str(e)} Traceback:{traceback.format_exc()}",
                    "error_code": CANNOT_LOAD_IMAGE,
                    "tool_reward": tool_reward + correct_param_content_num/required_keys_num
                }
                
        except Exception as e:
            logger.error(f"Error during SAM Around Point inference: {e}")
            logger.error(traceback.format_exc())
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error: {str(e)} Traceback:{traceback.format_exc()}",
                "error_code": TOOL_RUN_FAILED,
                "tool_reward": tool_reward + (correct_param_content_num/required_keys_num if required_keys_num > 0 else 0)
            }
    
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
                    result = future.result(timeout=120)
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
    parser = HfArgumentParser((SAMAroundPointArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")
    
    worker = SAMAroundPointWorker(worker_arguments=args)
    worker.run()