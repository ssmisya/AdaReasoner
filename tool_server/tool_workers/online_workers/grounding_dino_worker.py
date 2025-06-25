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

from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

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

class GroundingDinoWorker(BaseToolWorker):
    def __init__(self, worker_arguments: GroundingDinoArguments = None):
        # 在调用父类初始化前先设置模型名称
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "GroundingDINO"
        
        # 保存model_config到实例变量
        self.model_config = worker_arguments.model_config if worker_arguments else None
        
        super().__init__(worker_arguments)
            
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
                            "description": "The identifier of the image in which to locate the object, e.g., 'img_1'."
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
        logger.info(f"Initializing model {self.model_name}...")
        logger.info(f"CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}")
        try:
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
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            logger.error(traceback.format_exc())
            raise e

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
        iou_threshold = 0.8
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
        # 设置matplotlib使用非交互模式
        matplotlib.use('Agg')
        
        # 创建一个新的图形和轴
        fig, ax = plt.subplots(1)
        ax.imshow(image_np)
        
        # 获取图像尺寸
        h, w, _ = image_np.shape
        
        # 为不同类别设置不同颜色
        unique_phrases = list(set(phrases))
        colors = plt.cm.hsv(np.linspace(0, 1, len(unique_phrases) if len(unique_phrases) > 0 else 1))
        color_map = {phrase: colors[i % len(colors)] for i, phrase in enumerate(unique_phrases)}
        
        # 绘制每个检测框
        for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
            # 获取边界框坐标
            x_min, y_min, x_max, y_max = box
            # 将归一化坐标转换为像素坐标
            x_min_px, y_min_px = int(x_min * w), int(y_min * h)
            x_max_px, y_max_px = int(x_max * w), int(y_max * h)
            
            # 计算宽度和高度
            width = x_max_px - x_min_px
            height = y_max_px - y_min_px
            
            # 获取当前类别的颜色
            color = color_map.get(phrase, 'red')
            
            # 创建一个矩形
            rect = patches.Rectangle(
                (x_min_px, y_min_px), width, height, 
                linewidth=2, edgecolor=color, facecolor='none'
            )
            
            # 添加矩形到轴
            ax.add_patch(rect)
            
            # 添加标签文本
            confidence = f"{logit:.2f}"
            label = f"{phrase}: {confidence}"
            plt.text(
                x_min_px, y_min_px - 5, label, 
                bbox=dict(facecolor=color, alpha=0.5),
                fontsize=8, color='white'
            )
        
        # 移除轴标签
        plt.axis('off')
        
        # 将图形保存到内存缓冲区
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # 从缓冲区加载图像
        annotated_image = Image.open(buf).convert('RGB')
        return annotated_image
    
    def image_to_base64(self, image):
        """将PIL图像转换为base64编码的字符串"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    @torch.inference_mode()
    def generate(self, params):
        # Extract inputs
        try:
            image_path = params["image"]
            description = params.get("description")
            if not description:
                raise KeyError("缺少必要参数 'description'")
        except Exception as e:
            message = f"无效参数: 需要的参数: image, description. 错误: {str(e)}"
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": message,
                "error_code": INVALID_PARAMETERS
            }
            return pred_dict
        
        # If the params have been parsed successfully.
        box_threshold = params.get("box_threshold", 0.25)
        text_threshold = params.get("text_threshold", 0.25)
        try:
            # Load image
            try:
                image_np, image = self.load_image(image_path)
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Failed to load image: {str(e)}",
                    "error_code": CANNOT_LOAD_IMAGE
                }
                return pred_dict
            
            # Run model
            try:
                boxes, logits, phrases = predict(
                    model=self.model, 
                    image=image, 
                    caption=description, 
                    box_threshold=box_threshold, 
                    text_threshold=text_threshold,
                    device=self.device
                )
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Model inference failed: {str(e)}",
                    "error_code": TOOL_RUN_FAILED
                }
                logger.error(f"Error during GroundingDINO inference: {e}")
                logger.error(traceback.format_exc())
                return pred_dict
            
            # Apply NMS to boxes
            boxes, logits, phrases = self.nms(boxes, logits, phrases)
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)

            # Format output
            boxes_list = [[round(x, 2) for x in box] for box in boxes.tolist()]
            logits_list = [round(x, 2) for x in logits.tolist()]

            h, w, _ = image_np.shape
    
            detect_res_num = len(boxes_list)
            detections = []
            for detect_res_idx in range(detect_res_num):
                x_min = int(boxes_list[detect_res_idx][0] * w)
                y_min = int(boxes_list[detect_res_idx][1] * h)
                x_max = int(boxes_list[detect_res_idx][2] * w)
                y_max = int(boxes_list[detect_res_idx][3] * h)
                detections.append({
                    "label": phrases[detect_res_idx],
                    "confidence": logits_list[detect_res_idx],
                    "bbox": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max
                    }
                })
            
            # 生成带有边界框的图像
            annotated_image = self.annotate_image(image_np, boxes.tolist(), logits.tolist(), phrases)
            annotated_image_base64 = self.image_to_base64(annotated_image)
            
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "success",
                "detections": detections,
                "image_dimensions_pixels": {
                    "width": w,
                    "height": h
                },
                "edited_image": annotated_image_base64,
                "message": f"Successfully detected {len(detections)} objects.",
                "error_code": SUCCESS
            }
            
            return pred_dict
            
        except Exception as e:
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error: {str(e)}\nTraceback: {traceback.format_exc()}",
                "error_code": TOOL_RUN_FAILED
            }
            logger.error(f"Error during GroundingDINO inference: {e}")
            logger.error(traceback.format_exc())
            return pred_dict
        
    
    def get_tool_instruction(self):
        return self.instruction  


if __name__ == "__main__":
    parser = HfArgumentParser((GroundingDinoArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")

    worker = GroundingDinoWorker(
        worker_arguments=args
    )
    worker.run()
