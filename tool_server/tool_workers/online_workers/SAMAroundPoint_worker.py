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
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from tool_server.utils.worker_arguments import WorkerArguments


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

def extract_points(generate_param, image_w, image_h):
    all_points = []
    pattern = r'x\d*=\s*\\?"?([0-9]+(?:\.[0-9]*)?)\\?"?\s*y\d*=\s*\\?"?([0-9]+(?:\.[0-9]*)?)\\?"?'
    
    for match in re.finditer(pattern, generate_param):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            continue
        else:
            point = np.array(point)
            if np.max(point) > 100:
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    
    return all_points

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
        show_mask(mask, ax, borders=borders)
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

class SAMAroundPointWorker(BaseToolWorker):
    def __init__(self, worker_arguments: SAMAroundPointArguments = None):
        # 在调用父类初始化前先设置模型名称和必要的属性
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "SegmentRegionAroundPoint"
        
        # 将这两行移到super().__init__之前
        self.sam2_checkpoint = worker_arguments.sam2_checkpoint
        self.sam2_model_cfg = worker_arguments.sam2_model_cfg
        
        super().__init__(worker_arguments)
        
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Segment the region around a given point in the image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The base64 encoded image or path to the image file."
                        },
                        "param": {
                            "type": "string",
                            "description": "The point coordinates in format 'x=50 y=50' (as percentage of image dimensions)."
                        }
                    },
                    "required": ["image", "param"]
                }
            }
        }

    def init_model(self):
        logger.info(f"Initializing model {self.model_name}...")
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

    @torch.inference_mode()
    def generate(self, params):
        try:
            # Extract inputs
            try:
                image_data = params["image"]
                point_param = params.get("param", "")
                if not point_param:
                    raise KeyError("Required parameter 'param' not found in params")
            except Exception as e:
                message = f"Invalid parameters: expected keys: image, param. Error: {str(e)}"
                return {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": message,
                }
            
            # Load and process the image
            try:
                if os.path.exists(image_data):
                    img = Image.open(image_data).convert("RGB")
                else:
                    img = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
                
                width, height = img.size
                self.predictor.set_image(img)

                # Extract points from input parameters
                points = extract_points(point_param, width, height)

                if not points:
                    return {
                        "tool_response_from": self.model_name,
                        "status": "failed",
                        "message": "No valid points extracted from the parameters."
                    }

                # Run segmentation
                input_labels = np.ones(len(points))
                masks, scores, _ = self.predictor.predict(
                    point_coords=points,
                    point_labels=input_labels,
                    box=None,
                    multimask_output=False,
                )

                # Generate the visualization
                edited_img = show_masks(img, masks, scores, point_coords=np.array(points), input_labels=input_labels)
                
                # Prepare the response
                buffered = BytesIO()
                image_format = img.format if img.format else 'PNG'
                edited_img.save(buffered, format=image_format)
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                return {
                    "tool_response_from": self.model_name,
                    "status": "success",
                    "edited_image": img_str,
                    "image_dimensions_pixels": {
                        "width": width,
                        "height": height
                    },
                }
                
            except Exception as e:
                return {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
        except Exception as e:
            logger.error(f"Error during SAM Around Point inference: {e}")
            logger.error(traceback.format_exc())
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def get_tool_instruction(self):
        return self.instruction

if __name__ == "__main__":
    parser = HfArgumentParser((SAMAroundPointArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")
    
    worker = SAMAroundPointWorker(worker_arguments=args)
    worker.run()