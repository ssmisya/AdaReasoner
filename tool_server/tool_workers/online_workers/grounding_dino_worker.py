import torch
import numpy as np
from PIL import Image
import base64
import uuid
import os
import torchvision
from io import BytesIO

from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops

from tool_server.utils.server_utils import build_logger
from tool_server.utils.worker_arguments import WorkerArguments
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
        super().__init__(worker_arguments)
        if self.model_name is None:
            self.model_name = "grounding_dino"
            
        self.model_config = worker_arguments.model_config
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Locate objects in the image based on a natural language description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier or path of the image in which to locate the object, e.g., 'img_1'."
                        },
                        "description": {
                            "type": "string",
                            "description": "A natural language description of the object to locate, e.g., 'a red car', 'a man holding a dog'."
                        }
                    },
                    "required": ["image", "description"]
                },
                # "returns": {
                #     "edited_image": "An image with bounding boxes drawn around the located objects.",
                # }
            }
        }

    def init_model(self):
        logger.info(f"Initializing model {self.model_name}...")
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
        iou_threshold = 0.8
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
        logger.info(f"Before NMS: {boxes_xyxy.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_xyxy, logits, iou_threshold)

        boxes = boxes[nms_idx]
        logits = logits[nms_idx]
        phrases = [phrases[idx] for idx in nms_idx]
        logger.info(f"After NMS: {boxes.shape[0]} boxes")

        return boxes, logits, phrases
    
    @torch.inference_mode()
    def generate(self, params):
        # Extract inputs
        try:
            image_path = params["image"]
            description = params["description"]
        except:
            message = f"Invalid parameters: expected keys: image, description. Please reference the tool instruction: {self.get_tool_instruction()}"
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": message,
            }
            return pred_dict
        
        # If the params have been parsed successfully.
        box_threshold = params.get("box_threshold", 0.25)
        text_threshold = params.get("text_threshold", 0.25)
        try:
            # Load image and run model
            image_np, image = self.load_image(image_path)
            boxes, logits, phrases = predict(
                model=self.model, 
                image=image, 
                caption=description, 
                box_threshold=box_threshold, 
                text_threshold=text_threshold,
                device=self.device
            )
            
            # Apply NMS to boxes
            boxes, logits, phrases = self.nms(boxes, logits, phrases)
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)

            # Format output
            boxes = [[round(x, 2) for x in box] for box in boxes.tolist()]
            logits = [round(x, 2) for x in logits.tolist()]

            h, w, _ = image_np.shape
    
            detect_res_num = len(boxes)
            detections = []
            for detect_res_idx in range(detect_res_num):
                x_min = int(boxes[detect_res_idx][0] * w)
                y_min = int(boxes[detect_res_idx][1] * h)
                x_max = int(boxes[detect_res_idx][2] * w)
                y_max = int(boxes[detect_res_idx][3] * h)
                detections.append({
                    "label": phrases[detect_res_idx],
                    "confidence": logits[detect_res_idx],
                    "normalized_bbox": {
                        "x_min": boxes[detect_res_idx][0],
                        "y_min": boxes[detect_res_idx][1],
                        "x_max": boxes[detect_res_idx][2],
                        "y_max": boxes[detect_res_idx][3],
                    },
                    "pixel_bbox": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max
                    }
                })
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "success",
                "detections": detections,
                "image_dimensions_pixels": {
                    "width": w,
                    "height": h
                },
            }
            
            return pred_dict
            
        except Exception as e:
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"Error during GroundingDINO inference: {e}")
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
