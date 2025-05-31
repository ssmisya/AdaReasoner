import torch
import numpy as np
from PIL import Image
import base64
import argparse
import torchvision
import uuid
import os
from io import BytesIO
from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

import groundingdino.datasets.transforms as T


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"grounding_dino_worker_{worker_id}.log")

class GroundingDinoWorker(BaseToolWorker):
    def __init__(self, 
                 controller_addr, 
                 worker_addr="auto",
                 worker_id=worker_id, 
                 no_register=False,
                 model_path="/mnt/petrelfs/songmingyang/songmingyang/model/mm/groundingdino_official/groundingdino_swint_ogc.pth", 
                 model_config="/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                 model_name="grounding_dino",
                 load_8bit=False,
                 load_4bit=False,
                 device="cuda",
                 limit_model_concurrency=5,
                 host="0.0.0.0",
                 port=None,
                 model_semaphore=None,
                 wait_timeout=120.0,
                 task_timeout=30.0,
                 ):
        self.model_config = model_config
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            model_path,
            None,
            model_name,
            load_8bit,
            load_4bit,
            device,
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

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
        text_prompt = params.get("param", params.get("caption", ""))
        image_path = params.get("image")
        box_threshold = params.get("box_threshold", 0.25)
        text_threshold = params.get("text_threshold", 0.25)
        
        if not image_path or not text_prompt:
            logger.error("Missing required parameters: image or text prompt")
            return {"text": "Missing required parameters: image or text prompt", "error_code": 1}

        try:
            # Load image and run model
            image_np, image = self.load_image(image_path)
            boxes, logits, phrases = predict(
                model=self.model, 
                image=image, 
                caption=text_prompt, 
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
            pred_dict = {
                "boxes": boxes,
                "logits": logits,
                "phrases": phrases,
                "size": [h, w],  # H,W
            }
            return {"text": pred_dict, "error_code": 0}
            
        except Exception as e:
            logger.error(f"Error during GroundingDINO inference: {e}")
            return {"text": f"Error during GroundingDINO inference: {e}", "error_code": 1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20003)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://SH-IDCA1404-10-140-54-119:20001")
    parser.add_argument("--model-path", type=str, default="/mnt/petrelfs/songmingyang/songmingyang/model/mm/groundingdino_official/groundingdino_swint_ogc.pth")
    parser.add_argument("--model-config", type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--model-name", type=str, default="grounding_dino")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = GroundingDinoWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_path=args.model_path,
        model_config=args.model_config,
        model_name=args.model_name,
        device=args.device,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()