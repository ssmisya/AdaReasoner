import torch
import numpy as np
from PIL import Image
import base64
import uuid
import os
import traceback
import re
from io import BytesIO
import matplotlib.pyplot as plt
import sys
from pathlib import Path


from transformers import HfArgumentParser, AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig
from dataclasses import dataclass, field
from typing import Optional

from tool_server.utils.server_utils import build_logger
from tool_server.utils.worker_arguments import WorkerArguments
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"molmo_point_worker_{worker_id}.log")

@dataclass
class MolmoPointArguments(WorkerArguments):
    max_length: Optional[int] = field(
        default=2048,
        metadata={"help": "Maximum length for token generation"}
    )

class MolmoPointWorker(BaseToolWorker):
    def __init__(self, worker_arguments: MolmoPointArguments = None):
        # 在调用父类初始化前先设置模型名称
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "Point"
        super().__init__(worker_arguments)
            
        self.max_length = worker_arguments.max_length
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
                            "description": "The identifier or path of the image in which to locate the point, e.g., 'img_1'."
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
        logger.info(f"Initializing model {self.model_name}...")
        logger.info(f"CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}")
        
        quant_config = None
        # 在worker_arguments中定义了load_4bit和load_8bit，可以在all_service_example.yaml中设置
        # 原本没修改的情况下，两个都是False
        if self.load_4bit or self.load_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=self.load_8bit,
                load_in_4bit=self.load_4bit,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            logger.info(f"Using quantization config: {quant_config}")

        # load the processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto',
            quantization_config=quant_config
        )
        self.model.eval()

        
    def extract_points(self, molmo_output, image_w, image_h):
        all_points = []
        for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)
        return np.array(all_points)  # Ensure it's always a NumPy array
    
    def show_points(self, coords, labels, ax, marker_size=375):
        # Only plot if there are points
        if len(coords) == 0:
            return
        
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        
        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        if len(neg_points) > 0:
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    
    def create_image_with_points(self, image, coords, labels, marker_size=375):
        fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100), dpi=100)
        ax.imshow(image)
        # 修复image.format为None的情况
        image_format = 'png'  # 默认使用png格式
        if image.format is not None:
            image_format = image.format.lower()
            if image_format not in ['png', 'jpeg', 'jpg']:
                image_format = 'png'

        # Only show points if there are any valid coordinates
        self.show_points(coords, labels, ax, marker_size)

        plt.axis('off')  # Turn off axis

        # Convert the figure to a PIL image
        buf = BytesIO()
        plt.savefig(buf, format=image_format, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    @torch.inference_mode()
    def generate(self, params):
        # Extract inputs
        try:
            image_data = params["image"]
            description = params.get("description", params.get("param", ""))  # 兼容两种参数名
            if not description:
                raise KeyError("Neither 'description' nor 'param' found in params")
        except Exception as e:
            message = f"Invalid parameters: expected keys: image, description or param. Error: {str(e)}"
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": message,
            }
            return pred_dict
        
        try:
            # Convert base64 to PIL image
            if os.path.exists(image_data):
                image = Image.open(image_data).convert("RGB")
            else:
                image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
            
            text_prompt = f"Point to the {description} in the scene."
            
            # Process inputs and run model
            with torch.no_grad():
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
                    
                    # Only get generated tokens and decode them to text
                    generated_tokens = output[0, inputs['input_ids'].size(1):]
                    response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Extract points from the response
            if response:
                points = self.extract_points(response, image.width, image.height)
                
                # Create point representations
                point_data = []
                if len(points) > 0:
                    for point in points:
                        point_data.append({
                            "x": float(point[0]),
                            "y": float(point[1]),
                        })
                
                # 创建带有标记点的图片
                # 在create_image_with_points中，coords应该是点的数组，labels决定点的颜色
                # 这里我们默认所有点都是正向的（绿色），因为我们没有负向标签信息
                if len(points) > 0:
                    labels = np.ones(len(points), dtype=np.int32)  # 设置所有点为正向 (1)
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
                        "raw_response": response,
                        "image_with_points": img_str,
                        "image_dimensions_pixels": {
                            "width": image.width,
                            "height": image.height
                        },
                    }
                else:
                    pred_dict = {
                        "tool_response_from": self.model_name,
                        "status": "success",
                        "points": point_data,
                        "raw_response": response,
                        "message": "No valid points detected in the response.",
                        "image_dimensions_pixels": {
                            "width": image.width,
                            "height": image.height
                        },
                    }
                
                return pred_dict
            else:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": "Model did not generate any response."
                }
                return pred_dict
                
        except Exception as e:
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            logger.error(f"Error during Molmo Point inference: {e}")
            logger.error(traceback.format_exc())
            return pred_dict
        
    
    def get_tool_instruction(self):
        return self.instruction  


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = HfArgumentParser((MolmoPointArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")

    worker = MolmoPointWorker(
        worker_arguments=args
    )
    worker.run() 