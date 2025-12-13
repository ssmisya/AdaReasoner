# rotate_image.py
from PIL import Image
import ast

from tool_server.utils.server_utils import build_logger
from tool_server.utils.utils import load_image, pil_to_base64
from tool_server.tool_workers.offline_workers.base_offline_worker import BaseOfflineWorker
from tool_server.utils.error_codes import *

logger = build_logger("rotate_image_worker")

class RotateImage(BaseOfflineWorker):
    """
    旋转图像指定角度
    支持90度、180度、270度旋转
    """
    
    def __init__(self):
        super().__init__(model_name="RotateImage")
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Rotate an image by a specified angle (90, 180, or 270 degrees clockwise)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The image to rotate (image identifier)"
                        },
                        "angle": {
                            "type": "integer",
                            "description": "Rotation angle in degrees. Must be one of: 90, 180, 270 (clockwise)",
                            "enum": [90, 180, 270]
                        }
                    },
                    "required": ["image", "angle"]
                }
            }
        }
    
    def _execute(self, params):
        """执行图像旋转"""
        try:
            # 提取参数
            image = params["image"]
            angle = params["angle"]
            
            # 确保图像是PIL Image对象
            if isinstance(image, str):
                img = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                img = image.copy().convert("RGB")
            else:
                return {
                    "status": "failed",
                    "message": "Invalid image format",
                    "error_code": INVALID_PARAMETERS
                }
            
            # 执行旋转
            # PIL的rotate方法是逆时针旋转，我们需要顺时针，所以取负数
            # 或者使用transpose方法更高效
            if angle == 90:
                rotated_image = img.transpose(Image.ROTATE_270)  # 顺时针90度
            elif angle == 180:
                rotated_image = img.transpose(Image.ROTATE_180)
            elif angle == 270:
                rotated_image = img.transpose(Image.ROTATE_90)   # 顺时针270度
            else:
                return {
                    "status": "failed",
                    "message": f"Invalid angle: {angle}. Must be 90, 180, or 270",
                    "error_code": INVALID_PARAMETERS
                }
            
            # 转换为base64并返回结果
            image_base64 = pil_to_base64(rotated_image)
            
            return {
                "status": "success",
                "edited_image": image_base64,
                "message": f"Image rotated {angle} degrees clockwise successfully",
                "original_size": {"width": img.width, "height": img.height},
                "rotated_size": {"width": rotated_image.width, "height": rotated_image.height},
                "error_code": SUCCESS
            }
            
        except Exception as e:
            logger.error(f"Error rotating image: {str(e)}")
            return {
                "status": "failed",
                "message": f"Error rotating image: {str(e)}",
                "error_code": TOOL_RUN_FAILED
            }
    
    def verify_tool_parameter(self, params):
        """验证工具的输入参数"""
        try:
            # 检查必要参数是否存在
            if "image" not in params:
                raise ValueError("Missing required parameter: image")
            if "angle" not in params:
                raise ValueError("Missing required parameter: angle")
            
            # 加载图像
            image = params["image"]
            loaded_image = load_image(image)
            
            # 验证角度参数
            angle = params["angle"]
            
            # 如果angle是字符串，尝试转换为整数
            if isinstance(angle, str):
                try:
                    angle = int(angle)
                except ValueError:
                    raise ValueError(f"angle must be an integer, got: {angle}")
            
            # 检查angle是否为有效值
            if angle not in [90, 180, 270]:
                raise ValueError(f"angle must be one of [90, 180, 270], got: {angle}")
            
            # 创建新的参数字典
            new_params = {
                "image": loaded_image,
                "angle": angle
            }
            
            # 验证通过，返回成功结果
            res = {
                "params_qualified_reward": 1,
                "params_qualified": True,
                "new_params": new_params
            }
            return res
            
        except Exception as e:
            error_info = str(e)
            res = {
                "params_qualified_reward": 0,
                "params_qualified": False,
                "error_info": error_info,
                "new_params": None,
            }
            return res
        