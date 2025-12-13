# get_obstacles.py
from PIL import Image

from tool_server.utils.server_utils import build_logger
from tool_server.utils.utils import *
from tool_server.tool_workers.offline_workers.base_offline_worker import BaseOfflineWorker
from tool_server.utils.error_codes import *

logger = build_logger("get_obstacles_worker")

class GetObstacles(BaseOfflineWorker):
    """
    获取障碍物位置的工具
    通过识别图像中的障碍物标记来确定所有障碍物的坐标
    """
    
    def __init__(self):
        super().__init__(model_name="GetObstacles")
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Identify all obstacle locations in the maze image by detecting hazard markers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The image to analyze (image identifier)"
                        }
                    },
                    "required": ["image"]
                }
            }
        }

    def _execute(self, params):
        """执行障碍物检测"""
        try:
            # 提取参数
            image = params["image"]
            tool_manager = params.get("tool_manager")
            
            if tool_manager is None:
                return {
                    "status": "failed",
                    "message": "tool_manager is required but not provided",
                    "error_code": INVALID_PARAMETERS
                }
            
            # 检查 Point 工具是否可用
            if "Point" not in tool_manager.available_tools:
                return {
                    "status": "failed",
                    "message": "Point tool is not available in tool_manager",
                    "error_code": TOOL_NOT_FOUND
                }
                
                
                
            image_base64 = load_image(image)
            image_base64 = pil_to_base64(image)
            
            # 调用 Point 工具来定位 "Ice Holes"
            point_params = {
                "image": image_base64,
                "description": "Ice Holes"
            }
            
            logger.info("Calling Point tool to locate Ice Holes...")
            point_result = tool_manager.call_tool("Point", point_params)
            
            # 检查 Point 工具的返回结果
            if point_result.get("status") != "success":
                error_msg = point_result.get("message", "Point tool failed")
                return {
                    "status": "failed",
                    "message": f"Failed to locate obstacles: {error_msg}",
                    "error_code": point_result.get("error_code", TOOL_RUN_FAILED)
                }
            
            # 提取所有障碍物点坐标
            obstacles = point_result.get("points", [])
            
            # 即使没有检测到障碍物，也返回成功（可能是没有障碍物的迷宫）
            if not obstacles or len(obstacles) == 0:
                logger.info("No obstacles detected in the image")
                obstacles = []
            
            # 准备成功响应
            result = {
                "status": "success",
                "obstacles": obstacles,
                "obstacle_count": len(obstacles),
                "message": f"Detected {len(obstacles)} obstacle(s)",
                "error_code": SUCCESS
            }
            
            # 如果 Point 返回了带标注的图像，也一并返回
            if "edited_image" in point_result:
                result["edited_image"] = point_result["edited_image"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting obstacles: {str(e)}")
            return {
                "status": "failed",
                "message": f"Error detecting obstacles: {str(e)}",
                "error_code": TOOL_RUN_FAILED
            }
    
    def verify_tool_parameter(self, params):
        """验证工具的输入参数"""
        try:
            # 检查必要参数
            if "image" not in params:
                raise ValueError("Missing required parameter: image")
            
            # 加载图像
            image = params["image"]
            loaded_image = load_image(image)
            
            # 创建新的参数字典
            new_params = {
                "image": loaded_image
            }
            
            # 验证通过
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